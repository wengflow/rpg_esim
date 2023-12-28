#pragma once

#include <complex>
#include <vector>
#include <Eigen/Core>
#include <esim/common/types.hpp>

namespace control {

using Scalar = FloatType;

using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using RowVectorXs = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

using ArrayXXs = Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using ArrayXs = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
using Array1Xs = Eigen::Array<Scalar, 1, Eigen::Dynamic>;

using ComplexScalar = std::complex<Scalar>;
using ArrayXXcs = Eigen::Array<ComplexScalar, Eigen::Dynamic, Eigen::Dynamic>;
using ArrayXcs = Eigen::Array<ComplexScalar, Eigen::Dynamic, 1>;
using Array1Xcs = Eigen::Array<ComplexScalar, 1, Eigen::Dynamic>;

/* 
 * Linear Time-Invariant (LTI) system state-space model in standard form:
 *    1. Continuous
 *        x-dot(t) = A x(t) + B u(t)
 *            y(t) = C x(t) + D u(t)
 *    2. Discrete
 *        x[k+1] = A x[k] + B u[k]
 *          y[k] = C x[k] + D u[k]
 * or discrete LTI system state-space model in non-standard form:
 *        x[k+1] = A x[k] + B u[k] + B-tilde u[k+1]
 *          y[k] = C x[k] + D u[k]
 */
class LtiStateSpace{
 public:
  LtiStateSpace() = default;
  LtiStateSpace(const Eigen::Ref<const MatrixXs>& A,
                const Eigen::Ref<const MatrixXs>& B,
                const Eigen::Ref<const MatrixXs>& C,
                const Eigen::Ref<const MatrixXs>& D);
  LtiStateSpace(const Eigen::Ref<const MatrixXs>& A,
                const Eigen::Ref<const MatrixXs>& B,
                const Eigen::Ref<const MatrixXs>& B_tilde,
                const Eigen::Ref<const MatrixXs>& C,
                const Eigen::Ref<const MatrixXs>& D);
  bool isInit() const;
  unsigned int N_u() const;
  unsigned int N_x() const;
  unsigned int N_y() const;

  // TODO: prevent invalid modifications
  MatrixXs A;                                                                     // (N_x(), N_x())
  MatrixXs B;                                                                     // (N_x(), N_u())
  MatrixXs B_tilde;                                                               // (N_x(), N_u())
  MatrixXs C;                                                                     // (N_y(), N_x())
  MatrixXs D;                                                                     // (N_y(), N_u())
};

/* 
 * Batched variant of `LtiStateSpace`. For a batch size of `N_b`:
 *    1. `A[i]` gives an `N_b x N_x` array, where each row corresponds to the
 *       i-th row of `A` for different `LtiStateSpace` instances in the batch,
 *       and `0 <= i < N_x`
 *    2. `B/B_tilde[i]` gives an `N_b x N_u` array, where each row corresponds
 *       to the i-th row of `B/B_tilde` for different `LtiStateSpace` instances
 *       in the batch, and `0 <= i < N_x`
 *    3. `C[i]` gives an `N_b x N_x` array, where each row corresponds to the
 *       i-th row of `C` for different `LtiStateSpace` instances in the batch,
 *       and `0 <= i < N_y`
 *    4. `D[i]` gives an `N_b x N_u` array, where each row corresponds to the
 *       i-th row of `A` for different `LtiStateSpace` instances in the batch,
 *       and `0 <= i < N_y`
 */
class BatchLtiStateSpace{
 public:
  BatchLtiStateSpace() = default;
  BatchLtiStateSpace(const std::vector<ArrayXXs>& A,
                     const std::vector<ArrayXXs>& B,
                     const std::vector<ArrayXXs>& C,
                     const std::vector<ArrayXXs>& D);
  BatchLtiStateSpace(const std::vector<ArrayXXs>& A,
                     const std::vector<ArrayXXs>& B,
                     const std::vector<ArrayXXs>& B_tilde,
                     const std::vector<ArrayXXs>& C,
                     const std::vector<ArrayXXs>& D);
  bool isInit() const;
  unsigned int N_b() const;
  unsigned int N_u() const;
  unsigned int N_x() const;
  unsigned int N_y() const;

  // TODO: prevent invalid modifications
  std::vector<ArrayXXs> A;                                                        // (N_x()) vector of (N_b(), N_x()) arrays
  std::vector<ArrayXXs> B;                                                        // (N_x()) vector of (N_b(), N_u()) arrays
  std::vector<ArrayXXs> B_tilde;                                                  // (N_x()) vector of (N_b(), N_u()) arrays
  std::vector<ArrayXXs> C;                                                        // (N_y()) vector of (N_b(), N_x()) arrays
  std::vector<ArrayXXs> D;                                                        // (N_y()) vector of (N_b(), N_u()) arrays
};

/* 
 * Let the continuous LTI system state be `x`, input be `u`, and First-Order
 * Hold (FOH)-discretized LTI system state be `xi`. If `is_state_preserved`,
 * then `xi[k] = x[k]`. Else, `xi[k] = x[k] - gamma2 * u[k]`.
 *
 * References:
 *   1. G. F. Franklin, J. D. Powell, and M. L. Workman, Digital control of 
 *      dynamic systems, 3rd ed. Menlo Park, Calif: Addison-Wesley, 
 *      pp. 204-206, 1998.
 *   2. `scipy.signal.cont2discrete(method='foh')`
 *      (https://github.com/scipy/scipy/blob/v1.11.3/scipy/signal/_lti_conversion.py#L498-L518)
 */
LtiStateSpace fohCont2discrete(const LtiStateSpace& cont_sys, FloatType dt,
                               bool is_state_preserved = false);

class DiscreteLtiFilter{
 public:
  DiscreteLtiFilter() = default;
  DiscreteLtiFilter(const LtiStateSpace& system,
                    const Eigen::Ref<const MatrixXs>& init_input,
                    const Eigen::Ref<const MatrixXs>& init_state);
  void update(const Eigen::Ref<const MatrixXs>& next_input);
  void update(MatrixXs&& next_input);
  MatrixXs output() const;
  MatrixXs output(const Eigen::Ref<const MatrixXs>& C,
                  const Eigen::Ref<const MatrixXs>& D) const;

  LtiStateSpace system;
  MatrixXs input;                                                                 // (system.N_u() [, N_b])
  MatrixXs state;                                                                 // (system.N_x() [, N_b])
};

class BatchDiscreteLtiFilter{
 public:
  BatchDiscreteLtiFilter() = default;
  BatchDiscreteLtiFilter(const BatchLtiStateSpace& system,
                         const Eigen::Ref<const MatrixXs>& init_input,
                         const Eigen::Ref<const MatrixXs>& init_state);
  void update(const Eigen::Ref<const MatrixXs>& next_input);
  void update(MatrixXs&& next_input);
  MatrixXs output() const;
  MatrixXs output(const std::vector<ArrayXXs>& C,
                  const std::vector<ArrayXXs>& D) const;

  BatchLtiStateSpace system;
  MatrixXs input;                                                                 // (system.N_u(), system.N_b())
  MatrixXs state;                                                                 // (system.N_x(), system.N_b())
};

} // namespace control
