#pragma once

#include <Eigen/Core>
#include <esim/common/types.hpp>

namespace control {

using MatrixXs = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXs = Eigen::Matrix<FloatType, Eigen::Dynamic, 1>;
using RowVectorXs = Eigen::Matrix<FloatType, 1, Eigen::Dynamic>;

using ArrayXXs = Eigen::Array<FloatType, Eigen::Dynamic, Eigen::Dynamic>;
using ArrayXs = Eigen::Array<FloatType, Eigen::Dynamic, 1>;
using Array1Xs = Eigen::Array<FloatType, 1, Eigen::Dynamic>;

/* 
 * Linear Time-Invariant (LTI) system state-space model in standard form:
 *  1. Continuous
 *      x-dot(t) = A x(t) + B u(t)
 *          y(t) = C x(t) + D u(t)
 *  2. Discrete
 *      x[k+1] = A x[k] + B u[k]
 *        y[k] = C x[k] + D u[k]
 * or discrete LTI system state-space model in non-standard form:
 *      x[k+1] = A x[k] + B u[k] + B' u[k+1]
 *        y[k] = C x[k] + D u[k]
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
                const Eigen::Ref<const MatrixXs>& B_prime,
                const Eigen::Ref<const MatrixXs>& C,
                const Eigen::Ref<const MatrixXs>& D);
  int N() const;
  int M() const;

  // TODO: prevent invalid modifications
  MatrixXs A;
  MatrixXs B;
  MatrixXs B_prime;
  MatrixXs C;
  MatrixXs D;
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
LtiStateSpace foh_cont2discrete(const LtiStateSpace& cont_sys, FloatType dt,
                                bool is_state_preserved = false);

class DiscreteLtiFilter{
 public:
  DiscreteLtiFilter() = default;
  DiscreteLtiFilter(const LtiStateSpace& system,
                    const Eigen::Ref<const MatrixXs>& init_input,
                    const Eigen::Ref<const MatrixXs>& init_state);
  MatrixXs update(const Eigen::Ref<const MatrixXs>& next_input);

  LtiStateSpace system;
  MatrixXs input;
  MatrixXs state;
};

/*
 * Discretized Non-Linear Time-Invariant (NLTI) Low-Pass Filter (LPF). The 
 * continuous-time (i.e. non-discretized with sampling interval `dt`) state
 * space model takes the following form:
 *    x-dot(t) = -w_c(u(t)) x(t) + w_c(u(t)) u(t)
 *    y(t) = x(t)
 * where:
 *    w_c(u(t)) = 2 * pi * f_c(u(t))
 *    f_c(u(t)) = input_exp_cutoff_freq_prop_constant * exp(u(t))
 */
class DiscreteNltiLpf {
 public:
  DiscreteNltiLpf() = default;
  DiscreteNltiLpf(double input_exp_cutoff_freq_prop_constant);
  DiscreteNltiLpf(double input_exp_cutoff_freq_prop_constant,
                  const control::RowVectorXs& init_input,
                  const control::RowVectorXs& init_input_exp,
                  const control::RowVectorXs& init_state);
  const control::RowVectorXs& update(
      const control::RowVectorXs& next_input,
      const control::RowVectorXs& next_input_exp,
      FloatType dt);
  const control::RowVectorXs& update(
      const control::RowVectorXs&& next_input,
      const control::RowVectorXs&& next_input_exp,
      FloatType dt);
  static FloatType expint(FloatType z);

  double input_exp_cutoff_freq_prop_constant;
  control::RowVectorXs input;
  control::RowVectorXs input_exp;
  control::RowVectorXs state;
};

} // namespace control
