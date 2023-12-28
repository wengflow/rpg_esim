#include <esim/esim/control.hpp>
#include <glog/logging.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/math/special_functions/expint.hpp>

namespace control {

LtiStateSpace::LtiStateSpace(
    const Eigen::Ref<const MatrixXs>& A,
    const Eigen::Ref<const MatrixXs>& B,
    const Eigen::Ref<const MatrixXs>& C,
    const Eigen::Ref<const MatrixXs>& D)
    : LtiStateSpace(A, B, MatrixXs::Zero(N_x(), N_u()), C, D) {
}

LtiStateSpace::LtiStateSpace(
    const Eigen::Ref<const MatrixXs>& A,
    const Eigen::Ref<const MatrixXs>& B,
    const Eigen::Ref<const MatrixXs>& B_tilde,
    const Eigen::Ref<const MatrixXs>& C,
    const Eigen::Ref<const MatrixXs>& D)
    : A{A}, B{B}, B_tilde{B_tilde}, C{C}, D{D} {
  CHECK_EQ(A.rows(), N_x());
  CHECK_EQ(B.rows(), N_x());
  CHECK_EQ(B_tilde.rows(), N_x());
  CHECK_EQ(B_tilde.cols(), N_u());
  CHECK_EQ(C.cols(), N_x());
  CHECK_EQ(D.cols(), N_u());
  CHECK_EQ(D.rows(), N_y());
}

bool LtiStateSpace::isInit() const {
  return (A.size() > 0);
}

unsigned int LtiStateSpace::N_u() const {
  CHECK(isInit());
  return B.cols();
}

unsigned int LtiStateSpace::N_x() const {
  CHECK(isInit());
  return A.cols();
}

unsigned int LtiStateSpace::N_y() const {
  CHECK(isInit());
  return C.rows();
}

BatchLtiStateSpace::BatchLtiStateSpace(
    const std::vector<ArrayXXs>& A,
    const std::vector<ArrayXXs>& B,
    const std::vector<ArrayXXs>& C,
    const std::vector<ArrayXXs>& D)
    : BatchLtiStateSpace(
          A, B, std::vector<ArrayXXs>{N_x(), MatrixXs::Zero(N_b(), N_u())},
          C, D) {
}

BatchLtiStateSpace::BatchLtiStateSpace(
    const std::vector<ArrayXXs>& A,
    const std::vector<ArrayXXs>& B,
    const std::vector<ArrayXXs>& B_tilde,
    const std::vector<ArrayXXs>& C,
    const std::vector<ArrayXXs>& D)
    : A{A}, B{B}, B_tilde{B_tilde}, C{C}, D{D} {
  for (int i=0; i<N_x(); ++i) {
    CHECK_EQ(A[i].rows(), N_b());
    CHECK_EQ(B[i].rows(), N_b());
    CHECK_EQ(B_tilde[i].rows(), N_b());

    CHECK_EQ(A[i].cols(), N_x());
    CHECK_EQ(B[i].cols(), N_u());
    CHECK_EQ(B_tilde[i].cols(), N_u());
  }
  for (int i=0; i<N_y(); ++i) {
    CHECK_EQ(C[i].rows(), N_b());
    CHECK_EQ(D[i].rows(), N_b());

    CHECK_EQ(C[i].cols(), N_x());
    CHECK_EQ(D[i].cols(), N_u());
  }
  CHECK_EQ(A.size(), N_x());
  CHECK_EQ(B.size(), N_x());
  CHECK_EQ(B_tilde.size(), N_x());
  CHECK_EQ(D.size(), N_y());
}

bool BatchLtiStateSpace::isInit() const {
  return (A.size() > 0);
}

unsigned int BatchLtiStateSpace::N_b() const {
  CHECK(isInit());
  return A[0].rows();
}

unsigned int BatchLtiStateSpace::N_u() const {
  CHECK(isInit());
  return B[0].cols();
}

unsigned int BatchLtiStateSpace::N_x() const {
  CHECK(isInit());
  return A[0].cols();
}

unsigned int BatchLtiStateSpace::N_y() const {
  CHECK(isInit());
  return C.size();
}

LtiStateSpace fohCont2discrete(const LtiStateSpace& cont_sys, FloatType dt,
                               bool is_state_preserved) {
  // build an exponential matrix
  const int N_x = cont_sys.N_x();
  const int N_u = cont_sys.N_u();

  const int em_size = 2 * N_u + N_x;
  MatrixXs em = MatrixXs::Zero(em_size, em_size);                                 // (em_size, em_size)

  em.block(0, 0, N_x, N_x) = cont_sys.A * dt;                                     // (N_x, N_x)
  em.block(0, N_x, N_x, N_u) = cont_sys.B * dt;                                   // (N_x, N_u)
  em.block(N_x, N_x + N_u, N_u, N_u) = MatrixXs::Identity(N_u, N_u);              // (N_u, N_u)

  const MatrixXs ms = em.exp();                                                   // (em_size, em_size)

  // get the three blocks from upper rows to form the FOH-discretized LTI state
  // space model
  MatrixXs::ConstBlockXpr phi = ms.block(0, 0, N_x, N_x);                         // (N_x, N_x)
  MatrixXs::ConstBlockXpr gamma1 = ms.block(0, N_x, N_x, N_u);                    // (N_x, N_u)
  MatrixXs::ConstBlockXpr gamma2 = ms.block(0, N_x + N_u, N_x, N_u);              // (N_x, N_u)

  // return the FOH-discretized LTI state space model in the desired form
  if (is_state_preserved) {
    return LtiStateSpace{
        phi,
        gamma1 - gamma2,
        gamma2,
        cont_sys.C,
        cont_sys.D};
  } else {
    return LtiStateSpace{
        phi,
        gamma1 + phi * gamma2 - gamma2,
        cont_sys.C,
        cont_sys.D + cont_sys.C * gamma2};
  }
}

DiscreteLtiFilter::DiscreteLtiFilter(
    const LtiStateSpace& system,
    const Eigen::Ref<const MatrixXs>& init_input,                                 // (system.N_u() [, N_b])
    const Eigen::Ref<const MatrixXs>& init_state)                                 // (system.N_x() [, N_b])
    : system{system},
      input{init_input},
      state{init_state} {
  CHECK_EQ(init_input.cols(), init_state.cols());
  CHECK_EQ(init_input.rows(), system.N_u());
  CHECK_EQ(init_state.rows(), system.N_x());
}

void DiscreteLtiFilter::update(const Eigen::Ref<const MatrixXs>& next_input) {    // (system.N_u() [, N_b])
  update(std::move(MatrixXs{next_input}));
}

void DiscreteLtiFilter::update(MatrixXs&& next_input) {                           // (system.N_u() [, N_b])
  // update the current input & state with the next
  state = system.A * state + system.B * input + system.B_tilde * next_input;      // (system.N_x() [, N_b])
  input = std::move(next_input);
}

MatrixXs DiscreteLtiFilter::output() const {
  return output(system.C, system.D);
}

MatrixXs DiscreteLtiFilter::output(
    const Eigen::Ref<const MatrixXs>& C,
    const Eigen::Ref<const MatrixXs>& D) const {
  return C * state + D * input;                                                   // (system.N_y() [, N_b])
}

BatchDiscreteLtiFilter::BatchDiscreteLtiFilter(
    const BatchLtiStateSpace& system,
    const Eigen::Ref<const MatrixXs>& init_input,                                 // (system.N_u(), system.N_b())
    const Eigen::Ref<const MatrixXs>& init_state)                                 // (system.N_x(), system.N_b())
    : system{system},
      input{init_input},
      state{init_state} {
  CHECK_EQ(init_input.cols(), system.N_b());
  CHECK_EQ(init_state.cols(), system.N_b());
  CHECK_EQ(init_input.rows(), system.N_u());
  CHECK_EQ(init_state.rows(), system.N_x());
}

void BatchDiscreteLtiFilter::update(
    const Eigen::Ref<const MatrixXs>& next_input) {                               // (system.N_u(), system.N_b())
  update(std::move(MatrixXs{next_input}));
}

void BatchDiscreteLtiFilter::update(MatrixXs&& next_input) {                      // (system.N_u(), system.N_b())
  // update the current input & state with the next
  MatrixXs next_state{system.N_x(), system.N_b()};                                // (system.N_x(), system.N_b())
  for (int i=0; i<system.N_x(); ++i) {
    next_state.row(i) = (                                                         // (1, system.N_b())
        (system.A[i].transpose() * state.array()).colwise().sum()
        + (system.B[i].transpose() * input.array()).colwise().sum()
        + (system.B_tilde[i].transpose() * next_input.array()).colwise().sum()
    );
  }
  state = std::move(next_state);
  input = std::move(next_input);
}

MatrixXs BatchDiscreteLtiFilter::output() const {
  return output(system.C, system.D);
}

MatrixXs BatchDiscreteLtiFilter::output(
    const std::vector<ArrayXXs>& C,
    const std::vector<ArrayXXs>& D) const {
  MatrixXs output{system.N_y(), system.N_b()};                                    // (system.N_y(), system.N_b())
  for (int i=0; i<system.N_y(); ++i) {
    output.row(i) = (C[i].transpose() * state.array()).colwise().sum()            // (1, system.N_b())
                    + (D[i].transpose() * input.array()).colwise().sum();
  }
  return output;
}

} // namespace control
