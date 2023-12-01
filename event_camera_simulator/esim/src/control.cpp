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
    : A{A}, B{B}, B_prime{MatrixXs::Zero(N(), M())}, C{C}, D{D} {
  CHECK_EQ(A.rows(), N());
  CHECK_EQ(B.rows(), N());
  CHECK_EQ(C.cols(), N());
  CHECK_EQ(D.cols(), M());
}

LtiStateSpace::LtiStateSpace(
    const Eigen::Ref<const MatrixXs>& A,
    const Eigen::Ref<const MatrixXs>& B,
    const Eigen::Ref<const MatrixXs>& B_prime,
    const Eigen::Ref<const MatrixXs>& C,
    const Eigen::Ref<const MatrixXs>& D)
    : A{A}, B{B}, B_prime{B_prime}, C{C}, D{D} {
  CHECK_EQ(A.rows(), N());
  CHECK_EQ(B.rows(), N());
  CHECK_EQ(B_prime.rows(), N());
  CHECK_EQ(C.cols(), N());
  CHECK_EQ(D.cols(), M());
}

int LtiStateSpace::N() const {
  return A.cols();
}

int LtiStateSpace::M() const {
  return B.cols();
}

LtiStateSpace foh_cont2discrete(const LtiStateSpace& cont_sys, FloatType dt,
                                bool is_state_preserved) {
  // build an exponential matrix
  const int N = cont_sys.N();
  const int M = cont_sys.M();

  const int em_size = 2 * M + N;
  MatrixXs em = MatrixXs::Zero(em_size, em_size);

  em.block(0, 0, N, N) = cont_sys.A * dt;
  em.block(0, N, N, M) = cont_sys.B * dt;
  em.block(N, N + M, M, M) = MatrixXs::Identity(M, M);

  const MatrixXs ms = em.exp();

  // get the three blocks from upper rows to form the FOH-discretized LTI state
  // space model
  MatrixXs::ConstBlockXpr phi = ms.block(0, 0, N, N);
  MatrixXs::ConstBlockXpr gamma1 = ms.block(0, N, N, M);
  MatrixXs::ConstBlockXpr gamma2 = ms.block(0, N + M, N, M);

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
    const Eigen::Ref<const MatrixXs>& init_input,
    const Eigen::Ref<const MatrixXs>& init_state)
    : system{system},
      input{init_input},
      state{init_state} {
  CHECK_EQ(init_input.rows(), system.M());
  CHECK_EQ(init_state.rows(), system.N());
}

MatrixXs DiscreteLtiFilter::update(
    const Eigen::Ref<const MatrixXs>& next_input) {
  // update the current input & state with the next & return the next output
  state = system.A * state + system.B * input + system.B_prime * next_input;
  input = next_input;
  return system.C * state + system.D * input;
}

DiscreteNltiLpf::DiscreteNltiLpf(double input_exp_cutoff_freq_prop_constant)
    : input_exp_cutoff_freq_prop_constant{input_exp_cutoff_freq_prop_constant},
      input{},
      input_exp{},
      state{} {
}

DiscreteNltiLpf::DiscreteNltiLpf(double input_exp_cutoff_freq_prop_constant,
                                 const control::RowVectorXs& init_input,
                                 const control::RowVectorXs& init_input_exp,
                                 const control::RowVectorXs& init_state)
    : input_exp_cutoff_freq_prop_constant{input_exp_cutoff_freq_prop_constant},
      input{init_input},
      input_exp{init_input_exp},
      state{init_state} {
  CHECK_EQ(init_input.cols(), init_input_exp.cols());
  CHECK_EQ(init_input.cols(), init_state.cols());
}

const control::RowVectorXs& DiscreteNltiLpf::update(
    const control::RowVectorXs& next_input,
    const control::RowVectorXs& next_input_exp,
    FloatType dt) {
  // update the current state / output
  control::Array1Xs scale = (
      2 * EIGEN_PI * input_exp_cutoff_freq_prop_constant * dt
      / (next_input - input).array());
  control::Array1Xs scaled_input_exp = scale * input_exp.array();
  control::Array1Xs scaled_next_input_exp = scale * next_input_exp.array();

  state = (scaled_input_exp - scaled_next_input_exp).exp()
          * (state - input).array()
          + next_input.array()
          + (-scaled_next_input_exp).exp()
          * (scaled_input_exp.unaryExpr(std::ref(expint))
             - scaled_next_input_exp.unaryExpr(std::ref(expint)));
  
  // update the current input & input exponential (cache) via copy assignment
  input = next_input;
  input_exp = next_input_exp;

  // return the next state / output
  return state;
}

// TODO: check move successful
const control::RowVectorXs& DiscreteNltiLpf::update(
    const control::RowVectorXs&& next_input,
    const control::RowVectorXs&& next_input_exp,
    FloatType dt) {
  // update the current state / output
  control::Array1Xs scale = (
      2 * EIGEN_PI * input_exp_cutoff_freq_prop_constant * dt
      / (next_input - input).array());
  control::Array1Xs scaled_input_exp = scale * input_exp.array();
  control::Array1Xs scaled_next_input_exp = scale * next_input_exp.array();

  state = (scaled_input_exp - scaled_next_input_exp).exp()
          * (state - input).array()
          + next_input.array()
          + (-scaled_next_input_exp).exp()
          * (scaled_input_exp.unaryExpr(std::ref(expint))
             - scaled_next_input_exp.unaryExpr(std::ref(expint)));
  
  // update the current input & input exponential (cache) via move assignment
  input = std::move(next_input);
  input_exp = std::move(next_input_exp);

  // return the next state / output
  return state;
}

FloatType DiscreteNltiLpf::expint(FloatType z) {
  return boost::math::expint(z);
}


} // namespace control