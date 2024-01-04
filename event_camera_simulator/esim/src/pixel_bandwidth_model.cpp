#include <esim/esim/pixel_bandwidth_model.hpp>
#include <functional>
#include <glog/logging.h>
#include <ze/common/time_conversions.hpp>
#include <opencv2/core/eigen.hpp>

namespace event_camera_simulator {

FloatTypeImage eigen2cv(const control::MatrixXs& eigen_mat,
                        int output_channels,
                        int output_height) {
  FloatTypeImage cv_mat;
  cv::eigen2cv(eigen_mat, cv_mat);                                                // copy involved
  return cv_mat.reshape(output_channels, output_height);                          // (output_height, cv_mat.total() / cv_mat.channels()) OpenCV matrix with `output_channels` channel(s)
}

StdVectorIntPair getIndexPair(const Eigen::Ref<const ArrayXb>& condition) {
  StdVectorIntPair index_pair{};                                                  // pair of true & false indices
  for (int i=0; i<condition.size(); ++i) {
    if (condition(i))
      index_pair.first.push_back(i);
    else
      index_pair.second.push_back(i);
  }
  return index_pair;
}

PixelBandwidthModel::PixelBandwidthModel(
    double I_p_to_it_ratio_fa,
    double A_amp,
    double kappa,
    double V_T_mv,
    double C_p_ff,
    double C_mil_ff,
    double tau_out_us,
    double f_c_lower_hz,
    double f_c_sf_hz,
    double f_c_diff_hz,
    double assumed_log_it_mean)
    : nlti_lpf_order_{0},
      lti_hpf_order_{0},
      lti_lpf_order_{0},
      is_diff_amp_inf_fast_{static_cast<bool>(std::isinf(f_c_diff_hz))},
      I_p_to_it_ratio_{I_p_to_it_ratio_fa * kFromFemto},
      A_amp_{A_amp},
      A_cl_{1 / kappa},
      Q_in_{(C_p_ff * kFromFemto) * (V_T_mv *  kFromMilli)},
      Q_mil_{(C_mil_ff * kFromFemto) * (V_T_mv *  kFromMilli)},
      tau_out_{tau_out_us * kFromMicro},
      cont_lti_subsys_{},
      assumed_log_it_mean_{assumed_log_it_mean},
      discrete_nlti_lpf_{},
      discrete_lti_filter_{} {
  // asserts
  const std::array<double, 4> lower_bounded = {I_p_to_it_ratio_fa, A_amp,         // (4)
                                               f_c_sf_hz, f_c_diff_hz};
  const std::array<double, 4> upper_bounded = {C_p_ff, C_mil_ff, tau_out_us,      // (4)
                                               f_c_lower_hz};
  const std::array<double, 2> bounded = {kappa, V_T_mv};                          // (2)

  for (double val : lower_bounded) {
    CHECK_GT(val, 0);
  }
  for (double val : upper_bounded) {
    CHECK(!std::isinf(val));
  }
  for (double val : bounded) {
    CHECK_GT(val, 0);
    CHECK(!std::isinf(val));
  }

  // infer the desired system order & 1st-order LTI LPF/HPF cutoff (angular)
  // frequencies for the pixel bandwidth model
  const std::array<double, 2> fo_lti_lpf_omega_c = {                              // (2)
      static_cast<double>(2 * EIGEN_PI * f_c_sf_hz),
      static_cast<double>(2 * EIGEN_PI * f_c_diff_hz)};
  std::vector<double> finite_fo_lti_lpf_omega_c = {};
  for (const double omega_c : fo_lti_lpf_omega_c) {
    if (!std::isinf(omega_c))
      finite_fo_lti_lpf_omega_c.push_back(omega_c);
  }

  if (((is_fb_amp_inf_fast() || is_A_amp_inf()) && is_input_inf_fast())
      || (is_A_amp_inf() && is_mil_negligible())) {
    nlti_lpf_order_ = 0;
  } else if (is_input_inf_fast()) {
    nlti_lpf_order_ = 0;
    finite_fo_lti_lpf_omega_c.insert(finite_fo_lti_lpf_omega_c.begin(),
                                     (A_loop() + 1) / tau_out_);
  } else if (is_fb_amp_inf_fast() || is_A_amp_inf()) {
    nlti_lpf_order_ = 1;
  } else {
    nlti_lpf_order_ = 2;
  }

  lti_lpf_order_ = finite_fo_lti_lpf_omega_c.size();
  lti_hpf_order_ = static_cast<int>(f_c_lower_hz != 0);

  // return, if the LTI sub-system of the pixel bandwidth model is not required
  if (lti_subsys_order() == 0)
    return;

  // initialize the continuous-time state-space model for the LTI sub-system
  cont_lti_subsys_.A = control::MatrixXs::Zero(lti_subsys_order(),                // (lti_subsys_order(), lti_subsys_order())
                                               lti_subsys_order());
  cont_lti_subsys_.B = control::MatrixXs::Zero(lti_subsys_order(), 1);            // (lti_subsys_order(), 1)
  cont_lti_subsys_.B_tilde = control::MatrixXs::Zero(lti_subsys_order(), 1);      // (lti_subsys_order(), 1)
  cont_lti_subsys_.C = control::MatrixXs::Zero(1, lti_subsys_order());            // (1, lti_subsys_order())
  cont_lti_subsys_.D = control::MatrixXs::Zero(1, 1);                             // (1, 1)

  if (lti_hpf_order_ == 1) {
    const double omega_c_lower = 2 * EIGEN_PI * f_c_lower_hz;
    cont_lti_subsys_.A(0, 0) = -omega_c_lower;
    cont_lti_subsys_.B(0) = -omega_c_lower;

    if (lti_lpf_order_ == 0)
      cont_lti_subsys_.D(0) = 1;
  }

  for (int i=0; i<lti_lpf_order_; ++i) {
    const int coord = i + lti_hpf_order_;
    const double omega_c_upper = finite_fo_lti_lpf_omega_c[i];
    cont_lti_subsys_.A(coord, coord) = -omega_c_upper;
    if (coord-1 >= 0)
      cont_lti_subsys_.A(coord, coord-1) = omega_c_upper;
    if (i == 0)
      cont_lti_subsys_.B(coord) = omega_c_upper;
  }
  cont_lti_subsys_.C(lti_subsys_order() - 1) = 1;
}

unsigned int PixelBandwidthModel::order() const {
  return lti_subsys_order() + nlti_subsys_order();
}

unsigned int PixelBandwidthModel::nlti_subsys_order() const {
  return nlti_lpf_order_;
}

unsigned int PixelBandwidthModel::lti_subsys_order() const {
  return lti_hpf_order_ + lti_lpf_order_;
}

bool PixelBandwidthModel::is_input_inf_fast() const {
  return (std::isinf(I_p_to_it_ratio_) || (Q_in_ == 0 && Q_mil_ == 0));
}

bool PixelBandwidthModel::is_fb_amp_inf_fast() const {
  return (tau_out_ == 0);
}

bool PixelBandwidthModel::is_A_amp_inf() const {
  return std::isinf(A_amp_);
}

bool PixelBandwidthModel::is_mil_negligible() const {
  return (Q_mil_ == 0);
}

void PixelBandwidthModel::initState(const Image& init_log_img) {                  // (height, width) OpenCV matrix with 1 channel
  CHECK_EQ(init_log_img.channels(), 1);

  // convert & cast the given OpenCV `ImageFloatType` 2D log-image matrix to
  // Eigen `FloatType` log-image row vector
  const Image init_cv_log_img_rvec = init_log_img.reshape(1, 1);                  // (1, height * width) or (1, num_pixels) OpenCV matrix with 1 channel
  control::RowVectorXs init_eigen_log_img_rvec{};
  cv::cv2eigen(init_cv_log_img_rvec, init_eigen_log_img_rvec);                    // copy involved, (1, num_pixels)

  // initialize the NLTI LPF sub-system, assuming steady-state input given by
  // the initial log-image, if applicable
  const int num_pixels = init_eigen_log_img_rvec.cols();
  if (nlti_subsys_order() > 0) {
    // initialize the input to the NLTI LPF sub-system to the given initial
    // log-image
    discrete_nlti_lpf_.input = init_eigen_log_img_rvec;                           // (1, num_pixels)
    discrete_nlti_lpf_.state.resize(nlti_subsys_order(), num_pixels);             // (nlti_subsys_order(), num_pixels)

    // initialize the discretized NLTI LPF sub-system `A`, `B` & `B_tilde` array
    // sizes, and `D` array to 0
    discrete_nlti_lpf_.system.A.clear();
    discrete_nlti_lpf_.system.B.clear();
    for (int i=0; i<nlti_subsys_order(); ++i) {
      discrete_nlti_lpf_.system.A.emplace_back(num_pixels, nlti_subsys_order());  // (num_pixels, nlti_subsys_order())
      discrete_nlti_lpf_.system.B.emplace_back(num_pixels, 1);                    // (num_pixels, 1)
      discrete_nlti_lpf_.system.B_tilde.emplace_back(num_pixels, 1);              // (num_pixels, 1)
    }
    discrete_nlti_lpf_.system.D = {control::ArrayXXs::Zero(num_pixels, 1)};       // (1) vector of (num_pixels, 1) array
  }
  if (nlti_subsys_order() == 1) {
    // initialize the state of the 1st-order NLTI LPF sub-system (i.e. initial
    // output) to the initial log-image
    discrete_nlti_lpf_.state = init_eigen_log_img_rvec;                           // (1, num_pixels)

    // initialize the discretized 1st-order NLTI LPF sub-system `C` array to 1
    discrete_nlti_lpf_.system.C = {control::ArrayXXs::Ones(num_pixels, 1)};       // (1) vector of (num_pixels, 1) array
  } else if (nlti_subsys_order() == 2) {
    // initialize the state of the 2nd-order NLTI LPF sub-system
    // (i.e. [initial rate of change of output, initial output]T) to
    // `[0, initial log-image]T`
    discrete_nlti_lpf_.state << control::RowVectorXs::Zero(num_pixels),           // (2, num_pixels)
                                init_eigen_log_img_rvec;

    // initialize the auxiliary 2x2 identity matrix
    identity_ = {control::ArrayXXs{num_pixels, 2},                                // (2) vector of (num_pixels, 2) arrays
                 control::ArrayXXs{num_pixels, 2}};                               
    identity_[0] << control::ArrayXs::Ones(num_pixels),                           // (num_pixels, 2)
                    control::ArrayXs::Zero(num_pixels);
    identity_[1] << control::ArrayXs::Zero(num_pixels),                           // (num_pixels, 2)
                    control::ArrayXs::Ones(num_pixels);

    // initialize the discretized 2nd-order NLTI LPF sub-system `C` array to
    // `[0, 1]T`
    discrete_nlti_lpf_.system.C = {identity_[1]};                                 // (1) vector of (num_pixels, 2) array
  }

  if (lti_subsys_order() == 0)
    return;

  // initialize the input to the LTI sub-system with the given init. log-image,
  // which is also the init. (& steady-state) output of the NLTI LPF sub-system
  discrete_lti_filter_.input = init_eigen_log_img_rvec;                           // (1, num_pixels)

  if (lti_hpf_order_ == 0) {
    // initialize the state of each LPF in the LTI sub-system (i.e. initial
    // output) to the given initial log-image, which assumes such steady-state
    // input
    discrete_lti_filter_.state = (                                                // (lti_lpf_order_, num_pixels)
        init_eigen_log_img_rvec.replicate(lti_lpf_order_, 1));
  } else if (lti_hpf_order_ == 1) {
    // initialize the state of the LTI sub-system HPF (i.e. initial output - 
    // initial input) to negative of the assumed input mean / DC component, and
    // the state of each LPF in the LTI sub-system (i.e. initial output) to
    // the given initial log-image - assumed input mean, which assumes the
    // spectrum of the input lies within the passband of the LTI sub-system
    // with negligible phase shift, except for the DC component (filtered out)
    discrete_lti_filter_.state.resize(lti_subsys_order(), num_pixels);            // (lti_subsys_order(), num_pixels)
    discrete_lti_filter_.state.row(0).setConstant(-assumed_log_it_mean_);         // (1, num_pixels)
    discrete_lti_filter_.state.bottomRows(lti_lpf_order_) = (                     // (lti_lpf_order_, num_pixels)
        (init_eigen_log_img_rvec.array() - assumed_log_it_mean_).replicate(
            lti_lpf_order_, 1));
  }
}

void PixelBandwidthModel::linearizeFohNltiLpf(
    const Eigen::Ref<const control::RowVectorXs>& linearization_img_rvec,         // (1, num_pixels)
    FloatType dt) {
  const auto linearization_I_p = I_p_to_it_ratio_
                                 * linearization_img_rvec.transpose().array();    // (num_pixels, 1)
  if (nlti_subsys_order() == 0) {
    return;
  } else if (nlti_subsys_order() == 1) {
    // linearize the 1st-order NLTI LPF continuous-time sub-system at the
    // steady-state, where sub-system output = sub-system input
    // = `linearization_img_rvec`
    control::ArrayXs omega_c;
    if (is_fb_amp_inf_fast()) {
      omega_c = (A_loop() + 1) / (Q_in_ + (A_amp_ + 1) * Q_mil_)                  // (num_pixels, 1)
                * linearization_I_p;
    } else {  // else if (is_A_amp_inf()) {
      omega_c = linearization_I_p / (A_cl_ * Q_mil_);                             // (num_pixels, 1)
    }

    // FOH-discretize the linearized 1st-order NLTI LPF continuous-time
    // sub-system, according to the next image sampling interval `dt`
    const auto A_dt = -omega_c * dt;                                              // (num_pixels, 1)
    const auto B_dt = omega_c * dt;                                               // (num_pixels, 1)
    control::ArrayXXs phi = exp(A_dt);                                            // (num_pixels, 1)
    control::ArrayXXs gamma2 = (phi - 1 - A_dt) / A_dt.square() * B_dt;           // (num_pixels, 1)
    control::ArrayXXs gamma1_minus_gamma2 = (A_dt - 1) * gamma2 + B_dt;           // (num_pixels, 1)

    discrete_nlti_lpf_.system.A[0] = std::move(phi);
    discrete_nlti_lpf_.system.B[0] = std::move(gamma1_minus_gamma2);
    discrete_nlti_lpf_.system.B_tilde[0] = std::move(gamma2);
    
    return;
  } // else if (nlti_subsys_order() == 2) {

  // linearize the 2nd-order NLTI LPF continuous-time sub-system at the
  // steady-state, where sub-system output = sub-system input
  // = `linearization_img_rvec`
  const control::ArrayXs tau_in_plus_tau_mil = (                                  // (num_pixels, 1)
      (Q_in_ + Q_mil_) / linearization_I_p);
  const auto A_amp_tau_mil = A_amp_ * Q_mil_ / linearization_I_p;                 // (num_pixels, 1)
  const control::ArrayXs zeta_omega_n = (                                         // (num_pixels, 1)
      (tau_in_plus_tau_mil + A_amp_tau_mil + tau_out_)
      / (2 * tau_in_plus_tau_mil * tau_out_));
  const control::ArrayXs omega_n_square = (                                       // (num_pixels, 1)
      (A_loop() + 1) / (tau_in_plus_tau_mil * tau_out_));
  const control::ArrayXs zeta_omega_n_square_minus_omega_n_square = (             // (num_pixels, 1)
      zeta_omega_n.square() - omega_n_square);  

  // FOH-discretize the linearized 2nd-order NLTI LPF continuous-time sub-system
  // analytically, according to the next image sampling interval `dt`
  const StdVectorIntPair index_pair = getIndexPair(                               // pair of `zeta >= 1` & `zeta < 1` indices
      zeta_omega_n_square_minus_omega_n_square >= 0);
  const std::vector<int>& is_zeta_ge_one = index_pair.first;                      // (num_zeta_ge_one)
  const std::vector<int>& is_zeta_lt_one = index_pair.second;                     // (num_zeta_lt_one)

  // A = phi (zeta >= 1, critically damped or overdamped)
  const auto sigma_ge = zeta_omega_n(is_zeta_ge_one);                             // (num_zeta_ge_one, 1)
  const control::ArrayXs j_omega_d = sqrt(                                        // (num_zeta_ge_one, 1)
      zeta_omega_n_square_minus_omega_n_square(is_zeta_ge_one));
  const control::ArrayXs pos_pole = -sigma_ge + j_omega_d;                        // (num_zeta_ge_one, 1)
  const control::ArrayXs neg_pole = -sigma_ge - j_omega_d;                        // (num_zeta_ge_one, 1)
  const auto exp_pos_pole_dt = exp(pos_pole * dt);                                // (num_zeta_ge_one, 1)
  const auto exp_neg_pole_dt = exp(neg_pole * dt);                                // (num_zeta_ge_one, 1)
  const control::ArrayXs normalized_exp_pos_pole_dt = exp_pos_pole_dt             // (num_zeta_ge_one, 1)
                                                      / (2 * j_omega_d);
  const control::ArrayXs normalized_exp_neg_pole_dt = exp_neg_pole_dt             // (num_zeta_ge_one, 1)
                                                      / (2 * j_omega_d);

  discrete_nlti_lpf_.system.A[1].col(0)(is_zeta_ge_one) = (                       // (num_zeta_ge_one, 1)
      normalized_exp_pos_pole_dt - normalized_exp_neg_pole_dt);
  discrete_nlti_lpf_.system.A[1].col(1)(is_zeta_ge_one) = (                       // (num_zeta_ge_one, 1)
      pos_pole * normalized_exp_neg_pole_dt
      - neg_pole * normalized_exp_pos_pole_dt);
  discrete_nlti_lpf_.system.A[0].col(0)(is_zeta_ge_one) = (                       // (num_zeta_ge_one, 1)
      pos_pole * normalized_exp_pos_pole_dt
      - neg_pole * normalized_exp_neg_pole_dt);

  // A = phi (zeta < 1, undamped or underdamped)
  const auto sigma_lt = zeta_omega_n(is_zeta_lt_one);                             // (num_zeta_lt_one, 1)
  const control::ArrayXs omega_d = sqrt(                                          // (num_zeta_lt_one, 1)
      -zeta_omega_n_square_minus_omega_n_square(is_zeta_lt_one));
  const auto sin_omega_d_dt = sin(omega_d * dt);                                  // (num_zeta_lt_one, 1)
  const auto cos_omega_d_dt = cos(omega_d * dt);                                  // (num_zeta_lt_one, 1)
  const control::ArrayXs exp_neg_sigma_dt_over_omega_d = exp(-sigma_lt * dt)      // (num_zeta_lt_one, 1)
                                                         / omega_d;
  const control::ArrayXs normalized_sin_omega_d_dt = (                            // (num_zeta_lt_one, 1)
      exp_neg_sigma_dt_over_omega_d * sin_omega_d_dt);
  const control::ArrayXs normalized_cos_omega_d_dt = (                            // (num_zeta_lt_one, 1)
      exp_neg_sigma_dt_over_omega_d * cos_omega_d_dt);
  
  discrete_nlti_lpf_.system.A[1].col(0)(is_zeta_lt_one) = (                       // (num_zeta_lt_one, 1)
      normalized_sin_omega_d_dt);
  discrete_nlti_lpf_.system.A[1].col(1)(is_zeta_lt_one) = (                       // (num_zeta_lt_one, 1)
      sigma_lt * normalized_sin_omega_d_dt
      + omega_d * normalized_cos_omega_d_dt);
  discrete_nlti_lpf_.system.A[0].col(0)(is_zeta_lt_one) = (                       // (num_zeta_lt_one, 1)
      omega_d * normalized_cos_omega_d_dt
      - sigma_lt * normalized_sin_omega_d_dt);

  discrete_nlti_lpf_.system.A[0].col(1) = -discrete_nlti_lpf_.system.A[1].col(0)  // (num_pixels, 1)
                                          * omega_n_square;
  CHECK(!discrete_nlti_lpf_.system.A[0].isNaN().any());
  CHECK(!discrete_nlti_lpf_.system.A[1].isNaN().any());

  /*
  // Equivalent, but less efficient implementation
  // A = phi (any zeta)
  const control::ArrayXs& sigma = zeta_omega_n;                                   // (num_pixels, 1)
  const control::ArrayXcs j_omega_d = sqrt(                                       // (num_pixels, 1)
      (zeta_omega_n.square() - omega_n_square).cast<control::ComplexScalar>());
  const control::ArrayXcs pos_pole = -sigma + j_omega_d;                          // (num_pixels, 1)
  const control::ArrayXcs neg_pole = -sigma - j_omega_d;                          // (num_pixels, 1)
  const auto exp_pos_pole_dt = exp(pos_pole * dt);                                // (num_pixels, 1)
  const auto exp_neg_pole_dt = exp(neg_pole * dt);                                // (num_pixels, 1)
  const control::ArrayXcs normalized_exp_pos_pole_dt = exp_pos_pole_dt            // (num_pixels, 1)
                                                       / (2 * j_omega_d);
  const control::ArrayXcs normalized_exp_neg_pole_dt = exp_neg_pole_dt            // (num_pixels, 1)
                                                       / (2 * j_omega_d);

  discrete_nlti_lpf_.system.A[1].col(0) = real(                                   // (num_pixels, 1)
      normalized_exp_pos_pole_dt - normalized_exp_neg_pole_dt);
  discrete_nlti_lpf_.system.A[1].col(1) = real(                                   // (num_pixels, 1)
      pos_pole * normalized_exp_neg_pole_dt
      - neg_pole * normalized_exp_pos_pole_dt);
  discrete_nlti_lpf_.system.A[0].col(0) = real(                                   // (num_pixels, 1)
      pos_pole * normalized_exp_pos_pole_dt
      - neg_pole * normalized_exp_neg_pole_dt);
  discrete_nlti_lpf_.system.A[0].col(1) = -discrete_nlti_lpf_.system.A[1].col(0)  // (num_pixels, 1)
                                          * omega_n_square;
  */

  // B-tilde = gamma2, B = gamma1 - gamma2
  const auto A_dt_inv_B_dt = -identity_[1].transpose();                           // (2, num_pixels)

  const int num_pixels = linearization_img_rvec.cols();
  control::ArrayXXs A_dt_square_inv_B_dt {2, num_pixels};                         // (2, num_pixels)
  A_dt_square_inv_B_dt.row(0).setConstant(-1/dt);                                 // (1, num_pixels)   
  A_dt_square_inv_B_dt.row(1) = 1/dt * 2 * zeta_omega_n.transpose()               // (1, num_pixels)
                                / omega_n_square.transpose();

  for (int i=0; i<2; ++i) {
    discrete_nlti_lpf_.system.B_tilde[i] = (                                      // (num_pixels, 1)
        ((discrete_nlti_lpf_.system.A[i] - identity_[i])
        * A_dt_square_inv_B_dt.transpose()).rowwise().sum()
        - A_dt_inv_B_dt.row(i).transpose());
    discrete_nlti_lpf_.system.B[i] = (                                            // (num_pixels, 1)
        ((discrete_nlti_lpf_.system.A[i] - identity_[i])
        * A_dt_inv_B_dt.transpose()).rowwise().sum()
        - discrete_nlti_lpf_.system.B_tilde[i]);
  }
}

FloatTypeImagePair PixelBandwidthModel::filter(
    const Image& log_img,                                                         // (height, width) OpenCV matrix with 1 channel
    const Image& img,                                                             // (height, width) OpenCV matrix with 1 channel
    Duration dt_ns) {
  CHECK_EQ(log_img.channels(), 1);
  CHECK_EQ(img.channels(), 1);

  // convert & cast the given OpenCV `ImageFloatType` 2D image matrices to
  // Eigen `FloatType` image row vectors
  const Image input_cv_log_img_rvec = log_img.reshape(1, 1);                      // (1, height * width) or (1, num_pixels) OpenCV matrix with 1 channel
  const Image input_cv_img_rvec = img.reshape(1, 1);                              // (1, height * width) or (1, num_pixels) OpenCV matrix with 1 channel
  control::MatrixXs input_eigen_log_img_rvec{};
  control::RowVectorXs input_eigen_img_rvec{};
  cv::cv2eigen(input_cv_log_img_rvec, input_eigen_log_img_rvec);                  // copy involved, (1, num_pixels)
  cv::cv2eigen(input_cv_img_rvec, input_eigen_img_rvec);                          // copy involved, (1, num_pixels)

  // convert the sampling interval from nanoseconds to seconds
  const FloatType dt = ze::nanosecToSecTrunc(dt_ns);

  control::MatrixXs nlti_subsys_output;
  if (nlti_subsys_order() > 0) {
    // linearize the NLTI continuous-time sub-system at the steady-state, where
    // sub-system output = sub-system input = next system input, & then FOH
    // -discretize it, according to the next image sampling interval `dt`
    linearizeFohNltiLpf(input_eigen_img_rvec, dt);

    // filter the log-image through the NLTI sub-system
    discrete_nlti_lpf_.update(std::move(input_eigen_log_img_rvec));

    // `nlti_subsys_output` is equivalent to `discrete_nlti_lpf_.output()`
    nlti_subsys_output = discrete_nlti_lpf_.state.bottomRows(1);                  // (1, num_pixels)
  } else {
    nlti_subsys_output = std::move(input_eigen_log_img_rvec);                     // (1, num_pixels)
  }

  FloatTypeImagePair output_cv_log_img_pair;                                      // pair of source follower & differencing amplifier output OpenCV log-images
  auto eigen2cv_log_img = std::bind(eigen2cv, std::placeholders::_1,
                                    1, log_img.size().height);
  if (lti_subsys_order() > 0) {
    // FOH-discretize the LTI continuous-time sub-system, according to the
    // next image sampling interval `dt`
    static const bool is_state_preserved = true;
    discrete_lti_filter_.system = control::fohCont2discrete(
        cont_lti_subsys_, dt, is_state_preserved);

    // filter the log-image through the LTI sub-system
    discrete_lti_filter_.update(std::move(nlti_subsys_output));

    // `output_cv_log_img_pair.second` is equivalent to
    // `eigen2cv_log_img(discrete_lti_filter_.output())`
    output_cv_log_img_pair.second = eigen2cv_log_img(                             // (height, width) OpenCV matrix with 1 channel
        discrete_lti_filter_.state.row(lti_subsys_order() - 1));
    output_cv_log_img_pair.first = [&] {                                          // (height, width) OpenCV matrix with 1 channel
      if (is_diff_amp_inf_fast_) {
        return output_cv_log_img_pair.second.clone();
      } else if (lti_hpf_order_ == 1 && lti_lpf_order_ == 1) {
        return eigen2cv_log_img(discrete_lti_filter_.state.row(0)
                                + discrete_lti_filter_.input);
      } else if (lti_lpf_order_ == 1) {
        return eigen2cv_log_img(discrete_lti_filter_.input);
      } else {
        return eigen2cv_log_img(
            discrete_lti_filter_.state.row(lti_subsys_order() - 2));
      }
    }();
  } else {
    output_cv_log_img_pair.second = eigen2cv_log_img(nlti_subsys_output);         // (height, width) OpenCV matrix with 1 channel
    output_cv_log_img_pair.first = output_cv_log_img_pair.second.clone();         // (height, width) OpenCV matrix with 1 channel
  }

  return output_cv_log_img_pair;
}

} // namespace event_camera_simulator
