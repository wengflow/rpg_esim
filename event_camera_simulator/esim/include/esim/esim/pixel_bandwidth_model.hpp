#pragma once

#include <vector>
#include <utility>
#include <esim/esim/control.hpp>

namespace event_camera_simulator {

using FloatTypeImagePair = std::pair<FloatTypeImage, FloatTypeImage>;
using StdVectorIntPair = std::pair<std::vector<int>, std::vector<int>>;
using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;
static constexpr double kFromMilli = 1e-3;
static constexpr double kFromMicro = 1e-6;
static constexpr double kFromFemto = 1e-15;

FloatTypeImage eigen2cv(const control::MatrixXs& eigen_mat,
                        int output_channels, int output_height);
StdVectorIntPair getIndexPair(const Eigen::Ref<const ArrayXb>& condition);

/*
 * The pixel bandwidth model is, in general, a 5th-order unity-gain Non-Linear
 * Time-Invariant (NLTI) continuous-time system that is formed by a cascade of:
 *    1. 1 x 2nd-order unity-gain NLTI Low-Pass Filter (LPF), which:
 *          a. Damping Ratio
 *             `zeta = (tau_in + tau_out + (A_amp + 1) * tau_mil)
 *                     / (2 * sqrt((tau_in + tau_mil) * tau_out * (A_loop + 1)))`
 *          b. Natural (Angular) Frequency
 *             `omega_n = sqrt((A_loop + 1) / ((tau_in + tau_mil) * tau_out))`
 *       It models the dynamic/transient response of the photoreceptor circuit.
 *    2. 1 x 1st-order unity-gain LTI High-Pass Filter (HPF), which cutoff
 *       frequency is given by:
 *          a. `f_c_lower`
 *       It models the high-pass characteristics of the pixel circuit or
 *       dynamic/transient response of the high-pass filter present in certain
 *       event cameras.
 *    3. 2 x 1st-order unity-gain Linear Time-Invariant (LTI) LPFs, which cutoff
 *       frequencies are, respectively, given by:
 *          a. `f_c_sf`
 *          b. `f_c_diff`
 *       They, respectively, model the dynamic/transient response of the:
 *          a. source follower buffer
 *          b. change/differencing amplifier
 * 
 * A component/sub-system of the pixel bandwidth model is removed/bypassed if:
 *    1. 2nd-order NLTI LPF: Either
 *          a. `tau_out` is 0 and `I_p_to_it_ratio` is infinite 
 *          b. `tau_out`, `C_in` and `C_mil` are 0
 *          c. `A_amp` and `I_p_to_it_ratio` are infinite
 *          d. `A_amp` is infinite and `C_mil` is 0
 *    2. 1st-order LTI HPF: `f_c` is 0
 *    3. 1st-order LTI LPF: `f_c` is infinite
 * 
 * Furthermore, the 2nd-order NLTI LPF reduces to a 1st-order LPF that is:
 *    1. LTI with `omega_c = (A_loop + 1) / tau_out`,
 *       when `I_p_to_it_ratio` is infinite, or `C_in` and `C_mil` are 0
 *    2. NLTI with `omega_c = (A_loop + 1) / (tau_in + (A_amp + 1) * tau_mil)`,
 *       when `tau_out` is 0
 *    3. NLTI with `omega_c = 1 / (A_cl * tau_mil)`,
 *       when `A_amp` is infinite
 * where `omega_c = 2 * pi * f_c` is the cutoff angular frequency.
 * 
 * The input to the model is the dark intensity-accounted image pixel
 * log-intensity `log(it + dark_it)`, which is equal to the photocurrent
 * -equivalent image pixel log-intensity `log(I / I_p_to_it_ratio)`. The outputs
 * of the model are the outputs of the source follower buffer & differencing
 * amplifier 1st-order LTI LPFs.
 *
 * The pixel bandwidth model is implemented as a Linear Time-Varying (LTV)
 * discrete-time system by:
 *    1. Linearizing the NLTI LPF continuous-time sub-system at the steady
 *       -state, where the sub-system output = sub-system input = next system
 *       input
 *    2. Discretizing the linearized NLTI LPF & LTI continuous-time sub-systems
 *       separately, assuming First-Order Hold (FOH) on their respective inputs
 *    3. Cascading the discretized NLTI & LTI sub-systems
 * at each instant where the next input is available.
 * 
 * NOTE:
 *    1. The linearized 2nd-order NLTI & 3rd-order LTI continuous-time
 *       sub-systems are not discretized jointly, as it would incur high
 *       computational & memory cost, due to varying linearized 5th-order NLTI
 *       system for each pixel (more precisely, for each different system input)
 *    2. Discretized sub-systems are represented as LTV state-space models in
 *       non-standard form, where their states match that of their continuous
 *       -time counterpart, to accommodate variations in image sampling interval
 *       `dt` and linearization steady-states (for NLTI sub-system)
 */
class PixelBandwidthModel
{
 public:
  PixelBandwidthModel(double I_p_to_it_ratio_fa,
                      double A_amp,
                      double kappa,
                      double V_T_mv,
                      double C_p_ff,
                      double C_mil_ff,
                      double tau_out_us,
                      double f_c_lower_hz,
                      double f_c_sf_hz,
                      double f_c_diff_hz,
                      double assumed_log_it_mean);
  unsigned int order() const;
  unsigned int nlti_subsys_order() const;
  unsigned int lti_subsys_order() const;

  bool is_input_inf_fast() const;
  bool is_fb_amp_inf_fast() const;
  bool is_A_amp_inf() const;
  bool is_mil_negligible() const;

  void initState(const Image& init_log_img);
  FloatTypeImagePair filter(const Image& log_img, const Image& img,
                            Duration dt_nanosec);

 private:
  constexpr double A_loop() {
    return A_amp_ / A_cl_;
  }
  void linearizeFohNltiLpf(
      const Eigen::Ref<const control::RowVectorXs>& linearization_img_rvec,
      FloatType dt);

  unsigned int nlti_lpf_order_;
  unsigned int lti_hpf_order_;
  unsigned int lti_lpf_order_;
  const bool is_diff_amp_inf_fast_;

  // all quantities represented in SI units (i.e. no milli-, femto- etc.)
  const double I_p_to_it_ratio_;
  const double A_amp_;
  const double A_cl_;
  const double Q_in_;
  const double Q_mil_;
  const double tau_out_;
  control::LtiStateSpace cont_lti_subsys_;

  const double assumed_log_it_mean_;

  std::vector<control::ArrayXXs> identity_;
  control::BatchDiscreteLtiFilter discrete_nlti_lpf_;
  control::DiscreteLtiFilter discrete_lti_filter_;
};

} // namespace event_camera_simulator
