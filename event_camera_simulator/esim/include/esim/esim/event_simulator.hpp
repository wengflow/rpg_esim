#pragma once

#include <esim/common/types.hpp>
#include <esim/esim/control.hpp>

namespace event_camera_simulator {

using FloatTypeImage = cv::Mat_<FloatType>;

/*
 * The pixel bandwidth model is, in general, a 5th order Non-Linear Time
 * -Invariant (NLTI) system, formed by a cascade of:
 *    1. 1 x NLTI Low-Pass Filter (LPF), which bandwidth / cutoff-frequency is
 *       linear to the pixel intensity with proportionality constant
 *       `intensity_cutoff_freq_hz_prop_constant`
 *    2. 3 x Linear Time-Invariant (LTI) LPFs, which cutoff-frequencies are,
 *       respectively, given by:
 *          a. `I_pr_cutoff_freq_hz`
 *          b. `I_sf_cutoff_freq_hz`
 *          c. `chg_amp_cutoff_freq_hz`
 *    3. 1 x LTI High-Pass Filter (HPF), which cutoff-frequency is given by
 *       `hpf_cutoff_freq_hz`
 * 
 * A component/sub-system of the pixel bandwidth model is removed/bypassed if:
 *    1. NLTI LPF: Proportionality constant is set to infinity
 *    2. LTI LPF: Cutoff-frequency is set to infinity
 *    3. LTI HPF: Cutoff-frequency is set to 0
 */
class PixelBandwidthModel
{
 public:
  using MatrixXsMap = Eigen::Map<const Eigen::Matrix<ImageFloatType,
                                                     Eigen::Dynamic,
                                                     Eigen::Dynamic,
                                                     Eigen::RowMajor>>;
  using VectorXsMap = Eigen::Map<const Eigen::Matrix<ImageFloatType,
                                                     Eigen::Dynamic, 1>>;
  using RowVectorXsMap = Eigen::Map<const Eigen::Matrix<ImageFloatType,
                                                        1, Eigen::Dynamic>>;

  PixelBandwidthModel(double intensity_cutoff_freq_hz_prop_constant,
                      double I_pr_cutoff_freq_hz,
                      double I_sf_cutoff_freq_hz,
                      double chg_amp_cutoff_freq_hz,
                      double hpf_cutoff_freq_hz);
  int order() const;
  int lti_subsys_order() const;
  int nlti_subsys_order() const;

  // RowVectorXsMap cv2eigen(const Image& img) const;
  // FloatTypeImage eigen2cv(Eigen::Ref<control::RowVectorXs> eigen_img,
  //                         const int width, const int height) const;
  void initState(const Image& init_log_img, const Image& init_img);
  FloatTypeImage filter(const Image& log_img, const Image& img,
                        Duration dt_nanosec);

 private:
  int nlti_lpf_order_;
  int lti_lpf_order_;
  int lti_hpf_order_;

  control::LtiStateSpace cont_lti_subsys_;
  control::DiscreteNltiLpf discrete_nlti_lpf_;
  control::DiscreteLtiFilter discrete_lti_filter_;
};

/*
 * The EventSimulator takes as input a sequence of stamped images,
 * assumed to be sampled at a "sufficiently high" framerate,
 * and simulates the principle of operation of an idea event camera
 * with a constant contrast threshold C.
 * Pixel-wise intensity values are linearly interpolated in time.
 *
 * The pixel-wise voltages are reset with the values from the first image
 * which is passed to the simulator.
 */
class EventSimulator
{
public:

  struct Config
  {
    double Cp;
    double Cm;
    double sigma_Cp;
    double sigma_Cm;
    Duration refractory_period_ns;
    double leak_rate_hz;
    double intensity_cutoff_freq_hz_prop_constant;
    double I_pr_cutoff_freq_hz;
    double I_sf_cutoff_freq_hz;
    double chg_amp_cutoff_freq_hz;
    double hpf_cutoff_freq_hz;
    bool use_log_image;
    double log_eps;
    bool simulate_color_events;
  };

  using TimestampImage = Eigen::Matrix<Time, Eigen::Dynamic, Eigen::Dynamic>;

  EventSimulator(const Config& config)
    : is_initialized_(false),
      pixel_bandwidth_model_(config.intensity_cutoff_freq_hz_prop_constant,
                             config.I_pr_cutoff_freq_hz,
                             config.I_sf_cutoff_freq_hz,
                             config.chg_amp_cutoff_freq_hz,
                             config.hpf_cutoff_freq_hz),
      last_time_(0),
      config_(config)
  {}

  void init(const Image& log_img, const Image& img, Time time);
  Events imageCallback(const ColorImage& color_img, Time time);

private:

  bool is_initialized_;
  PixelBandwidthModel pixel_bandwidth_model_;
  FloatTypeImage last_filtered_log_img_;
  Time last_time_;
  FloatTypeImage ref_filtered_log_img_;
  TimestampImage ref_timestamp_;
  FloatTypeImage per_pixel_Cp_;
  FloatTypeImage per_pixel_Cm_;
  FloatType leak_gradient_;
  cv::Size size_;

  Config config_;
};

} // namespace event_camera_simulator
