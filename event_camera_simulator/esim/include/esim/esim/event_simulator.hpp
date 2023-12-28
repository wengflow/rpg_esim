#pragma once

#include <esim/common/types.hpp>
#include <esim/esim/pixel_bandwidth_model.hpp>

namespace event_camera_simulator {

using FloatTypeImage = cv::Mat_<FloatType>;

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

    double I_p_to_it_ratio_fa;
    double I_dark_fa;
    double A_amp;
    double kappa;
    double V_T_mv;
    double C_p_ff;
    double C_mil_ff;
    double tau_out_us;
    double f_c_lower_hz;
    double f_c_sf_hz;
    double f_c_diff_hz;

    bool use_log_image;
    double log_eps;
    bool simulate_color_events;
  };

  using TimestampImage = Eigen::Matrix<Time, Eigen::Dynamic, Eigen::Dynamic>;

  EventSimulator(const Config& config)
    : is_initialized_(false),
      pixel_bandwidth_model_(config.I_p_to_it_ratio_fa,
                             config.A_amp,
                             config.kappa,
                             config.V_T_mv,
                             config.C_p_ff,
                             config.C_mil_ff,
                             config.tau_out_us,
                             config.f_c_lower_hz,
                             config.f_c_sf_hz,
                             config.f_c_diff_hz,
                             log(0.5 + config.log_eps)),
      last_time_(0),
      config_(config)
  {}

  void init(const Image& log_img, Time time);
  Events imageCallback(const ColorImage& color_img, Time time);

private:

  bool is_initialized_;
  PixelBandwidthModel pixel_bandwidth_model_;
  FloatTypeImage last_sf_log_img_;
  FloatTypeImage last_diff_log_img_;
  Time last_time_;
  FloatTypeImage ref_sf_log_img_;
  TimestampImage ref_timestamp_;
  FloatTypeImage per_pixel_Cp_;
  FloatTypeImage per_pixel_Cm_;
  FloatType leak_gradient_;
  cv::Size size_;

  Config config_;
};

} // namespace event_camera_simulator
