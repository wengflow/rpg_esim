#pragma once

#include <esim/common/types.hpp>

namespace event_camera_simulator {

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
    bool use_log_image;
    double log_eps;
  };

  using TimestampImage = Eigen::Matrix<Time, Eigen::Dynamic, Eigen::Dynamic>;

  EventSimulator(const Config& config)
    : config_(config),
      is_initialized_(false),
      last_time_(0)
  {}

  void init(const Image& img, Time time);
  Events imageCallback(const Image& img, Time time);

private:
  bool is_initialized_;
  Image last_img_;
  Time last_time_;
  Image ref_it_;
  TimestampImage ref_timestamp_;
  Image per_pixel_Cp_;
  Image per_pixel_Cm_;
  FloatType leak_gradient_;
  cv::Size size_;

  Config config_;
};

} // namespace event_camera_simulator
