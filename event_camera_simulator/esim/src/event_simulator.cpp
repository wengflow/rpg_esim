#include <esim/esim/event_simulator.hpp>
#include <esim/common/utils.hpp>
#include <array>
#include <cmath>
#include <ze/common/random.hpp>
#include <ze/common/time_conversions.hpp>
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace event_camera_simulator {

void EventSimulator::init(const Image& log_img, Time time) {
  VLOG(1) << "Initialized event camera simulator with sensor size: "
          << log_img.size();
  VLOG(1) << "and contrast thresholds: C+ = " << config_.Cp
          << " , C- = " << config_.Cm;
  is_initialized_ = true;
  last_time_ = time;
  size_ = log_img.size();

  // initialize reference timestamps with the given timestamp plus a random
  // offset bounded by the refractory period to desynchronize event generation
  ref_timestamp_ = TimestampImage::Constant(log_img.rows, log_img.cols, time);
  if (config_.refractory_period_ns > 0) {
    for (int y = 0; y < size_.height; ++y) {
      for (int x = 0; x < size_.width; ++x) {
        ref_timestamp_(y, x) += ze::sampleUniformIntDistribution<Time>(
            false, 0, config_.refractory_period_ns - 1);
      }
    }
  }

  // initialize static per-pixel contrast sensitivity thresholds
  const FloatType MINIMUM_CONTRAST_THRESHOLD = 0.01;
  CHECK_GE(config_.sigma_Cp, 0.0);
  CHECK_GE(config_.sigma_Cm, 0.0);

  per_pixel_Cp_ = FloatTypeImage(log_img.size(), config_.Cp);
  per_pixel_Cm_ = FloatTypeImage(log_img.size(), config_.Cm);
  for (int y = 0; y < size_.height; ++y) {
    for (int x = 0; x < size_.width; ++x) {
      if (config_.sigma_Cp > 0) {
        per_pixel_Cp_(y, x) += ze::sampleNormalDistribution<FloatType>(
            false, 0, config_.sigma_Cp);
        per_pixel_Cp_(y, x) = std::max(MINIMUM_CONTRAST_THRESHOLD,
                                       per_pixel_Cp_(y, x));
      }
      if (config_.sigma_Cm > 0) {
        per_pixel_Cm_(y, x) += ze::sampleNormalDistribution<FloatType>(
            false, 0, config_.sigma_Cm);
        per_pixel_Cm_(y, x) = std::max(MINIMUM_CONTRAST_THRESHOLD,
                                       per_pixel_Cm_(y, x));
      }
    }
  }

  // initialize the state & outputs of the pixel bandwidth model, as well as the
  // reference source follower log-image
  pixel_bandwidth_model_.initState(log_img);
  log_img.convertTo(last_sf_log_img_, last_sf_log_img_.depth());
  last_diff_log_img_ = last_sf_log_img_.clone();
  ref_sf_log_img_ = last_sf_log_img_.clone();
  ref_diff_log_img_ = last_sf_log_img_.clone();
}

Events EventSimulator::imageCallback(const ColorImage& color_img, Time time)
{
  CHECK_GE(time, 0);

  Image img;
  if(config_.simulate_color_events) {
    // Convert BGR image to bayered image (for color event simulation)
    colorImageToGrayscaleBayer(color_img, &img);
  } else {
    cv::cvtColor(color_img, img, cv::COLOR_BGR2GRAY);
  }

  // if the input is a log-image, convert it back to an (linear-)image
  if (!config_.use_log_image) {
    cv::exp(img, img);
  }

  // add the dark current-equivalent image pixel intensity (i.e. dark
  // intensity), and if necessary an epsilon value, to the image & then convert
  // it to a log-image
  const FloatType dark_it = (config_.I_dark_fa * kFromFemto)
                            / (config_.I_p_to_it_ratio_fa * kFromFemto);
  img += dark_it;
  if(config_.use_log_image) {
    LOG_FIRST_N(INFO, 1) << "Adding eps = " << config_.log_eps 
                         << " to the image before log-image conversion.";
    img += config_.log_eps;
  }
  Image log_img;
  cv::log(img, log_img);

  if(!is_initialized_) {
    init(log_img, time);
    return {};
  }

  // filter the log-image with the pixel bandwidth model, if necessary
  FloatTypeImage sf_log_img;
  FloatTypeImage diff_log_img;
  Duration dt_ns = time - last_time_;
  if (pixel_bandwidth_model_.order() > 0) {
    FloatTypeImagePair output_log_img_pair = pixel_bandwidth_model_.filter(
        log_img, img, dt_ns);
    sf_log_img = std::move(output_log_img_pair.first);
    diff_log_img = std::move(output_log_img_pair.second);
  } else {
    // bypass the pixel bandwidth model, cast the log-image from `Image` to
    // `FloatTypeImage` & duplicate it
    log_img.convertTo(sf_log_img, sf_log_img.depth());
    diff_log_img = sf_log_img.clone();
  }

  // for each pixel, check if new events need to be generated
  /**
   * NOTE:
   *    Events are generated based on changes in log-intensity, measured between
   *    the differencing amplifier 1st-order LTI LPF output, and the source
   *    follower buffer 1st-order LTI LPF output at a given reference timestamp.
   *    It also approximately considers state (i.e. output) reset of the
   *    differencing amplifier LPF to its input (i.e. source follower buffer LPF
   *    output) after an event is generated, throughout the refractory period.
   **/
  static constexpr Time TIMESTAMP_TOLERANCE = 100;
  Events events;

  CHECK_GT(dt_ns, 0u);
  CHECK_EQ(color_img.size(), size_);

  for (int y = 0; y < size_.height; ++y) {
    for (int x = 0; x < size_.width; ++x) {
      FloatType diff_log_it_dt = diff_log_img(y, x);
      FloatType diff_log_it = last_diff_log_img_(y, x);
      FloatType diff_log_it_grad = (diff_log_it_dt - diff_log_it) / dt_ns;
      FloatType leaky_diff_log_it_grad = diff_log_it_grad + leak_log_it_grad_;
      std::array<FloatType, 2> C = {-per_pixel_Cm_(y, x), per_pixel_Cp_(y, x)};
      bool pol;

      while (ref_timestamp_(y, x) <= time) {
        // update the source follower buffer LPF reference output log-intensity,
        // if the reference timestamp has been updated
        if (ref_timestamp_(y, x) > last_time_) {   // && (ref_timestamp_(y, x) <= time)
          FloatType sf_log_it_dt = sf_log_img(y, x);
          FloatType sf_log_it = last_sf_log_img_(y, x);
          FloatType sf_log_it_grad = (sf_log_it_dt - sf_log_it) / dt_ns;

          Time dt_ref_last_ns = ref_timestamp_(y, x) - last_time_;
          ref_sf_log_img_(y, x) = sf_log_it + sf_log_it_grad * dt_ref_last_ns;
          ref_diff_log_img_(y, x) = (
              diff_log_it + diff_log_it_grad * dt_ref_last_ns);
        }

        // predict the event timestamp
        Time event_timestamp;
        FloatType ref_change = ref_diff_log_img_(y, x) - ref_sf_log_img_(y, x);
        if (ref_timestamp_(y, x) < last_time_) {
          Time dt_last_ref_ns = last_time_ - ref_timestamp_(y, x);
          FloatType last_diff_change = diff_log_it - ref_diff_log_img_(y, x);
          FloatType last_leaky_diff_change = (
              last_diff_change + leak_log_it_grad_ * dt_last_ref_ns);
          FloatType last_exp_converging_offset = (
              exp_converging_line::evalOffset(dt_last_ref_ns, omega_c_diff_,
                                              ref_change));
          FloatType last_leaky_change = last_leaky_diff_change
                                        + last_exp_converging_offset;
          CHECK_GT(last_leaky_change, C[0]);
          CHECK_LT(last_leaky_change, C[1]);

          // stop event generation, if the leaky differencing amplifier LPF
          // output log-intensity gradient is zero, and the effective
          // exponentially converging log-intensity offset has smaller magnitude
          // than the effective contast sensitivity thresholds
          std::array<FloatType, 2> eff_C = {C[0] - last_leaky_change,
                                            C[1] - last_leaky_change};
          FloatType eff_exp_converging_offset = ref_change
                                                - last_exp_converging_offset;
          if ((leaky_diff_log_it_grad == 0)
              && (eff_C[0] < eff_exp_converging_offset)
              && (eff_exp_converging_offset < eff_C[1])) {
            break;
          }

          bool has_critical_log_it = exp_converging_line::hasCriticalPt(
              leaky_diff_log_it_grad, omega_c_diff_, eff_exp_converging_offset);
          FloatType critical_log_it;
          if (has_critical_log_it) {
            critical_log_it = exp_converging_line::evalCriticalPt(
                leaky_diff_log_it_grad, omega_c_diff_,
                eff_exp_converging_offset);
          }

          FloatType interval_from_last_time;
          if (!has_critical_log_it || ((eff_C[0] < critical_log_it)
                                       && (critical_log_it < eff_C[1]))) {
            pol = (leaky_diff_log_it_grad >= 0);
          } else {
            pol = (critical_log_it > 0);
          }
          interval_from_last_time = exp_converging_line::solve(
              eff_C[pol], leaky_diff_log_it_grad, omega_c_diff_,
              eff_exp_converging_offset);
          CHECK_GT(interval_from_last_time, 0);

          // to prevent integer overflow from casting
          if (interval_from_last_time >= INT64_MAX - last_time_) {
            break;
          }
          event_timestamp = (
              last_time_
              + static_cast<Time>(std::ceil(interval_from_last_time)));
          CHECK_GT(event_timestamp, last_time_);
        }
        else {  // else if (ref_timestamp_(y, x) >= last_time_) {
          // stop event generation, if the leaky differencing amplifier LPF
          // output log-intensity gradient is zero, and the change at reference
          // timestamp / exponentially converging log-intensity offset has
          // smaller magnitude than the contast sensitivity thresholds
          if ((leaky_diff_log_it_grad == 0)
              && (C[0] < ref_change) && (ref_change < C[1])) {
            break;
          }

          bool has_critical_log_it = exp_converging_line::hasCriticalPt(
              leaky_diff_log_it_grad, omega_c_diff_, ref_change);
          FloatType critical_log_it;
          if (has_critical_log_it) {
            critical_log_it = exp_converging_line::evalCriticalPt(
                leaky_diff_log_it_grad, omega_c_diff_, ref_change);
          }

          FloatType interval_from_ref_ts;
          if (!has_critical_log_it || ((C[0] < critical_log_it)
                                       && (critical_log_it < C[1]))) {
            pol = (leaky_diff_log_it_grad >= 0);
          } else {
            pol = (critical_log_it > 0);
          }
          interval_from_ref_ts = exp_converging_line::solve(
              C[pol], leaky_diff_log_it_grad, omega_c_diff_, ref_change);
          CHECK_GT(interval_from_ref_ts, 0);

          // to prevent integer overflow from casting
          if (interval_from_ref_ts >= INT64_MAX - ref_timestamp_(y, x)) {
            break;
          }
          event_timestamp = (
              ref_timestamp_(y, x)
              + static_cast<Time>(std::ceil(interval_from_ref_ts)));
          CHECK_GT(event_timestamp, ref_timestamp_(y, x));
        }

        // stop event generation, if the predicted event timestamp exceeds the 
        // current image timestamp & the maximum possible leaky change has
        // smaller magnitude than the contrast sensitivity thresholds (the
        // latter condition is necessary as the computation of the predicted
        // event timestamp is less numerically precise than that of the max.
        // possible leaky change)
        if (event_timestamp > time){
          Time dt_curr_ref_ns = time - ref_timestamp_(y, x);
          FloatType max_diff_change = diff_log_it_dt - ref_diff_log_img_(y, x);
          FloatType max_leaky_diff_change = (
              max_diff_change + leak_log_it_grad_ * dt_curr_ref_ns);
          FloatType max_exp_converging_offset = exp_converging_line::evalOffset(
              dt_curr_ref_ns, omega_c_diff_, ref_change);
          FloatType max_leaky_change = max_leaky_diff_change
                                       + max_exp_converging_offset;

          if (C[0] < max_leaky_change && max_leaky_change < C[1]) {
            break;
          }
          CHECK_LT(event_timestamp, time + TIMESTAMP_TOLERANCE);
        }

        // save the generated event
        events.emplace_back(x, y, event_timestamp, pol);

        // update the reference timestamp
        ref_timestamp_(y, x) = event_timestamp + config_.refractory_period_ns;
      }

    } // end for each pixel
  }

  // update simvars for next loop
  last_time_ = time;
  last_sf_log_img_ = std::move(sf_log_img);     // it is now the latest image
  last_diff_log_img_ = std::move(diff_log_img);

  // Sort the events by increasing timestamps, since this is what
  // most event processing algorithms expect
  sort(events.begin(), events.end(),
       [](const Event& a, const Event& b) -> bool { return a.t < b.t; });

  return events;
}

} // namespace event_camera_simulator
