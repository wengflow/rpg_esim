#include <esim/esim/event_simulator.hpp>
#include <esim/common/utils.hpp>
#include <ze/common/random.hpp>
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <ze/common/time_conversions.hpp>

namespace event_camera_simulator {

void EventSimulator::init(const Image &img, Time time)
{
  VLOG(1) << "Initialized event camera simulator with sensor size: " << img.size();
  VLOG(1) << "and contrast thresholds: C+ = " << config_.Cp << " , C- = " << config_.Cm;
  is_initialized_ = true;
  last_img_ = img.clone();
  last_time_ = time;
  ref_it_ = img.clone();
  size_ = img.size();
  leak_gradient_ = 1e-9 * config_.leak_rate_hz * config_.Cp;  // in log-intensity per nanoseconds

  // initialize reference timestamps with the given timestamp plus a random
  // offset bounded by the refractory period to desynchronize event generation
  ref_timestamp_ = TimestampImage::Constant(img.rows, img.cols, time);
  if (config_.refractory_period_ns > 0) {
    for (int y = 0; y < size_.height; ++y) {
      for (int x = 0; x < size_.width; ++x) {
        ref_timestamp_(y, x) += ze::sampleUniformIntDistribution<Time>(false, 0, config_.refractory_period_ns - 1);
      }
    }
  }

  // initialize static per-pixel contrast sensitivity thresholds
  constexpr ImageFloatType MINIMUM_CONTRAST_THRESHOLD = 0.01;
  CHECK_GE(config_.sigma_Cp, 0.0);
  CHECK_GE(config_.sigma_Cm, 0.0);

  per_pixel_Cp_ = Image(img.size(), config_.Cp);
  per_pixel_Cm_ = Image(img.size(), config_.Cm);
  for (int y = 0; y < size_.height; ++y) {
    for (int x = 0; x < size_.width; ++x) {
      if (config_.sigma_Cp > 0) {
        per_pixel_Cp_(y, x) += ze::sampleNormalDistribution<ImageFloatType>(false, 0, config_.sigma_Cp);
        per_pixel_Cp_(y, x) = std::max(MINIMUM_CONTRAST_THRESHOLD, per_pixel_Cp_(y, x));
      }
      if (config_.sigma_Cm > 0) {
        per_pixel_Cm_(y, x) += ze::sampleNormalDistribution<ImageFloatType>(false, 0, config_.sigma_Cm);
        per_pixel_Cm_(y, x) = std::max(MINIMUM_CONTRAST_THRESHOLD, per_pixel_Cm_(y, x));
      }
    }
  }

}

Events EventSimulator::imageCallback(const ColorImage& img, Time time)
{
  CHECK_GE(time, 0);

  Image preprocessed_img;

  if(config_.simulate_color_events)
  {
    // Convert BGR image to bayered image (for color event simulation)
    colorImageToGrayscaleBayer(img, &preprocessed_img);
  }
  else
  {
    cv::cvtColor(img, preprocessed_img, cv::COLOR_BGR2GRAY);
  }

  if(config_.use_log_image)
  {
    LOG_FIRST_N(INFO, 1) << "Converting the image to log image with eps = " << config_.log_eps << ".";
    cv::log(config_.log_eps + preprocessed_img, preprocessed_img);
  }

  if(!is_initialized_)
  {
    init(preprocessed_img, time);
    return {};
  }

  // For each pixel, check if new events need to be generated since the last image sample
  static constexpr Time TIMESTAMP_TOLERANCE = 100;
  Events events;
  Duration delta_t_ns = time - last_time_;

  CHECK_GT(delta_t_ns, 0u);
  CHECK_EQ(img.size(), size_);

  for (int y = 0; y < size_.height; ++y) {
    for (int x = 0; x < size_.width; ++x) {
      FloatType itdt = preprocessed_img(y, x);
      FloatType it = last_img_(y, x);
      FloatType gradient_at_xy = (itdt - it) / delta_t_ns;
      FloatType leaky_gradient_at_xy = gradient_at_xy + leak_gradient_;
      FloatType pol = (leaky_gradient_at_xy >= 0) ? +1.0 : -1.0;
      FloatType C = (pol > 0) ? per_pixel_Cp_(y, x) : per_pixel_Cm_(y, x);

      while (ref_timestamp_(y, x) <= time) {
        // update the reference log-intensity, if the reference timestamp has been updated
        if (ref_timestamp_(y, x) > last_time_) {   // && (ref_timestamp_(y, x) <= time)
          ref_it_(y, x) = it + gradient_at_xy * (ref_timestamp_(y, x) - last_time_);
        }

        // prevent undefined divide-by-zero behavior when computing intervals
        if (leaky_gradient_at_xy == 0) {
          break;
        }

        // predict the event timestamp
        Time event_timestamp;
        if (ref_timestamp_(y, x) < last_time_) {
          FloatType last_change_at_xy = it - ref_it_(y, x);
          FloatType last_leaky_change_at_xy = last_change_at_xy + leak_gradient_ * (last_time_ - ref_timestamp_(y, x));

          if (last_leaky_change_at_xy >= 0) {
            CHECK_LT(last_leaky_change_at_xy, per_pixel_Cp_(y, x));
          } else {
            CHECK_LT(-last_leaky_change_at_xy, per_pixel_Cm_(y, x));
          }

          FloatType interval_from_last_time = (pol * C - last_leaky_change_at_xy) / leaky_gradient_at_xy;
          CHECK_GT(interval_from_last_time, 0);
          if (interval_from_last_time >= INT64_MAX - last_time_) {    // to prevent integer overflow from casting
            break;
          }
          event_timestamp = last_time_ + static_cast<Time>(std::ceil(interval_from_last_time));
        }
        else {  // else if (ref_timestamp_(y, x) >= last_time_) {
          FloatType interval_from_ref_ts = (pol * C) / leaky_gradient_at_xy;
          CHECK_GT(interval_from_ref_ts, 0);
          if (interval_from_ref_ts >= INT64_MAX - event_timestamp) {  // to prevent integer overflow from casting
            break;
          }
          event_timestamp = ref_timestamp_(y, x) + static_cast<Time>(std::ceil(interval_from_ref_ts));
        }
        CHECK_GT(event_timestamp, last_time_);

        // stop event generation, if the predicted event timestamp exceeds the current image timestamp
        // & the maximum possible leaky change is smaller than the contrast sensitivity threshold
        // (the latter condition is necessary as the computation of the predicted event timestamp 
        //  is less numerically precise than that of the maximum possible leaky change)
        FloatType max_change_at_xy = itdt - ref_it_(y, x);
        FloatType max_leaky_change_at_xy = max_change_at_xy + leak_gradient_ * (time - ref_timestamp_(y, x));
        if (event_timestamp > time){
          if (max_leaky_change_at_xy * pol <= 0 || std::fabs(max_leaky_change_at_xy) < C) {
            break;
          }
          CHECK_LT(event_timestamp, time + TIMESTAMP_TOLERANCE);
        }

        // save the generated event
        events.push_back(Event(x, y, event_timestamp, pol > 0));
        
        // update the reference timestamp
        ref_timestamp_(y, x) = event_timestamp + config_.refractory_period_ns;
      }

    } // end for each pixel
  }

  // update simvars for next loop
  last_time_ = time;
  last_img_ = preprocessed_img.clone(); // it is now the latest image

  // Sort the events by increasing timestamps, since this is what
  // most event processing algorithms expect
  sort(events.begin(), events.end(),
       [](const Event& a, const Event& b) -> bool
  {
    return a.t < b.t;
  });

  return events;
}

} // namespace event_camera_simulator
