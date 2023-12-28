#include <esim/esim/event_simulator.hpp>
#include <esim/common/utils.hpp>
#include <ze/common/random.hpp>
#include <ze/common/time_conversions.hpp>
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace event_camera_simulator {

void EventSimulator::init(const Image& log_img, Time time) {
  VLOG(1) << "Initialized event camera simulator with sensor size: " << log_img.size();
  VLOG(1) << "and contrast thresholds: C+ = " << config_.Cp << " , C- = " << config_.Cm;
  is_initialized_ = true;
  last_time_ = time;
  size_ = log_img.size();
  leak_gradient_ = 1e-9 * config_.leak_rate_hz * config_.Cp;  // in log-intensity per nanoseconds

  // initialize reference timestamps with the given timestamp plus a random
  // offset bounded by the refractory period to desynchronize event generation
  ref_timestamp_ = TimestampImage::Constant(log_img.rows, log_img.cols, time);
  if (config_.refractory_period_ns > 0) {
    for (int y = 0; y < size_.height; ++y) {
      for (int x = 0; x < size_.width; ++x) {
        ref_timestamp_(y, x) += ze::sampleUniformIntDistribution<Time>(false, 0, config_.refractory_period_ns - 1);
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
        per_pixel_Cp_(y, x) += ze::sampleNormalDistribution<FloatType>(false, 0, config_.sigma_Cp);
        per_pixel_Cp_(y, x) = std::max(MINIMUM_CONTRAST_THRESHOLD, per_pixel_Cp_(y, x));
      }
      if (config_.sigma_Cm > 0) {
        per_pixel_Cm_(y, x) += ze::sampleNormalDistribution<FloatType>(false, 0, config_.sigma_Cm);
        per_pixel_Cm_(y, x) = std::max(MINIMUM_CONTRAST_THRESHOLD, per_pixel_Cm_(y, x));
      }
    }
  }

  // initialize the state & outputs of the pixel bandwidth model, as well as the
  // reference source follower log-image
  pixel_bandwidth_model_.initState(log_img);
  log_img.convertTo(last_sf_log_img_, last_sf_log_img_.depth());
  last_diff_log_img_ = last_sf_log_img_.clone();
  ref_sf_log_img_ = last_sf_log_img_.clone();
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
  Duration dt_nanosec = time - last_time_;
  if (pixel_bandwidth_model_.order() > 0) {
    FloatTypeImagePair output_log_img_pair = pixel_bandwidth_model_.filter(
        log_img, img, dt_nanosec);
    sf_log_img = std::move(output_log_img_pair.first);
    diff_log_img = std::move(output_log_img_pair.second);
  } else {
    // bypass the pixel bandwidth model, cast the log-image from `Image` to
    // `FloatTypeImage` & duplicate it
    log_img.convertTo(sf_log_img, sf_log_img.depth());
    diff_log_img = sf_log_img.clone();
  }

  // For each pixel, check if new events need to be generated since the last image sample
  static constexpr Time TIMESTAMP_TOLERANCE = 100;
  Events events;

  CHECK_GT(dt_nanosec, 0u);
  CHECK_EQ(color_img.size(), size_);

  for (int y = 0; y < size_.height; ++y) {
    for (int x = 0; x < size_.width; ++x) {
      FloatType diff_log_it_dt = diff_log_img(y, x);
      FloatType diff_log_it = last_diff_log_img_(y, x);
      FloatType diff_gradient_at_xy = (diff_log_it_dt - diff_log_it) / dt_nanosec;
      FloatType leaky_diff_gradient_at_xy = diff_gradient_at_xy + leak_gradient_;
      FloatType pol = (leaky_diff_gradient_at_xy >= 0) ? +1.0 : -1.0;
      FloatType C = (pol > 0) ? per_pixel_Cp_(y, x) : per_pixel_Cm_(y, x);

      while (ref_timestamp_(y, x) <= time) {
        // update the reference source follower output log-intensity, if the reference timestamp has been updated
        if (ref_timestamp_(y, x) > last_time_) {   // && (ref_timestamp_(y, x) <= time)
          FloatType sf_log_it_dt = sf_log_img(y, x);
          FloatType sf_log_it = last_sf_log_img_(y, x);
          FloatType sf_gradient_at_xy = (sf_log_it_dt - sf_log_it) / dt_nanosec;

          ref_sf_log_img_(y, x) = sf_log_it + sf_gradient_at_xy * (ref_timestamp_(y, x) - last_time_);
        }

        // prevent undefined divide-by-zero behavior when computing intervals
        if (leaky_diff_gradient_at_xy == 0) {
          break;
        }

        // predict the event timestamp
        Time event_timestamp;
        if (ref_timestamp_(y, x) < last_time_) {
          FloatType last_change_at_xy = diff_log_it - ref_sf_log_img_(y, x);
          FloatType last_leaky_change_at_xy = last_change_at_xy + leak_gradient_ * (last_time_ - ref_timestamp_(y, x));

          if (last_leaky_change_at_xy >= 0) {
            CHECK_LT(last_leaky_change_at_xy, per_pixel_Cp_(y, x));
          } else {
            CHECK_LT(-last_leaky_change_at_xy, per_pixel_Cm_(y, x));
          }

          FloatType interval_from_last_time = (pol * C - last_leaky_change_at_xy) / leaky_diff_gradient_at_xy;
          CHECK_GT(interval_from_last_time, 0);
          if (interval_from_last_time >= INT64_MAX - last_time_) {        // to prevent integer overflow from casting
            break;
          }
          event_timestamp = last_time_ + static_cast<Time>(std::ceil(interval_from_last_time));
          CHECK_GT(event_timestamp, last_time_);
        }
        else {  // else if (ref_timestamp_(y, x) >= last_time_) {
          FloatType ref_diff_log_it = diff_log_it + diff_gradient_at_xy * (ref_timestamp_(y, x) - last_time_);
          FloatType ref_change_at_xy = ref_diff_log_it - ref_sf_log_img_(y, x);
          FloatType interval_from_ref_ts = (pol * C - ref_change_at_xy) / leaky_diff_gradient_at_xy;

          CHECK_GT(interval_from_ref_ts, 0);
          CHECK_LT(ref_change_at_xy, per_pixel_Cp_(y, x));
          CHECK_LT(-ref_change_at_xy, per_pixel_Cm_(y, x));

          if (interval_from_ref_ts >= INT64_MAX - ref_timestamp_(y, x)) { // to prevent integer overflow from casting
            break;
          }
          event_timestamp = ref_timestamp_(y, x) + static_cast<Time>(std::ceil(interval_from_ref_ts));
          CHECK_GT(event_timestamp, ref_timestamp_(y, x));
        }

        // stop event generation, if the predicted event timestamp exceeds the current image timestamp
        // & the maximum possible leaky change is smaller than the contrast sensitivity threshold
        // (the latter condition is necessary as the computation of the predicted event timestamp 
        //  is less numerically precise than that of the maximum possible leaky change)
        FloatType max_change_at_xy = diff_log_it_dt - ref_sf_log_img_(y, x);
        FloatType max_leaky_change_at_xy = max_change_at_xy + leak_gradient_ * (time - ref_timestamp_(y, x));
        if (event_timestamp > time){
          if (max_leaky_change_at_xy * pol <= 0 || std::fabs(max_leaky_change_at_xy) < C) {
            break;
          }
          CHECK_LT(event_timestamp, time + TIMESTAMP_TOLERANCE);
        }

        // save the generated event
        events.emplace_back(x, y, event_timestamp, pol > 0);

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
