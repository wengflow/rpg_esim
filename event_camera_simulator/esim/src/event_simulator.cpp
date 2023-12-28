#include <esim/esim/event_simulator.hpp>
#include <esim/common/utils.hpp>
#include <ze/common/random.hpp>
#include <ze/common/time_conversions.hpp>
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <boost/math/special_functions/expint.hpp>

namespace event_camera_simulator {

void EventSimulator::init(const Image& log_img, const Image& img, Time time)
{
  VLOG(1) << "Initialized event camera simulator with sensor size: " << img.size();
  VLOG(1) << "and contrast thresholds: C+ = " << config_.Cp << " , C- = " << config_.Cm;
  is_initialized_ = true;
  last_time_ = time;
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
  const FloatType MINIMUM_CONTRAST_THRESHOLD = 0.01;
  CHECK_GE(config_.sigma_Cp, 0.0);
  CHECK_GE(config_.sigma_Cm, 0.0);

  per_pixel_Cp_ = Image(img.size(), config_.Cp);
  per_pixel_Cm_ = Image(img.size(), config_.Cm);
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

  // initialize the state & output of the pixel bandwidth model, as well as the
  // reference filtered log-image
  pixel_bandwidth_model_.initState(log_img, img);
  last_filtered_log_img_ = log_img.clone();
  ref_filtered_log_img_ = log_img.clone();
}

Events EventSimulator::imageCallback(const ColorImage& color_img, Time time)
{
  CHECK_GE(time, 0);

  Image img;
  if(config_.simulate_color_events)
  {
    // Convert BGR image to bayered image (for color event simulation)
    colorImageToGrayscaleBayer(color_img, &img);
  }
  else
  {
    cv::cvtColor(color_img, img, cv::COLOR_BGR2GRAY);
  }

  Image log_img;
  if(config_.use_log_image)
  {
    LOG_FIRST_N(INFO, 1) << "Converting the image to log image with eps = " << config_.log_eps << ".";
    cv::log(config_.log_eps + img, log_img);
  }
  else
  {
    log_img = std::move(img);
    cv::exp(log_img, img);
  }

  if(!is_initialized_)
  {
    init(log_img, img, time);
    return {};
  }

  // filter the log-image with the pixel bandwidth model, if necessary
  FloatTypeImage filtered_log_img;
  Duration dt_nanosec = time - last_time_;
  if (pixel_bandwidth_model_.order() > 0) {
    filtered_log_img = pixel_bandwidth_model_.filter(log_img, img, dt_nanosec);
  } else {
    // bypass the pixel bandwidth model & cast the log-image from `Image` to
    // `FloatTypeImage`
    log_img.convertTo(filtered_log_img, filtered_log_img.depth());
  }

  // For each pixel, check if new events need to be generated since the last image sample
  static constexpr Time TIMESTAMP_TOLERANCE = 100;
  Events events;

  CHECK_GT(dt_nanosec, 0u);
  CHECK_EQ(color_img.size(), size_);

  for (int y = 0; y < size_.height; ++y) {
    for (int x = 0; x < size_.width; ++x) {
      FloatType log_it_dt = filtered_log_img(y, x);
      FloatType log_it = last_filtered_log_img_(y, x);
      FloatType gradient_at_xy = (log_it_dt - log_it) / dt_nanosec;
      FloatType leaky_gradient_at_xy = gradient_at_xy + leak_gradient_;
      FloatType pol = (leaky_gradient_at_xy >= 0) ? +1.0 : -1.0;
      FloatType C = (pol > 0) ? per_pixel_Cp_(y, x) : per_pixel_Cm_(y, x);

      while (ref_timestamp_(y, x) <= time) {
        // update the reference log-intensity, if the reference timestamp has been updated
        if (ref_timestamp_(y, x) > last_time_) {   // && (ref_timestamp_(y, x) <= time)
          ref_filtered_log_img_(y, x) = log_it + gradient_at_xy * (ref_timestamp_(y, x) - last_time_);
        }

        // prevent undefined divide-by-zero behavior when computing intervals
        if (leaky_gradient_at_xy == 0) {
          break;
        }

        // predict the event timestamp
        Time event_timestamp;
        if (ref_timestamp_(y, x) < last_time_) {
          FloatType last_change_at_xy = log_it - ref_filtered_log_img_(y, x);
          FloatType last_leaky_change_at_xy = last_change_at_xy + leak_gradient_ * (last_time_ - ref_timestamp_(y, x));

          if (last_leaky_change_at_xy >= 0) {
            CHECK_LT(last_leaky_change_at_xy, per_pixel_Cp_(y, x));
          } else {
            CHECK_LT(-last_leaky_change_at_xy, per_pixel_Cm_(y, x));
          }

          FloatType interval_from_last_time = (pol * C - last_leaky_change_at_xy) / leaky_gradient_at_xy;
          CHECK_GT(interval_from_last_time, 0);
          if (interval_from_last_time >= INT64_MAX - last_time_) {        // to prevent integer overflow from casting
            break;
          }
          event_timestamp = last_time_ + static_cast<Time>(std::ceil(interval_from_last_time));
          CHECK_GT(event_timestamp, last_time_);
        }
        else {  // else if (ref_timestamp_(y, x) >= last_time_) {
          FloatType interval_from_ref_ts = (pol * C) / leaky_gradient_at_xy;
          CHECK_GT(interval_from_ref_ts, 0);
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
        FloatType max_change_at_xy = log_it_dt - ref_filtered_log_img_(y, x);
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
  last_filtered_log_img_ = std::move(filtered_log_img);   // it is now the latest image

  // Sort the events by increasing timestamps, since this is what
  // most event processing algorithms expect
  sort(events.begin(), events.end(),
       [](const Event& a, const Event& b) -> bool { return a.t < b.t; });

  return events;
}

PixelBandwidthModel::PixelBandwidthModel(
    double intensity_cutoff_freq_hz_prop_constant,
    double I_pr_cutoff_freq_hz,
    double I_sf_cutoff_freq_hz,
    double chg_amp_cutoff_freq_hz,
    double hpf_cutoff_freq_hz) 
    : lti_lpf_order_{0},
      lti_hpf_order_{0},
      nlti_lpf_order_{0},
      cont_lti_subsys_{},
      discrete_nlti_lpf_{intensity_cutoff_freq_hz_prop_constant},
      discrete_lti_filter_{} { 
  // infer the desired system order & cutoff freq. for the pixel bandwidth model
  const std::array<double, 3> lti_lpf_cutoff_freqs_hz = {
      I_pr_cutoff_freq_hz, I_sf_cutoff_freq_hz, chg_amp_cutoff_freq_hz};
  std::vector<double> finite_lti_lpf_cutoff_freqs_hz = {};
  for (const double cutoff_freq_hz : lti_lpf_cutoff_freqs_hz) {
    if (!std::isinf(cutoff_freq_hz))
      finite_lti_lpf_cutoff_freqs_hz.push_back(cutoff_freq_hz);
  }

  lti_lpf_order_ = finite_lti_lpf_cutoff_freqs_hz.size();
  lti_hpf_order_ = static_cast<int>(hpf_cutoff_freq_hz != 0);
  const int c_lti_subsys_order = lti_subsys_order();  // cached copy

  const int nlti_lpf_order = static_cast<int>(
      !std::isinf(intensity_cutoff_freq_hz_prop_constant));
  nlti_lpf_order_ = nlti_lpf_order;

  // return, if the LTI sub-system of the pixel bandwidth model is not required
  if (c_lti_subsys_order == 0)
    return;

  // initialize the continuous state-space model for the LTI sub-system
  cont_lti_subsys_.A = control::MatrixXs::Zero(c_lti_subsys_order,
                                               c_lti_subsys_order);
  cont_lti_subsys_.B = control::VectorXs::Zero(c_lti_subsys_order);
  cont_lti_subsys_.C = control::RowVectorXs::Zero(c_lti_subsys_order);
  cont_lti_subsys_.D = control::RowVectorXs::Zero(1);

  if (lti_hpf_order_ == 1) {
    cont_lti_subsys_.A(0, 0) = -2 * EIGEN_PI * hpf_cutoff_freq_hz;
    cont_lti_subsys_.B(0) = cont_lti_subsys_.A(0, 0);

    if (c_lti_subsys_order == 1)
      cont_lti_subsys_.D(0) = 1;
  }

  for (int i=0; i<lti_lpf_order_; ++i) {
    const int coord = i + lti_hpf_order_;
    cont_lti_subsys_.A(coord, coord) = (
        -2 * EIGEN_PI * finite_lti_lpf_cutoff_freqs_hz[i]);
    if (coord-1 >= 0)
      cont_lti_subsys_.A(coord, coord-1) = -cont_lti_subsys_.A(coord, coord);
    if (i == 0)
      cont_lti_subsys_.B(coord) = cont_lti_subsys_.A(coord, coord-1);
  }
  cont_lti_subsys_.C(c_lti_subsys_order - 1) = 1;
}

int PixelBandwidthModel::order() const {
  return lti_subsys_order() + nlti_subsys_order();
}

int PixelBandwidthModel::lti_subsys_order() const {
  return lti_lpf_order_ + lti_hpf_order_;
}

int PixelBandwidthModel::nlti_subsys_order() const {
  return nlti_lpf_order_;
}

// PixelBandwidthModel::RowVectorXsMap PixelBandwidthModel::cv2eigen(
//     const Image& img) const {
//   const cv::Size img_size = img.size();
//   const int num_pixels = img_size.height * img_size.width;

//   /* 
//    * NOTE: 
//    *    OpenCV & Eigen matrix entries are stored in row-major & column-major 
//    *    order, respectively. 
//    */
//   // `img` OpenCV matrix is flattened to an Eigen row vector in row-major order
//   // TODO: map from `ImageFloatType` to `FloatType` invalid
//   return RowVectorXsMap{img.ptr<ImageFloatType>(), num_pixels};
// }

// PixelBandwidthModel::FloatTypeImage PixelBandwidthModel::eigen2cv(
//     Eigen::Ref<control::RowVectorXs> eigen_img,
//     const int height, const int width) const {
//   return FloatTypeImage{height, width, eigen_img.data()};
// }

void PixelBandwidthModel::initState(const Image& init_log_img,
                                    const Image& init_img) {
  // copy the given OpenCV images to Eigen image row vectors
  const Image init_cv_log_img_rvec = init_log_img.reshape(1, 1);
  control::RowVectorXs init_eigen_log_img_rvec{};
  cv::cv2eigen(init_cv_log_img_rvec, init_eigen_log_img_rvec);  // copy involved

  const Image init_cv_img_rvec = init_img.reshape(1, 1);
  control::RowVectorXs init_eigen_img_rvec{};
  cv::cv2eigen(init_cv_img_rvec, init_eigen_img_rvec);          // copy involved

  if (nlti_subsys_order() > 0) {
    // define the initial input & state of NLTI sub-system Low-Pass Filter (LPF)
    // to be the given initial log-image, and its initial input exponential to
    // be the given initial image
    discrete_nlti_lpf_.input = init_eigen_log_img_rvec;
    discrete_nlti_lpf_.input_exp = init_eigen_img_rvec;
    discrete_nlti_lpf_.state = init_eigen_log_img_rvec;
  }

  const int c_lti_subsys_order = lti_subsys_order();  // cached copy
  if (c_lti_subsys_order > 0) {
    // define the initial input to the LTI sub-system to be given initial
    // log-image (initial NLTI sub-system state)
    discrete_lti_filter_.input = init_eigen_log_img_rvec;

    // define the initial state of each LPF in the LTI sub-system to be the 
    // given initial log-image, and the initial state of the LTI sub-system
    // High-Pass Filter (HPF) to be zero
    const int num_pixels = init_eigen_log_img_rvec.cols();
    discrete_lti_filter_.state = control::MatrixXs{c_lti_subsys_order,
                                                   num_pixels};
    discrete_lti_filter_.state.topRows(lti_hpf_order_) = (
        control::MatrixXs::Zero(lti_hpf_order_, num_pixels));
    discrete_lti_filter_.state.bottomRows(lti_lpf_order_) = (
        init_eigen_log_img_rvec.replicate(lti_lpf_order_, 1));
  }
}

FloatTypeImage PixelBandwidthModel::filter(
    const Image& log_img, const Image& img, Duration dt_nanosec) {
  // copy the given OpenCV 2D image matrices to Eigen image row vectors
  const Image input_cv_log_img_rvec = log_img.reshape(1, 1);
  control::RowVectorXs input_eigen_log_img_rvec{};
  cv::cv2eigen(input_cv_log_img_rvec, input_eigen_log_img_rvec);  // copy involved

  const Image input_cv_img_rvec = img.reshape(1, 1);
  control::RowVectorXs input_eigen_img_rvec{};
  cv::cv2eigen(input_cv_img_rvec, input_eigen_img_rvec);          // copy involved

  // convert the sampling interval from nanoseconds to seconds
  const FloatType dt_sec = ze::nanosecToSecTrunc(dt_nanosec);

  // filter the log-image through the NLTI sub-system LPF, if available
  const control::RowVectorXs& nlti_subsys_output = (
      [&]() -> const control::RowVectorXs& {
        if (nlti_subsys_order() > 0) {
          return discrete_nlti_lpf_.update(std::move(input_eigen_log_img_rvec),
                                           std::move(input_eigen_img_rvec),
                                           dt_sec);
        } else {
          return input_eigen_log_img_rvec;
        }}()
  );

  // filter the log-image through the LTI sub-system, if available
  FloatTypeImage output_cv_img_rvec;
  if (lti_subsys_order() > 0) {
    // discretize the continuous LTI sub-system assuming First-Order Hold (FOH)
    // on its input (i.e. `nlti_subsys_state_`), according to the current image
    // sampling interval
    /*
     * NOTE:
     *    Interpretation of the state is preserved to accommodate variations in
     *    the image sampling interval across iterations.
     */
    static const bool is_state_preserved = true;
    discrete_lti_filter_.system = control::foh_cont2discrete(
        cont_lti_subsys_, dt_sec, is_state_preserved);

    // filter the log-image
    const control::RowVectorXs output_eigen_img_rvec = (
        discrete_lti_filter_.update(nlti_subsys_output));

    // copy the output Eigen log-image row vector to an OpenCV row vector
    cv::eigen2cv(output_eigen_img_rvec, output_cv_img_rvec);  // copy involved
  } else {
    const control::RowVectorXs& output_eigen_img_rvec = nlti_subsys_output;

    // copy the output Eigen log-image row vector to an OpenCV row vector
    cv::eigen2cv(output_eigen_img_rvec, output_cv_img_rvec);  // copy involved
  }
  // reshape the output OpenCV row vector to a 2D matrix & return
  return output_cv_img_rvec.reshape(1, log_img.size().height);
}

} // namespace event_camera_simulator
