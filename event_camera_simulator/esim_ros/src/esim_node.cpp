#include <esim/esim/simulator.hpp>
#include <esim/visualization/ros_publisher.hpp>
#include <esim/visualization/rosbag_writer.hpp>
#include <esim/visualization/adaptive_sampling_benchmark_publisher.hpp>
#include <esim/visualization/synthetic_optic_flow_publisher.hpp>
#include <esim/data_provider/data_provider_factory.hpp>

#include <glog/logging.h>
#include <gflags/gflags.h>

DEFINE_double(contrast_threshold_pos, 1.0,
              "Contrast threshold (positive)");

DEFINE_double(contrast_threshold_neg, 1.0,
              "Contrast threshold  (negative))");

DEFINE_double(contrast_threshold_sigma_pos, 0.021,
              "Standard deviation of contrast threshold (positive)");

DEFINE_double(contrast_threshold_sigma_neg, 0.021,
              "Standard deviation of contrast threshold  (negative))");

DEFINE_int64(refractory_period_ns, 0,
             "Refractory period (time during which a pixel cannot fire events just after it fired one), in nanoseconds");

DEFINE_double(leak_rate_hz, 0.0,
              "Nominal rate of ON leak events, in Hz");

DEFINE_double(I_p_to_intensity_ratio_fa, std::numeric_limits<double>::infinity(),
              "Ratio of the signal photocurrent `I_p`, in fA, to image pixel"
              " intensity `it`, `I_p_to_it_ratio`");

DEFINE_double(dark_current_fa, 0.0,
              "Photodiode dark current `I_dark`, in fA. The photocurrent"
              " `I = I_p + I_dark`. When `I_p_to_it_ratio` approaches"
              " infinity, then `I_dark` is effectively 0 / dark current"
              " -equivalent image pixel intensity (i.e. dark intensity)"
              " `dark_it = I_dark / I_p_to_it_ratio` is 0.");

DEFINE_double(amplifier_gain, std::numeric_limits<double>::infinity(),
              "Amplifier gain of the photoreceptor circuit `A_amp`");

DEFINE_double(back_gate_coeff, 0.7,
              "Back-gate coefficient `kappa`. The closed-loop gain of the"
              " photoreceptor circuit `A_cl = 1 / kappa`, and the total loop"
              " gain of the photoreceptor circuit `A_loop = A_amp / A_cl`.");

DEFINE_double(thermal_voltage_mv, 25,
              "Thermal voltage `V_T`, in mV");

DEFINE_double(photodiode_cap_ff, 0.0,
              "(Lumped) Parasitic capacitance on the photodiode / input node of"
              " the photoreceptor circuit `C_p`, in fF. The time constant"
              " associated to the input node of the photoreceptor circuit"
              " `tau_in = C_p * V_T / I = Q_in / I`.");

DEFINE_double(miller_cap_ff, 0.0,
              "Miller capacitance in the photoreceptor circuit `C_mil`, in fF."
              " In the absence of a cascode transistor, `C_mil = C_fb + C_n`,"
              " where `C_fb` is the Miller capacitance from the gate to the"
              " source of the feedback transistor M_fb, and `C_n` is the Miller"
              " capacitance from the gate to the drain of the inverting"
              " amplifier transistor M_n. Else, `C_mil = C_fb`. "
              " The time constant associated to the Miller capacitance"
              " `tau_mil = C_mil * V_T / I = Q_mil / I`.");

DEFINE_double(output_time_const_us, 0.0,
              "Time constant `tau_out` associated to the output node of the"
              " photoreceptor circuit / photoreceptor bias current `I_pr`, in"
              " microseconds");

DEFINE_double(lower_cutoff_freq_hz, 0.0,
              "Lower cutoff frequency of the pixel circuit / high-pass filter"
              " present in certain event cameras `f_c_lower`, in Hz");

DEFINE_double(sf_cutoff_freq_hz, std::numeric_limits<double>::infinity(),
              "(Upper) Cutoff frequency of the source follower buffer"
              " `f_c_sf`, associated to the source follower buffer bias current"
              " `I_sf`, in Hz");

DEFINE_double(diff_amp_cutoff_freq_hz, std::numeric_limits<double>::infinity(),
              "(Upper) Cutoff frequency of the differencing/change amplifier"
              " `f_c_diff`, in Hz");

DEFINE_double(exposure_time_ms, 10.0,
              "Exposure time in milliseconds, used to simulate motion blur");

DEFINE_bool(use_log_image, true,
            "Whether to convert images to log images in the preprocessing step.");

DEFINE_double(log_eps, 0.001,
              "Epsilon value used to convert images to log: L = log(eps + I / 255.0).");

DEFINE_bool(simulate_color_events, false,
              "Whether to simulate color events or not (default: false)");

DEFINE_int32(random_seed, 0,
              "Random seed used to generate the trajectories. If set to 0 the current time(0) is taken as seed.");

using namespace event_camera_simulator;

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  if (FLAGS_random_seed == 0) FLAGS_random_seed = (unsigned int) time(0);
  srand(FLAGS_random_seed);

  DataProviderBase::Ptr data_provider_ =
      loadDataProviderFromGflags();
  CHECK(data_provider_);

  EventSimulator::Config event_sim_config;
  event_sim_config.Cp = FLAGS_contrast_threshold_pos;
  event_sim_config.Cm = FLAGS_contrast_threshold_neg;
  event_sim_config.sigma_Cp = FLAGS_contrast_threshold_sigma_pos;
  event_sim_config.sigma_Cm = FLAGS_contrast_threshold_sigma_neg;
  event_sim_config.refractory_period_ns = FLAGS_refractory_period_ns;
  event_sim_config.leak_rate_hz = FLAGS_leak_rate_hz;

  event_sim_config.I_p_to_it_ratio_fa = FLAGS_I_p_to_intensity_ratio_fa;
  event_sim_config.I_dark_fa = FLAGS_dark_current_fa;
  event_sim_config.A_amp = FLAGS_amplifier_gain;
  event_sim_config.kappa = FLAGS_back_gate_coeff;
  event_sim_config.V_T_mv = FLAGS_thermal_voltage_mv;
  event_sim_config.C_p_ff = FLAGS_photodiode_cap_ff;
  event_sim_config.C_mil_ff = FLAGS_miller_cap_ff;
  event_sim_config.tau_out_us = FLAGS_output_time_const_us;
  event_sim_config.f_c_lower_hz = FLAGS_lower_cutoff_freq_hz;
  event_sim_config.f_c_sf_hz = FLAGS_sf_cutoff_freq_hz;
  event_sim_config.f_c_diff_hz = FLAGS_diff_amp_cutoff_freq_hz;

  event_sim_config.use_log_image = FLAGS_use_log_image;
  event_sim_config.log_eps = FLAGS_log_eps;
  event_sim_config.simulate_color_events = FLAGS_simulate_color_events;

  std::shared_ptr<Simulator> sim;
  sim.reset(new Simulator(data_provider_->numCameras(),
                          event_sim_config,
                          FLAGS_exposure_time_ms));
  CHECK(sim);

  Publisher::Ptr ros_publisher = std::make_shared<RosPublisher>(data_provider_->numCameras());
  Publisher::Ptr rosbag_writer = RosbagWriter::createBagWriterFromGflags(data_provider_->numCameras());
  Publisher::Ptr adaptive_sampling_benchmark_publisher
      = AdaptiveSamplingBenchmarkPublisher::createFromGflags();

  Publisher::Ptr synthetic_optic_flow_publisher
      = SyntheticOpticFlowPublisher::createFromGflags();

  if(ros_publisher) sim->addPublisher(ros_publisher);
  if(rosbag_writer) sim->addPublisher(rosbag_writer);
  if(adaptive_sampling_benchmark_publisher) sim->addPublisher(adaptive_sampling_benchmark_publisher);
  if(synthetic_optic_flow_publisher) sim->addPublisher(synthetic_optic_flow_publisher);

  data_provider_->registerCallback(
        std::bind(&Simulator::dataProviderCallback, sim.get(),
                  std::placeholders::_1));

  data_provider_->spin();

}
