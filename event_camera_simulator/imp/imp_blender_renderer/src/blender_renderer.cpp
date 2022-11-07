#include <string>
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <esim/imp_blender_renderer/blender_renderer.hpp>

DEFINE_string(blend_file, "",
              "Path to the blend-file of the synthetic Blender scene.");
DEFINE_int32(blender_bridge_port, 5555,
             "TCP port number of the Blender bridge server.");
DEFINE_int32(blender_render_device_type, 0,
             "Compute device type for rendering the synthetic Blender scene. 0: CPU, 1: CUDA, 2: OptiX");
DEFINE_int32(blender_render_device_id, 0,
             "Compute device ID for rendering the synthetic Blender scene.");
DEFINE_int32(blender_interm_color_space, 0,
             "Color space of the intermediate output RGBA image. 0: Display (Filmic sRGB by default), 1: Linear");
DEFINE_string(blender_interm_rgba_file, "/tmp/esim_blender_rgba",
              "Filepath (image extension omitted) of the intermediate output RGBA image.");
DEFINE_string(blender_interm_depth_file, "/tmp/esim_blender_depth",
              "Filepath (suffix of 0000 & image extension omitted) of the intermediate output depth image.");

namespace event_camera_simulator {

BlenderRenderer::BlenderRenderer()
{
  // initialize ZeroMQ connection to Blender bridge server
  socket_ = std::unique_ptr<zmq::socket_t>(
    new zmq::socket_t(context_, zmq::socket_type::req)
  );
  socket_->connect("tcp://127.0.0.1:" + std::to_string(FLAGS_blender_bridge_port));

  // open the scene blend-file
  LOG(INFO) << "Opening scene blend-file: " << FLAGS_blend_file;
  sendBpyCmd("bpy.ops.wm.open_mainfile(filepath='" + FLAGS_blend_file + "')");

  // initialize render settings
  sendBpyCmd("scene = bpy.data.scenes['Scene']");
  sendBpyCmd("scene.render.engine = 'CYCLES'");
  sendBpyCmd("scene.render.use_persistent_data = True");
  switch (FLAGS_blender_render_device_type)
  {
  case 0: // CPU
    sendBpyCmd("bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'NONE'");
    sendBpyCmd("bpy.context.scene.cycles.device = 'CPU'");
    break;
  case 1: // CUDA
    sendBpyCmd("bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'");
    sendBpyCmd("bpy.context.scene.cycles.device = 'GPU'");
    break;
  case 2: // OptiX
    sendBpyCmd("bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'");
    sendBpyCmd("bpy.context.scene.cycles.device = 'GPU'");
    break;
  default:
    LOG(FATAL) << "Invalid compute device for Blender rendering: "
               << FLAGS_blender_render_device_type;
    break;
  }
  /**
   * NOTE:
   *    Multi-GPU Cycles rendering (CUDA & OptiX) leads to random crashes in Blender/bpy.
   * 
   * Reference:
   *    https://developer.blender.org/T94052
   */
  useDevice(FLAGS_blender_render_device_id);

  // create compositing nodes
  sendBpyCmd("scene.view_layers[0].use_pass_combined = True");
  sendBpyCmd("scene.view_layers[0].use_pass_z = True");
  sendBpyCmd("scene.use_nodes = True");
  sendBpyCmd("tree = scene.node_tree");

  sendBpyCmd("render_layers = tree.nodes.new('CompositorNodeRLayers')");
  sendBpyCmd("combine_color = tree.nodes.new('CompositorNodeCombineColor')");
  sendBpyCmd("depth_output = tree.nodes.new('CompositorNodeOutputFile')");

  // initialize RGBA render image output settings
  sendBpyCmd("scene.render.filepath = '" + FLAGS_blender_interm_rgba_file + "'");
  sendBpyCmd("scene.render.use_file_extension = True");
  sendBpyCmd("scene.render.use_overwrite = True");
  sendBpyCmd("scene.render.image_settings.color_mode = 'RGBA'");

  switch (FLAGS_blender_interm_color_space)
  {
  case 0: // Display
    sendBpyCmd("scene.render.image_settings.file_format = 'PNG'");
    sendBpyCmd("scene.render.image_settings.color_depth = '8'");
    sendBpyCmd("scene.render.image_settings.compression = 0");
    sendBpyCmd("scene.render.image_settings.color_management = 'FOLLOW_SCENE'");  // Filmic sRGB by default
    break;
  case 1: // Linear
    sendBpyCmd("scene.render.image_settings.file_format = 'OPEN_EXR'");
    sendBpyCmd("scene.render.image_settings.color_depth = '32'"); // matches 'ImageFloatType`
    sendBpyCmd("scene.render.image_settings.exr_codec = 'NONE'");
    sendBpyCmd("scene.render.image_settings.use_zbuffer = False");
    break;
  default:
    LOG(FATAL) << "Invalid color space of the intermediate output RGBA image: "
               << FLAGS_blender_interm_color_space;
    break;
  }

  // initialize depth render image output settings
  boost::filesystem::path depth_path(FLAGS_blender_interm_depth_file);
  sendBpyCmd("depth_output.base_path = '" + depth_path.parent_path().string() + "'");
  sendBpyCmd("depth_output.file_slots[0].path = '" + depth_path.filename().string() + "'");
  sendBpyCmd("depth_output.file_slots[0].use_node_format = True");
  sendBpyCmd("scene.frame_set(0)");

  sendBpyCmd("depth_output.format.file_format = 'OPEN_EXR'");
  sendBpyCmd("depth_output.format.color_mode = 'RGB'");
  sendBpyCmd("depth_output.format.color_depth = '32'");           // matches 'ImageFloatType`
  sendBpyCmd("depth_output.format.exr_codec = 'NONE'");
  sendBpyCmd("depth_output.format.use_zbuffer = False");
  
  // link compositing nodes
  sendBpyCmd("links = tree.links");

  // output depth image (RGB image is output via the existing composite node)
  sendBpyCmd("combine_color.mode = 'RGB'");
  sendBpyCmd("links.new(render_layers.outputs['Depth'], combine_color.inputs['Red'])");
  sendBpyCmd("combine_color.inputs['Green'].default_value = 0");
  sendBpyCmd("combine_color.inputs['Blue'].default_value = 0");
  sendBpyCmd("combine_color.inputs['Alpha'].default_value = 1");

  sendBpyCmd("links.new(combine_color.outputs['Image'], depth_output.inputs['Image'])");
}

std::string BlenderRenderer::sendBpyCmd(const std::string& cmd) const
{
  // send bpy command
  zmq::message_t request(cmd.length());
  memcpy(request.data(), cmd.c_str(), cmd.length());
  socket_->send(request, zmq::send_flags::none);
  
  // receive command execution status and returned message
  zmq::message_t status_reply, return_reply;
  socket_->recv(status_reply, zmq::recv_flags::none);
  CHECK(status_reply.more());
  socket_->recv(return_reply, zmq::recv_flags::none);
  CHECK(!return_reply.more());

  std::string status_reply_string = status_reply.to_string();
  std::string return_reply_string = return_reply.to_string();
  if (status_reply_string == "OK") {
    return return_reply_string;
  }
  else if (status_reply_string == "ERROR") {
    LOG(FATAL) << "Invalid bpy command: " << return_reply_string;
  }
  else {
    LOG(FATAL) << "Invalid reply from Blender bridge";
  }
}

void BlenderRenderer::useDevice(const int target_device_id) const
{
  sendBpyCmd("bpy.context.preferences.addons['cycles'].preferences.get_devices()");
  const int num_devices = std::stoi(sendBpyCmd(
    "len(bpy.context.preferences.addons['cycles'].preferences.devices)"
  ));
  CHECK_LT(target_device_id, num_devices);

  for (int device_id = 0; device_id < num_devices; ++device_id)
  {
    const std::string is_device_used = (device_id == target_device_id)
                                       ? "True" : "False";
    sendBpyCmd("bpy.context.preferences.addons['cycles'].preferences.devices["
               + std::to_string(device_id) + "].use = " + is_device_used);
  }
}

void BlenderRenderer::setCamera(const ze::Camera::Ptr& camera)
{
  camera_ = camera;

  // compute the horizontal field of view of the camera
  ze::VectorX intrinsics = camera_->projectionParameters();
  const FloatType fx = intrinsics(0);
  const FloatType hfov = 2 * std::atan(0.5 * (FloatType) camera_->width() / fx);

  // initialize camera settings
  sendBpyCmd("scene.render.dither_intensity = 0.0");
  sendBpyCmd("scene.render.film_transparent = True");
  sendBpyCmd("scene.render.resolution_percentage = 100");
  sendBpyCmd("scene.render.resolution_x = " + std::to_string(camera->width()));
  sendBpyCmd("scene.render.resolution_y = " + std::to_string(camera->height()));

  sendBpyCmd("camera = bpy.data.objects['Camera']");
  sendBpyCmd("camera.rotation_mode = 'QUATERNION'");
  sendBpyCmd("camera.data.angle_x = " + std::to_string(hfov));
}

void BlenderRenderer::render(const Transformation& T_W_C, const ColorImagePtr& out_image, const DepthmapPtr& out_depthmap) const
{
  CHECK_EQ(out_image->rows, camera_->height());
  CHECK_EQ(out_image->cols, camera_->width());

  // compute the rigid body transformation of the Blender camera frame (x-axis
  // points to the right, y-axis points upwards & z-axis points to the
  // back/outwards, of an image respectively) wrt. the world frame, from the
  // rigid body transformation of the ESIM / ZE camera frame (x-axis points to
  // the right, y-axis points downwards & z-axis points to the front/inwards, of
  // an image respectively) wrt. the world frame
  const Transformation::TransformationMatrix mT_W_Cesim = T_W_C.getTransformationMatrix();
  Transformation::TransformationMatrix mT_Cesim_Cblender;
  mT_Cesim_Cblender << 1,  0,  0, 0,
                       0, -1,  0, 0,
                       0,  0, -1, 0,
                       0,  0,  0, 1;
  const Transformation::TransformationMatrix mT_W_Cblender = mT_W_Cesim * mT_Cesim_Cblender;
  const Transformation T_W_Cblender(mT_W_Cblender);

  // set the camera pose
  sendBpyCmd("camera.location = ("
             + std::to_string(T_W_Cblender.getPosition()[0]) + ", "
             + std::to_string(T_W_Cblender.getPosition()[1]) + ", "
             + std::to_string(T_W_Cblender.getPosition()[2]) + ")");
  sendBpyCmd("camera.rotation_quaternion = ("
             + std::to_string(T_W_Cblender.getRotation().w()) + ", "
             + std::to_string(T_W_Cblender.getRotation().x()) + ", "
             + std::to_string(T_W_Cblender.getRotation().y()) + ", "
             + std::to_string(T_W_Cblender.getRotation().z()) + ")");

  // render the scene & save the RGBA & depth images
  sendBpyCmd("bpy.ops.render.render(write_still=True)");

  // read and output the saved RGBA image
  cv::Mat bgra_img;
  switch (FLAGS_blender_interm_color_space)
  {
  case 0: // Display
    bgra_img = cv::imread(FLAGS_blender_interm_rgba_file + ".png",
                          cv::IMREAD_UNCHANGED);
    bgra_img.convertTo(bgra_img, cv::DataType<ImageFloatType>::type, 1./255.);
    break;
  case 1: // Linear
    bgra_img = cv::imread(FLAGS_blender_interm_rgba_file + ".exr",
                          cv::IMREAD_UNCHANGED);
    break;
  }

  // alpha-composite the intermediate image over a white background (in the output color space)
  cv::Mat bgr_img, alpha_img;
  cv::cvtColor(bgra_img, bgr_img, cv::COLOR_BGRA2BGR);
  cv::extractChannel(bgra_img, alpha_img, 3);
  cv::cvtColor(alpha_img, alpha_img, cv::COLOR_GRAY2BGR); // expand single-channel alpha image to three channels
  switch (FLAGS_blender_interm_color_space)
  {
  case 0: // Display
    // straight alpha
    *out_image = alpha_img.mul(bgr_img) + (1 - alpha_img);
    break;
  case 1: // Linear
    // premultiplied alpha
    *out_image = bgr_img + (1 - alpha_img);
    break;
  }

  // read and output the saved depth image
  cv::Mat depth_img = cv::imread(FLAGS_blender_interm_depth_file + "0000.exr",
                                 cv::IMREAD_UNCHANGED);
  cv::extractChannel(depth_img, *out_depthmap, 2);  // extract "Red" channel
}

} // namespace event_camera_simulator
