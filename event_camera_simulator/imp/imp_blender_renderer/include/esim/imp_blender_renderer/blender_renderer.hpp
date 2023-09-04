#pragma once

#include <esim/rendering/renderer_base.hpp>
#include <zmq.hpp>

namespace event_camera_simulator {

class BlenderRenderer : public Renderer
{
public:
  BlenderRenderer();

  //! Render image and depth map for a given camera pose
  virtual void render(const Transformation& T_W_C, const ColorImagePtr &out_image, const DepthmapPtr &out_depthmap) const;

  void render(const Transformation& T_W_C, const std::vector<Transformation>& T_W_OBJ, const ColorImagePtr &out_image, const DepthmapPtr &out_depthmap) const
  {
      render(T_W_C, out_image, out_depthmap);
  }

  //! Returns true if the rendering engine can compute optic flow, false otherwise
  virtual bool canComputeOpticFlow() const override { return false; }

  virtual void setCamera(const ze::Camera::Ptr& camera) override;

private:
  zmq::context_t context_;
  std::unique_ptr<zmq::socket_t> socket_;

  std::string sendBpyCmd(const std::string& cmd) const;
  void useDevice(const int target_device_id) const;
};


}
