#pragma once

#include <esim/rendering/renderer_base.hpp>

namespace event_camera_simulator {

class UnrealCvClient; // fwd

class UnrealCvRenderer : public Renderer
{
public:
  /**
   * The Unreal Engine camera ID is set to 1 to circumvent issues on scene
   * capture & object masks, in Unreal Engine > 4.16 (latest officially
   * supported version by UnrealCV). This requires additional "Fusion Camera
   * Actor"(s) added to the scene.
   * 
   * References:
   * 1) https://github.com/unrealcv/unrealcv/issues/198
   * 2) https://github.com/unrealcv/unrealcv/issues/186
   */
  const uint32_t UE_CAMERA_ID = 1;

  UnrealCvRenderer();

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
  std::shared_ptr<UnrealCvClient> client_;
  mutable size_t frame_idx_;
};


}
