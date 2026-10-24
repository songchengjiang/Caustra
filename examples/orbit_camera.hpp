#pragma once

#include <caustra/renderer.hpp>

namespace caustra::example {

// Application-side controls; the renderer itself has no input or window state.
class OrbitCamera {
public:
  OrbitCamera() = default;
  Camera camera() const;
  // Deltas are client-area pixels. Dragging moves the object with the pointer.
  void orbit(float dxPixels, float dyPixels);
  void pan(float dxPixels, float dyPixels, float viewportHeight);
  // Positive wheel steps move toward the target; fractional steps are supported.
  void zoom(float wheelSteps);
  void reset();

private:
  double targetX_ = 0, targetY_ = 1.1, targetZ_ = 0;
  double distance_ = 4.8, yaw_ = 0.4, pitch_ = 0.12;
};

} // namespace caustra::example
