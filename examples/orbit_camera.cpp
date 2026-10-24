#include "orbit_camera.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace caustra::example {
namespace {
constexpr double fieldOfView = 45;
constexpr double minDistance = 0.6, maxDistance = 40;
constexpr double poleLimit = std::numbers::pi / 2 - 0.02;
constexpr double orbitSpeed = 0.006;
constexpr double zoomSpeed = 0.14;
constexpr double panLimit = 1000;
}

Camera OrbitCamera::camera() const {
  const double horizontalDistance = distance_ * std::cos(pitch_);
  return {{float(targetX_ + horizontalDistance * std::sin(yaw_)),
           float(targetY_ + distance_ * std::sin(pitch_)),
           float(targetZ_ + horizontalDistance * std::cos(yaw_)), 0},
          {float(targetX_), float(targetY_), float(targetZ_), 0},
          {0, 1, 0, 0}, float(fieldOfView)};
}

void OrbitCamera::orbit(float dxPixels, float dyPixels) {
  if (!std::isfinite(dxPixels) || !std::isfinite(dyPixels)) return;
  yaw_ = std::remainder(yaw_ - double(dxPixels) * orbitSpeed, 2 * std::numbers::pi);
  pitch_ = std::clamp(pitch_ - double(dyPixels) * orbitSpeed, -poleLimit, poleLimit);
}

void OrbitCamera::pan(float dxPixels, float dyPixels, float viewportHeight) {
  if (!std::isfinite(dxPixels) || !std::isfinite(dyPixels) ||
      !std::isfinite(viewportHeight) || viewportHeight <= 0) return;
  // One screen-height spans 2*d*tan(fov/2) world units at the orbit target.
  const double scale = 2 * distance_ * std::tan(fieldOfView * std::numbers::pi / 360)
                     / std::max(1.0, double(viewportHeight));
  const double horizontal = -double(dxPixels) * scale;
  const double vertical = double(dyPixels) * scale;
  const double sinYaw = std::sin(yaw_), cosYaw = std::cos(yaw_);
  const double sinPitch = std::sin(pitch_), cosPitch = std::cos(pitch_);
  targetX_ = std::clamp(targetX_ + horizontal * cosYaw - vertical * sinYaw * sinPitch, -panLimit, panLimit);
  targetY_ = std::clamp(targetY_ + vertical * cosPitch, -panLimit, panLimit);
  targetZ_ = std::clamp(targetZ_ - horizontal * sinYaw - vertical * cosYaw * sinPitch, -panLimit, panLimit);
}

void OrbitCamera::zoom(float wheelSteps) {
  if (!std::isfinite(wheelSteps)) return;
  const double logarithm = std::clamp(std::log(distance_) - double(wheelSteps) * zoomSpeed,
                                    std::log(minDistance), std::log(maxDistance));
  distance_ = std::exp(logarithm);
}

void OrbitCamera::reset() { *this = OrbitCamera{}; }

} // namespace caustra::example
