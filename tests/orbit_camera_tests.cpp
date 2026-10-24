#include "orbit_camera.hpp"
#include "viewer_scene.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
double length(caustra::Float4 a, caustra::Float4 b) {
  return std::sqrt(double(a.x-b.x)*(a.x-b.x)+double(a.y-b.y)*(a.y-b.y)+double(a.z-b.z)*(a.z-b.z));
}
bool sameCamera(const caustra::Camera& a, const caustra::Camera& b) {
  return length(a.position,b.position) < 1e-5 && length(a.target,b.target) < 1e-5 && a.verticalFov == b.verticalFov;
}
double distance(const caustra::Camera& camera) { return length(camera.position,camera.target); }
}

int main() {
  try {
    using caustra::example::OrbitCamera;
    OrbitCamera orbit;
    const auto initial = orbit.camera();
    OrbitCamera imported;
    auto importedHome=initial;importedHome.position={8,4,10,0};importedHome.target={1,1,0,0};importedHome.verticalFov=30;
    imported.setHome(importedHome);const auto importedInitial=imported.camera();
    imported.orbit(100,25);imported.zoom(2);imported.pan(30,20,480);imported.reset();
    require(sameCamera(imported.camera(),importedInitial),"Reset must restore the imported inspection pose.");
    caustra::validateCamera(initial);
    caustra::validateScene(caustra::example::makeViewerScene());

    orbit.orbit(150,30);
    require(!sameCamera(initial,orbit.camera()),"Orbit must change the camera pose.");
    require(length(initial.target,orbit.camera().target) == 0,"Orbit must preserve its target.");
    require(std::abs(distance(initial)-distance(orbit.camera())) < 1e-5,"Orbit must preserve distance.");
    orbit.orbit(-150,-30);
    require(sameCamera(initial,orbit.camera()),"Reversing an unclamped orbit must restore the camera.");

    orbit.zoom(3.5f);
    require(distance(orbit.camera()) < distance(initial),"Positive wheel steps must zoom in.");
    orbit.zoom(-3.5f);
    require(sameCamera(initial,orbit.camera()),"Reversing zoom must restore the camera.");

    orbit.pan(120,60,640);
    require(length(initial.target,orbit.camera().target) > 0.1,"Pan must translate the target.");
    require(std::abs(distance(initial)-distance(orbit.camera())) < 1e-5,"Pan must preserve orbit distance.");
    const auto panDisplacement = length(initial.target,orbit.camera().target);
    orbit.reset();
    orbit.pan(120,60,1280);
    require(std::abs(panDisplacement-2*length(initial.target,orbit.camera().target)) < 1e-5,
            "Pan must scale with viewport height.");
    orbit.pan(-120,-60,1280);
    require(sameCamera(initial,orbit.camera()),"Reversing pan must restore the camera.");

    for (const float sign : {-1.f,1.f}) {
      orbit.reset();
      orbit.orbit(sign*std::numeric_limits<float>::max(),sign*std::numeric_limits<float>::max());
      caustra::validateCamera(orbit.camera());
      orbit.zoom(sign*std::numeric_limits<float>::max());
      caustra::validateCamera(orbit.camera());
      require(distance(orbit.camera()) >= 0.59 && distance(orbit.camera()) <= 40.01,
              "Extreme wheel input must respect distance bounds.");
      orbit.pan(sign*std::numeric_limits<float>::max(),sign*std::numeric_limits<float>::max(),1);
      caustra::validateCamera(orbit.camera());
    }
    orbit.reset();
    require(sameCamera(initial,orbit.camera()),"Reset must restore the initial camera.");
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto infinity = std::numeric_limits<float>::infinity();
    orbit.orbit(nan,0);
    orbit.orbit(0,infinity);
    orbit.zoom(nan);
    orbit.pan(2,3,0);
    orbit.pan(2,3,-640);
    orbit.pan(2,3,infinity);
    orbit.pan(nan,3,640);
    require(sameCamera(initial,orbit.camera()),"Invalid input must leave the camera unchanged.");
    std::cout << "Orbit camera and viewer scene tests passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
