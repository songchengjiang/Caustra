#include <caustra/renderer.hpp>
#include "demo_scene.hpp"

#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
size_t checks = 0;
void check(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
void invalid(const std::function<void()>& action, const std::string& message) {
  ++checks;
  try { action(); }
  catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("Expected input rejection: " + message);
}
caustra::Scene triangleScene() {
  caustra::Scene scene;
  scene.materials.emplace_back();
  caustra::Mesh mesh;
  mesh.vertices = {{{0,0,0,0},{}}, {{1,0,0,0},{}}, {{0,1,0,0},{}}};
  mesh.indices = {0,1,2};
  scene.meshes.push_back(mesh);
  scene.instances.emplace_back();
  return scene;
}
void testScene() {
  auto scene = triangleScene();
  caustra::validateScene(scene); // Zero normals intentionally exercise face-normal fallback.
  caustra::validateScene(caustra::example::makeDemoScene());
  invalid([] { caustra::validateScene({}); }, "empty scene");
  for (const auto& mutation : std::initializer_list<std::function<void(caustra::Scene&)>>{
    [](auto& s) { s.materials.clear(); },
    [](auto& s) { s.meshes.clear(); },
    [](auto& s) { s.instances.clear(); },
    [](auto& s) { s.meshes[0].vertices.clear(); },
    [](auto& s) { s.meshes[0].indices.clear(); },
    [](auto& s) { s.meshes[0].indices.push_back(0); },
    [](auto& s) { s.meshes[0].indices[2] = 3; },
    [](auto& s) { s.meshes[0].indices[2] = 0; },
    [](auto& s) { s.meshes[0].vertices[2].position = {2,0,0,0}; },
    [](auto& s) { s.meshes[0].material = 1; },
    [](auto& s) { s.instances[0].mesh = 1; },
    [](auto& s) { s.instances[0].transform[0] = 0; },
    [](auto& s) { s.instances[0].transform[0] = std::numeric_limits<float>::quiet_NaN(); },
    [](auto& s) { s.meshes[0].vertices[0].position.x = std::numeric_limits<float>::quiet_NaN(); },
    [](auto& s) { s.meshes[0].vertices[0].normal.y = std::numeric_limits<float>::infinity(); },
    [](auto& s) { s.materials[0].baseColor.x = std::numeric_limits<float>::quiet_NaN(); },
    [](auto& s) { s.materials[0].emission.z = std::numeric_limits<float>::infinity(); },
    [](auto& s) { s.materials[0].surface.w = std::numeric_limits<float>::quiet_NaN(); },
    [](auto& s) { s.materials[0].baseColor.x = -0.1f; },
    [](auto& s) { s.materials[0].baseColor.z = 1.1f; },
    [](auto& s) { s.materials[0].emission.y = -1; },
    [](auto& s) { s.materials[0].surface.x = 1.1f; },
    [](auto& s) { s.materials[0].surface.y = -0.1f; }
  }) {
    auto broken = scene;
    mutation(broken);
    invalid([&] { caustra::validateScene(broken); }, "invalid scene component");
  }
  // Translation, rotation, nonuniform scale, and mirrored transforms are supported.
  scene.instances[0].transform = {-2,0,0,3, 0,3,0,-1, 0,0,0.5f,2};
  caustra::validateScene(scene);
}
void testCameraAndSettings() {
  caustra::validateDlssSettings({});
  caustra::DlssSettings dlss;
  dlss.quality=static_cast<caustra::DlssQuality>(99);
  invalid([&] { caustra::validateDlssSettings(dlss); }, "unknown DLSS quality");
  dlss={}; dlss.preset=static_cast<caustra::DlssPreset>(99);
  invalid([&] { caustra::validateDlssSettings(dlss); }, "unknown DLSS RR preset");
  caustra::validateCamera({});
  caustra::validateCamera(caustra::example::makeDemoCamera());
  caustra::Camera camera;
  camera.target = camera.position;
  invalid([&] { caustra::validateCamera(camera); }, "zero camera direction");
  camera = {}; camera.up = {};
  invalid([&] { caustra::validateCamera(camera); }, "zero camera up");
  camera = {}; camera.up = {camera.target.x-camera.position.x, camera.target.y-camera.position.y, camera.target.z-camera.position.z,0};
  invalid([&] { caustra::validateCamera(camera); }, "parallel camera up");
  camera = {}; camera.verticalFov = 180;
  invalid([&] { caustra::validateCamera(camera); }, "invalid FOV");
  camera = {}; camera.position.w = std::numeric_limits<float>::quiet_NaN();
  invalid([&] { caustra::validateCamera(camera); }, "nonfinite camera");
  caustra::RenderSettings settings;
  caustra::validateSettings(settings);
  settings.width = 0;
  invalid([&] { caustra::validateSettings(settings); }, "zero width");
  settings = {}; settings.height = 16385;
  invalid([&] { caustra::validateSettings(settings); }, "oversized height");
  settings = {}; settings.maxBounces = 0;
  invalid([&] { caustra::validateSettings(settings); }, "zero bounce depth");
  settings = {}; settings.maxBounces = 65;
  invalid([&] { caustra::validateSettings(settings); }, "excessive bounce depth");
  settings = {}; settings.environment.y = -1;
  invalid([&] { caustra::validateSettings(settings); }, "negative environment");
  settings = {}; settings.environment.w = std::numeric_limits<float>::quiet_NaN();
  invalid([&] { caustra::validateSettings(settings); }, "nonfinite environment");
}
void testImage() {
  const auto unique = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto folder = std::filesystem::temp_directory_path() / ("caustra-tests-" + unique);
  std::filesystem::create_directory(folder);
  struct Cleanup { std::filesystem::path folder; ~Cleanup() { std::error_code error; std::filesystem::remove_all(folder, error); } } cleanup{folder};
  caustra::Image image{2,2,1, {{0,0,0,1},{1,0,0,1},{0,2,0,1},{0,0,4,1}}};
  const auto ppm = folder / "image.ppm", pfm = folder / "image.pfm";
  image.savePpm(ppm);
  image.savePfm(pfm);
  std::ifstream stream(ppm, std::ios::binary);
  std::string magic;
  uint32_t width = 0, height = 0, maxValue = 0;
  stream >> magic >> width >> height >> maxValue;
  stream.get();
  unsigned char rgb[12]{};
  stream.read(reinterpret_cast<char*>(rgb), sizeof(rgb));
  check(magic == "P6" && width == 2 && height == 2 && maxValue == 255, "PPM header mismatch");
  check(stream.gcount() == sizeof(rgb) && stream.peek() == std::char_traits<char>::eof(), "PPM payload length mismatch");
  check(rgb[0] == 0 && rgb[3] > 0 && rgb[4] == 0 && rgb[7] > rgb[3] && rgb[11] > rgb[7], "PPM tone mapping or top-row order mismatch");
  std::ifstream hdr(pfm, std::ios::binary);
  float scale = 0;
  hdr >> magic >> width >> height >> scale;
  hdr.get();
  float linear[12]{};
  hdr.read(reinterpret_cast<char*>(linear), sizeof(linear));
  check(magic == "PF" && width == 2 && height == 2 && (scale < 0) == (std::endian::native == std::endian::little), "PFM header mismatch");
  check(hdr.gcount() == sizeof(linear) && hdr.peek() == std::char_traits<char>::eof(), "PFM payload length mismatch");
  check(linear[0] == 0 && linear[1] == 2 && linear[5] == 4 && linear[9] == 1, "PFM must preserve HDR values and store bottom row first");
  auto broken = image; broken.width = 0;
  invalid([&] { broken.savePpm(ppm); }, "empty image");
  broken = image; broken.pixels.pop_back();
  invalid([&] { broken.savePfm(pfm); }, "wrong image size");
  broken = image; broken.pixels[0].x = std::numeric_limits<float>::quiet_NaN();
  invalid([&] { broken.savePpm(ppm); }, "nonfinite display image");
  invalid([&] { broken.savePfm(pfm); }, "nonfinite HDR image");
  invalid([&] { image.savePpm(ppm, std::numeric_limits<float>::infinity()); }, "nonfinite exposure");
  bool failed = false;
  try { image.savePpm(folder / "missing" / "image.ppm"); }
  catch (const std::runtime_error&) { failed = true; }
  check(failed, "Output errors must propagate");
}
}

int main() {
  try {
    testScene();
    testCameraAndSettings();
    testImage();
    std::cout << "CPU scene/image tests passed (" << checks << " checks).\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CPU test failed: " << error.what() << '\n';
    return 1;
  }
}
