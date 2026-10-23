#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace caustra {

struct alignas(16) Float4 { float x = 0, y = 0, z = 0, w = 0; };
struct alignas(16) Uint4 { uint32_t x = 0, y = 0, z = 0, w = 0; };
struct Vertex { Float4 position; Float4 normal; };
// Linear RGB. These PODs are also the shader storage-buffer layout.
struct Material {
  Float4 baseColor{0.8f, 0.8f, 0.8f, 1};
  Float4 emission{0, 0, 0, 0};
  Float4 surface{0, 0.5f, 0, 0}; // metallic, perceptual roughness, reserved, reserved
};
struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices; // triangle list, local vertex indices
  uint32_t material = 0;
};
struct Instance {
  uint32_t mesh = 0;
  // Row-major affine 3x4, matching VkTransformMatrixKHR; identity by default.
  std::array<float, 12> transform{1,0,0,0, 0,1,0,0, 0,0,1,0};
};
struct Scene {
  std::vector<Material> materials;
  std::vector<Mesh> meshes;
  std::vector<Instance> instances;
};
struct Camera {
  Float4 position{0, 1.3f, 4, 0};
  Float4 target{0, 1, 0, 0};
  Float4 up{0, 1, 0, 0};
  float verticalFov = 45;
};
struct RenderSettings {
  uint32_t width = 640, height = 480;
  uint32_t maxBounces = 8;
  Float4 environment{0.03f, 0.03f, 0.03f, 0}; // constant linear radiance
};
struct RendererOptions { bool validation = false; };
enum class DlssQuality { Native, Quality, Balanced, Performance };
enum class DlssPreset : uint32_t { Default = 0, D = 4, E = 5 };
struct DlssSettings {
  bool enabled = false;
  DlssQuality quality = DlssQuality::Native;
  DlssPreset preset = DlssPreset::Default;
};
struct Image {
  uint32_t width = 0, height = 0, samples = 0;
  std::vector<Float4> pixels; // linear HDR, top row first
  void savePpm(const std::filesystem::path& path, float exposure = 0) const;
  void savePfm(const std::filesystem::path& path) const;
};

// Owns Vulkan device/resources. Single-thread owned; methods synchronize GPU work.
class Renderer {
public:
  explicit Renderer(const RendererOptions& options = {});
  ~Renderer();
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;
  void setScene(const Scene& scene);
  void setCamera(const Camera& camera);
  void setSettings(const RenderSettings& settings);
  // Optional NVIDIA DLSS Ray Reconstruction. Enabling unsupported hardware or
  // a build without DLSS throws; the previously working configuration is kept.
  void setDlssSettings(const DlssSettings& settings);
  DlssSettings dlssSettings() const;
  std::string dlssStatus() const;
  void reset();
  void render(uint32_t samples = 1); // progressive accumulation; preserves sample sequence
  Image readback(); // DLSS output when enabled, otherwise progressive raw HDR
  Image readbackRaw(); // progressive raw HDR at the internal render resolution
  uint32_t sampleCount() const;
  std::string deviceName() const;
  uint32_t validationErrorCount() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// CPU-only input checks, also performed by Renderer before GPU upload/dispatch.
void validateScene(const Scene& scene);
void validateCamera(const Camera& camera);
void validateSettings(const RenderSettings& settings);
void validateDlssSettings(const DlssSettings& settings);

} // namespace caustra
