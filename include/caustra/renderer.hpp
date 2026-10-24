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
struct Vertex { Float4 position; Float4 normal; Float4 texcoord; Float4 tangent; }; // tangent.w: handedness; zero tangent derives it from UVs
// A compact, renderer-neutral program produced by the external MaterialX compiler.
// Instructions execute in order; instruction i writes register i (float4).
// Scalar values are replicated in all lanes. Arguments address earlier registers.
enum class MaterialOp : uint32_t {
  Constant=0, Texcoord=1, Position=2, ObjectPosition=3, Normal=4, Tangent=5,
  Add=10, Subtract=11, Multiply=12, Divide=13, Mix=14, Clamp=15, Power=16,
  Min=17, Max=18, Abs=19, OneMinus=20, Sin=21, Cos=22, Floor=23, Ceil=24,
  Fract=25, Dot=26, Normalize=27, Length=28, Cross=29, Combine=30,
  Swizzle=31, Image=32, Checker=33, Noise3D=34, Normalmap=35,
  RgbToHsv=36, HsvToRgb=37, HexImage=38, Output=63
};
enum class MaterialParameter : uint32_t {
  BaseColor=0, Metalness=1, Roughness=2, Emission=3, Normal=4,
  BaseWeight=5, DiffuseRoughness=6, SpecularWeight=7, SpecularColor=8,
  SpecularIOR=9, Anisotropy=10, Rotation=11, CoatWeight=12, CoatColor=13,
  CoatRoughness=14, CoatIOR=15, CoatNormal=16, SheenWeight=17,
  SheenColor=18, SheenRoughness=19, Opacity=20, Tangent=21, CoatTangent=22,
  CoatAnisotropy=23, CoatDarkening=24,
  TransmissionWeight=25, TransmissionColor=26, TransmissionDepth=27,
  TransmissionScatter=28, TransmissionAnisotropy=29, TransmissionExtraRoughness=30,
  SubsurfaceWeight=31, SubsurfaceColor=32, SubsurfaceRadius=33,
  SubsurfaceScale=34, SubsurfaceAnisotropy=35, ThinWalled=36,
  ThinFilmWeight=37, ThinFilmThickness=38, ThinFilmIOR=39,
  FuzzWeight=40, FuzzColor=41, FuzzRoughness=42, Dispersion=43,
  VolumeDensity=44, VolumeAbsorption=45, VolumeScattering=46,
  VolumeAnisotropy=47, VolumeEmission=48, VolumeBoundaryOnly=49,
  CoatRotation=50, CoatAffectColor=51, CoatAffectRoughness=52
};
struct MaterialInstruction {
  Uint4 code; // opcode, argument A, argument B, argument C (opcode-dependent)
  Float4 value; // constant; Swizzle lane indices; Image LOD bias; Normalmap flags
};
constexpr uint32_t maxMaterialInstructions=128;
struct Texture {
  uint32_t width=0, height=0;
  std::vector<Float4> pixels; // linear RGBA, top row first; renderer builds mipmaps
};
enum class MaterialModel : uint32_t {
  StandardSurface=0, OpenPbr=1, DisneyPrincipled=2, GltfPbr=3,
  UsdPreviewSurface=4, SimpleHair=5
};
enum MaterialProgramFlag : uint32_t {
  HeterogeneousInterior=1u, // Object-space 32^3 coefficient grid, evaluated from the graph at scene setup.
  SphericalInteriorUv=2u   // Radial UV extension; requires HeterogeneousInterior. Default extension is planar XY.
};
// Linear RGB. These PODs are also the shader storage-buffer layout.
struct Material {
  Float4 baseColor{0.8f, 0.8f, 0.8f, 1};
  Float4 emission{0, 0, 0, 0};
  Float4 surface{0, 0.5f, 0, 0}; // metallic, perceptual roughness, one-sided emission (0/1), reserved
  Float4 specular{1,1.5f,0,0}; // weight, IOR, anisotropy, rotation (turns)
  Float4 specularColor{1,1,1,0};
  Float4 coat{0,0.1f,1.5f,0}; // weight, perceptual roughness, IOR, anisotropy
  Float4 coatColor{1,1,1,0};
  Float4 sheen{0,0.3f,0,0}; // weight, roughness, reserved
  Float4 sheenColor{1,1,1,0};
  Float4 factors{1,0,1,1}; // base weight, diffuse roughness, opacity, OpenPBR coat darkening
  Float4 transmission{0,0,0,0}; // weight, attenuation depth (m), HG anisotropy, extra roughness
  Float4 transmissionColor{1,1,1,0};
  Float4 transmissionScatter{}; // SS: inverse meters; OpenPBR: scattering per attenuation depth
  Float4 subsurface{0,1,0,0}; // weight, radius scale (m), HG anisotropy, thin-walled (0/1)
  Float4 subsurfaceColor{1,1,1,0};
  Float4 subsurfaceRadius{1,1,1,0}; // per-channel radius multiplier
  Float4 thinFilm{1,0,1.5f,0}; // coverage, thickness (nm), IOR, inverse effective Abbe number
  Float4 fuzz{0,0.5f,0,0}; // weight, roughness, coat rotation (turns), coat affect color
  Float4 fuzzColor{1,1,1,0};
  Float4 volume{}; // density multiplier, HG anisotropy, invisible boundary (0/1), coat affect roughness
  Float4 volumeAbsorption{}; // homogeneous coefficients, inverse meters
  Float4 volumeScattering{};
  Float4 volumeEmission{}; // emitted radiance per meter
  Uint4 program; // first instruction, count, MaterialModel tag, MaterialProgramFlag bits
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
  std::vector<MaterialInstruction> materialInstructions;
  std::vector<Texture> textures;
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
  // Camera cut: clears raw accumulation and DLSS temporal history.
  void setCamera(const Camera& camera);
  // Continuous camera motion in a static scene: clears raw accumulation when
  // changed, preserving DLSS history with camera reprojection motion vectors.
  void updateCamera(const Camera& camera);
  void setSettings(const RenderSettings& settings);
  // Optional NVIDIA DLSS Ray Reconstruction. Enabling unsupported hardware or
  // a build without DLSS throws; the previously working configuration is kept.
  // May be switched off and on repeatedly; disabling releases feature resources.
  void setDlssSettings(const DlssSettings& settings);
  DlssSettings dlssSettings() const;
  std::string dlssStatus() const;
  void reset();
  void render(uint32_t samples = 1); // progressive accumulation; preserves sample sequence
  Image readback(); // DLSS output when enabled, otherwise progressive raw HDR
  Image readbackRaw(); // progressive raw HDR at the internal render resolution
  uint32_t sampleCount() const;
  uint64_t frameCount() const; // frames since full reset; updateCamera preserves it
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
