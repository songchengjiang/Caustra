#include <caustra/renderer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace caustra {
namespace {
bool finite(const Float4& v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
}
bool nonnegativeRgb(const Float4& v) { return v.x >= 0 && v.y >= 0 && v.z >= 0; }
struct D3 { double x, y, z; };
D3 subtract(const Float4& a, const Float4& b) {
  return {double(a.x) - b.x, double(a.y) - b.y, double(a.z) - b.z};
}
D3 cross(D3 a, D3 b) {
  return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
double lengthSquared(D3 a) { return a.x*a.x + a.y*a.y + a.z*a.z; }
void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}
}

void validateScene(const Scene& scene) {
  require(!scene.materials.empty(), "Scene must contain at least one material.");
  require(!scene.meshes.empty(), "Scene must contain at least one mesh.");
  require(!scene.instances.empty(), "Scene must contain at least one instance.");
  constexpr auto maxCount = std::numeric_limits<uint32_t>::max();
  require(scene.materials.size() <= maxCount && scene.meshes.size() <= maxCount &&
          scene.instances.size() <= 0x00ffffffu, "Scene exceeds GPU index limits.");

  for (size_t i = 0; i < scene.materials.size(); ++i) {
    const auto& m = scene.materials[i];
    const auto prefix = "Material " + std::to_string(i) + ": ";
    require(finite(m.baseColor) && finite(m.emission) && finite(m.surface), prefix + "all components must be finite.");
    require(nonnegativeRgb(m.baseColor) && m.baseColor.x <= 1 && m.baseColor.y <= 1 && m.baseColor.z <= 1,
            prefix + "base color must be in [0, 1].");
    require(nonnegativeRgb(m.emission), prefix + "emission must be nonnegative.");
    require(m.surface.x >= 0 && m.surface.x <= 1 && m.surface.y >= 0 && m.surface.y <= 1,
            prefix + "metallic and roughness must be in [0, 1].");
  }
  uint64_t totalVertices = 0, totalIndices = 0;
  for (size_t i = 0; i < scene.meshes.size(); ++i) {
    const auto& mesh = scene.meshes[i];
    const auto prefix = "Mesh " + std::to_string(i) + ": ";
    require(mesh.material < scene.materials.size(), prefix + "material index is out of range.");
    require(!mesh.vertices.empty(), prefix + "vertices are empty.");
    require(!mesh.indices.empty() && mesh.indices.size() % 3 == 0, prefix + "indices must be a nonempty triangle list.");
    totalVertices += mesh.vertices.size();
    totalIndices += mesh.indices.size();
    require(totalVertices <= maxCount && totalIndices <= maxCount, "Combined mesh data exceeds 32-bit GPU indices.");
    for (const auto& vertex : mesh.vertices)
      require(finite(vertex.position) && finite(vertex.normal), prefix + "vertex components must be finite.");
    for (size_t triangle = 0; triangle < mesh.indices.size(); triangle += 3) {
      const auto a = mesh.indices[triangle], b = mesh.indices[triangle + 1], c = mesh.indices[triangle + 2];
      require(a < mesh.vertices.size() && b < mesh.vertices.size() && c < mesh.vertices.size(),
              prefix + "triangle index is out of range.");
      const auto& p = mesh.vertices[a].position;
      require(lengthSquared(cross(subtract(mesh.vertices[b].position, p), subtract(mesh.vertices[c].position, p))) > 0,
              prefix + "triangle " + std::to_string(triangle / 3) + " is degenerate.");
    }
  }
  for (size_t i = 0; i < scene.instances.size(); ++i) {
    const auto& instance = scene.instances[i];
    const auto prefix = "Instance " + std::to_string(i) + ": ";
    require(instance.mesh < scene.meshes.size(), prefix + "mesh index is out of range.");
    const auto& t = instance.transform;
    for (float value : t) require(std::isfinite(value), prefix + "transform must be finite.");
    const double determinant = double(t[0]) * (double(t[5])*t[10] - double(t[6])*t[9])
                             - double(t[1]) * (double(t[4])*t[10] - double(t[6])*t[8])
                             + double(t[2]) * (double(t[4])*t[9] - double(t[5])*t[8]);
    double scale = 0;
    for (size_t row = 0; row < 3; ++row)
      for (size_t col = 0; col < 3; ++col) scale = std::max(scale, std::abs(double(t[row*4 + col])));
    require(scale > 0 && std::abs(determinant) > 1e-12 * scale * scale * scale,
            prefix + "transform is singular or numerically unstable.");
    for (const auto& vertex : scene.meshes[instance.mesh].vertices) {
      for (size_t row = 0; row < 3; ++row) {
        const auto& p = vertex.position;
        const double world = double(t[row*4])*p.x + double(t[row*4+1])*p.y + double(t[row*4+2])*p.z + t[row*4+3];
        require(std::isfinite(world) && std::abs(world) <= std::numeric_limits<float>::max(),
                prefix + "transformed vertex exceeds float range.");
      }
    }
  }
}

void validateCamera(const Camera& camera) {
  require(finite(camera.position) && finite(camera.target) && finite(camera.up) && std::isfinite(camera.verticalFov),
          "Camera components must be finite.");
  require(camera.verticalFov > 0 && camera.verticalFov < 179, "Camera vertical FOV must be in (0, 179) degrees.");
  const D3 direction = subtract(camera.target, camera.position);
  const D3 up{camera.up.x, camera.up.y, camera.up.z};
  require(lengthSquared(direction) > 0, "Camera position and target must differ.");
  require(lengthSquared(up) > 0 && lengthSquared(cross(direction, up)) > 1e-12 * lengthSquared(direction) * lengthSquared(up),
          "Camera up must be nonzero and must not be parallel to the viewing direction.");
}

void validateSettings(const RenderSettings& settings) {
  require(settings.width > 0 && settings.height > 0 && settings.width <= 16384 && settings.height <= 16384,
          "Render dimensions must each be in [1, 16384].");
  require(settings.maxBounces >= 1 && settings.maxBounces <= 64, "Bounce count must be in [1, 64].");
  require(finite(settings.environment) && nonnegativeRgb(settings.environment), "Environment radiance must be finite and nonnegative.");
}

void validateDlssSettings(const DlssSettings& settings) {
  require(settings.quality==DlssQuality::Native || settings.quality==DlssQuality::Quality ||
          settings.quality==DlssQuality::Balanced || settings.quality==DlssQuality::Performance,
          "Invalid DLSS quality mode.");
  require(settings.preset==DlssPreset::Default || settings.preset==DlssPreset::D || settings.preset==DlssPreset::E,
          "DLSS RR preset must be Default, D or E.");
}

} // namespace caustra
