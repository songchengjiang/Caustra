#pragma once

#include <caustra/renderer.hpp>
#include <vulkan/vulkan.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace caustra {

// Internal NGX adapter. The caller owns device/queue synchronization and must
// finish all recorded commands before destroying or reconfiguring this object.
class DlssBackend {
public:
  static std::vector<std::string> requiredInstanceExtensions();
  static std::vector<std::string> requiredDeviceExtensions(VkInstance instance, VkPhysicalDevice physical);

  DlssBackend(VkInstance instance, VkPhysicalDevice physical, VkDevice device);
  ~DlssBackend();
  DlssBackend(const DlssBackend&) = delete;
  DlssBackend& operator=(const DlssBackend&) = delete;

  VkExtent2D inputSize(VkExtent2D output, DlssQuality quality) const;
  void configure(VkCommandBuffer cmd, VkExtent2D input, VkExtent2D output, DlssQuality quality, DlssPreset preset);

  // Guides contain planar images: RGBA32F color/diffuse/specular/normal+roughness,
  // RG32F motion, R32F linear depth, R32F specular hit distance (80 bytes/pixel).
  // Output is tightly packed RGBA32F at the display extent. Matrices use the
  // column-vector convention in row-major memory (transposed to NGX's row-vector
  // convention). Jitter is the actual sample offset in pixels (negated for NGX).
  void evaluate(VkCommandBuffer cmd, VkBuffer guides, VkBuffer output, bool reset,
                float jitterX, float jitterY, const std::array<float,16>& view,
                const std::array<float,16>& projection);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace caustra
