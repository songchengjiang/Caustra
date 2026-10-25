#pragma once

// Optional native Vulkan interop. The portable renderer API does not require
// Vulkan headers. Window systems and presentation remain with the caller.
#include <caustra/renderer.hpp>
#include <vulkan/vulkan.h>
#include <functional>

namespace caustra {

struct VulkanDeviceOptions {
  std::vector<std::string> instanceExtensions;
  std::vector<std::string> deviceExtensions;
  // Create instance-owned application resources before device/queue selection.
  // Destruction runs after the device is released, including failed startup.
  // Captured objects must outlive Renderer. Callbacks must not reenter Renderer;
  // instanceDestroying must not throw and must tolerate partial initialization.
  std::function<void(VkInstance)> instanceCreated;
  std::function<void(VkInstance)> instanceDestroying;
  std::function<bool(VkPhysicalDevice, uint32_t)> supportsQueue;
};

struct VulkanContext {
  VkInstance instance{};
  VkPhysicalDevice physicalDevice{};
  VkDevice device{};
  VkQueue queue{};
  uint32_t queueFamily = 0;
};

struct VulkanFrame {
  VkBuffer buffer{}; // tightly packed, linear RGBA32F, top row first
  uint32_t width = 0, height = 0;
};

class VulkanAccess {
public:
  // Borrowed handles: never destroy them or submit concurrently with Renderer.
  static VulkanContext context(Renderer& renderer);
  // Requires a rendered sample. Returns native accumulation or DLSS output,
  // both device-local STORAGE_BUFFER | TRANSFER_SRC buffers. The synchronous
  // render call has completed; caller must insert a producer-write -> consumer
  // read barrier and finish all buffer uses before another Renderer mutation.
  // Refetch after render/settings/DLSS changes; no CPU image transfer occurs.
  static VulkanFrame output(Renderer& renderer);
};

} // namespace caustra
