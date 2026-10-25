#pragma once

#include <caustra/vulkan.hpp>
#include <memory>

namespace caustra::example {

// Private viewer interop between two devices on the same physical GPU. Both
// contexts and their queues are borrowed and must outlive this single owner.
// stage() finishes reading the renderer output before returning. The returned
// buffer belongs to the consumer device and contains a GPU copy, never CPU HDR.
class VulkanFrameBridge {
public:
  VulkanFrameBridge(VulkanContext producer, VulkanContext consumer);
  ~VulkanFrameBridge();
  VulkanFrameBridge(const VulkanFrameBridge&) = delete;
  VulkanFrameBridge& operator=(const VulkanFrameBridge&) = delete;

  VulkanFrame stage(VulkanFrame source);
  void acquire(VkCommandBuffer command);
  void release(VkCommandBuffer command);
  // The consumer submission must wait ready() at ALL_COMMANDS and signal
  // consumed() after release(). Call submitted() only after QueueSubmit succeeds.
  VkSemaphore ready() const;
  VkSemaphore consumed() const;
  void submitted();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace caustra::example
