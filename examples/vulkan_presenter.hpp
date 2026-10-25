#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <caustra/renderer.hpp>
#include <memory>
#include <vector>

namespace caustra::example {

// The example owns the window surface, display transform and swapchain. The
// renderer remains usable without any window-system or presentation dependency.
class VulkanPresenter {
public:
  explicit VulkanPresenter(HWND window, bool validation = false);
  VulkanPresenter(HWND window, const RendererOptions& options);
  ~VulkanPresenter();
  VulkanPresenter(const VulkanPresenter&) = delete;
  VulkanPresenter& operator=(const VulkanPresenter&) = delete;
  Renderer& renderer();
  // Can republish the last rendered output, including while paused or resizing.
  // Returns false while minimized or when an out-of-date surface needs a retry.
  bool present(uint32_t width, uint32_t height);
  uint32_t width() const;
  uint32_t height() const;
  // Explicit screenshot/test operation only. Display-encoded BGRX, top row first.
  std::vector<uint32_t> readbackLdr();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace caustra::example
