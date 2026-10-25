#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include "vulkan_presenter.hpp"
#include <caustra/vulkan.hpp>
#include "present_spv.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace caustra::example {
namespace {
void check(VkResult result, const char* operation) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed (VkResult " + std::to_string(result) + ")");
}
bool srgb(VkFormat format) {
  return format == VK_FORMAT_R8G8B8A8_SRGB || format == VK_FORMAT_B8G8R8A8_SRGB;
}
struct PresentConstants { uint32_t sourceWidth, sourceHeight, width, height; };
}

struct VulkanPresenter::Impl {
  HWND window;
  VkSurfaceKHR surface{};
  std::unique_ptr<Renderer> renderer;
  VulkanContext context{};
  VkCommandPool commandPool{};
  VkCommandBuffer command{};
  VkFence fence{};
  VkSemaphore acquired{};
  VkDescriptorSetLayout descriptorLayout{};
  VkDescriptorPool descriptorPool{};
  VkDescriptorSet descriptors{};
  VkPipelineLayout pipelineLayout{};
  VkPipeline pipeline{};
  VkSwapchainKHR swapchain{};
  std::vector<VkImage> images;
  std::vector<VkSemaphore> finished;
  std::vector<bool> initialized;
  VkExtent2D extent{};
  VkFormat swapFormat{};
  VkImage display{};
  VkImageView displayView{};
  VkDeviceMemory displayMemory{};
  bool displayInitialized = false;
  bool rebuild = true;
  uint32_t requestedWidth = 0, requestedHeight = 0;

  explicit Impl(HWND handle, const RendererOptions& rendererOptions) : window(handle) {
    try {
      VulkanDeviceOptions options;
      options.instanceExtensions = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
      options.deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
      options.instanceCreated = [this](VkInstance instance) {
        VkWin32SurfaceCreateInfoKHR info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        info.hinstance = reinterpret_cast<HINSTANCE>(GetWindowLongPtrW(window, GWLP_HINSTANCE));
        info.hwnd = window;
        check(vkCreateWin32SurfaceKHR(instance, &info, nullptr, &surface), "Create viewer Vulkan surface");
      };
      options.instanceDestroying = [this](VkInstance instance) {
        if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
      };
      options.supportsQueue = [this](VkPhysicalDevice physical, uint32_t family) {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
        VkBool32 supported = VK_FALSE;
        return family < count && (queues[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)
          && vkGetPhysicalDeviceSurfaceSupportKHR(physical, family, surface, &supported) == VK_SUCCESS && supported;
      };
      renderer = std::make_unique<Renderer>(rendererOptions, options);
      context = VulkanAccess::context(*renderer);
      createResources();
    } catch (...) { cleanup(); throw; }
  }
  ~Impl() { cleanup(); }

  uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred = 0) const {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(context.physicalDevice, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & (required | preferred)) == (required | preferred)) return i;
    if (preferred) return memoryType(bits, required);
    throw std::runtime_error("No compatible Vulkan presentation memory type");
  }

  void createResources() {
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = context.queueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(context.device, &poolInfo, nullptr, &commandPool), "Create presentation command pool");
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = commandPool; allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocation.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(context.device, &allocation, &command), "Allocate presentation commands");
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(vkCreateFence(context.device, &fenceInfo, nullptr, &fence), "Create presentation fence");
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &acquired), "Create acquire semaphore");
    const VkDescriptorSetLayoutBinding bindings[]{
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo descriptorInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptorInfo.bindingCount = 2; descriptorInfo.pBindings = bindings;
    check(vkCreateDescriptorSetLayout(context.device, &descriptorInfo, nullptr, &descriptorLayout), "Create presentation descriptor layout");
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
    VkDescriptorPoolCreateInfo descriptorPoolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptorPoolInfo.maxSets = 1; descriptorPoolInfo.poolSizeCount = 2; descriptorPoolInfo.pPoolSizes = sizes;
    check(vkCreateDescriptorPool(context.device, &descriptorPoolInfo, nullptr, &descriptorPool), "Create presentation descriptor pool");
    VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setInfo.descriptorPool = descriptorPool; setInfo.descriptorSetCount = 1; setInfo.pSetLayouts = &descriptorLayout;
    check(vkAllocateDescriptorSets(context.device, &setInfo, &descriptors), "Allocate presentation descriptors");
    VkPushConstantRange constants{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PresentConstants)};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1; layoutInfo.pSetLayouts = &descriptorLayout;
    layoutInfo.pushConstantRangeCount = 1; layoutInfo.pPushConstantRanges = &constants;
    check(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &pipelineLayout), "Create presentation pipeline layout");
    VkShaderModule module{};
    VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderInfo.codeSize = sizeof(presentSpv); shaderInfo.pCode = presentSpv;
    check(vkCreateShaderModule(context.device, &shaderInfo, nullptr, &module), "Create presentation shader");
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pipelineInfo.stage.module = module; pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;
    const auto result = vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(context.device, module, nullptr);
    check(result, "Create presentation pipeline");
  }

  void destroySwapchain() noexcept {
    if (!context.device) return;
    if (displayView) vkDestroyImageView(context.device, displayView, nullptr);
    if (display) vkDestroyImage(context.device, display, nullptr);
    if (displayMemory) vkFreeMemory(context.device, displayMemory, nullptr);
    displayView = VK_NULL_HANDLE; display = VK_NULL_HANDLE; displayMemory = VK_NULL_HANDLE;
    for (auto semaphore : finished) if (semaphore) vkDestroySemaphore(context.device, semaphore, nullptr);
    finished.clear(); images.clear(); initialized.clear();
    if (swapchain) vkDestroySwapchainKHR(context.device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE; extent = {}; displayInitialized = false;
  }

  void cleanup() noexcept {
    if (context.device) {
      vkDeviceWaitIdle(context.device);
      destroySwapchain();
      if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
      if (pipelineLayout) vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
      if (descriptorPool) vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
      if (descriptorLayout) vkDestroyDescriptorSetLayout(context.device, descriptorLayout, nullptr);
      if (acquired) vkDestroySemaphore(context.device, acquired, nullptr);
      if (fence) vkDestroyFence(context.device, fence, nullptr);
      if (commandPool) vkDestroyCommandPool(context.device, commandPool, nullptr);
    }
    // The renderer callback destroys the instance-owned surface after releasing
    // its device. All resources borrowing that device must already be gone.
    renderer.reset(); context = {};
  }

  bool createSwapchain(uint32_t width, uint32_t height) {
    VkSurfaceCapabilitiesKHR capabilities{};
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context.physicalDevice, surface, &capabilities), "Query viewer surface");
    VkExtent2D size = capabilities.currentExtent;
    if (size.width == std::numeric_limits<uint32_t>::max()) {
      size.width = std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
      size.height = std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }
    if (!size.width || !size.height) return false;
    if (!(capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
      throw std::runtime_error("The viewer surface does not support GPU transfer presentation");
    uint32_t count = 0;
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(context.physicalDevice, surface, &count, nullptr), "Query surface format count");
    std::vector<VkSurfaceFormatKHR> formats(count);
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(context.physicalDevice, surface, &count, formats.data()), "Query surface formats");
    VkSurfaceFormatKHR selected{};
    for (auto preferred : {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB}) {
      for (auto format : formats) {
        if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR && (format.format == preferred || format.format == VK_FORMAT_UNDEFINED)) {
          selected = {preferred, format.colorSpace}; break;
        }
      }
      if (selected.format != VK_FORMAT_UNDEFINED) break;
    }
    if (selected.format == VK_FORMAT_UNDEFINED) throw std::runtime_error("No compatible SDR RGBA/BGRA viewer surface format");
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(context.physicalDevice, selected.format, &properties);
    if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
      throw std::runtime_error("The viewer surface format does not support GPU blits");
    check(vkGetPhysicalDeviceSurfacePresentModesKHR(context.physicalDevice, surface, &count, nullptr), "Query present mode count");
    std::vector<VkPresentModeKHR> modes(count);
    check(vkGetPhysicalDeviceSurfacePresentModesKHR(context.physicalDevice, surface, &count, modes.data()), "Query present modes");
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end()) mode = VK_PRESENT_MODE_MAILBOX_KHR;
    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount) imageCount = std::min(imageCount, capabilities.maxImageCount);
    VkCompositeAlphaFlagBitsKHR composite = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(capabilities.supportedCompositeAlpha & composite))
      for (auto candidate : {VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
        if (capabilities.supportedCompositeAlpha & candidate) { composite = candidate; break; }
    check(vkDeviceWaitIdle(context.device), "Wait before rebuilding viewer swapchain");
    destroySwapchain();
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = surface; info.minImageCount = imageCount; info.imageFormat = selected.format;
    info.imageColorSpace = selected.colorSpace; info.imageExtent = size; info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT; info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = capabilities.currentTransform; info.compositeAlpha = composite; info.presentMode = mode; info.clipped = VK_TRUE;
    check(vkCreateSwapchainKHR(context.device, &info, nullptr, &swapchain), "Create viewer swapchain");
    extent = size; swapFormat = selected.format;
    check(vkGetSwapchainImagesKHR(context.device, swapchain, &count, nullptr), "Query swapchain image count");
    images.resize(count);
    check(vkGetSwapchainImagesKHR(context.device, swapchain, &count, images.data()), "Query swapchain images");
    initialized.assign(count, false); finished.resize(count, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto& semaphore : finished) check(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &semaphore), "Create image presentation semaphore");

    // Storage writes use the UNORM view. For an SRGB-only surface the image's
    // base format is SRGB, so the blit decodes/re-encodes the already encoded
    // bytes rather than applying the display transfer function twice.
    const VkFormat imageFormat = srgb(swapFormat) ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    vkGetPhysicalDeviceFormatProperties(context.physicalDevice, imageFormat, &properties);
    if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT))
      throw std::runtime_error("The GPU display image format does not support blits");
    vkGetPhysicalDeviceFormatProperties(context.physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &properties);
    if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
      throw std::runtime_error("The GPU display image format does not support storage writes");
    const VkFormat viewFormats[]{imageFormat, VK_FORMAT_R8G8B8A8_UNORM};
    VkImageFormatListCreateInfo formatList{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
    formatList.viewFormatCount = srgb(swapFormat) ? 2u : 1u; formatList.pViewFormats = viewFormats;
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    if (srgb(swapFormat)) {
      imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
      imageInfo.pNext = &formatList;
    }
    imageInfo.imageType = VK_IMAGE_TYPE_2D; imageInfo.format = imageFormat; imageInfo.extent = {size.width, size.height, 1};
    imageInfo.mipLevels = 1; imageInfo.arrayLayers = 1; imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL; imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateImage(context.device, &imageInfo, nullptr, &display), "Create GPU display image");
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(context.device, display, &requirements);
    VkMemoryAllocateInfo memoryInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    memoryInfo.allocationSize = requirements.size; memoryInfo.memoryTypeIndex = memoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(context.device, &memoryInfo, nullptr, &displayMemory), "Allocate GPU display image");
    check(vkBindImageMemory(context.device, display, displayMemory, 0), "Bind GPU display image");
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = display; viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(context.device, &viewInfo, nullptr, &displayView), "Create GPU display image view");
    requestedWidth = width; requestedHeight = height; rebuild = false;
    return true;
  }

  void beginCommands() {
    check(vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX), "Wait for viewer GPU commands");
    check(vkResetCommandBuffer(command, 0), "Reset viewer GPU commands");
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &info), "Begin viewer GPU commands");
  }
  void submitAndWait(VkSemaphore wait = VK_NULL_HANDLE, VkSemaphore signal = VK_NULL_HANDLE) {
    check(vkEndCommandBuffer(command), "End viewer GPU commands");
    check(vkResetFences(context.device, 1, &fence), "Reset viewer GPU fence");
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.commandBufferCount = 1; info.pCommandBuffers = &command;
    if (wait) { info.waitSemaphoreCount = 1; info.pWaitSemaphores = &wait; info.pWaitDstStageMask = &waitStage; }
    if (signal) { info.signalSemaphoreCount = 1; info.pSignalSemaphores = &signal; }
    check(vkQueueSubmit(context.queue, 1, &info, fence), "Submit viewer GPU commands");
    check(vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX), "Finish viewer GPU commands");
  }

  bool present(uint32_t width, uint32_t height) {
    if (!width || !height || IsIconic(GetAncestor(window, GA_ROOT))) return false;
    const auto frame = VulkanAccess::output(*renderer);
    if ((rebuild || requestedWidth != width || requestedHeight != height) && !createSwapchain(width, height)) return false;
    uint32_t imageIndex = 0;
    const auto acquireResult = vkAcquireNextImageKHR(context.device, swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &imageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) { rebuild = true; return false; }
    if (acquireResult != VK_SUBOPTIMAL_KHR) check(acquireResult, "Acquire viewer image");
    rebuild = acquireResult == VK_SUBOPTIMAL_KHR;
    beginCommands();
    const VkDescriptorBufferInfo hdrInfo{frame.buffer, 0, VkDeviceSize(frame.width) * frame.height * sizeof(Float4)};
    const VkDescriptorImageInfo displayInfo{VK_NULL_HANDLE, displayView, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) { writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = descriptors; writes[i].dstBinding = i; writes[i].descriptorCount = 1; }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[0].pBufferInfo = &hdrInfo;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].pImageInfo = &displayInfo;
    vkUpdateDescriptorSets(context.device, 2, writes, 0, nullptr);
    VkBufferMemoryBarrier hdrBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    hdrBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; hdrBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    hdrBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; hdrBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hdrBarrier.buffer = frame.buffer; hdrBarrier.offset = 0; hdrBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &hdrBarrier, 0, nullptr);
    VkImageMemoryBarrier displayBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    displayBarrier.srcAccessMask = displayInitialized ? VK_ACCESS_TRANSFER_READ_BIT : 0;
    displayBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    displayBarrier.oldLayout = displayInitialized ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    displayBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    displayBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; displayBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    displayBarrier.image = display; displayBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, displayInitialized ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &displayBarrier);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptors, 0, nullptr);
    const PresentConstants constants{frame.width, frame.height, extent.width, extent.height};
    vkCmdPushConstants(command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    vkCmdDispatch(command, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
    displayBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; displayBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    displayBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL; displayBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &displayBarrier);
    VkImageMemoryBarrier swapBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    swapBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    swapBarrier.oldLayout = initialized[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;
    swapBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; swapBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapBarrier.image = images[imageIndex]; swapBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &swapBarrier);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; blit.dstSubresource = blit.srcSubresource;
    blit.srcOffsets[1] = {int32_t(extent.width), int32_t(extent.height), 1}; blit.dstOffsets[1] = blit.srcOffsets[1];
    vkCmdBlitImage(command, display, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    swapBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; swapBarrier.dstAccessMask = 0;
    swapBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; swapBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &swapBarrier);
    submitAndWait(acquired, finished[imageIndex]);
    displayInitialized = true; initialized[imageIndex] = true;
    VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    info.waitSemaphoreCount = 1; info.pWaitSemaphores = &finished[imageIndex];
    info.swapchainCount = 1; info.pSwapchains = &swapchain; info.pImageIndices = &imageIndex;
    const auto result = vkQueuePresentKHR(context.queue, &info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) rebuild = true;
    else check(result, "Present viewer image");
    return result != VK_ERROR_OUT_OF_DATE_KHR;
  }

  std::vector<uint32_t> readbackLdr() {
    if (!displayInitialized) throw std::logic_error("No GPU display frame available for capture");
    VkBuffer buffer{}; VkDeviceMemory memory{}; void* mapped = nullptr;
    auto release = [&] {
      if (mapped) vkUnmapMemory(context.device, memory);
      if (buffer) vkDestroyBuffer(context.device, buffer, nullptr);
      if (memory) vkFreeMemory(context.device, memory, nullptr);
    };
    try {
      const VkDeviceSize size = VkDeviceSize(extent.width) * extent.height * 4;
      VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      info.size = size; info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT; info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      check(vkCreateBuffer(context.device, &info, nullptr, &buffer), "Create explicit display capture buffer");
      VkMemoryRequirements requirements{};
      vkGetBufferMemoryRequirements(context.device, buffer, &requirements);
      VkMemoryAllocateInfo memoryInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      memoryInfo.allocationSize = requirements.size;
      memoryInfo.memoryTypeIndex = memoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      check(vkAllocateMemory(context.device, &memoryInfo, nullptr, &memory), "Allocate explicit display capture");
      check(vkBindBufferMemory(context.device, buffer, memory, 0), "Bind explicit display capture");
      beginCommands();
      VkBufferImageCopy copy{};
      copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageExtent = {extent.width, extent.height, 1};
      vkCmdCopyImageToBuffer(command, display, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
      VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = buffer; barrier.size = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
      submitAndWait();
      check(vkMapMemory(context.device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "Map explicit display capture");
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; range.memory = memory; range.size = VK_WHOLE_SIZE;
      check(vkInvalidateMappedMemoryRanges(context.device, 1, &range), "Invalidate explicit display capture");
      std::vector<uint32_t> result(size_t(extent.width) * extent.height);
      const auto* rgba = static_cast<const uint8_t*>(mapped);
      for (size_t i = 0; i < result.size(); ++i)
        result[i] = uint32_t(rgba[4*i+2]) | (uint32_t(rgba[4*i+1]) << 8) | (uint32_t(rgba[4*i]) << 16);
      release(); return result;
    } catch (...) { release(); throw; }
  }
};

VulkanPresenter::VulkanPresenter(HWND window, bool validation) : VulkanPresenter(window, RendererOptions{validation}) {}
VulkanPresenter::VulkanPresenter(HWND window, const RendererOptions& options) : impl_(std::make_unique<Impl>(window, options)) {}
VulkanPresenter::~VulkanPresenter() = default;
Renderer& VulkanPresenter::renderer() { return *impl_->renderer; }
bool VulkanPresenter::present(uint32_t width, uint32_t height) { return impl_->present(width, height); }
uint32_t VulkanPresenter::width() const { return impl_->extent.width; }
uint32_t VulkanPresenter::height() const { return impl_->extent.height; }
std::vector<uint32_t> VulkanPresenter::readbackLdr() { return impl_->readbackLdr(); }

} // namespace caustra::example
