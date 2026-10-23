// Vulkan/NGX integration derived from vk_gltf_renderer's dlss_wrapper.
// Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
// Caustra adaptation: standalone allocation, planar guide uploads, synchronous
// readback integration, project identity, and exception-safe per-device lifetime.

#include "dlss_backend.hpp"

#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>
#include <nvsdk_ngx_helpers_dlssd.h>

#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifndef CAUSTRA_DLSS_PROJECT_ID
#define CAUSTRA_DLSS_PROJECT_ID "0d804d1c-880b-4bb2-bf53-e38a3d7ed6b8"
#endif

namespace caustra {
namespace {

std::mutex ngxMutex;
std::map<VkDevice, unsigned> ngxUsers;

void checkVk(VkResult result, const char* operation) {
  if(result != VK_SUCCESS)
    throw std::runtime_error(std::string("DLSS: ") + operation + " failed (VkResult " + std::to_string(result) + ")");
}

void checkNgx(NVSDK_NGX_Result result, const char* operation) {
  if(NVSDK_NGX_FAILED(result)) {
    std::string description;
    if(const wchar_t* text = GetNGXResultAsString(result))
      while(*text) { description += *text <= 127 ? char(*text) : '?'; ++text; }
    std::ostringstream message;
    message << "DLSS Ray Reconstruction: " << operation << " failed (" << description << ", NGX 0x" << std::hex << unsigned(result) << ").";
    throw std::runtime_error(message.str());
  }
}

void NVSDK_CONV ngxLog(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
  if(message) std::fprintf(stderr, "[NGX] %s\n", message);
}

const std::wstring& dataPath() {
  static const std::wstring path = [] {
    auto directory = std::filesystem::temp_directory_path() / "caustra-ngx";
    std::filesystem::create_directories(directory);
    return directory.wstring();
  }();
  return path;
}

NVSDK_NGX_FeatureDiscoveryInfo discoveryInfo(const NVSDK_NGX_FeatureCommonInfo* common) {
  NVSDK_NGX_FeatureDiscoveryInfo info{};
  info.SDKVersion = NVSDK_NGX_Version_API;
  info.FeatureID = NVSDK_NGX_Feature_RayReconstruction;
#ifdef CAUSTRA_DLSS_APPLICATION_ID
  info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
  info.Identifier.v.ApplicationId = CAUSTRA_DLSS_APPLICATION_ID;
#else
  info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
  info.Identifier.v.ProjectDesc.ProjectId = CAUSTRA_DLSS_PROJECT_ID;
  info.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
  info.Identifier.v.ProjectDesc.EngineVersion = "0.1.0";
#endif
  info.ApplicationDataPath = dataPath().c_str();
  info.FeatureInfo = common;
  return info;
}

NVSDK_NGX_PerfQuality_Value performance(DlssQuality quality) {
  switch(quality) {
    case DlssQuality::Native: return NVSDK_NGX_PerfQuality_Value_DLAA;
    case DlssQuality::Quality: return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    case DlssQuality::Balanced: return NVSDK_NGX_PerfQuality_Value_Balanced;
    case DlssQuality::Performance: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
  }
  throw std::invalid_argument("Unknown DLSS quality.");
}

std::vector<std::string> extensionNames(uint32_t count, const VkExtensionProperties* properties) {
  std::vector<std::string> names;
  names.reserve(count);
  for(uint32_t i = 0; i < count; ++i) names.emplace_back(properties[i].extensionName);
  return names;
}

constexpr VkImageSubresourceRange colorRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
constexpr VkImageSubresourceLayers colorLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};

struct NgxImage {
  VkDevice device{};
  VkImage image{};
  VkDeviceMemory memory{};
  VkImageView view{};
  VkFormat format{};
  VkExtent2D extent{};
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;

  NgxImage() = default;
  NgxImage(const NgxImage&) = delete;
  NgxImage& operator=(const NgxImage&) = delete;
  ~NgxImage() { clear(); }

  void clear() noexcept {
    if(view) vkDestroyImageView(device, view, nullptr);
    if(image) vkDestroyImage(device, image, nullptr);
    if(memory) vkFreeMemory(device, memory, nullptr);
    view = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    layout = VK_IMAGE_LAYOUT_UNDEFINED;
  }

  void create(VkDevice inDevice, VkPhysicalDevice physical, VkFormat inFormat, VkExtent2D inExtent) {
    device = inDevice;
    format = inFormat;
    extent = inExtent;
    VkFormatProperties formatProperties{};
    vkGetPhysicalDeviceFormatProperties(physical, format, &formatProperties);
    constexpr VkFormatFeatureFlags features = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    if((formatProperties.optimalTilingFeatures & features) != features)
      throw std::runtime_error("DLSS: required guide image format lacks storage/sampling support.");

    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {extent.width, extent.height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    checkVk(vkCreateImage(device, &info, nullptr, &image), "vkCreateImage");

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, image, &requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    uint32_t memoryType = properties.memoryTypeCount;
    for(uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      if((requirements.memoryTypeBits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        memoryType = i;
        break;
      }
    }
    if(memoryType == properties.memoryTypeCount) throw std::runtime_error("DLSS: no device-local image memory.");
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType;
    checkVk(vkAllocateMemory(device, &allocation, nullptr, &memory), "vkAllocateMemory");
    checkVk(vkBindImageMemory(device, image, memory, 0), "vkBindImageMemory");

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = colorRange;
    checkVk(vkCreateImageView(device, &viewInfo, nullptr, &view), "vkCreateImageView");
  }

  NVSDK_NGX_Resource_VK resource(bool writable) const {
    return NVSDK_NGX_Create_ImageView_Resource_VK(view, image, colorRange, format, extent.width, extent.height, writable);
  }

  void barrier(VkCommandBuffer cmd, VkImageLayout target, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = layout;
    barrier.newLayout = target;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = colorRange;
    vkCmdPipelineBarrier(cmd, layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    layout = target;
  }
};

enum Resource : size_t { Color, Diffuse, Specular, NormalRoughness, Motion, Depth, SpecularDistance, Output, ResourceCount };

} // namespace

struct DlssBackend::Impl {
  VkInstance instance{};
  VkPhysicalDevice physical{};
  VkDevice device{};
  bool initialized = false;
  NVSDK_NGX_Parameter* parameters{};
  NVSDK_NGX_Handle* feature{};
  std::array<NgxImage, ResourceCount> images;
  VkExtent2D input{}, output{};

  ~Impl() {
    // Renderer owns completion before teardown. Destroy the feature before its resources.
    if(feature) NVSDK_NGX_VULKAN_ReleaseFeature(feature);
    for(auto& image : images) image.clear();
    std::lock_guard lock(ngxMutex);
    if(parameters) NVSDK_NGX_VULKAN_DestroyParameters(parameters);
    if(initialized) {
      auto found = ngxUsers.find(device);
      if(found != ngxUsers.end() && --found->second == 0) {
        NVSDK_NGX_VULKAN_Shutdown1(device);
        ngxUsers.erase(found);
      }
    }
  }
};

std::vector<std::string> DlssBackend::requiredInstanceExtensions() {
  NVSDK_NGX_FeatureCommonInfo common{};
  const auto discovery = discoveryInfo(&common);
  uint32_t count = 0;
  VkExtensionProperties* properties{};
  std::lock_guard lock(ngxMutex);
  checkNgx(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&discovery, &count, &properties), "query instance extensions");
  return extensionNames(count, properties);
}

std::vector<std::string> DlssBackend::requiredDeviceExtensions(VkInstance instance, VkPhysicalDevice physical) {
  NVSDK_NGX_FeatureCommonInfo common{};
  const auto discovery = discoveryInfo(&common);
  uint32_t count = 0;
  VkExtensionProperties* properties{};
  std::lock_guard lock(ngxMutex);
  checkNgx(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, physical, &discovery, &count, &properties), "query device extensions");
  return extensionNames(count, properties);
}

DlssBackend::DlssBackend(VkInstance instance, VkPhysicalDevice physical, VkDevice device) : impl_(std::make_unique<Impl>()) {
  impl_->instance = instance;
  impl_->physical = physical;
  impl_->device = device;
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical, &properties);
  if(properties.vendorID != 0x10de) throw std::runtime_error("DLSS Ray Reconstruction requires a supported NVIDIA RTX GPU.");

  NVSDK_NGX_FeatureCommonInfo common{};
  common.LoggingInfo.LoggingCallback = ngxLog;
  common.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
  const auto discovery = discoveryInfo(&common);
  std::lock_guard lock(ngxMutex);
  NVSDK_NGX_FeatureRequirement requirements{};
  checkNgx(NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physical, &discovery, &requirements), "query feature requirements");
  if(requirements.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported)
    throw std::runtime_error("DLSS Ray Reconstruction is unsupported by the GPU/driver/OS (support flags " + std::to_string(unsigned(requirements.FeatureSupported)) + ").");

  auto found = ngxUsers.find(device);
  if(found == ngxUsers.end()) {
#ifdef CAUSTRA_DLSS_APPLICATION_ID
    checkNgx(NVSDK_NGX_VULKAN_Init(CAUSTRA_DLSS_APPLICATION_ID, dataPath().c_str(), instance, physical, device,
                                 vkGetInstanceProcAddr, vkGetDeviceProcAddr, &common), "initialize NGX");
#else
    checkNgx(NVSDK_NGX_VULKAN_Init_with_ProjectID(CAUSTRA_DLSS_PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1.0", dataPath().c_str(),
                                                instance, physical, device, vkGetInstanceProcAddr, vkGetDeviceProcAddr, &common), "initialize NGX");
#endif
    ngxUsers.emplace(device, 1);
  } else {
    ++found->second;
  }
  impl_->initialized = true;
  checkNgx(NVSDK_NGX_VULKAN_GetCapabilityParameters(&impl_->parameters), "query capabilities");
  if(!impl_->parameters) throw std::runtime_error("DLSS: NGX returned null capability parameters.");

  int needsDriver = 0;
  if(NVSDK_NGX_SUCCEED(impl_->parameters->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver, &needsDriver)) && needsDriver) {
    unsigned major = 0, minor = 0;
    impl_->parameters->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor, &major);
    impl_->parameters->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor, &minor);
    throw std::runtime_error("DLSS Ray Reconstruction requires a driver update (minimum " + std::to_string(major) + "." + std::to_string(minor) + ").");
  }
  int available = 0;
  checkNgx(impl_->parameters->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &available), "query Ray Reconstruction availability");
  if(!available) throw std::runtime_error("DLSS Ray Reconstruction is unavailable. Check the NVIDIA driver and nvngx_dlssd runtime deployment.");
}

DlssBackend::~DlssBackend() = default;

VkExtent2D DlssBackend::inputSize(VkExtent2D output, DlssQuality quality) const {
  if(quality == DlssQuality::Native) return output;
  VkExtent2D optimal{}, minimum{}, maximum{};
  float sharpness = 0;
  checkNgx(NGX_DLSSD_GET_OPTIMAL_SETTINGS(impl_->parameters, output.width, output.height, performance(quality),
                                       &optimal.width, &optimal.height, &maximum.width, &maximum.height,
                                       &minimum.width, &minimum.height, &sharpness), "query recommended input resolution");
  if(!optimal.width || !optimal.height || optimal.width > output.width || optimal.height > output.height)
    throw std::runtime_error("DLSS: NGX returned an invalid input resolution.");
  return optimal;
}

void DlssBackend::configure(VkCommandBuffer cmd, VkExtent2D input, VkExtent2D output, DlssQuality quality, DlssPreset preset) {
  auto& state = *impl_;
  if(state.feature) throw std::logic_error("DLSS backend is already configured; create a new backend for a size/preset change.");
  state.input = input;
  state.output = output;
  constexpr std::array<VkFormat, ResourceCount> formats{
    VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R32G32B32A32_SFLOAT};
  for(size_t i = 0; i < ResourceCount; ++i) {
    state.images[i].create(state.device, state.physical, formats[i], i == Output ? output : input);
    // Configure is submitted before publication of this backend. Keep GENERAL as
    // the stable layout at every API boundary, including an abandoned evaluate
    // command buffer if NGX rejects that evaluation.
    state.images[i].barrier(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            i == Output ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT);
  }

  const unsigned presetValue = static_cast<unsigned>(preset);
  for(const char* key : {NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
                        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
                        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
                        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
                        NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance})
    state.parameters->Set(key, presetValue);

  NVSDK_NGX_DLSSD_Create_Params create{};
  create.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
  create.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
  create.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;
  create.InWidth = input.width;
  create.InHeight = input.height;
  create.InTargetWidth = output.width;
  create.InTargetHeight = output.height;
  create.InPerfQualityValue = performance(quality);
  // The RR integration guide explicitly excludes exposure/auto-exposure and
  // sharpening; radiance is passed unchanged in linear HDR.
  create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
  checkNgx(NGX_VULKAN_CREATE_DLSSD_EXT1(state.device, cmd, 1, 1, &state.feature, state.parameters, &create), "create Ray Reconstruction feature");
}

void DlssBackend::evaluate(VkCommandBuffer cmd, VkBuffer guides, VkBuffer output, bool reset, float jitterX, float jitterY,
                           const std::array<float,16>& view, const std::array<float,16>& projection) {
  auto& state = *impl_;
  if(!state.feature) throw std::logic_error("DLSS backend is not configured.");
  VkBufferMemoryBarrier ready{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  ready.buffer = guides;
  ready.offset = 0;
  ready.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &ready, 0, nullptr);

  const VkDeviceSize pixels = VkDeviceSize(state.input.width) * state.input.height;
  constexpr std::array<uint32_t, Output> offsets{0, 16, 32, 48, 64, 72, 76};
  std::array<NVSDK_NGX_Resource_VK, ResourceCount> resources{};
  for(size_t i = 0; i < Output; ++i) {
    auto& image = state.images[i];
    image.barrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy copy{};
    copy.bufferOffset = offsets[i] * pixels;
    copy.imageSubresource = colorLayers;
    copy.imageExtent = {state.input.width, state.input.height, 1};
    vkCmdCopyBufferToImage(cmd, guides, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    image.barrier(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    resources[i] = image.resource(false);
  }
  auto& result = state.images[Output];
  result.barrier(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
  resources[Output] = result.resource(true);

  NVSDK_NGX_VK_DLSSD_Eval_Params evaluate{};
  evaluate.pInColor = &resources[Color];
  evaluate.pInDiffuseAlbedo = &resources[Diffuse];
  evaluate.pInSpecularAlbedo = &resources[Specular];
  evaluate.pInNormals = &resources[NormalRoughness];
  evaluate.pInRoughness = &resources[NormalRoughness];
  evaluate.pInMotionVectors = &resources[Motion];
  evaluate.pInDepth = &resources[Depth];
  evaluate.pInSpecularHitDistance = &resources[SpecularDistance];
  evaluate.pInOutput = &resources[Output];
  evaluate.InJitterOffsetX = -jitterX;
  evaluate.InJitterOffsetY = -jitterY;
  evaluate.InMVScaleX = 1;
  evaluate.InMVScaleY = 1;
  evaluate.InRenderSubrectDimensions = {state.input.width, state.input.height};
  evaluate.InReset = reset ? 1 : 0;
  evaluate.InPreExposure = 1;
  evaluate.InExposureScale = 1;
  evaluate.InFrameTimeDeltaInMsec = 1000.f / 60.f;
  // Caustra supplies conventional column-vector transforms in row-major memory.
  // NGX expects row-major matrices with the vector on the LEFT, so transpose the
  // mathematical transforms (translation lands in the final row).
  std::array<float,16> viewCopy{}, projectionCopy{};
  for(size_t row = 0; row < 4; ++row)
    for(size_t column = 0; column < 4; ++column) {
      viewCopy[4 * row + column] = view[4 * column + row];
      projectionCopy[4 * row + column] = projection[4 * column + row];
    }
  evaluate.pInWorldToViewMatrix = viewCopy.data();
  evaluate.pInViewToClipMatrix = projectionCopy.data();
  checkNgx(NGX_VULKAN_EVALUATE_DLSSD_EXT(cmd, state.feature, state.parameters, &evaluate), "evaluate Ray Reconstruction");

  result.barrier(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  VkBufferImageCopy readback{};
  readback.imageSubresource = colorLayers;
  readback.imageExtent = {state.output.width, state.output.height, 1};
  vkCmdCopyImageToBuffer(cmd, result.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, output, 1, &readback);
  result.barrier(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);

  VkBufferMemoryBarrier visible{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  visible.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  visible.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  visible.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  visible.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  visible.buffer = output;
  visible.offset = 0;
  visible.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &visible, 0, nullptr);
}

} // namespace caustra
