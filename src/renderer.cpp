// Vulkan Ray Query backend adapted from the vk_gltf_renderer rendering architecture.
// Scene import and application/editor state remain outside this library.
#include <caustra/renderer.hpp>
#include <caustra/vulkan.hpp>
#include "shader_types.hpp"
#include "material_coverage.hpp"
#include "shutdown_trace.hpp"
#include "pathtrace_spv.hpp"
#include "pathtrace_profile_spv.hpp"
#include <vulkan/vulkan.h>
#if CAUSTRA_HAS_DLSS
#include "dlss_backend.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace caustra {
namespace {
void check(VkResult result, const char* operation) {
  if(result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + " failed (VkResult " + std::to_string(result) + ")");
}
Float4 sub(Float4 a, Float4 b) { return {a.x-b.x,a.y-b.y,a.z-b.z,0}; }
Float4 mul(Float4 a, float b) { return {a.x*b,a.y*b,a.z*b,0}; }
Float4 cross(Float4 a, Float4 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x,0}; }
float length(Float4 a) { return std::hypot(a.x,a.y,a.z); }
Float4 normalize(Float4 a) {
  const double magnitude=std::hypot(double(a.x),double(a.y),double(a.z));
  return {float(a.x/magnitude),float(a.y/magnitude),float(a.z/magnitude),0};
}
Float4 cameraDirection(Float4 target,Float4 origin) {
  double x=double(target.x)-origin.x,y=double(target.y)-origin.y,z=double(target.z)-origin.z;
  double magnitude=std::hypot(x,y,z);
  return {float(x/magnitude),float(y/magnitude),float(z/magnitude),0};
}
bool sameCamera(const Camera& a,const Camera& b) {
  const auto near=[](float x,float y) { return std::abs(x-y)<=1e-6f*std::max({1.0f,std::abs(x),std::abs(y)}); };
  const auto near3=[&](Float4 x,Float4 y) { return near(x.x,y.x) && near(x.y,y.y) && near(x.z,y.z); };
  return near3(a.position,b.position) && near3(a.target,b.target) && near3(a.up,b.up) && near(a.verticalFov,b.verticalFov);
}
bool sameCameraOrigin(const Camera& a,const Camera& b) {
  return a.position.x==b.position.x && a.position.y==b.position.y && a.position.z==b.position.z;
}
float halton(uint32_t index,uint32_t base) {
  float value=0,weight=1;
  while(index) { weight/=base; value+=weight*(index%base); index/=base; }
  return value;
}
Float4 transform(const Instance& i, Float4 p) {
  const auto& m=i.transform;
  return {m[0]*p.x+m[1]*p.y+m[2]*p.z+m[3],m[4]*p.x+m[5]*p.y+m[6]*p.z+m[7],m[8]*p.x+m[9]*p.y+m[10]*p.z+m[11],0};
}
Float4 transformNormal(const Instance& i, Float4 n) {
  const auto& m=i.transform;
  Float4 a{m[0],m[4],m[8],0}, b{m[1],m[5],m[9],0}, c{m[2],m[6],m[10],0};
  auto x=cross(b,c), y=cross(c,a), z=cross(a,b);
  const float determinant=a.x*x.x+a.y*x.y+a.z*x.z;
  return {(x.x*n.x+y.x*n.y+z.x*n.z)/determinant,
          (x.y*n.x+y.y*n.y+z.y*n.z)/determinant,
          (x.z*n.x+y.z*n.y+z.z*n.z)/determinant,0};
}
float instanceOrientation(const Instance& i) {
  const auto& m=i.transform;
  const double determinant=double(m[0])*(double(m[5])*m[10]-double(m[6])*m[9])
    -double(m[1])*(double(m[4])*m[10]-double(m[6])*m[8])+double(m[2])*(double(m[4])*m[9]-double(m[5])*m[8]);
  return determinant<0?-1.0f:1.0f;
}
Float4 transformTangent(const Instance& i,Float4 tangent) {
  const auto& m=i.transform;
  return {m[0]*tangent.x+m[1]*tangent.y+m[2]*tangent.z,
          m[4]*tangent.x+m[5]*tangent.y+m[6]*tangent.z,
          m[8]*tangent.x+m[9]*tangent.y+m[10]*tangent.z,tangent.w*instanceOrientation(i)};
}

enum class BufferMemory { Device, Upload, Readback };
struct Buffer {
  VkDevice device{}; VkBuffer handle{}; VkDeviceMemory memory{}; void* mapped{}; VkDeviceSize size{};
  VkMemoryPropertyFlags memoryFlags{};
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& b) noexcept { *this=std::move(b); }
  Buffer& operator=(Buffer&& b) noexcept {
    if(this != &b) { clear(); device=b.device; handle=std::exchange(b.handle,{}); memory=std::exchange(b.memory,{}); mapped=std::exchange(b.mapped,nullptr); size=b.size; memoryFlags=b.memoryFlags; }
    return *this;
  }
  void clear() {
    if(mapped) vkUnmapMemory(device,memory);
    if(handle) vkDestroyBuffer(device,handle,nullptr);
    if(memory) vkFreeMemory(device,memory,nullptr);
    mapped=nullptr; handle={}; memory={};
  }
  void flush() const {
    if(!mapped || (memoryFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory=memory; range.size=VK_WHOLE_SIZE;
    check(vkFlushMappedMemoryRanges(device,1,&range),"Flush buffer memory");
  }
  void invalidate() const {
    if(!mapped || (memoryFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory=memory; range.size=VK_WHOLE_SIZE;
    check(vkInvalidateMappedMemoryRanges(device,1,&range),"Invalidate buffer memory");
  }
  ~Buffer() { clear(); }
};
struct PendingUpload { Buffer staging; VkBuffer destination{}; };
struct Acceleration {
  VkDevice device{}; VkAccelerationStructureKHR handle{};
  PFN_vkDestroyAccelerationStructureKHR destroy{};
  Buffer storage;
  Acceleration()=default;
  Acceleration(const Acceleration&)=delete;
  Acceleration& operator=(const Acceleration&)=delete;
  Acceleration(Acceleration&& a) noexcept { *this=std::move(a); }
  Acceleration& operator=(Acceleration&& a) noexcept {
    if(this!=&a) { clear(); device=a.device; handle=std::exchange(a.handle,{}); destroy=a.destroy; storage=std::move(a.storage); }
    return *this;
  }
  void clear() { if(handle) destroy(device,handle,nullptr); handle={}; storage.clear(); }
  ~Acceleration() { clear(); }
};
struct SceneResources {
  Buffer vertices, indices, instances, materials, lights;
  Buffer materialInstructions, textureInfos, texturePixels, materialMediumTable;
  Buffer coveragePrograms, coverageInstructions;
  std::vector<Acceleration> blas;
  Acceleration tlas;
  uint32_t lightCount=0, materialInitX=0, materialInitY=0;
  bool hasMedia=false, materialMediumDirty=true;
  std::vector<uint32_t> heterogeneousInstances;
};
}

struct Renderer::Impl {
  VulkanDeviceOptions vulkanOptions;
  VkInstance instance{}; VkDebugUtilsMessengerEXT messenger{};
  VkPhysicalDevice physical{}; VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceMemoryProperties memoryProperties{};
  VkDevice device{}; VkQueue queue{}; uint32_t queueFamily=0;
  VkCommandPool commandPool{}; VkCommandBuffer commandBuffer{}; VkFence fence{};
  VkQueryPool timestampPool{};
  uint32_t timestampValidBits=0;
  VkDescriptorSetLayout descriptorLayout{}; VkDescriptorPool descriptorPool{}; VkDescriptorSet descriptorSet{};
  VkPipelineLayout pipelineLayout{}; VkPipeline pipeline{};
  PFN_vkCreateAccelerationStructureKHR createAS{};
  PFN_vkDestroyAccelerationStructureKHR destroyAS{};
  PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes{};
  PFN_vkCmdBuildAccelerationStructuresKHR cmdBuild{};
  PFN_vkGetAccelerationStructureDeviceAddressKHR getASAddress{};
  uint32_t scratchAlignment=256;
  uint64_t maxPrimitiveCount=0, maxInstanceCount=0;
  std::atomic<uint32_t> validationErrors{0};
  RenderSettings settings; Camera camera, previousCamera; uint32_t samples=0;
  RendererOptions rendererOptions;
  RendererDiagnostics diagnostics;
  RendererProfile lastProfile;
  uint64_t frames=0;
  std::chrono::steady_clock::time_point previousFrameStart{};
  DlssSettings dlss;
  VkExtent2D renderSize{640,480};
  bool dlssExtensions=false;
  std::string dlssUnavailable="DLSS is not compiled into this build";
#if CAUSTRA_HAS_DLSS
  // Keep the NGX device session alive across Off/On and feature replacements.
  std::shared_ptr<DlssSession> dlssSession;
  std::unique_ptr<DlssBackend> denoiser;
#endif
  std::unique_ptr<SceneResources> scene;
  Buffer output, guides, denoisedOutput, rawReadback, denoisedReadback, cameraHistory, renderErrors, cameraMediumCache;
  bool rawReadbackCurrent=false, denoisedReadbackCurrent=false, cameraMediumDirty=true;

  ~Impl() {
    detail::ShutdownTrace shutdown("renderer");
    {
      detail::ShutdownTrace phase("renderer.waitIdle");
      if(device) vkDeviceWaitIdle(device);
    }
#if CAUSTRA_HAS_DLSS
    {
      detail::ShutdownTrace phase("renderer.denoiser");
      denoiser.reset();
    }
    {
      detail::ShutdownTrace phase("renderer.dlssSession");
      dlssSession.reset();
    }
#endif
    {
      detail::ShutdownTrace phase("renderer.sceneAndBuffers");
      scene.reset(); output.clear(); guides.clear(); denoisedOutput.clear(); rawReadback.clear(); denoisedReadback.clear();
      cameraHistory.clear(); renderErrors.clear(); cameraMediumCache.clear();
    }
    {
      detail::ShutdownTrace phase("renderer.pipelineAndCommands");
      if(pipeline) vkDestroyPipeline(device,pipeline,nullptr);
      if(pipelineLayout) vkDestroyPipelineLayout(device,pipelineLayout,nullptr);
      if(descriptorPool) vkDestroyDescriptorPool(device,descriptorPool,nullptr);
      if(descriptorLayout) vkDestroyDescriptorSetLayout(device,descriptorLayout,nullptr);
      if(fence) vkDestroyFence(device,fence,nullptr);
      if(timestampPool) vkDestroyQueryPool(device,timestampPool,nullptr);
      if(commandPool) vkDestroyCommandPool(device,commandPool,nullptr);
    }
    {
      detail::ShutdownTrace phase("renderer.destroyDevice");
      if(device) vkDestroyDevice(device,nullptr);
    }
    {
      detail::ShutdownTrace phase("renderer.instanceDestroyingCallback");
      if(instance && vulkanOptions.instanceDestroying) vulkanOptions.instanceDestroying(instance);
    }
    {
      detail::ShutdownTrace phase("renderer.destroyDebugMessenger");
      if(messenger) {
        auto fn=reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance,"vkDestroyDebugUtilsMessengerEXT"));
        if(fn) fn(instance,messenger,nullptr);
      }
    }
    {
      detail::ShutdownTrace phase("renderer.destroyInstance");
      if(instance) vkDestroyInstance(instance,nullptr);
    }
  }
  static VKAPI_ATTR VkBool32 VKAPI_CALL debug(VkDebugUtilsMessageSeverityFlagBitsEXT severity,VkDebugUtilsMessageTypeFlagsEXT,const VkDebugUtilsMessengerCallbackDataEXT* data,void* user) {
    if(severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ++static_cast<Impl*>(user)->validationErrors;
    std::cerr << "[Vulkan validation] " << data->pMessage << '\n'; return VK_FALSE;
  }
  void init(const RendererOptions& options) {
    if(options.mediumErrorPolicy!=MediumErrorPolicy::Strict && options.mediumErrorPolicy!=MediumErrorPolicy::SkipInvalidPath)
      throw std::invalid_argument("Unknown medium error policy");
    if(!std::isfinite(options.mediumBoundaryTolerance) || options.mediumBoundaryTolerance<0)
      throw std::invalid_argument("Medium boundary tolerance must be finite and nonnegative (meters)");
    rendererOptions=options;
    const char* validation="VK_LAYER_KHRONOS_validation";
    if(options.validation) {
      uint32_t count=0; check(vkEnumerateInstanceLayerProperties(&count,nullptr),"Enumerate layers");
      std::vector<VkLayerProperties> layers(count); check(vkEnumerateInstanceLayerProperties(&count,layers.data()),"Enumerate layers");
      if(std::none_of(layers.begin(),layers.end(),[&](const auto& p){return std::strcmp(p.layerName,validation)==0;}))
        throw std::runtime_error("Validation requested but VK_LAYER_KHRONOS_validation is unavailable. Install Vulkan SDK or omit --validation.");
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.pApplicationName="Caustra"; app.apiVersion=VK_API_VERSION_1_2;
    VkDebugUtilsMessengerCreateInfoEXT debugInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    debugInfo.messageSeverity=VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    debugInfo.messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugInfo.pfnUserCallback=debug; debugInfo.pUserData=this;
    const char* debugExt=VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    std::vector<std::string> instanceExtensions;
    if(options.validation) instanceExtensions.emplace_back(debugExt);
    for(const auto& extension:vulkanOptions.instanceExtensions) {
      if(extension.empty()) throw std::invalid_argument("Vulkan instance extension name must not be empty");
      if(std::find(instanceExtensions.begin(),instanceExtensions.end(),extension)==instanceExtensions.end()) instanceExtensions.push_back(extension);
    }
#if CAUSTRA_HAS_DLSS
    try {
      uint32_t n=0; check(vkEnumerateInstanceExtensionProperties(nullptr,&n,nullptr),"Enumerate instance extensions");
      std::vector<VkExtensionProperties> available(n); check(vkEnumerateInstanceExtensionProperties(nullptr,&n,available.data()),"Enumerate instance extensions");
      for(const auto& ext:DlssBackend::requiredInstanceExtensions()) {
        if(std::none_of(available.begin(),available.end(),[&](auto p){return ext==p.extensionName;})) throw std::runtime_error("Missing DLSS instance extension: "+ext);
        if(std::find(instanceExtensions.begin(),instanceExtensions.end(),ext)==instanceExtensions.end()) instanceExtensions.push_back(ext);
      }
      dlssExtensions=true; dlssUnavailable.clear();
    } catch(const std::exception& e) { dlssUnavailable=e.what(); }
#endif
    std::vector<const char*> instanceNames; for(const auto& ext:instanceExtensions) instanceNames.push_back(ext.c_str());
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo=&app;
    ci.enabledExtensionCount=static_cast<uint32_t>(instanceNames.size()); ci.ppEnabledExtensionNames=instanceNames.data();
    if(options.validation) { ci.enabledLayerCount=1; ci.ppEnabledLayerNames=&validation; ci.pNext=&debugInfo; }
    check(vkCreateInstance(&ci,nullptr,&instance),"Create Vulkan 1.2 instance");
    if(options.validation) {
      auto fn=reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance,"vkCreateDebugUtilsMessengerEXT"));
      if(!fn) throw std::runtime_error("VK_EXT_debug_utils unavailable");
      check(fn(instance,&debugInfo,nullptr,&messenger),"Create debug messenger");
    }
    if(vulkanOptions.instanceCreated) vulkanOptions.instanceCreated(instance);
    const char* extensions[]={VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,VK_KHR_RAY_QUERY_EXTENSION_NAME,VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
    uint32_t count=0; check(vkEnumeratePhysicalDevices(instance,&count,nullptr),"Enumerate GPUs");
    std::vector<VkPhysicalDevice> devices(count); check(vkEnumeratePhysicalDevices(instance,&count,devices.data()),"Enumerate GPUs");
    int best=-1;
    for(auto candidate:devices) {
      VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(candidate,&props);
      if(props.apiVersion<VK_API_VERSION_1_2) continue;
      uint32_t n=0; check(vkEnumerateDeviceExtensionProperties(candidate,nullptr,&n,nullptr),"Enumerate GPU extensions");
      std::vector<VkExtensionProperties> available(n); check(vkEnumerateDeviceExtensionProperties(candidate,nullptr,&n,available.data()),"Enumerate GPU extensions");
      bool supported=true;
      for(auto ext:extensions) supported &= std::any_of(available.begin(),available.end(),[&](auto p){return std::strcmp(ext,p.extensionName)==0;});
      for(const auto& ext:vulkanOptions.deviceExtensions) {
        if(ext.empty()) throw std::invalid_argument("Vulkan device extension name must not be empty");
        supported &= std::any_of(available.begin(),available.end(),[&](auto p){return ext==p.extensionName;});
      }
      VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
      VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR}; as.pNext=&rq;
      VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext=&as;
      VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f.pNext=&f12; vkGetPhysicalDeviceFeatures2(candidate,&f);
      if(!supported || !rq.rayQuery || !as.accelerationStructure || !f12.bufferDeviceAddress ||
         props.limits.maxPerStageDescriptorStorageBuffers<15 || props.limits.maxDescriptorSetStorageBuffers<15) continue;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,nullptr); std::vector<VkQueueFamilyProperties> families(n); vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,families.data());
      for(uint32_t i=0;i<n;++i) if(families[i].queueFlags&VK_QUEUE_COMPUTE_BIT) {
        if(vulkanOptions.supportsQueue && !vulkanOptions.supportsQueue(candidate,i)) continue;
        int score=(props.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU?10:1)+(props.vendorID==0x10de?4:0)+(families[i].queueFlags&VK_QUEUE_GRAPHICS_BIT?1:0);
        if(score>best) { best=score; physical=candidate; properties=props; queueFamily=i; timestampValidBits=families[i].timestampValidBits; }
      }
    }
    if(!physical) throw std::runtime_error("No Vulkan 1.2 GPU with rayQuery, accelerationStructure, bufferDeviceAddress and 15 storage-buffer bindings was found.");
    vkGetPhysicalDeviceMemoryProperties(physical,&memoryProperties);
    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; props2.pNext=&asProps; vkGetPhysicalDeviceProperties2(physical,&props2);
    scratchAlignment=asProps.minAccelerationStructureScratchOffsetAlignment;
    maxPrimitiveCount=asProps.maxPrimitiveCount; maxInstanceCount=asProps.maxInstanceCount;
    float priority=1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=queueFamily; qci.queueCount=1; qci.pQueuePriorities=&priority;
    VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR}; rq.rayQuery=VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR}; as.accelerationStructure=VK_TRUE; as.pNext=&rq;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.bufferDeviceAddress=VK_TRUE; f12.pNext=&as;
    std::vector<std::string> deviceExtensions(std::begin(extensions),std::end(extensions));
    for(const auto& ext:vulkanOptions.deviceExtensions)
      if(std::find(deviceExtensions.begin(),deviceExtensions.end(),ext)==deviceExtensions.end()) deviceExtensions.push_back(ext);
#if CAUSTRA_HAS_DLSS
    if(dlssExtensions) {
      try {
        if(properties.vendorID!=0x10de) throw std::runtime_error("DLSS Ray Reconstruction requires a supported NVIDIA RTX GPU");
        uint32_t n=0; check(vkEnumerateDeviceExtensionProperties(physical,nullptr,&n,nullptr),"Enumerate DLSS extensions");
        std::vector<VkExtensionProperties> available(n); check(vkEnumerateDeviceExtensionProperties(physical,nullptr,&n,available.data()),"Enumerate DLSS extensions");
        auto required=DlssBackend::requiredDeviceExtensions(instance,physical);
        for(const auto& ext:required) if(std::none_of(available.begin(),available.end(),[&](auto p){return ext==p.extensionName;})) throw std::runtime_error("Missing DLSS device extension: "+ext);
        for(const auto& ext:required) if(std::find(deviceExtensions.begin(),deviceExtensions.end(),ext)==deviceExtensions.end()) deviceExtensions.push_back(ext);
      } catch(const std::exception& e) { dlssExtensions=false; dlssUnavailable=e.what(); }
    }
#endif
    std::vector<const char*> deviceNames; for(const auto& ext:deviceExtensions) deviceNames.push_back(ext.c_str());
    VkPhysicalDeviceFeatures supportedFeatures{}, enabledFeatures{};
    vkGetPhysicalDeviceFeatures(physical,&supportedFeatures);
    if(dlssExtensions) {
      if(!supportedFeatures.shaderStorageImageWriteWithoutFormat) {
        dlssExtensions=false; dlssUnavailable="DLSS requires shaderStorageImageWriteWithoutFormat support";
      } else {
        enabledFeatures.shaderStorageImageWriteWithoutFormat=VK_TRUE;
        enabledFeatures.shaderStorageImageReadWithoutFormat=supportedFeatures.shaderStorageImageReadWithoutFormat;
      }
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext=&f12; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.enabledExtensionCount=static_cast<uint32_t>(deviceNames.size()); dci.ppEnabledExtensionNames=deviceNames.data();
    dci.pEnabledFeatures=&enabledFeatures;
    check(vkCreateDevice(physical,&dci,nullptr,&device),"Create ray-query device"); vkGetDeviceQueue(device,queueFamily,0,&queue);
#define LOAD_AS(member, name) member=reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device,#name)); if(!member) throw std::runtime_error("Missing " #name)
    LOAD_AS(createAS,vkCreateAccelerationStructureKHR);
    LOAD_AS(destroyAS,vkDestroyAccelerationStructureKHR);
    LOAD_AS(getBuildSizes,vkGetAccelerationStructureBuildSizesKHR);
    LOAD_AS(cmdBuild,vkCmdBuildAccelerationStructuresKHR);
    LOAD_AS(getASAddress,vkGetAccelerationStructureDeviceAddressKHR);
#undef LOAD_AS
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex=queueFamily;
    pci.flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT|VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(device,&pci,nullptr,&commandPool),"Create command pool");
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool=commandPool; cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount=1;
    check(vkAllocateCommandBuffers(device,&cai,&commandBuffer),"Allocate command buffer");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; check(vkCreateFence(device,&fci,nullptr,&fence),"Create fence");
    invalidateProfile();
    if(rendererOptions.enableGpuTiming && lastProfile.gpuTimingSupported) {
      VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      qpi.queryType=VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount=4;
      check(vkCreateQueryPool(device,&qpi,nullptr,&timestampPool),"Create timestamp query pool");
    }
    createPipeline();
    cameraHistory=buffer(sizeof(GpuCameraHistory),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    renderErrors=buffer(renderDiagnosticWords()*sizeof(uint32_t),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    cameraMediumCache=buffer(sizeof(GpuCameraMediumCache),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,BufferMemory::Device);
    resizeOutput(settings,dlss);
  }
  Buffer buffer(VkDeviceSize bytes,VkBufferUsageFlags usage,BufferMemory memory=BufferMemory::Upload,const void* data=nullptr) {
    Buffer b; b.device=device; b.size=std::max<VkDeviceSize>(bytes,16);
    if((usage&VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && b.size>properties.limits.maxStorageBufferRange)
      throw std::invalid_argument("Buffer exceeds this GPU's maxStorageBufferRange");
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=b.size; ci.usage=usage; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device,&ci,nullptr,&b.handle),"Create buffer");
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device,b.handle,&req);
    const bool host=memory!=BufferMemory::Device;
    const VkMemoryPropertyFlags required=host?VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t type=UINT32_MAX; int bestScore=-1;
    for(uint32_t i=0;i<memoryProperties.memoryTypeCount;++i) {
      const auto flags=memoryProperties.memoryTypes[i].propertyFlags;
      if(!(req.memoryTypeBits&(1u<<i)) || (flags&required)!=required) continue;
      // CPU readback favors cached memory even when explicit invalidation is
      // required. Upload/uniform buffers favor coherent host writes instead.
      int score=0;
      if(memory==BufferMemory::Readback) {
        if(flags&VK_MEMORY_PROPERTY_HOST_CACHED_BIT) score+=4;
        if(flags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) score+=2;
      } else if(memory==BufferMemory::Upload && (flags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) score+=2;
      if(score>bestScore) {type=i; bestScore=score;}
    }
    if(type==UINT32_MAX) throw std::runtime_error("No suitable Vulkan memory type");
    b.memoryFlags=memoryProperties.memoryTypes[type].propertyFlags;
    VkMemoryAllocateFlagsInfo addressFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; addressFlags.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=req.size; ai.memoryTypeIndex=type;
    if(usage&VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ai.pNext=&addressFlags;
    check(vkAllocateMemory(device,&ai,nullptr,&b.memory),"Allocate buffer memory"); check(vkBindBufferMemory(device,b.handle,b.memory,0),"Bind buffer memory");
    if(host) {
      check(vkMapMemory(device,b.memory,0,VK_WHOLE_SIZE,0,&b.mapped),"Map buffer");
      if(memory==BufferMemory::Upload) {
        std::memset(b.mapped,0,static_cast<size_t>(b.size));
        if(data && bytes) std::memcpy(b.mapped,data,static_cast<size_t>(bytes));
        b.flush();
      }
    } else if(data) throw std::logic_error("Device-local data must be uploaded through staging");
    return b;
  }
  Buffer stageBuffer(VkDeviceSize bytes,VkBufferUsageFlags usage,const void* data,std::vector<PendingUpload>& pending) {
    if(bytes && !data) throw std::invalid_argument("Nonempty buffer upload requires source data");
    auto destination=buffer(bytes,usage|VK_BUFFER_USAGE_TRANSFER_DST_BIT,BufferMemory::Device);
    auto staging=buffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,BufferMemory::Upload,data);
    pending.push_back({std::move(staging),destination.handle});
    return destination;
  }
  void uploadBuffers(std::vector<PendingUpload>& pending) {
    if(pending.empty()) return;
    submit([&](VkCommandBuffer cmd) {
      barrier(cmd,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_HOST_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
      for(const auto& upload:pending) {
        const VkBufferCopy copy{0,0,upload.staging.size};
        vkCmdCopyBuffer(cmd,upload.staging.handle,upload.destination,1,&copy);
      }
      barrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR|VK_ACCESS_SHADER_READ_BIT);
    });
    pending.clear();
  }
  VkDeviceAddress address(const Buffer& b) { VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; info.buffer=b.handle; return vkGetBufferDeviceAddress(device,&info); }
  VkDeviceAddress address(const Acceleration& a) { VkAccelerationStructureDeviceAddressInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR}; info.accelerationStructure=a.handle; return getASAddress(device,&info); }
  void submit(const std::function<void(VkCommandBuffer)>& record) {
    const auto cmd=commandBuffer;
    check(vkResetCommandBuffer(cmd,0),"Reset command buffer");
    try {
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      check(vkBeginCommandBuffer(cmd,&bi),"Begin command buffer"); record(cmd); check(vkEndCommandBuffer(cmd),"End command buffer");
      check(vkResetFences(device,1,&fence),"Reset fence");
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
      check(vkQueueSubmit(queue,1,&si,fence),"Submit GPU work");
      check(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX),"Wait for GPU");
    } catch(...) { vkDeviceWaitIdle(device); vkResetCommandBuffer(cmd,0); throw; }
  }
  static void barrier(VkCommandBuffer cmd,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkAccessFlags source,VkAccessFlags dest) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; b.srcAccessMask=source; b.dstAccessMask=dest; vkCmdPipelineBarrier(cmd,src,dst,0,1,&b,0,nullptr,0,nullptr);
  }
  Acceleration build(VkAccelerationStructureTypeKHR type,const VkAccelerationStructureGeometryKHR& geometry,uint32_t count) {
    if(count>(type==VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR?maxPrimitiveCount:maxInstanceCount))
      throw std::invalid_argument("Scene exceeds this GPU's acceleration structure primitive/instance limit");
    VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    info.type=type; info.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR; info.mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR; info.geometryCount=1; info.pGeometries=&geometry;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR}; getBuildSizes(device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&info,&count,&sizes);
    Acceleration result; result.device=device; result.destroy=destroyAS;
    result.storage=buffer(sizes.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,BufferMemory::Device);
    VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR}; ci.type=type; ci.buffer=result.storage.handle; ci.size=sizes.accelerationStructureSize;
    check(createAS(device,&ci,nullptr,&result.handle),"Create acceleration structure");
    Buffer scratch=buffer(sizes.buildScratchSize+scratchAlignment,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,BufferMemory::Device);
    VkDeviceAddress base=address(scratch); info.scratchData.deviceAddress=(base+scratchAlignment-1)&~VkDeviceAddress(scratchAlignment-1); info.dstAccelerationStructure=result.handle;
    VkAccelerationStructureBuildRangeInfoKHR range{}; range.primitiveCount=count; const auto* ranges=&range;
    submit([&](VkCommandBuffer cmd){
      barrier(cmd,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
      cmdBuild(cmd,1,&info,&ranges);
      barrier(cmd,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    });
    return result;
  }
  void createPipeline() {
    std::array<VkDescriptorSetLayoutBinding,17> bindings{};
    for(uint32_t i=0;i<bindings.size();++i) { bindings[i].binding=i; bindings[i].descriptorCount=1; bindings[i].descriptorType=i==0?VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:(i==8?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; li.bindingCount=static_cast<uint32_t>(bindings.size()); li.pBindings=bindings.data();
    check(vkCreateDescriptorSetLayout(device,&li,nullptr,&descriptorLayout),"Create descriptor layout");
    VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,15},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pi.maxSets=1; pi.poolSizeCount=3; pi.pPoolSizes=sizes;
    check(vkCreateDescriptorPool(device,&pi,nullptr,&descriptorPool),"Create descriptor pool");
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool=descriptorPool; ai.descriptorSetCount=1; ai.pSetLayouts=&descriptorLayout;
    check(vkAllocateDescriptorSets(device,&ai,&descriptorSet),"Allocate descriptor set");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pci.setLayoutCount=1; pci.pSetLayouts=&descriptorLayout; pci.pushConstantRangeCount=1; pci.pPushConstantRanges=&range;
    check(vkCreatePipelineLayout(device,&pci,nullptr,&pipelineLayout),"Create pipeline layout");
    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sci.codeSize=rendererOptions.enableRayStatistics?sizeof(pathtraceProfileSpv):sizeof(pathtraceSpv);
    sci.pCode=rendererOptions.enableRayStatistics?pathtraceProfileSpv:pathtraceSpv;
    VkShaderModule shader{}; check(vkCreateShaderModule(device,&sci,nullptr,&shader),"Create shader module");
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.layout=pipelineLayout; ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module=shader; ci.stage.pName="main";
    auto result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&ci,nullptr,&pipeline); vkDestroyShaderModule(device,shader,nullptr); check(result,"Create path-trace pipeline");
  }
  void descriptors() {
    if(!scene) return;
    VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR}; as.accelerationStructureCount=1; as.pAccelerationStructures=&scene->tlas.handle;
    std::array<VkWriteDescriptorSet,17> writes{}; std::array<VkDescriptorBufferInfo,16> infos{};
    Buffer* buffers[]={&scene->vertices,&scene->indices,&scene->instances,&scene->materials,&scene->lights,&output,&guides,&cameraHistory,&scene->materialInstructions,&scene->textureInfos,&scene->texturePixels,&renderErrors,&cameraMediumCache,&scene->materialMediumTable,&scene->coveragePrograms,&scene->coverageInstructions};
    for(uint32_t i=0;i<writes.size();++i) {
      auto& w=writes[i]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=descriptorSet; w.dstBinding=i; w.descriptorCount=1;
      w.descriptorType=i==0?VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:(i==8?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
      if(i==0) w.pNext=&as;
      else { infos[i-1]={buffers[i-1]->handle,0,buffers[i-1]->size}; w.pBufferInfo=&infos[i-1]; }
    }
    vkUpdateDescriptorSets(device,static_cast<uint32_t>(writes.size()),writes.data(),0,nullptr);
  }
  void resizeOutput(const RenderSettings& desired,const DlssSettings& desiredDlss) {
    if(desiredDlss.enabled && (desired.width<32 || desired.height<32))
      throw std::invalid_argument("DLSS output dimensions must each be at least 32 pixels");
    VkExtent2D extent{desired.width,desired.height};
#if CAUSTRA_HAS_DLSS
    std::unique_ptr<DlssBackend> candidate;
    if(desiredDlss.enabled) {
      if(!dlssExtensions) throw std::runtime_error(dlssUnavailable);
      if(!dlssSession) dlssSession=std::make_shared<DlssSession>(instance,physical,device);
      candidate=std::make_unique<DlssBackend>(dlssSession);
      extent=candidate->inputSize({desired.width,desired.height},desiredDlss.quality);
    }
#else
    if(desiredDlss.enabled) throw std::runtime_error(dlssUnavailable+". Reconfigure with CAUSTRA_ENABLE_DLSS=ON.");
#endif
    const VkDeviceSize bytes=VkDeviceSize(extent.width)*extent.height*sizeof(Float4);
    if(bytes>properties.limits.maxStorageBufferRange || (extent.width+7)/8>properties.limits.maxComputeWorkGroupCount[0] || (extent.height+7)/8>properties.limits.maxComputeWorkGroupCount[1])
      throw std::invalid_argument("Image size exceeds this GPU's storage-buffer or dispatch limit");
    auto next=buffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,BufferMemory::Device);
    auto nextGuides=buffer(desiredDlss.enabled?VkDeviceSize(extent.width)*extent.height*80:16,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,BufferMemory::Device);
    Buffer nextDenoised;
#if CAUSTRA_HAS_DLSS
    if(candidate) {
      nextDenoised=buffer(VkDeviceSize(desired.width)*desired.height*sizeof(Float4),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,BufferMemory::Device);
      submit([&](VkCommandBuffer cmd){candidate->configure(cmd,extent,{desired.width,desired.height},desiredDlss.quality,desiredDlss.preset);});
    }
    denoiser=std::move(candidate);
#endif
    output=std::move(next); guides=std::move(nextGuides); denoisedOutput=std::move(nextDenoised);
    rawReadback.clear();
    denoisedReadback.clear();
    renderSize=extent; settings=desired; dlss=desiredDlss; resetHistory(); descriptors();
  }
  void upload(const Scene& input) {
    validateScene(input);
    auto next=std::make_unique<SceneResources>();
    std::vector<PendingUpload> pending;
    for(const auto& material:input.materials) {
      next->hasMedia=next->hasMedia || material.transmission.x>0 || material.subsurface.x>0 || material.volume.x>0 || material.volume.z>0;
      for(uint32_t j=0;j<material.program.y;++j) {
        const auto& op=input.materialInstructions[material.program.x+j].code;
        if(op.x==uint32_t(MaterialOp::Output) && ((op.z>=25 && op.z<=29) || (op.z>=31 && op.z<=36) || (op.z>=44 && op.z<=49))) next->hasMedia=true;
      }
    }
    std::vector<Vertex> vertices; std::vector<uint32_t> indices;
    std::vector<Uint4> meshOffsets;
    std::vector<std::pair<Float4,Float4>> meshBounds;
    for(const auto& mesh:input.meshes) {
      if(vertices.size()+mesh.vertices.size()>UINT32_MAX || indices.size()+mesh.indices.size()>UINT32_MAX) throw std::invalid_argument("Scene exceeds 32-bit geometry indexing");
      meshOffsets.push_back({static_cast<uint32_t>(vertices.size()),static_cast<uint32_t>(indices.size()),mesh.material,0});
      vertices.insert(vertices.end(),mesh.vertices.begin(),mesh.vertices.end()); indices.insert(indices.end(),mesh.indices.begin(),mesh.indices.end());
      Float4 low=mesh.vertices.front().position,high=low;
      for(const auto& v:mesh.vertices) {
        low={std::min(low.x,v.position.x),std::min(low.y,v.position.y),std::min(low.z,v.position.z),0};
        high={std::max(high.x,v.position.x),std::max(high.y,v.position.y),std::max(high.z,v.position.z),0};
      }
      meshBounds.emplace_back(low,high);
    }
    auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    next->vertices=stageBuffer(vertices.size()*sizeof(Vertex),usage,vertices.data(),pending);
    next->indices=stageBuffer(indices.size()*sizeof(uint32_t),usage,indices.data(),pending);
    next->materials=stageBuffer(input.materials.size()*sizeof(Material),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,input.materials.data(),pending);
    next->materialInstructions=stageBuffer(input.materialInstructions.size()*sizeof(MaterialInstruction),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,input.materialInstructions.data(),pending);
    const auto coverage=buildCoveragePrograms(input);
    next->coveragePrograms=stageBuffer(coverage.ranges.size()*sizeof(Uint4),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,coverage.ranges.data(),pending);
    next->coverageInstructions=stageBuffer(coverage.instructions.size()*sizeof(MaterialInstruction),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,coverage.instructions.data(),pending);
    const uint32_t materialGroups=static_cast<uint32_t>((uint64_t(input.materials.size())+63)/64);
    next->materialInitX=std::min(materialGroups,properties.limits.maxComputeWorkGroupCount[0]);
    next->materialInitY=(materialGroups+next->materialInitX-1)/next->materialInitX;
    if(next->materialInitY>properties.limits.maxComputeWorkGroupCount[1])
      throw std::invalid_argument("Material medium table initialization exceeds this GPU's dispatch limit");
    std::vector<GpuTextureInfo> textureInfos;
    std::vector<Float4> texturePixels;
    for(const auto& texture:input.textures) {
      uint64_t texelCount=0;
      for(uint32_t w=texture.width,h=texture.height;;w=std::max(1u,w/2),h=std::max(1u,h/2)) {
        texelCount+=uint64_t(w)*h;if(w==1 && h==1) break;
      }
      if((texelCount+texturePixels.size())*sizeof(Float4)>properties.limits.maxStorageBufferRange)
        throw std::invalid_argument("Texture mip atlas exceeds this GPU's storage-buffer range");
      GpuTextureInfo info{{texture.width,texture.height,static_cast<uint32_t>(texturePixels.size()),0}};
      auto level=texture.pixels;uint32_t w=texture.width,h=texture.height;
      for(;;) {
        texturePixels.insert(texturePixels.end(),level.begin(),level.end());++info.size.w;
        if(w==1 && h==1) break;
        const uint32_t nw=std::max(1u,w/2),nh=std::max(1u,h/2);
        std::vector<Float4> smaller(size_t(nw)*nh);
        // Area filtering includes the last row/column of odd-sized images.
        for(uint32_t y=0;y<nh;++y) for(uint32_t x=0;x<nw;++x) {
          const double x0=double(x)*w/nw,x1=double(x+1)*w/nw,y0=double(y)*h/nh,y1=double(y+1)*h/nh;
          double sum[4]{};
          for(uint32_t sy=uint32_t(y0);sy<std::min(h,uint32_t(std::ceil(y1)));++sy)
            for(uint32_t sx=uint32_t(x0);sx<std::min(w,uint32_t(std::ceil(x1)));++sx) {
              const double weight=(std::min(x1,double(sx+1))-std::max(x0,double(sx)))*(std::min(y1,double(sy+1))-std::max(y0,double(sy)));
              const auto& p=level[size_t(sy)*w+sx];sum[0]+=p.x*weight;sum[1]+=p.y*weight;sum[2]+=p.z*weight;sum[3]+=p.w*weight;
            }
          const double area=(x1-x0)*(y1-y0);
          smaller[size_t(y)*nw+x]={float(sum[0]/area),float(sum[1]/area),float(sum[2]/area),float(sum[3]/area)};
        }
        level=std::move(smaller);w=nw;h=nh;
      }
      textureInfos.push_back(info);
    }
    if(textureInfos.empty()) {textureInfos.push_back({{1,1,0,1}});texturePixels.push_back({1,1,1,1});}
    next->textureInfos=stageBuffer(textureInfos.size()*sizeof(GpuTextureInfo),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,textureInfos.data(),pending);
    next->texturePixels=stageBuffer(texturePixels.size()*sizeof(Float4),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,texturePixels.data(),pending);
    uploadBuffers(pending);
    for(size_t m=0;m<input.meshes.size();++m) {
      const auto& mesh=input.meshes[m];
      VkAccelerationStructureGeometryKHR geo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR}; geo.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR; geo.flags=VK_GEOMETRY_OPAQUE_BIT_KHR;
      auto& tri=geo.geometry.triangles; tri.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
      tri.vertexFormat=VK_FORMAT_R32G32B32_SFLOAT; tri.vertexData.deviceAddress=address(next->vertices)+VkDeviceSize(meshOffsets[m].x)*sizeof(Vertex); tri.vertexStride=sizeof(Vertex); tri.maxVertex=static_cast<uint32_t>(mesh.vertices.size()-1);
      tri.indexType=VK_INDEX_TYPE_UINT32; tri.indexData.deviceAddress=address(next->indices)+VkDeviceSize(meshOffsets[m].y)*sizeof(uint32_t);
      next->blas.push_back(build(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,geo,static_cast<uint32_t>(mesh.indices.size()/3)));
    }
    std::vector<GpuInstance> instanceData; std::vector<VkAccelerationStructureInstanceKHR> asInstances; std::vector<GpuLight> lights;
    std::vector<double> lightWeights,emissionWeights(input.materials.size());
    for(size_t index=0;index<input.materials.size();++index) {
      const auto& material=input.materials[index];bool dynamic=false;
      for(uint32_t instruction=0;instruction<material.program.y;++instruction) {
        const auto& op=input.materialInstructions[material.program.x+instruction].code;
        dynamic|=op.x==uint32_t(MaterialOp::Output) && op.z==uint32_t(MaterialParameter::Emission);
      }
      const double luminance=.2126*material.emission.x+.7152*material.emission.y+.0722*material.emission.z;
      emissionWeights[index]=luminance>0?luminance:(dynamic?1.0:0.0);
    }
    if(input.instances.size()>0x1000000) throw std::invalid_argument("Too many instances for Vulkan's 24-bit custom index");
    for(size_t n=0;n<input.instances.size();++n) {
      const auto& i=input.instances[n]; const auto& mesh=input.meshes[i.mesh]; const auto& mat=input.materials[mesh.material];
      GpuInstance gpu{};gpu.offsets=meshOffsets[i.mesh];
      const auto& m=i.transform;
      Float4 column0{m[0],m[4],m[8],0},column1{m[1],m[5],m[9],0},column2{m[2],m[6],m[10],0};
      auto x=cross(column1,column2),y=cross(column2,column0),z=cross(column0,column1);
      const float invDet=1.0f/(column0.x*x.x+column0.y*x.y+column0.z*x.z);
      gpu.worldToObject={mul(x,invDet),mul(y,invDet),mul(z,invDet)};
      for(size_t row=0;row<3;++row) {
        auto& r=gpu.worldToObject[row];r.w=-(r.x*m[3]+r.y*m[7]+r.z*m[11]);
        gpu.objectToWorld[row]={m[row*4],m[row*4+1],m[row*4+2],m[row*4+3]};
      }
      gpu.objectMin=meshBounds[i.mesh].first;gpu.objectMax=meshBounds[i.mesh].second;
      const double emissionWeight=emissionWeights[mesh.material];
      if(emissionWeight>0 && lights.size()+mesh.indices.size()/3>=kLightSelectionTickets)
        throw std::invalid_argument("Emitter triangle count exceeds the 24-bit selection ticket capacity");
      gpu.objectMin.w=std::bit_cast<float>(emissionWeight>0?static_cast<uint32_t>(lights.size()):UINT32_MAX);
      if(mat.program.w&1) {
        const uint64_t base=(uint64_t(input.materials.size())+uint64_t(next->heterogeneousInstances.size())*kMediumGridCells)*8;
        if(base+uint64_t(kMediumGridCells)*8>UINT32_MAX) throw std::invalid_argument("Heterogeneous medium grid exceeds 32-bit row addressing");
        gpu.offsets.w=uint32_t(base);next->heterogeneousInstances.push_back(uint32_t(n));next->hasMedia=true;
      }
      instanceData.push_back(gpu);
      VkAccelerationStructureInstanceKHR vi{}; std::memcpy(vi.transform.matrix,i.transform.data(),sizeof(vi.transform.matrix));
      vi.instanceCustomIndex=static_cast<uint32_t>(n); vi.mask=0xFF; vi.flags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR; vi.accelerationStructureReference=address(next->blas[i.mesh]); asInstances.push_back(vi);
      if(emissionWeight>0) for(size_t t=0;t<mesh.indices.size();t+=3) {
        auto a=transform(i,mesh.vertices[mesh.indices[t]].position), b=transform(i,mesh.vertices[mesh.indices[t+1]].position), c=transform(i,mesh.vertices[mesh.indices[t+2]].position);
        auto e1=sub(b,a),e2=sub(c,a); float area=length(cross(e1,e2))*0.5f;
        if(!std::isfinite(area) || !(area>0)) throw std::invalid_argument("Transformed light area is invalid");
        lightWeights.push_back(double(area)*emissionWeight);
        a.w=area;e2.w=instanceOrientation(i);
        const auto offset=meshOffsets[i.mesh].x;
        auto normal0=transformNormal(i,mesh.vertices[mesh.indices[t]].normal);
        normal0.w=std::bit_cast<float>(static_cast<uint32_t>(n));
        lights.push_back({a,e1,e2,{offset+mesh.indices[t],offset+mesh.indices[t+1],offset+mesh.indices[t+2],mesh.material},
                          normal0,
                          transformNormal(i,mesh.vertices[mesh.indices[t+1]].normal),
                          transformNormal(i,mesh.vertices[mesh.indices[t+2]].normal),
                          transformTangent(i,mesh.vertices[mesh.indices[t]].tangent),
                          transformTangent(i,mesh.vertices[mesh.indices[t+1]].tangent),
                          transformTangent(i,mesh.vertices[mesh.indices[t+2]].tangent)});
      }
    }
    buildLightCdf(lights,lightWeights);
    next->lightCount=static_cast<uint32_t>(lights.size());
    const VkDeviceSize mediumBytes=(VkDeviceSize(input.materials.size())+VkDeviceSize(next->heterogeneousInstances.size())*kMediumGridCells)*sizeof(GpuMaterialMediumData);
    if(mediumBytes>properties.limits.maxStorageBufferRange) throw std::invalid_argument("Heterogeneous medium grids exceed this GPU's storage buffer range");
    next->materialMediumTable=buffer(mediumBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,BufferMemory::Device);
    next->instances=stageBuffer(instanceData.size()*sizeof(GpuInstance),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,instanceData.data(),pending);
    next->lights=stageBuffer(lights.size()*sizeof(GpuLight),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,lights.data(),pending);
    auto instances=stageBuffer(asInstances.size()*sizeof(VkAccelerationStructureInstanceKHR),usage,asInstances.data(),pending);
    uploadBuffers(pending);
    VkAccelerationStructureGeometryKHR geo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR}; geo.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geo.geometry.instances.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR; geo.geometry.instances.data.deviceAddress=address(instances);
    next->tlas=build(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,geo,static_cast<uint32_t>(asInstances.size()));
    scene=std::move(next); cameraMediumDirty=true; resetHistory(); descriptors();
  }
  size_t renderDiagnosticWords() const {
    return rendererOptions.enableRayStatistics?kRayProfileDiagnosticWords:kRenderDiagnosticWords;
  }
  void invalidateProfile() {
    lastProfile={};
    lastProfile.gpuTimingSupported=timestampValidBits>0 && properties.limits.timestampPeriod>0;
  }
  double gpuIntervalMs(uint64_t begin,uint64_t end) const {
    // Timestamp values have queue-family-specific precision and wrap modulo
    // that bit width. A measured submission must be shorter than one wrap.
    const uint64_t mask=timestampValidBits>=64?UINT64_MAX:(uint64_t{1}<<timestampValidBits)-1;
    return double((end-begin)&mask)*double(properties.limits.timestampPeriod)/1000000.0;
  }
  void collectRayStatistics(RayStatistics& rays) const {
    static constexpr std::pair<RayProfileCounter,uint64_t RayStatistics::*> sums[]{
      {RayProfileCounter::PathSamples,&RayStatistics::pathSamples},
      {RayProfileCounter::PrimaryRayQueries,&RayStatistics::primaryRayQueries},
      {RayProfileCounter::IndirectRayQueries,&RayStatistics::indirectRayQueries},
      {RayProfileCounter::ShadowRayQueries,&RayStatistics::shadowRayQueries},
      {RayProfileCounter::AuxiliaryRayQueries,&RayStatistics::auxiliaryRayQueries},
      {RayProfileCounter::BoundaryContinuationQueries,&RayStatistics::boundaryContinuationQueries},
      {RayProfileCounter::QueryHits,&RayStatistics::queryHits},
      {RayProfileCounter::QueryMisses,&RayStatistics::queryMisses},
      {RayProfileCounter::CandidateVisits,&RayStatistics::candidateVisits},
      {RayProfileCounter::CoverageEvaluations,&RayStatistics::coverageEvaluations},
      {RayProfileCounter::FullMaterialEvaluations,&RayStatistics::fullMaterialEvaluations},
      {RayProfileCounter::AlphaRejectedHits,&RayStatistics::alphaRejectedHits},
      {RayProfileCounter::ShadowTransparentHits,&RayStatistics::shadowTransparentHits},
      {RayProfileCounter::AcceptedSurfaceHits,&RayStatistics::acceptedSurfaceHits},
      {RayProfileCounter::SurfaceScatters,&RayStatistics::surfaceScatters},
      {RayProfileCounter::MediumScatters,&RayStatistics::mediumScatters},
      {RayProfileCounter::RussianRouletteTerminations,&RayStatistics::russianRouletteTerminations}
    };
    const auto* words=static_cast<const uint32_t*>(renderErrors.mapped);
    for(size_t shard=0;shard<kRayProfileShardCount;++shard) {
      const auto* counters=words+kRayProfileBaseWord+shard*kRayProfileShardWords;
      for(const auto& [counter,member]:sums) {
        const auto value=uint64_t(counters[static_cast<size_t>(counter)]);
        auto& total=rays.*member;
        if(value>UINT64_MAX-total) { rays.overflowed=true; total=UINT64_MAX; }
        else total+=value;
      }
      rays.maxScatteringDepth=std::max(rays.maxScatteringDepth,uint64_t(counters[static_cast<size_t>(RayProfileCounter::MaxScatteringDepth)]));
      rays.overflowed|=counters[kRayProfileOverflowOffset]!=0;
    }
  }
  void resetHistory() { samples=0; frames=0; rawReadbackCurrent=denoisedReadbackCurrent=false; invalidateProfile(); }
  void prepareReadback(const Buffer& source,Buffer& readback,bool& current) {
    if(current) return;
    if(!readback.handle || readback.size!=source.size)
      readback=buffer(source.size,VK_BUFFER_USAGE_TRANSFER_DST_BIT,BufferMemory::Readback);
    submit([&](VkCommandBuffer cmd) {
      barrier(cmd,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_ACCESS_HOST_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT);
      const VkBufferCopy copy{0,0,source.size};
      vkCmdCopyBuffer(cmd,source.handle,readback.handle,1,&copy);
      barrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
    });
    readback.invalidate();
    current=true;
  }
  PushConstants constants() const {
    const auto forward=cameraDirection(camera.target,camera.position);
    const auto right=normalize(cross(forward,normalize(camera.up))), up=cross(right,forward);
    float h=2*std::tan(camera.verticalFov*3.14159265358979323846f/360);
    PushConstants p{}; p.origin=camera.position; p.horizontal=mul(right,h*settings.width/settings.height); p.vertical=mul(up,h);
    p.lowerLeft=sub(sub(forward,mul(p.horizontal,0.5f)),mul(p.vertical,0.5f));
    p.environment=settings.environment; p.environment.w=scene->hasMedia?1.0f:0.0f;
    p.dimensions={renderSize.width,renderSize.height,samples,settings.maxBounces}; p.counts={scene->lightCount,dlss.enabled?1u:0u,static_cast<uint32_t>(frames),frames?1u:0u};
    uint32_t jitterIndex=static_cast<uint32_t>(frames%64)+1;
    p.params={scene->lightCount?float(kLightSelectionTickets):0,0.0001f,dlss.enabled?halton(jitterIndex,2)-0.5f:0,dlss.enabled?halton(jitterIndex,3)-0.5f:0}; return p;
  }
};

Renderer::Renderer(const RendererOptions& options):impl_(std::make_unique<Impl>()) { impl_->init(options); }
Renderer::Renderer(const RendererOptions& options,const VulkanDeviceOptions& vulkan):impl_(std::make_unique<Impl>()) {
  impl_->vulkanOptions=vulkan; impl_->init(options);
}
Renderer::~Renderer()=default;
void Renderer::setScene(const Scene& scene) { impl_->upload(scene); }
void Renderer::setCamera(const Camera& camera) {
  validateCamera(camera);
  if(!sameCameraOrigin(impl_->camera,camera)) impl_->cameraMediumDirty=true;
  impl_->camera=camera; reset();
}
void Renderer::updateCamera(const Camera& camera) {
  validateCamera(camera);
  if(sameCamera(impl_->camera,camera)) return;
  if(!sameCameraOrigin(impl_->camera,camera)) impl_->cameraMediumDirty=true;
  impl_->camera=camera;
  impl_->samples=0;
  impl_->rawReadbackCurrent=impl_->denoisedReadbackCurrent=false;
  impl_->invalidateProfile();
}
void Renderer::setSettings(const RenderSettings& settings) {
  validateSettings(settings); impl_->resizeOutput(settings,impl_->dlss);
}
void Renderer::setDlssSettings(const DlssSettings& settings) {
  validateDlssSettings(settings); impl_->resizeOutput(impl_->settings,settings);
}
DlssSettings Renderer::dlssSettings() const { return impl_->dlss; }
std::string Renderer::dlssStatus() const {
  if(impl_->dlss.enabled) return "DLSS Ray Reconstruction active: "+std::to_string(impl_->renderSize.width)+"x"+std::to_string(impl_->renderSize.height)+" -> "+std::to_string(impl_->settings.width)+"x"+std::to_string(impl_->settings.height);
  return impl_->dlssUnavailable.empty()?"DLSS Ray Reconstruction off (SDK compiled, runtime capability checked when enabled)":"DLSS Ray Reconstruction unavailable: "+impl_->dlssUnavailable;
}
void Renderer::reset() { impl_->resetHistory(); }
void Renderer::render(uint32_t count) {
  if(!impl_->scene) throw std::logic_error("Call setScene before render");
  if(count>UINT32_MAX-impl_->samples) throw std::overflow_error("Sample counter overflow");
  if(count>UINT64_MAX-impl_->frames) throw std::overflow_error("Frame counter overflow");
  if(!count) return;
  impl_->invalidateProfile();
  RendererProfile profile=impl_->lastProfile;
  profile.samples=count; profile.renderWidth=impl_->renderSize.width; profile.renderHeight=impl_->renderSize.height;
  for(uint32_t i=0;i<count;++i) {
    const auto frameStart=std::chrono::steady_clock::now();
    auto p=impl_->constants();
    if(impl_->rendererOptions.mediumErrorPolicy==MediumErrorPolicy::SkipInvalidPath) p.counts.w|=kSkipInvalidMediumPath;
    if(!impl_->rendererOptions.optimizeMaterialCoverage) p.counts.w|=kDisableMaterialCoverageOptimization;
    const auto current=makeCameraTransforms(impl_->camera,impl_->settings.width,impl_->settings.height);
    const auto previous=impl_->frames?makeCameraTransforms(impl_->previousCamera,impl_->settings.width,impl_->settings.height):current;
    const GpuCameraHistory history{current.worldToClip,previous.worldToClip,{impl_->rendererOptions.mediumBoundaryTolerance,0,0,0}};
    std::memcpy(impl_->cameraHistory.mapped,&history,sizeof(history));
    impl_->cameraHistory.flush();
    std::memset(impl_->renderErrors.mapped,0,impl_->renderDiagnosticWords()*sizeof(uint32_t));
    impl_->renderErrors.flush();
    impl_->rawReadbackCurrent=impl_->denoisedReadbackCurrent=false;
    const bool initializeMaterials=impl_->scene->materialMediumDirty;
    const bool initializeMedium=impl_->cameraMediumDirty && impl_->scene->hasMedia;
    impl_->submit([&](VkCommandBuffer cmd){
      if(impl_->timestampPool) {
        vkCmdResetQueryPool(cmd,impl_->timestampPool,0,4);
        vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,impl_->timestampPool,0);
      }
      Impl::barrier(cmd,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_UNIFORM_READ_BIT|VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
      vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,impl_->pipeline); vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,impl_->pipelineLayout,0,1,&impl_->descriptorSet,0,nullptr);
      if(initializeMaterials) {
        auto initialization=p; initialization.counts.w|=kInitMaterialMedium;
        initialization.dimensions.x=impl_->scene->materialInitX;
        vkCmdPushConstants(cmd,impl_->pipelineLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(initialization),&initialization);
        vkCmdDispatch(cmd,impl_->scene->materialInitX,impl_->scene->materialInitY,1);
        Impl::barrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        for(uint32_t instanceIndex:impl_->scene->heterogeneousInstances) {
          initialization=p;initialization.counts.w|=kInitHeterogeneousMedium;
          initialization.dimensions.x=instanceIndex;
          vkCmdPushConstants(cmd,impl_->pipelineLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(initialization),&initialization);
          vkCmdDispatch(cmd,kMediumGridCells/64,1,1);
          Impl::barrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        }
        Impl::barrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
      }
      if(initializeMedium) {
        auto initialization=p; initialization.counts.w|=kInitCameraMedium;
        vkCmdPushConstants(cmd,impl_->pipelineLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(initialization),&initialization);
        vkCmdDispatch(cmd,1,1,1);
        Impl::barrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
      }
      // Completion boundaries give additive elapsed regions without a TOP/BOTTOM
      // start for every pass (which could overlap earlier work). They are not
      // exclusive hardware-unit timings or pure traversal/ALU measurements.
      if(impl_->timestampPool) vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,impl_->timestampPool,1);
      vkCmdPushConstants(cmd,impl_->pipelineLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p); vkCmdDispatch(cmd,(p.dimensions.x+7)/8,(p.dimensions.y+7)/8,1);
      if(impl_->timestampPool) vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,impl_->timestampPool,2);
#if CAUSTRA_HAS_DLSS
      if(impl_->denoiser) {
        // Include readback, presentation and frame limiting between submissions:
        // NGX uses elapsed time with pixel motion to estimate temporal response.
        const float frameTime=impl_->frames?std::chrono::duration<float,std::milli>(frameStart-impl_->previousFrameStart).count():1000.0f/60.0f;
        impl_->denoiser->evaluate(cmd,impl_->guides.handle,impl_->denoisedOutput.handle,impl_->frames==0,p.params.z,p.params.w,current.view,current.projection,frameTime);
      }
#endif
      Impl::barrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
      if(impl_->timestampPool) vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,impl_->timestampPool,3);
    });
    impl_->renderErrors.invalidate();
    std::array<uint32_t,kRenderDiagnosticWords> diagnostic{};
    std::memcpy(diagnostic.data(),impl_->renderErrors.mapped,sizeof(diagnostic));
    impl_->diagnostics.invalidMediumBoundaries+=diagnostic[16];
    impl_->diagnostics.skippedNearMediumBoundaries+=diagnostic[17];
    const uint32_t errors=impl_->rendererOptions.mediumErrorPolicy==MediumErrorPolicy::SkipInvalidPath?
      diagnostic[0]&~2u:diagnostic[0];
    if(errors) {
      impl_->resetHistory();
      if(errors&1) throw std::runtime_error("Medium nesting exceeds the supported depth of 16");
      if(errors&2) {
        float point[4];std::memcpy(point,diagnostic.data()+4,sizeof(point));
        float origin[3],direction[3];std::memcpy(origin,diagnostic.data()+8,sizeof(origin));std::memcpy(direction,diagnostic.data()+11,sizeof(direction));
        std::ostringstream message;
        message<<std::setprecision(std::numeric_limits<float>::max_digits10)
          <<"Invalid medium boundary nesting: site="<<diagnostic[1]<<", instance="<<diagnostic[2]
          <<", depth="<<((diagnostic[3]>>24)&31)<<", top="<<(diagnostic[3]&0x00ffffffu)
          <<", entering="<<(diagnostic[3]>>31)<<", distance="<<point[0]
          <<", point=("<<point[1]<<','<<point[2]<<','<<point[3]<<')'
          <<", origin=("<<origin[0]<<','<<origin[1]<<','<<origin[2]<<')'
          <<", direction=("<<direction[0]<<','<<direction[1]<<','<<direction[2]<<')'
          <<", sample="<<diagnostic[14]<<", source="<<diagnostic[15]
          <<"; use closed, consistently oriented nested or disjoint volumes";
        throw std::runtime_error(message.str());
      }
      throw std::runtime_error("Invalid medium transport state (GPU diagnostic flags "+std::to_string(errors)+")");
    }
    if(impl_->timestampPool) {
      std::array<uint64_t,4> timestamps{};
      check(vkGetQueryPoolResults(impl_->device,impl_->timestampPool,0,4,sizeof(timestamps),timestamps.data(),sizeof(uint64_t),
                                 VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT),"Read render timestamps");
      const bool initialized=initializeMaterials || initializeMedium;
      if(initialized) profile.initializationGpuMs+=impl_->gpuIntervalMs(timestamps[0],timestamps[1]);
      // Absent regions have zero cost; their boundary overhead belongs to the
      // adjacent path-trace region, keeping total equal to the region sum.
      profile.pathTraceGpuMs+=impl_->gpuIntervalMs(timestamps[initialized?1:0],timestamps[impl_->dlss.enabled?2:3]);
      if(impl_->dlss.enabled) profile.dlssGpuMs+=impl_->gpuIntervalMs(timestamps[2],timestamps[3]);
      profile.totalGpuMs+=impl_->gpuIntervalMs(timestamps[0],timestamps[3]);
      profile.gpuTimingValid=true;
    }
    if(impl_->rendererOptions.enableRayStatistics) {
      impl_->collectRayStatistics(profile.rays);
      profile.rayStatisticsValid=true;
    }
    if(initializeMaterials) impl_->scene->materialMediumDirty=false;
    if(initializeMedium) impl_->cameraMediumDirty=false;
    ++impl_->samples;
    ++impl_->frames;
    impl_->previousCamera=impl_->camera;
    impl_->previousFrameStart=frameStart;
  }
  impl_->lastProfile=profile;
}
Image Renderer::readback() {
  if(!impl_->dlss.enabled) return readbackRaw();
  if(!impl_->samples) throw std::logic_error("Render at least one sample before readback");
  Image image; image.width=impl_->settings.width; image.height=impl_->settings.height; image.samples=impl_->samples;
  impl_->prepareReadback(impl_->denoisedOutput,impl_->denoisedReadback,impl_->denoisedReadbackCurrent);
  image.pixels.resize(size_t(image.width)*image.height); std::memcpy(image.pixels.data(),impl_->denoisedReadback.mapped,image.pixels.size()*sizeof(Float4)); return image;
}
Image Renderer::readbackRaw() {
  if(!impl_->samples) throw std::logic_error("Render at least one sample before readback");
  Image image; image.width=impl_->renderSize.width; image.height=impl_->renderSize.height; image.samples=impl_->samples;
  impl_->prepareReadback(impl_->output,impl_->rawReadback,impl_->rawReadbackCurrent);
  image.pixels.resize(size_t(image.width)*image.height); std::memcpy(image.pixels.data(),impl_->rawReadback.mapped,image.pixels.size()*sizeof(Float4)); return image;
}
uint32_t Renderer::sampleCount() const { return impl_->samples; }
uint64_t Renderer::frameCount() const { return impl_->frames; }
std::string Renderer::deviceName() const { return impl_->properties.deviceName; }
uint32_t Renderer::validationErrorCount() const { return impl_->validationErrors.load(); }
RendererDiagnostics Renderer::diagnostics() const { return impl_->diagnostics; }
RendererProfile Renderer::profile() const { return impl_->lastProfile; }
VulkanContext VulkanAccess::context(Renderer& renderer) {
  const auto& impl=*renderer.impl_;
  return {impl.instance,impl.physical,impl.device,impl.queue,impl.queueFamily};
}
VulkanFrame VulkanAccess::output(Renderer& renderer) {
  const auto& impl=*renderer.impl_;
  if(!impl.samples) throw std::logic_error("Render at least one sample before accessing GPU output");
  return {impl.dlss.enabled?impl.denoisedOutput.handle:impl.output.handle,
          impl.dlss.enabled?impl.settings.width:impl.renderSize.width,
          impl.dlss.enabled?impl.settings.height:impl.renderSize.height};
}
} // namespace caustra
