// Vulkan Ray Query backend adapted from the vk_gltf_renderer rendering architecture.
// Scene import and application/editor state remain outside this library.
#include <caustra/renderer.hpp>
#include "shader_types.hpp"
#include "pathtrace_spv.hpp"
#include <vulkan/vulkan.h>
#if CAUSTRA_HAS_DLSS
#include "dlss_backend.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
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
float halton(uint32_t index,uint32_t base) {
  float value=0,weight=1;
  while(index) { weight/=base; value+=weight*(index%base); index/=base; }
  return value;
}
Float4 transform(const Instance& i, Float4 p) {
  const auto& m=i.transform;
  return {m[0]*p.x+m[1]*p.y+m[2]*p.z+m[3],m[4]*p.x+m[5]*p.y+m[6]*p.z+m[7],m[8]*p.x+m[9]*p.y+m[10]*p.z+m[11],0};
}

struct Buffer {
  VkDevice device{}; VkBuffer handle{}; VkDeviceMemory memory{}; void* mapped{}; VkDeviceSize size{};
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& b) noexcept { *this=std::move(b); }
  Buffer& operator=(Buffer&& b) noexcept {
    if(this != &b) { clear(); device=b.device; handle=std::exchange(b.handle,{}); memory=std::exchange(b.memory,{}); mapped=std::exchange(b.mapped,nullptr); size=b.size; }
    return *this;
  }
  void clear() {
    if(mapped) vkUnmapMemory(device,memory);
    if(handle) vkDestroyBuffer(device,handle,nullptr);
    if(memory) vkFreeMemory(device,memory,nullptr);
    mapped=nullptr; handle={}; memory={};
  }
  ~Buffer() { clear(); }
};
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
  std::vector<Acceleration> blas;
  Acceleration tlas;
  uint32_t lightCount=0; float totalArea=0;
};
}

struct Renderer::Impl {
  VkInstance instance{}; VkDebugUtilsMessengerEXT messenger{};
  VkPhysicalDevice physical{}; VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceMemoryProperties memoryProperties{};
  VkDevice device{}; VkQueue queue{}; uint32_t queueFamily=0;
  VkCommandPool commandPool{}; VkFence fence{};
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
  uint64_t frames=0;
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
  Buffer output, guides, denoisedOutput, cameraHistory;

  ~Impl() {
    if(device) vkDeviceWaitIdle(device);
#if CAUSTRA_HAS_DLSS
    denoiser.reset();
    dlssSession.reset();
#endif
    scene.reset(); output.clear(); guides.clear(); denoisedOutput.clear(); cameraHistory.clear();
    if(pipeline) vkDestroyPipeline(device,pipeline,nullptr);
    if(pipelineLayout) vkDestroyPipelineLayout(device,pipelineLayout,nullptr);
    if(descriptorPool) vkDestroyDescriptorPool(device,descriptorPool,nullptr);
    if(descriptorLayout) vkDestroyDescriptorSetLayout(device,descriptorLayout,nullptr);
    if(fence) vkDestroyFence(device,fence,nullptr);
    if(commandPool) vkDestroyCommandPool(device,commandPool,nullptr);
    if(device) vkDestroyDevice(device,nullptr);
    if(messenger) {
      auto fn=reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance,"vkDestroyDebugUtilsMessengerEXT"));
      if(fn) fn(instance,messenger,nullptr);
    }
    if(instance) vkDestroyInstance(instance,nullptr);
  }
  static VKAPI_ATTR VkBool32 VKAPI_CALL debug(VkDebugUtilsMessageSeverityFlagBitsEXT severity,VkDebugUtilsMessageTypeFlagsEXT,const VkDebugUtilsMessengerCallbackDataEXT* data,void* user) {
    if(severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ++static_cast<Impl*>(user)->validationErrors;
    std::cerr << "[Vulkan validation] " << data->pMessage << '\n'; return VK_FALSE;
  }
  void init(const RendererOptions& options) {
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
      VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
      VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR}; as.pNext=&rq;
      VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext=&as;
      VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f.pNext=&f12; vkGetPhysicalDeviceFeatures2(candidate,&f);
      if(!supported || !rq.rayQuery || !as.accelerationStructure || !f12.bufferDeviceAddress) continue;
      vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,nullptr); std::vector<VkQueueFamilyProperties> families(n); vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,families.data());
      for(uint32_t i=0;i<n;++i) if(families[i].queueFlags&VK_QUEUE_COMPUTE_BIT) {
        int score=(props.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU?10:1)+(props.vendorID==0x10de?4:0)+(families[i].queueFlags&VK_QUEUE_GRAPHICS_BIT?1:0);
        if(score>best) { best=score; physical=candidate; properties=props; queueFamily=i; }
      }
    }
    if(!physical) throw std::runtime_error("No Vulkan 1.2 GPU with rayQuery, accelerationStructure and bufferDeviceAddress support was found.");
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
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex=queueFamily; pci.flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    check(vkCreateCommandPool(device,&pci,nullptr,&commandPool),"Create command pool");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; check(vkCreateFence(device,&fci,nullptr,&fence),"Create fence");
    createPipeline();
    cameraHistory=buffer(sizeof(GpuCameraHistory),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    resizeOutput(settings,dlss);
  }
  Buffer buffer(VkDeviceSize bytes,VkBufferUsageFlags usage,bool host=true,const void* data=nullptr) {
    Buffer b; b.device=device; b.size=std::max<VkDeviceSize>(bytes,16);
    if((usage&VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && b.size>properties.limits.maxStorageBufferRange)
      throw std::invalid_argument("Buffer exceeds this GPU's maxStorageBufferRange");
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size=b.size; ci.usage=usage; ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device,&ci,nullptr,&b.handle),"Create buffer");
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device,b.handle,&req);
    VkMemoryPropertyFlags flags=host?(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT):VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t type=UINT32_MAX;
    for(uint32_t i=0;i<memoryProperties.memoryTypeCount;++i) if((req.memoryTypeBits&(1u<<i)) && (memoryProperties.memoryTypes[i].propertyFlags&flags)==flags) {type=i; break;}
    if(type==UINT32_MAX) throw std::runtime_error("No suitable Vulkan memory type");
    VkMemoryAllocateFlagsInfo addressFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO}; addressFlags.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=req.size; ai.memoryTypeIndex=type;
    if(usage&VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ai.pNext=&addressFlags;
    check(vkAllocateMemory(device,&ai,nullptr,&b.memory),"Allocate buffer memory"); check(vkBindBufferMemory(device,b.handle,b.memory,0),"Bind buffer memory");
    if(host) { check(vkMapMemory(device,b.memory,0,VK_WHOLE_SIZE,0,&b.mapped),"Map buffer"); std::memset(b.mapped,0,static_cast<size_t>(b.size)); if(data && bytes) std::memcpy(b.mapped,data,static_cast<size_t>(bytes)); }
    return b;
  }
  VkDeviceAddress address(const Buffer& b) { VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; info.buffer=b.handle; return vkGetBufferDeviceAddress(device,&info); }
  VkDeviceAddress address(const Acceleration& a) { VkAccelerationStructureDeviceAddressInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR}; info.accelerationStructure=a.handle; return getASAddress(device,&info); }
  void submit(const std::function<void(VkCommandBuffer)>& record) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=commandPool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
    VkCommandBuffer cmd{}; check(vkAllocateCommandBuffers(device,&ai,&cmd),"Allocate command buffer");
    try {
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      check(vkBeginCommandBuffer(cmd,&bi),"Begin command buffer"); record(cmd); check(vkEndCommandBuffer(cmd),"End command buffer");
      check(vkResetFences(device,1,&fence),"Reset fence");
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
      check(vkQueueSubmit(queue,1,&si,fence),"Submit GPU work");
      check(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX),"Wait for GPU");
    } catch(...) { vkDeviceWaitIdle(device); vkFreeCommandBuffers(device,commandPool,1,&cmd); throw; }
    vkFreeCommandBuffers(device,commandPool,1,&cmd);
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
    result.storage=buffer(sizes.accelerationStructureSize,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,false);
    VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR}; ci.type=type; ci.buffer=result.storage.handle; ci.size=sizes.accelerationStructureSize;
    check(createAS(device,&ci,nullptr,&result.handle),"Create acceleration structure");
    Buffer scratch=buffer(sizes.buildScratchSize+scratchAlignment,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,false);
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
    std::array<VkDescriptorSetLayoutBinding,9> bindings{};
    for(uint32_t i=0;i<bindings.size();++i) { bindings[i].binding=i; bindings[i].descriptorCount=1; bindings[i].descriptorType=i==0?VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:(i==8?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; li.bindingCount=static_cast<uint32_t>(bindings.size()); li.pBindings=bindings.data();
    check(vkCreateDescriptorSetLayout(device,&li,nullptr,&descriptorLayout),"Create descriptor layout");
    VkDescriptorPoolSize sizes[]={{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,7},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pi.maxSets=1; pi.poolSizeCount=3; pi.pPoolSizes=sizes;
    check(vkCreateDescriptorPool(device,&pi,nullptr,&descriptorPool),"Create descriptor pool");
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool=descriptorPool; ai.descriptorSetCount=1; ai.pSetLayouts=&descriptorLayout;
    check(vkAllocateDescriptorSets(device,&ai,&descriptorSet),"Allocate descriptor set");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pci.setLayoutCount=1; pci.pSetLayouts=&descriptorLayout; pci.pushConstantRangeCount=1; pci.pPushConstantRanges=&range;
    check(vkCreatePipelineLayout(device,&pci,nullptr,&pipelineLayout),"Create pipeline layout");
    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sci.codeSize=sizeof(pathtraceSpv); sci.pCode=pathtraceSpv;
    VkShaderModule shader{}; check(vkCreateShaderModule(device,&sci,nullptr,&shader),"Create shader module");
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; ci.layout=pipelineLayout; ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module=shader; ci.stage.pName="main";
    auto result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&ci,nullptr,&pipeline); vkDestroyShaderModule(device,shader,nullptr); check(result,"Create path-trace pipeline");
  }
  void descriptors() {
    if(!scene) return;
    VkWriteDescriptorSetAccelerationStructureKHR as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR}; as.accelerationStructureCount=1; as.pAccelerationStructures=&scene->tlas.handle;
    std::array<VkWriteDescriptorSet,9> writes{}; std::array<VkDescriptorBufferInfo,8> infos{};
    Buffer* buffers[]={&scene->vertices,&scene->indices,&scene->instances,&scene->materials,&scene->lights,&output,&guides,&cameraHistory};
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
    auto next=buffer(bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto nextGuides=buffer(desiredDlss.enabled?VkDeviceSize(extent.width)*extent.height*80:16,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,false);
    Buffer nextDenoised;
#if CAUSTRA_HAS_DLSS
    if(candidate) {
      nextDenoised=buffer(VkDeviceSize(desired.width)*desired.height*sizeof(Float4),VK_BUFFER_USAGE_TRANSFER_DST_BIT);
      submit([&](VkCommandBuffer cmd){candidate->configure(cmd,extent,{desired.width,desired.height},desiredDlss.quality,desiredDlss.preset);});
    }
    denoiser=std::move(candidate);
#endif
    output=std::move(next); guides=std::move(nextGuides); denoisedOutput=std::move(nextDenoised);
    renderSize=extent; settings=desired; dlss=desiredDlss; resetHistory(); descriptors();
  }
  void upload(const Scene& input) {
    validateScene(input);
    auto next=std::make_unique<SceneResources>();
    std::vector<Vertex> vertices; std::vector<uint32_t> indices;
    std::vector<Uint4> meshOffsets;
    for(const auto& mesh:input.meshes) {
      if(vertices.size()+mesh.vertices.size()>UINT32_MAX || indices.size()+mesh.indices.size()>UINT32_MAX) throw std::invalid_argument("Scene exceeds 32-bit geometry indexing");
      meshOffsets.push_back({static_cast<uint32_t>(vertices.size()),static_cast<uint32_t>(indices.size()),mesh.material,0});
      vertices.insert(vertices.end(),mesh.vertices.begin(),mesh.vertices.end()); indices.insert(indices.end(),mesh.indices.begin(),mesh.indices.end());
    }
    auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    next->vertices=buffer(vertices.size()*sizeof(Vertex),usage,true,vertices.data());
    next->indices=buffer(indices.size()*sizeof(uint32_t),usage,true,indices.data());
    next->materials=buffer(input.materials.size()*sizeof(Material),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,true,input.materials.data());
    for(size_t m=0;m<input.meshes.size();++m) {
      const auto& mesh=input.meshes[m];
      VkAccelerationStructureGeometryKHR geo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR}; geo.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR; geo.flags=VK_GEOMETRY_OPAQUE_BIT_KHR;
      auto& tri=geo.geometry.triangles; tri.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
      tri.vertexFormat=VK_FORMAT_R32G32B32_SFLOAT; tri.vertexData.deviceAddress=address(next->vertices)+VkDeviceSize(meshOffsets[m].x)*sizeof(Vertex); tri.vertexStride=sizeof(Vertex); tri.maxVertex=static_cast<uint32_t>(mesh.vertices.size()-1);
      tri.indexType=VK_INDEX_TYPE_UINT32; tri.indexData.deviceAddress=address(next->indices)+VkDeviceSize(meshOffsets[m].y)*sizeof(uint32_t);
      next->blas.push_back(build(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,geo,static_cast<uint32_t>(mesh.indices.size()/3)));
    }
    std::vector<GpuInstance> instanceData; std::vector<VkAccelerationStructureInstanceKHR> asInstances; std::vector<GpuLight> lights;
    if(input.instances.size()>0x1000000) throw std::invalid_argument("Too many instances for Vulkan's 24-bit custom index");
    for(size_t n=0;n<input.instances.size();++n) {
      const auto& i=input.instances[n]; const auto& mesh=input.meshes[i.mesh]; const auto& mat=input.materials[mesh.material];
      instanceData.push_back({meshOffsets[i.mesh]});
      VkAccelerationStructureInstanceKHR vi{}; std::memcpy(vi.transform.matrix,i.transform.data(),sizeof(vi.transform.matrix));
      vi.instanceCustomIndex=static_cast<uint32_t>(n); vi.mask=0xFF; vi.flags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR; vi.accelerationStructureReference=address(next->blas[i.mesh]); asInstances.push_back(vi);
      if(std::max({mat.emission.x,mat.emission.y,mat.emission.z})>0) for(size_t t=0;t<mesh.indices.size();t+=3) {
        auto a=transform(i,mesh.vertices[mesh.indices[t]].position), b=transform(i,mesh.vertices[mesh.indices[t+1]].position), c=transform(i,mesh.vertices[mesh.indices[t+2]].position);
        auto e1=sub(b,a),e2=sub(c,a); float area=length(cross(e1,e2))*0.5f;
        next->totalArea+=area;
        if(!std::isfinite(next->totalArea) || !(area>0)) throw std::invalid_argument("Transformed light area is invalid");
        a.w=area; e1.w=next->totalArea; lights.push_back({a,e1,e2,mat.emission});
      }
    }
    next->lightCount=static_cast<uint32_t>(lights.size());
    next->instances=buffer(instanceData.size()*sizeof(GpuInstance),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,true,instanceData.data());
    next->lights=buffer(lights.size()*sizeof(GpuLight),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,true,lights.data());
    auto instances=buffer(asInstances.size()*sizeof(VkAccelerationStructureInstanceKHR),usage,true,asInstances.data());
    VkAccelerationStructureGeometryKHR geo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR}; geo.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geo.geometry.instances.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR; geo.geometry.instances.data.deviceAddress=address(instances);
    next->tlas=build(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,geo,static_cast<uint32_t>(asInstances.size()));
    scene=std::move(next); resetHistory(); descriptors();
  }
  void resetHistory() { samples=0; frames=0; }
  PushConstants constants() const {
    const auto forward=cameraDirection(camera.target,camera.position);
    const auto right=normalize(cross(forward,normalize(camera.up))), up=cross(right,forward);
    float h=2*std::tan(camera.verticalFov*3.14159265358979323846f/360);
    PushConstants p{}; p.origin=camera.position; p.horizontal=mul(right,h*settings.width/settings.height); p.vertical=mul(up,h);
    p.lowerLeft=sub(sub(forward,mul(p.horizontal,0.5f)),mul(p.vertical,0.5f));
    p.environment=settings.environment; p.dimensions={renderSize.width,renderSize.height,samples,settings.maxBounces}; p.counts={scene->lightCount,dlss.enabled?1u:0u,static_cast<uint32_t>(frames),frames?1u:0u};
    uint32_t jitterIndex=static_cast<uint32_t>(frames%64)+1;
    p.params={scene->totalArea,0.0001f,dlss.enabled?halton(jitterIndex,2)-0.5f:0,dlss.enabled?halton(jitterIndex,3)-0.5f:0}; return p;
  }
};

Renderer::Renderer(const RendererOptions& options):impl_(std::make_unique<Impl>()) { impl_->init(options); }
Renderer::~Renderer()=default;
void Renderer::setScene(const Scene& scene) { impl_->upload(scene); }
void Renderer::setCamera(const Camera& camera) { validateCamera(camera); impl_->camera=camera; reset(); }
void Renderer::updateCamera(const Camera& camera) {
  validateCamera(camera);
  if(sameCamera(impl_->camera,camera)) return;
  impl_->camera=camera;
  impl_->samples=0;
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
  for(uint32_t i=0;i<count;++i) {
    auto p=impl_->constants();
    const auto current=makeCameraTransforms(impl_->camera,impl_->settings.width,impl_->settings.height);
    const auto previous=impl_->frames?makeCameraTransforms(impl_->previousCamera,impl_->settings.width,impl_->settings.height):current;
    const GpuCameraHistory history{current.worldToClip,previous.worldToClip};
    std::memcpy(impl_->cameraHistory.mapped,&history,sizeof(history));
    impl_->submit([&](VkCommandBuffer cmd){
      Impl::barrier(cmd,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_UNIFORM_READ_BIT|VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
      vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,impl_->pipeline); vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,impl_->pipelineLayout,0,1,&impl_->descriptorSet,0,nullptr);
      vkCmdPushConstants(cmd,impl_->pipelineLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p); vkCmdDispatch(cmd,(p.dimensions.x+7)/8,(p.dimensions.y+7)/8,1);
#if CAUSTRA_HAS_DLSS
      if(impl_->denoiser) impl_->denoiser->evaluate(cmd,impl_->guides.handle,impl_->denoisedOutput.handle,impl_->frames==0,p.params.z,p.params.w,current.view,current.projection);
#endif
      Impl::barrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);
    });
    ++impl_->samples;
    ++impl_->frames;
    impl_->previousCamera=impl_->camera;
  }
}
Image Renderer::readback() {
  if(!impl_->dlss.enabled) return readbackRaw();
  if(!impl_->samples) throw std::logic_error("Render at least one sample before readback");
  Image image; image.width=impl_->settings.width; image.height=impl_->settings.height; image.samples=impl_->samples;
  image.pixels.resize(size_t(image.width)*image.height); std::memcpy(image.pixels.data(),impl_->denoisedOutput.mapped,image.pixels.size()*sizeof(Float4)); return image;
}
Image Renderer::readbackRaw() {
  if(!impl_->samples) throw std::logic_error("Render at least one sample before readback");
  Image image; image.width=impl_->renderSize.width; image.height=impl_->renderSize.height; image.samples=impl_->samples;
  image.pixels.resize(size_t(image.width)*image.height); std::memcpy(image.pixels.data(),impl_->output.mapped,image.pixels.size()*sizeof(Float4)); return image;
}
uint32_t Renderer::sampleCount() const { return impl_->samples; }
uint64_t Renderer::frameCount() const { return impl_->frames; }
std::string Renderer::deviceName() const { return impl_->properties.deviceName; }
uint32_t Renderer::validationErrorCount() const { return impl_->validationErrors.load(); }
} // namespace caustra
