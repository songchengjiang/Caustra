// SPDX-License-Identifier: Apache-2.0
// GPU integration test: raw ShaderGen inputs are evaluated independently of the
// production closure builder. This establishes routing/input fidelity, not an
// independent proof of the official scattering formulas themselves.
#include <caustra/renderer.hpp>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using caustra::Float4;
using caustra::Material;
constexpr std::array<const char*,6> modelNames{
  "StandardSurface", "OpenPBR", "DisneyPrincipled", "GltfPbr", "UsdPreviewSurface", "SimpleHair"};
struct alignas(16) ProbeRecord {
  Material material;
  Float4 normal, coatNormal, tangent, coatTangent, geometricNormal, view, light, random;
  Float4 runtimeValue, referenceValue, runtimeEmission, referenceEmission;
  Float4 sampleValue, sampleReference, sampleState;
};
static_assert(sizeof(Material)==384 && sizeof(ProbeRecord)==624);
static_assert(offsetof(ProbeRecord,normal)==384 && offsetof(ProbeRecord,runtimeValue)==512);
struct Case {
  ProbeRecord data{};
  std::string name;
  bool analyticEmission=false, analyticBurley=false;
};

void check(VkResult result,const char* operation) {
  if(result!=VK_SUCCESS) throw std::runtime_error(std::string(operation)+" failed: "+std::to_string(result));
}
void require(bool condition,const std::string& message) { if(!condition) throw std::runtime_error(message); }
Float4 normalized(float x,float y,float z) {
  const float length=std::sqrt(x*x+y*y+z*z); return {x/length,y/length,z/length,0};
}
float dot(Float4 a,Float4 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
Float4 emissionReference(const ProbeRecord& p) {
  const auto& m=p.material;
  const double root=(m.coat.z-1.0)/(m.coat.z+1.0);
  // Both official nodegraphs use closureData.N (the base geometry normal),
  // not the separately authored clearcoat normal, for generalized Schlick EDF.
  const double cosine=std::clamp(double(std::abs(dot(p.normal,p.view))),1.0e-8,1.0);
  const double transmittance=(1-root*root)*(1-std::pow(1-cosine,5));
  return {float(m.emission.x*((1-m.coat.x)+m.coat.x*m.coatColor.x*transmittance)),
          float(m.emission.y*((1-m.coat.x)+m.coat.x*m.coatColor.y*transmittance)),
          float(m.emission.z*((1-m.coat.x)+m.coat.x*m.coatColor.z*transmittance)),0};
}
Material material(uint32_t model,uint32_t variant) {
  Material m; m.program.z=model;
  m.baseColor={.63f,.27f,.11f,1}; m.emission={.7f,.3f,.15f,0};
  m.surface.x=(variant%4)*.3f; m.surface.y=std::array{.07f,.23f,.57f,.91f}[variant%4];
  m.specular={.65f,1.45f,(variant%3)*.35f,.173f};
  m.specularColor={.88f,.67f,.43f,0};
  m.coat={variant%2 ? .63f : 0.0f,.31f,1.62f,.24f}; m.coatColor={.81f,.54f,.29f,0};
  m.sheen={.21f,.43f,0,0};m.sheenColor={.37f,.19f,.08f,0};
  m.factors={.83f,.27f,1,.6f};m.fuzz={model==1 ? .18f : 0,.4f,.219f,.5f};
  m.fuzzColor={.74f,.55f,.33f,0};
  // These cases isolate surface reflection: no internal medium or BTDF model
  // adaptation is compared to the official real-time hardware approximation.
  m.transmission={};m.subsurface.x=0;m.subsurface.w=0;m.thinFilm={1,0,1.5f,0};
  if(model==2) {m.coat.y=.7f; m.fuzz.w=.4f;}
  if(model==3) {m.fuzz.z=0;m.specular.w=.37f;m.volume.w=.5f;}
  if(model==4) {m.specular.w=0;m.fuzz.w=1;m.factors.w=variant%2 ? 1.0f : 0.0f;}
  if(model==5) {
    m.surface.y=.25f+.12f*(variant%4);m.specular={1,1.55f,.35f,0};
    m.coat.y=.52f;m.factors={.25f,.4f,1,variant%2 ? 1.0f : 0.0f};
    m.specularColor={.8f,.6f,.4f,0};m.transmissionColor={.7f,.85f,.9f,0};
    m.sheenColor={.9f,.65f,.35f,0};m.emission={};
  }
  return m;
}
std::vector<Case> cases() {
  std::vector<Case> result;
  for(uint32_t model=0;model<6;++model) {
    for(uint32_t variant=0;variant<8;++variant) for(uint32_t direction=0;direction<4;++direction) {
      Case item;auto& p=item.data;p.material=material(model,variant);
      p.geometricNormal=p.normal={0,0,1,0};p.tangent={1,0,0,0};
      p.coatNormal=normalized(.25f,-.16f,1);p.coatTangent=normalized(.9f,.3f,-.177f);
      p.view=normalized(.07f+.19f*direction,.11f,.99f-.21f*direction);
      p.light=normalized(-.18f+.14f*variant,.23f-.1f*direction,.8f);
      if(direction==1) {
        p.normal=normalized(.45f,-.18f,1);p.tangent=normalized(.9f,.4f,-.333f);
      } else if(direction==2) {
        // Raw normal and both directions share a hemisphere, but raw N is in
        // the opposite hemisphere of Ng. Old closure reorientation changes it.
        p.geometricNormal=normalized(0,.8660254f,-.5f);
        p.view=normalized(.1f,.8660254f,.5f);p.light=normalized(-.2f,.8f,.6f);
        p.coatNormal=normalized(.2f,0,1);
      } else if(direction==3) {
        p.view=normalized(.83f,.1f,.2f);p.light=normalized(.68f,-.22f,.3f);
        p.normal=normalized(.12f,.16f,1);
      }
      p.random={.13f+.1f*(variant%5),.19f+.13f*(direction%4),.08f+.13f*(variant%7),0};
      item.name=std::string(modelNames[model])+" variant="+std::to_string(variant)+" direction="+std::to_string(direction);
      item.analyticEmission=model<2;
      result.push_back(item);
    }
  }
  // An independent directional reference for the official Disney Burley BRDF.
  // dot(V,L)=.06; (L.H)^2=(1+dot(V,L))/2=.53, hence F90=1.189.
  // At these grazing angles, a Lambert replacement differs substantially.
  Case burley;auto& p=burley.data;p.material=material(2,0);
  p.material.baseColor={.2f,.4f,.6f,1};p.material.surface={0,.65f,0,0};
  p.material.specular.x=0;p.material.coat.x=0;p.material.sheen.x=0;
  p.material.transmission.x=0;p.material.subsurface.x=0;
  p.normal=p.coatNormal=p.geometricNormal={0,0,1,0};
  p.tangent=p.coatTangent={1,0,0,0};
  p.view={std::sqrt(.96f),0,.2f,0};p.light={0,std::sqrt(.91f),.3f,0};
  p.random={.31f,.57f,.1f,0};
  burley.name="DisneyPrincipled independent grazing Burley reference";
  burley.analyticBurley=true;result.push_back(burley);
  return result;
}

struct Gpu {
  VkInstance instance{}; VkDebugUtilsMessengerEXT messenger{};VkPhysicalDevice physical{};
  VkDevice device{};VkQueue queue{};uint32_t queueFamily=0;VkPhysicalDeviceMemoryProperties memoryProperties{};
  VkBuffer buffer{};VkDeviceMemory memory{};void* mapped=nullptr;bool coherent=false;
  VkDescriptorSetLayout descriptorLayout{};VkDescriptorPool descriptorPool{};VkDescriptorSet descriptorSet{};
  VkPipelineLayout pipelineLayout{};VkPipeline pipeline{};VkShaderModule module{};
  VkCommandPool commandPool{};VkCommandBuffer command{};VkFence fence{};
  std::atomic<uint32_t> validationErrors{0};
  ~Gpu() {
    if(device) vkDeviceWaitIdle(device);
    if(fence)vkDestroyFence(device,fence,nullptr);
    if(commandPool)vkDestroyCommandPool(device,commandPool,nullptr);
    if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);
    if(module)vkDestroyShaderModule(device,module,nullptr);
    if(pipelineLayout)vkDestroyPipelineLayout(device,pipelineLayout,nullptr);
    if(descriptorPool)vkDestroyDescriptorPool(device,descriptorPool,nullptr);
    if(descriptorLayout)vkDestroyDescriptorSetLayout(device,descriptorLayout,nullptr);
    if(mapped)vkUnmapMemory(device,memory);
    if(buffer)vkDestroyBuffer(device,buffer,nullptr);
    if(memory)vkFreeMemory(device,memory,nullptr);
    if(device)vkDestroyDevice(device,nullptr);
    if(messenger) {
      auto destroy=reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance,"vkDestroyDebugUtilsMessengerEXT"));
      if(destroy)destroy(instance,messenger,nullptr);
    }
    if(instance)vkDestroyInstance(instance,nullptr);
  }
  static VKAPI_ATTR VkBool32 VKAPI_CALL debug(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,const VkDebugUtilsMessengerCallbackDataEXT* data,void* user) {
    if(severity&VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)++static_cast<Gpu*>(user)->validationErrors;
    std::cerr<<"[generated material validation] "<<data->pMessage<<'\n';return VK_FALSE;
  }
  void init(bool validation,VkDeviceSize bytes,const std::vector<uint32_t>& spirv) {
    const char* layer="VK_LAYER_KHRONOS_validation";const char* extension=VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    if(validation) {
      uint32_t count=0;check(vkEnumerateInstanceLayerProperties(&count,nullptr),"Enumerate layers");
      std::vector<VkLayerProperties> layers(count);check(vkEnumerateInstanceLayerProperties(&count,layers.data()),"Enumerate layers");
      require(std::any_of(layers.begin(),layers.end(),[&](auto& p){return std::strcmp(p.layerName,layer)==0;}),"Vulkan validation layer unavailable");
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="Caustra generated material probe";app.apiVersion=VK_API_VERSION_1_2;
    VkDebugUtilsMessengerCreateInfoEXT debugInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    debugInfo.messageSeverity=VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugInfo.messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugInfo.pfnUserCallback=debug;debugInfo.pUserData=this;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ici.pApplicationInfo=&app;
    if(validation){ici.enabledLayerCount=1;ici.ppEnabledLayerNames=&layer;ici.enabledExtensionCount=1;ici.ppEnabledExtensionNames=&extension;ici.pNext=&debugInfo;}
    check(vkCreateInstance(&ici,nullptr,&instance),"Create instance");
    if(validation) {
      auto create=reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance,"vkCreateDebugUtilsMessengerEXT"));
      require(create!=nullptr,"Debug utils entry point missing");check(create(instance,&debugInfo,nullptr,&messenger),"Create validation messenger");
    }
    uint32_t count=0;check(vkEnumeratePhysicalDevices(instance,&count,nullptr),"Enumerate physical devices");
    std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(instance,&count,devices.data()),"Enumerate physical devices");
    int best=-1;
    for(auto candidate:devices) {
      VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(candidate,&properties);
      if(properties.apiVersion<VK_API_VERSION_1_2)continue;
      uint32_t queues=0;vkGetPhysicalDeviceQueueFamilyProperties(candidate,&queues,nullptr);
      std::vector<VkQueueFamilyProperties> families(queues);vkGetPhysicalDeviceQueueFamilyProperties(candidate,&queues,families.data());
      for(uint32_t i=0;i<queues;++i)if(families[i].queueFlags&VK_QUEUE_COMPUTE_BIT) {
        int score=properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU?2:1;
        if(score>best){best=score;physical=candidate;queueFamily=i;}
        break;
      }
    }
    require(physical!=VK_NULL_HANDLE,"No Vulkan 1.2 compute device available");
    vkGetPhysicalDeviceMemoryProperties(physical,&memoryProperties);
    VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(physical,&properties);std::cout<<"Device: "<<properties.deviceName<<'\n';
    VkPhysicalDeviceFeatures features{};vkGetPhysicalDeviceFeatures(physical,&features);
    float priority=1;VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qci.queueFamilyIndex=queueFamily;qci.queueCount=1;qci.pQueuePriorities=&priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dci.queueCreateInfoCount=1;dci.pQueueCreateInfos=&qci;dci.pEnabledFeatures=&features;
    check(vkCreateDevice(physical,&dci,nullptr,&device),"Create compute device");vkGetDeviceQueue(device,queueFamily,0,&queue);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bci.size=bytes;bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device,&bci,nullptr,&buffer),"Create probe buffer");
    VkMemoryRequirements requirements{};vkGetBufferMemoryRequirements(device,buffer,&requirements);
    uint32_t type=UINT32_MAX;
    for(uint32_t i=0;i<memoryProperties.memoryTypeCount;++i)if((requirements.memoryTypeBits&(1u<<i))&&(memoryProperties.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
      type=i;if(memoryProperties.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)break;
    }
    require(type!=UINT32_MAX,"No host visible probe memory");coherent=(memoryProperties.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)!=0;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};mai.allocationSize=requirements.size;mai.memoryTypeIndex=type;
    check(vkAllocateMemory(device,&mai,nullptr,&memory),"Allocate probe memory");check(vkBindBufferMemory(device,buffer,memory,0),"Bind probe memory");
    check(vkMapMemory(device,memory,0,VK_WHOLE_SIZE,0,&mapped),"Map probe memory");
    VkDescriptorSetLayoutBinding binding{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};lci.bindingCount=1;lci.pBindings=&binding;
    check(vkCreateDescriptorSetLayout(device,&lci,nullptr,&descriptorLayout),"Create probe descriptor layout");
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1};VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pci.maxSets=1;pci.poolSizeCount=1;pci.pPoolSizes=&poolSize;
    check(vkCreateDescriptorPool(device,&pci,nullptr,&descriptorPool),"Create probe descriptor pool");
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};dai.descriptorPool=descriptorPool;dai.descriptorSetCount=1;dai.pSetLayouts=&descriptorLayout;
    check(vkAllocateDescriptorSets(device,&dai,&descriptorSet),"Allocate probe descriptor set");
    VkDescriptorBufferInfo info{buffer,0,bytes};VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};write.dstSet=descriptorSet;write.dstBinding=0;write.descriptorCount=1;write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;write.pBufferInfo=&info;vkUpdateDescriptorSets(device,1,&write,0,nullptr);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};plci.setLayoutCount=1;plci.pSetLayouts=&descriptorLayout;
    check(vkCreatePipelineLayout(device,&plci,nullptr,&pipelineLayout),"Create probe pipeline layout");
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};smci.codeSize=spirv.size()*sizeof(uint32_t);smci.pCode=spirv.data();
    check(vkCreateShaderModule(device,&smci,nullptr,&module),"Create probe shader module");
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cpci.layout=pipelineLayout;cpci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;cpci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cpci.stage.module=module;cpci.stage.pName="main";
    check(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cpci,nullptr,&pipeline),"Create probe compute pipeline");
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pool.queueFamilyIndex=queueFamily;
    check(vkCreateCommandPool(device,&pool,nullptr,&commandPool),"Create probe command pool");
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=commandPool;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;
    check(vkAllocateCommandBuffers(device,&cai,&command),"Allocate probe command buffer");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(device,&fci,nullptr,&fence),"Create probe fence");
  }
  void run(std::vector<ProbeRecord>& records) {
    check(vkResetCommandPool(device,commandPool,0),"Reset probe commands");
    check(vkResetFences(device,1,&fence),"Reset probe fence");
    std::memcpy(mapped,records.data(),records.size()*sizeof(ProbeRecord));
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};range.memory=memory;range.size=VK_WHOLE_SIZE;
    if(!coherent)check(vkFlushMappedMemoryRanges(device,1,&range),"Flush probe input");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;check(vkBeginCommandBuffer(command,&begin),"Begin probe commands");
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipelineLayout,0,1,&descriptorSet,0,nullptr);
    vkCmdDispatch(command,(static_cast<uint32_t>(records.size())+63)/64,1,1);
    barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    check(vkEndCommandBuffer(command),"End probe commands");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;check(vkQueueSubmit(queue,1,&submit,fence),"Submit material probes");
    check(vkWaitForFences(device,1,&fence,VK_TRUE,60'000'000'000ull),"Wait material probes");
    if(!coherent)check(vkInvalidateMappedMemoryRanges(device,1,&range),"Invalidate probe output");
    std::memcpy(records.data(),mapped,records.size()*sizeof(ProbeRecord));
  }
};
std::vector<uint32_t> readSpirv(const std::filesystem::path& path) {
  std::ifstream stream(path,std::ios::binary|std::ios::ate);require(bool(stream),"Cannot open probe SPIR-V: "+path.string());
  auto bytes=stream.tellg();require(bytes>0&&bytes%4==0,"Invalid probe SPIR-V length");
  std::vector<uint32_t> result(static_cast<size_t>(bytes)/4);stream.seekg(0);stream.read(reinterpret_cast<char*>(result.data()),bytes);
  require(bool(stream)&&result[0]==0x07230203u,"Invalid probe SPIR-V data");return result;
}
bool near3(Float4 actual,Float4 expected) {
  const std::array<float,3> a{actual.x,actual.y,actual.z},b{expected.x,expected.y,expected.z};
  for(size_t i=0;i<3;++i)if(!std::isfinite(a[i])||!std::isfinite(b[i])||std::abs(a[i]-b[i])>2e-5f+2e-4f*std::max(std::abs(a[i]),std::abs(b[i])))return false;
  return true;
}
std::string triple(Float4 p) {return "("+std::to_string(p.x)+", "+std::to_string(p.y)+", "+std::to_string(p.z)+")";}

void integrateTilted(bool validation,const std::filesystem::path& path,uint32_t count) {
  // Keep storage ranges below Vulkan's guaranteed 128 MiB limit even for a
  // large quadrature. Chunks retain global Fibonacci indices and one pipeline.
  const uint32_t batch=std::min(count,65536u);
  std::vector<ProbeRecord> records(batch);
  Gpu gpu;gpu.init(validation,records.size()*sizeof(ProbeRecord),readSpirv(path));
  ProbeRecord prototype{};
  prototype.material=Material{};prototype.material.baseColor={.8f,.6f,.3f,1};
  prototype.material.surface.y=.8f;prototype.material.specular.x=0;
  prototype.material.coat={1,.8f,2.5f,0};
  prototype.normal={.99f,0,.14106736f,0};prototype.coatNormal={-.99f,0,.14106736f,0};
  prototype.tangent=prototype.coatTangent={1,0,0,0};
  prototype.geometricNormal=prototype.view={0,0,1,0};prototype.random.w=1;
  constexpr double pi=3.1415926535897932384626433832795;
  constexpr double goldenConjugate=.6180339887498948482045868343656;
  std::array<double,3> sum{};
  for(uint32_t offset=0;offset<count;offset+=batch) {
    const uint32_t active=std::min(batch,count-offset);
    for(uint32_t j=0;j<batch;++j) {
      records[j]=prototype;
      // The unused tail remains valid input and is ignored in the host sum.
      const uint32_t i=offset+std::min(j,active-1);
      const double z=(double(i)+.5)/count;
      const double phi=2*pi*std::fmod(double(i)*goldenConjugate,1.0);
      const double radius=std::sqrt(std::max(0.0,1-z*z));
      records[j].light={float(radius*std::cos(phi)),float(radius*std::sin(phi)),float(z),0};
    }
    gpu.run(records);
    for(uint32_t j=0;j<active;++j) {
      const auto value=records[j].runtimeValue;
      require(std::isfinite(value.x)&&std::isfinite(value.y)&&std::isfinite(value.z),"Nonfinite official tilted-normal quadrature");
      sum[0]+=value.x;sum[1]+=value.y;sum[2]+=value.z;
    }
  }
  require(gpu.validationErrors==0,"Vulkan validation errors in official tilted-normal quadrature");
  const double measure=2*pi/count;
  std::cout<<std::setprecision(12)<<"Official tilted base/coat uniform-hemisphere integral: count="<<count
    <<", RGB="<<sum[0]*measure<<','<<sum[1]*measure<<','<<sum[2]*measure<<'\n';
}
}

int main(int argc,char** argv) {
  try {
    std::filesystem::path path;
#ifdef CAUSTRA_GENERATED_MATERIAL_PROBE_SPV
    path=CAUSTRA_GENERATED_MATERIAL_PROBE_SPV;
#endif
    bool validation=false,integrate=false;uint32_t integrationCount=65536;
    for(int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if(arg=="--validation")validation=true;
      else if(arg=="--spirv"&&i+1<argc)path=argv[++i];
      else if(arg=="--integrate-tilted")integrate=true;
      else if(arg=="--integration-count"&&i+1<argc) {
        const auto parsed=std::stoull(argv[++i]);
        require(parsed>0&&parsed<=16'777'216,"Integration count must be between 1 and 16777216");
        integrationCount=static_cast<uint32_t>(parsed);
      }
      else throw std::runtime_error("Usage: generated_material_tests [--spirv probe.spv] [--validation] [--integrate-tilted [--integration-count N]]");
    }
    require(!path.empty(),"Provide --spirv or CAUSTRA_GENERATED_MATERIAL_PROBE_SPV");
    if(integrate){integrateTilted(validation,path,integrationCount);return 0;}
    auto inputs=cases();std::vector<ProbeRecord> records;for(auto& item:inputs)records.push_back(item.data);
    Gpu gpu;gpu.init(validation,records.size()*sizeof(ProbeRecord),readSpirv(path));gpu.run(records);
    size_t failures=0;std::array<uint32_t,6> sampled{},nonzero{};
    auto compare=[&](const std::string& name,const char* operation,Float4 actual,Float4 expected) {
      if(near3(actual,expected))return;
      if(failures<16)std::cerr<<name<<" "<<operation<<": actual="<<triple(actual)<<", expected="<<triple(expected)<<'\n';
      ++failures;
    };
    for(size_t i=0;i<records.size();++i) {
      auto& p=records[i];auto model=inputs[i].data.material.program.z;
      compare(inputs[i].name,"reflection routing",p.runtimeValue,p.referenceValue);
      compare(inputs[i].name,"emission routing",p.runtimeEmission,p.referenceEmission);
      if(inputs[i].analyticEmission) {
        auto expected=emissionReference(inputs[i].data);
        compare(inputs[i].name,"independent coat EDF reference",p.referenceEmission,expected);
      } else {
        compare(inputs[i].name,"independent uniform EDF reference",p.runtimeEmission,inputs[i].data.material.emission);
      }
      if(inputs[i].analyticBurley) {
        const double burley=.3/3.14159265358979323846
          *(1+.189*std::pow(.8,5))*(1+.189*std::pow(.7,5));
        const auto color=inputs[i].data.material.baseColor;
        compare(inputs[i].name,"independent Burley reference",p.runtimeValue,
          {float(color.x*burley),float(color.y*burley),float(color.z*burley),0});
      }
      if(std::max({p.referenceValue.x,p.referenceValue.y,p.referenceValue.z})>1e-7f)++nonzero[model];
      if(p.sampleReference.w>0&&p.sampleState.w==0) {
        ++sampled[model];
        compare(inputs[i].name,"sample weight * PDF",p.sampleValue,p.sampleReference);
        require(std::isfinite(p.sampleValue.w)&&p.sampleValue.w>0,inputs[i].name+" returned invalid sample PDF");
      }
    }
    require(gpu.validationErrors==0,"Vulkan validation errors in generated material probe");
    for(size_t model=0;model<6;++model) {
      require(nonzero[model]>=8,std::string(modelNames[model])+" lacks nonzero reference coverage");
      require(sampled[model]>=8,std::string(modelNames[model])+" lacks accepted non-delta sample coverage");
      std::cout<<modelNames[model]<<": 32 directed cases"<<(model==2?" + 1 independent Burley case":"")
        <<", "<<sampled[model]<<" accepted sample checks\n";
    }
    require(failures==0,std::to_string(failures)+" generated material comparisons failed");
    std::cout<<records.size()<<" directed official ShaderGen integration cases passed\n";return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
