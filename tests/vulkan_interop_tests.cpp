// SPDX-License-Identifier: Apache-2.0
// Exercise the public borrowed-buffer contract independently of the viewer.
#include <caustra/vulkan.hpp>
#include "demo_scene.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
void require(bool condition,const char* message) {
  if(!condition) throw std::runtime_error(message);
}
void check(VkResult result,const char* operation) {
  if(result!=VK_SUCCESS) throw std::runtime_error(std::string(operation)+": "+std::to_string(result));
}
bool samePixels(const caustra::Image& a,const caustra::Image& b) {
  return a.width==b.width && a.height==b.height && a.pixels.size()==b.pixels.size() &&
    std::memcmp(a.pixels.data(),b.pixels.data(),a.pixels.size()*sizeof(caustra::Float4))==0;
}
void verifyImage(const caustra::Image& image) {
  require(image.width && image.height && image.pixels.size()==size_t(image.width)*image.height,
          "Exported image dimensions are invalid");
  double energy=0;
  for(const auto p:image.pixels) {
    require(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && std::isfinite(p.w),
            "Exported pixels are nonfinite");
    energy+=p.x+p.y+p.z;
  }
  require(energy>0,"Exported image is black");
}
void requireNoOutput(caustra::Renderer& renderer) {
  bool rejected=false;
  try { (void)caustra::VulkanAccess::output(renderer); }
  catch(const std::logic_error&) { rejected=true; }
  require(rejected,"GPU output must be unavailable before rendering or after accumulation reset");
}

// This consumer deliberately uses its own command pool, synchronization and
// allocation. It waits before returning, so no borrowed buffer remains in use
// across the next Renderer operation. A copy avoids duplicating its shader.
struct BufferConsumer {
  caustra::VulkanContext context;
  VkCommandPool pool{};
  VkCommandBuffer command{};
  VkFence fence{};
  VkBuffer buffer{};
  VkDeviceMemory memory{};
  void* mapped=nullptr;
  VkDeviceSize capacity=0;
  bool coherent=false;

  explicit BufferConsumer(caustra::VulkanContext value):context(value) {}
  ~BufferConsumer() {
    if(context.device) vkDeviceWaitIdle(context.device);
    releaseBuffer();
    if(fence) vkDestroyFence(context.device,fence,nullptr);
    if(pool) vkDestroyCommandPool(context.device,pool,nullptr);
  }
  void releaseBuffer() {
    if(mapped) vkUnmapMemory(context.device,memory);
    if(buffer) vkDestroyBuffer(context.device,buffer,nullptr);
    if(memory) vkFreeMemory(context.device,memory,nullptr);
    mapped=nullptr; buffer={}; memory={}; capacity=0;
  }
  void init() {
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex=context.queueFamily;
    check(vkCreateCommandPool(context.device,&poolInfo,nullptr,&pool),"Create consumer command pool");
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool=pool; allocate.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocate.commandBufferCount=1;
    check(vkAllocateCommandBuffers(context.device,&allocate,&command),"Allocate consumer command buffer");
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(context.device,&fenceInfo,nullptr,&fence),"Create consumer fence");
  }
  void reserve(VkDeviceSize bytes) {
    if(capacity==bytes) return;
    releaseBuffer();
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size=bytes; bufferInfo.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(context.device,&bufferInfo,nullptr,&buffer),"Create consumer readback buffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(context.device,buffer,&requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(context.physicalDevice,&properties);
    uint32_t type=UINT32_MAX; int best=-1;
    for(uint32_t i=0;i<properties.memoryTypeCount;++i) {
      const auto flags=properties.memoryTypes[i].propertyFlags;
      if(!(requirements.memoryTypeBits&(1u<<i)) || !(flags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
      const int score=((flags&VK_MEMORY_PROPERTY_HOST_CACHED_BIT)?4:0)+
                      ((flags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)?2:0);
      if(score>best) {type=i; best=score;}
    }
    require(type!=UINT32_MAX,"No host-visible consumer readback allocation available");
    coherent=(properties.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)!=0;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize=requirements.size; allocation.memoryTypeIndex=type;
    check(vkAllocateMemory(context.device,&allocation,nullptr,&memory),"Allocate consumer readback memory");
    check(vkBindBufferMemory(context.device,buffer,memory,0),"Bind consumer readback memory");
    check(vkMapMemory(context.device,memory,0,VK_WHOLE_SIZE,0,&mapped),"Map consumer readback memory");
    capacity=bytes;
  }
  caustra::Image copy(caustra::VulkanFrame frame) {
    require(frame.buffer && frame.width && frame.height,"Export contains an invalid buffer or extent");
    const VkDeviceSize bytes=VkDeviceSize(frame.width)*frame.height*sizeof(caustra::Float4);
    reserve(bytes);
    check(vkResetCommandPool(context.device,pool,0),"Reset consumer command pool");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command,&begin),"Begin consumer command buffer");
    // Raw accumulation is written by compute; RR output by image-to-buffer
    // transfer. Include both producers and the reused readback's host reads.
    VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    ready.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_HOST_READ_BIT;
    ready.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&ready,0,nullptr,0,nullptr);
    const VkBufferCopy region{0,0,bytes};
    vkCmdCopyBuffer(command,frame.buffer,buffer,1,&region);
    VkBufferMemoryBarrier visible{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    visible.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; visible.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    visible.srcQueueFamilyIndex=visible.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    visible.buffer=buffer; visible.size=VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&visible,0,nullptr);
    check(vkEndCommandBuffer(command),"End consumer command buffer");
    check(vkResetFences(context.device,1,&fence),"Reset consumer fence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&command;
    check(vkQueueSubmit(context.queue,1,&submit,fence),"Submit consumer copy");
    check(vkWaitForFences(context.device,1,&fence,VK_TRUE,UINT64_MAX),"Wait for consumer copy");
    if(!coherent) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; range.memory=memory; range.size=VK_WHOLE_SIZE;
      check(vkInvalidateMappedMemoryRanges(context.device,1,&range),"Invalidate consumer memory");
    }
    caustra::Image image; image.width=frame.width; image.height=frame.height;
    image.pixels.resize(size_t(frame.width)*frame.height);
    std::memcpy(image.pixels.data(),mapped,static_cast<size_t>(bytes));
    return image;
  }
};

caustra::Image verifyExport(caustra::Renderer& renderer,BufferConsumer& consumer) {
  // Read the borrowed buffer first, without relying on Renderer::readback to
  // establish its visibility. The public readback then supplies a byte oracle.
  auto exported=consumer.copy(caustra::VulkanAccess::output(renderer));
  verifyImage(exported);
  const auto first=renderer.readback(),cached=renderer.readback();
  require(first.samples==renderer.sampleCount() && cached.samples==first.samples,"Readback sample metadata is stale");
  require(samePixels(exported,first),"Borrowed GPU output differs from renderer readback");
  require(samePixels(first,cached),"Repeated readback changed pixels without rendering");
  require(samePixels(exported,consumer.copy(caustra::VulkanAccess::output(renderer))),
          "Readback altered the borrowed GPU output");
  return exported;
}

void verifyFailedCallbacks() {
  for(bool creationThrows:{false,true}) {
    unsigned created=0,destroyed=0,queueChecks=0;
    VkInstance tracked{}; bool mismatched=false;
    caustra::VulkanDeviceOptions integration;
    integration.instanceCreated=[&](VkInstance instance) {
      ++created; tracked=instance;
      if(creationThrows) throw std::runtime_error("Injected instance callback failure");
    };
    integration.instanceDestroying=[&](VkInstance instance) {
      ++destroyed; mismatched=mismatched || instance!=tracked;
    };
    integration.supportsQueue=[&](VkPhysicalDevice,uint32_t) {++queueChecks; return false;};
    bool rejected=false;
    try {caustra::Renderer renderer({true},integration);}
    catch(const std::runtime_error& error) {
      rejected=true;
      if(creationThrows) require(std::string_view(error.what())=="Injected instance callback failure",
                                "Creation callback exception was replaced");
    }
    require(rejected && created==1 && destroyed==1 && !mismatched,
            "Failed renderer creation did not pair instance callbacks exactly once");
    require(creationThrows ? queueChecks==0 : queueChecks>0,"Queue predicate ran at the wrong startup stage");
  }
}

void verifyRenderer(bool dlss) {
  unsigned created=0,destroyed=0,queueChecks=0;
  VkInstance tracked{}; bool mismatched=false;
  caustra::VulkanDeviceOptions integration;
  integration.instanceCreated=[&](VkInstance instance) {++created; tracked=instance;};
  integration.instanceDestroying=[&](VkInstance instance) {++destroyed; mismatched=mismatched || instance!=tracked;};
  integration.supportsQueue=[&](VkPhysicalDevice,uint32_t) {++queueChecks; return true;};
  {
    caustra::Renderer renderer({true},integration);
    const auto context=caustra::VulkanAccess::context(renderer);
    require(context.instance==tracked && context.physicalDevice && context.device && context.queue,
            "Vulkan context contains invalid or unrelated handles");
    require(created==1 && destroyed==0 && queueChecks>0,"Normal renderer callback lifecycle is invalid");
    std::cout<<"GPU: "<<renderer.deviceName()<<'\n';
    BufferConsumer consumer(context); consumer.init();
    requireNoOutput(renderer);
    caustra::RenderSettings settings; settings.width=64; settings.height=48; settings.maxBounces=4;
    auto camera=caustra::example::makeDemoCamera();
    renderer.setSettings(settings); renderer.setScene(caustra::example::makeDemoScene()); renderer.setCamera(camera);
    renderer.render(2); const auto first=verifyExport(renderer,consumer);
    require(samePixels(first,renderer.readbackRaw()),"Disabled DLSS export must contain raw accumulation");
    renderer.render(1); const auto accumulated=verifyExport(renderer,consumer);
    require(renderer.sampleCount()==3 && !samePixels(first,accumulated),"Render failed to invalidate cached output");
    camera.position.x+=.35f; renderer.updateCamera(camera); requireNoOutput(renderer);
    renderer.render(2); require(!samePixels(accumulated,verifyExport(renderer,consumer)),"Camera change retained stale output");
    renderer.reset(); requireNoOutput(renderer);
    settings.width=80; settings.height=56; renderer.setSettings(settings); requireNoOutput(renderer);
    renderer.render(1); auto resized=verifyExport(renderer,consumer);
    require(resized.width==80 && resized.height==56,"Raw exported extent did not follow resize");
    if(dlss) {
      settings.width=96; settings.height=64; renderer.setSettings(settings);
      for(auto quality:{caustra::DlssQuality::Native,caustra::DlssQuality::Quality}) {
        caustra::DlssSettings denoiser; denoiser.enabled=true; denoiser.quality=quality; denoiser.preset=caustra::DlssPreset::E;
        renderer.setDlssSettings(denoiser); requireNoOutput(renderer);
        renderer.render(2); const auto denoised=verifyExport(renderer,consumer);
        require(denoised.width==settings.width && denoised.height==settings.height,"DLSS exported extent must be output resolution");
        const auto raw=renderer.readbackRaw();
        require(quality==caustra::DlssQuality::Native ?
                raw.width==settings.width && raw.height==settings.height :
                raw.width<settings.width && raw.height<settings.height,"DLSS internal extent contract is invalid");
        renderer.render(1); verifyExport(renderer,consumer);
        settings.width+=16; settings.height+=8; renderer.setSettings(settings); requireNoOutput(renderer);
        renderer.render(1); const auto enlarged=verifyExport(renderer,consumer);
        require(enlarged.width==settings.width && enlarged.height==settings.height,"DLSS resized export is stale");
        denoiser.enabled=false; renderer.setDlssSettings(denoiser); requireNoOutput(renderer);
        renderer.render(2); require(samePixels(verifyExport(renderer,consumer),renderer.readbackRaw()),
                                   "Disabling DLSS did not restore raw buffer export");
      }
    }
    require(renderer.validationErrorCount()==0,"Vulkan validation errors in borrowed-buffer integration");
  }
  require(created==1 && destroyed==1 && !mismatched,"Normal renderer destruction did not pair instance callbacks");
}
}

int main(int argc,char** argv) {
  try {
    bool dlss=false;
    for(int i=1;i<argc;++i) {
      const std::string_view arg=argv[i];
      if(arg=="--dlss") dlss=true;
      else if(arg!="--validation") throw std::invalid_argument("Expected --dlss or --validation");
    }
    verifyFailedCallbacks(); verifyRenderer(dlss);
    std::cout<<"Vulkan interop passed: callback failure cleanup, queue selection, borrowed buffer copies, exact cached readback, accumulation/camera reset, resize"
             <<(dlss?", native/quality DLSS output and repeated off/on":"")<<".\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<"Vulkan interop: "<<error.what()<<'\n'; return 1;}
}
