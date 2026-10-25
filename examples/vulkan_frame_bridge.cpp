#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <windows.h>
#include "vulkan_frame_bridge.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace caustra::example {
namespace {
constexpr auto memoryHandleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr auto semaphoreHandleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr VkBufferUsageFlags bridgeUsage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
  VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

void check(VkResult result,const char* operation) {
  if(result!=VK_SUCCESS)
    throw std::runtime_error(std::string(operation)+" failed (VkResult "+std::to_string(result)+")");
}
template<class Function> Function deviceFunction(VkDevice device,const char* name) {
  auto function=reinterpret_cast<Function>(vkGetDeviceProcAddr(device,name));
  if(!function) throw std::runtime_error(std::string("Missing frame bridge device extension function: ")+name);
  return function;
}
struct Win32Handle {
  HANDLE value{};
  ~Win32Handle() {close();}
  void close() noexcept {if(value) {CloseHandle(value);value=nullptr;}}
  Win32Handle()=default;
  Win32Handle(const Win32Handle&)=delete;
  Win32Handle& operator=(const Win32Handle&)=delete;
};

// A candidate that failed allocation has never reached either queue. Active
// resources are destroyed only after Impl has waited for both borrowed devices.
struct SharedBuffer {
  VkDevice producer{},consumer{};
  VkBuffer exported{},imported{};
  VkDeviceMemory exportedMemory{},importedMemory{};
  VkDeviceSize size{};
  SharedBuffer(VkDevice from,VkDevice to):producer(from),consumer(to) {}
  ~SharedBuffer() {
    if(imported) vkDestroyBuffer(consumer,imported,nullptr);
    if(importedMemory) vkFreeMemory(consumer,importedMemory,nullptr);
    if(exported) vkDestroyBuffer(producer,exported,nullptr);
    if(exportedMemory) vkFreeMemory(producer,exportedMemory,nullptr);
  }
};
}

struct VulkanFrameBridge::Impl {
  VulkanContext producer,consumer;
  VkCommandPool commandPool{};
  VkCommandBuffer command{};
  VkFence copied{};
  VkSemaphore producerReady{},consumerReady{},consumerConsumed{},producerConsumed{};
  std::unique_ptr<SharedBuffer> buffer;
  bool dedicatedOnly=false,externalOwnership=false,readyPending=false,waitConsumed=false;
  bool acquireRecorded=false,releaseRecorded=false;
  VkCommandBuffer consumerCommand{};

  Impl(VulkanContext from,VulkanContext to):producer(from),consumer(to) {
    try {initialize();} catch(...) {cleanup();throw;}
  }
  ~Impl() {cleanup();}

  void initialize() {
    if(!producer.device || !consumer.device || !producer.queue || !consumer.queue ||
       !producer.physicalDevice || producer.physicalDevice!=consumer.physicalDevice || producer.device==consumer.device)
      throw std::invalid_argument("Frame bridge requires distinct Vulkan devices on the same physical GPU");
    VkPhysicalDeviceExternalBufferInfo info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
    info.usage=bridgeUsage;info.handleType=memoryHandleType;
    VkExternalBufferProperties properties{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
    vkGetPhysicalDeviceExternalBufferProperties(producer.physicalDevice,&info,&properties);
    const auto& memory=properties.externalMemoryProperties;
    constexpr VkExternalMemoryFeatureFlags memoryRequired=VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
    if((memory.externalMemoryFeatures & memoryRequired)!=memoryRequired || !(memory.compatibleHandleTypes & memoryHandleType))
      throw std::runtime_error("Physical GPU cannot import/export opaque Win32 HDR bridge buffers");
    dedicatedOnly=(memory.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT)!=0;

    VkPhysicalDeviceExternalSemaphoreInfo semaphoreInfo{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
    semaphoreInfo.handleType=semaphoreHandleType;
    VkExternalSemaphoreProperties semaphoreProperties{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
    vkGetPhysicalDeviceExternalSemaphoreProperties(producer.physicalDevice,&semaphoreInfo,&semaphoreProperties);
    constexpr VkExternalSemaphoreFeatureFlags semaphoreRequired=VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT | VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    if((semaphoreProperties.externalSemaphoreFeatures & semaphoreRequired)!=semaphoreRequired ||
       !(semaphoreProperties.compatibleHandleTypes & semaphoreHandleType))
      throw std::runtime_error("Physical GPU cannot import/export opaque Win32 frame semaphores");
    createSemaphorePair(producer.device,consumer.device,producerReady,consumerReady);
    createSemaphorePair(consumer.device,producer.device,consumerConsumed,producerConsumed);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex=producer.queueFamily;
    check(vkCreateCommandPool(producer.device,&poolInfo,nullptr,&commandPool),"Create HDR bridge command pool");
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool=commandPool;allocation.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;allocation.commandBufferCount=1;
    check(vkAllocateCommandBuffers(producer.device,&allocation,&command),"Allocate HDR bridge commands");
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(producer.device,&fenceInfo,nullptr,&copied),"Create HDR bridge copy fence");
  }

  void createSemaphorePair(VkDevice from,VkDevice to,VkSemaphore& exported,VkSemaphore& imported) {
    VkExportSemaphoreCreateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exportInfo.handleTypes=semaphoreHandleType;
    VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};info.pNext=&exportInfo;
    check(vkCreateSemaphore(from,&info,nullptr,&exported),"Create exported HDR bridge semaphore");
    info.pNext=nullptr;
    check(vkCreateSemaphore(to,&info,nullptr,&imported),"Create imported HDR bridge semaphore");
    auto getHandle=deviceFunction<PFN_vkGetSemaphoreWin32HandleKHR>(from,"vkGetSemaphoreWin32HandleKHR");
    auto importHandle=deviceFunction<PFN_vkImportSemaphoreWin32HandleKHR>(to,"vkImportSemaphoreWin32HandleKHR");
    VkSemaphoreGetWin32HandleInfoKHR getInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    getInfo.semaphore=exported;getInfo.handleType=semaphoreHandleType;
    Win32Handle handle;
    check(getHandle(from,&getInfo,&handle.value),"Export HDR bridge semaphore handle");
    VkImportSemaphoreWin32HandleInfoKHR importInfo{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
    importInfo.semaphore=imported;importInfo.handleType=semaphoreHandleType;importInfo.handle=handle.value;
    // Permanent import shares the binary payload. Win32 NT-handle ownership
    // stays with the application, so the local handle closes after import.
    check(importHandle(to,&importInfo),"Import HDR bridge semaphore handle");
  }

  void waitDevices() {
    check(vkDeviceWaitIdle(consumer.device),"Wait for HDR bridge consumer");
    check(vkDeviceWaitIdle(producer.device),"Wait for HDR bridge producer");
  }

  void ensureBuffer(VkDeviceSize size) {
    if(buffer && buffer->size==size) return;
    auto candidate=std::make_unique<SharedBuffer>(producer.device,consumer.device);
    candidate->size=size;
    VkExternalMemoryBufferCreateInfo externalInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    externalInfo.handleTypes=memoryHandleType;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.pNext=&externalInfo;info.size=size;info.usage=bridgeUsage;info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(producer.device,&info,nullptr,&candidate->exported),"Create exported HDR bridge buffer");
    check(vkCreateBuffer(consumer.device,&info,nullptr,&candidate->imported),"Create imported HDR bridge buffer");
    VkMemoryDedicatedRequirements producerDedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryDedicatedRequirements consumerDedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 producerRequirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};producerRequirements.pNext=&producerDedicated;
    VkMemoryRequirements2 consumerRequirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};consumerRequirements.pNext=&consumerDedicated;
    VkBufferMemoryRequirementsInfo2 requirementInfo{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};
    requirementInfo.buffer=candidate->exported;
    vkGetBufferMemoryRequirements2(producer.device,&requirementInfo,&producerRequirements);
    requirementInfo.buffer=candidate->imported;
    vkGetBufferMemoryRequirements2(consumer.device,&requirementInfo,&consumerRequirements);
    const uint32_t compatible=producerRequirements.memoryRequirements.memoryTypeBits & consumerRequirements.memoryRequirements.memoryTypeBits;
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(producer.physicalDevice,&memoryProperties);
    uint32_t memoryType=memoryProperties.memoryTypeCount;
    for(uint32_t index=0;index<memoryProperties.memoryTypeCount;++index)
      if((compatible & (1u<<index)) && (memoryProperties.memoryTypes[index].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        memoryType=index;break;
      }
    if(memoryType==memoryProperties.memoryTypeCount)
      throw std::runtime_error("No common device-local memory type for the HDR bridge");
    const bool dedicated=dedicatedOnly || producerDedicated.requiresDedicatedAllocation || consumerDedicated.requiresDedicatedAllocation ||
      producerDedicated.prefersDedicatedAllocation || consumerDedicated.prefersDedicatedAllocation;
    VkMemoryDedicatedAllocateInfo producerDedicatedInfo{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    producerDedicatedInfo.buffer=candidate->exported;
    VkExportMemoryAllocateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exportInfo.handleTypes=memoryHandleType;exportInfo.pNext=dedicated?&producerDedicatedInfo:nullptr;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.pNext=&exportInfo;allocation.memoryTypeIndex=memoryType;
    allocation.allocationSize=std::max(producerRequirements.memoryRequirements.size,consumerRequirements.memoryRequirements.size);
    check(vkAllocateMemory(producer.device,&allocation,nullptr,&candidate->exportedMemory),"Allocate exported HDR bridge memory");
    check(vkBindBufferMemory(producer.device,candidate->exported,candidate->exportedMemory,0),"Bind exported HDR bridge memory");
    auto getHandle=deviceFunction<PFN_vkGetMemoryWin32HandleKHR>(producer.device,"vkGetMemoryWin32HandleKHR");
    VkMemoryGetWin32HandleInfoKHR getInfo{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    getInfo.memory=candidate->exportedMemory;getInfo.handleType=memoryHandleType;
    Win32Handle handle;
    check(getHandle(producer.device,&getInfo,&handle.value),"Export HDR bridge memory handle");
    VkMemoryDedicatedAllocateInfo consumerDedicatedInfo{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    consumerDedicatedInfo.buffer=candidate->imported;
    VkImportMemoryWin32HandleInfoKHR importInfo{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    importInfo.handleType=memoryHandleType;importInfo.handle=handle.value;
    importInfo.pNext=dedicated?&consumerDedicatedInfo:nullptr;
    // Opaque handles exported by Vulkan retain allocation size/type. Querying
    // GetMemoryWin32HandleProperties for an opaque handle is invalid (00666).
    allocation.pNext=&importInfo;
    check(vkAllocateMemory(consumer.device,&allocation,nullptr,&candidate->importedMemory),"Import HDR bridge memory");
    handle.close();
    check(vkBindBufferMemory(consumer.device,candidate->imported,candidate->importedMemory,0),"Bind imported HDR bridge memory");
    if(buffer) waitDevices();
    buffer=std::move(candidate);
    externalOwnership=false;
  }

  VulkanFrame stage(VulkanFrame source) {
    if(readyPending) throw std::logic_error("Previous HDR bridge frame was not submitted by the consumer");
    if(!source.buffer || !source.width || !source.height) throw std::invalid_argument("HDR bridge requires a rendered frame");
    const uint64_t pixels=uint64_t(source.width)*source.height;
    if(pixels>std::numeric_limits<VkDeviceSize>::max()/sizeof(Float4)) throw std::invalid_argument("HDR bridge frame size overflow");
    const VkDeviceSize size=pixels*sizeof(Float4);
    ensureBuffer(size);
    check(vkResetCommandBuffer(command,0),"Reset HDR bridge commands");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command,&begin),"Begin HDR bridge commands");
    VkBufferMemoryBarrier sourceBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    sourceBarrier.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;sourceBarrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    sourceBarrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;sourceBarrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    sourceBarrier.buffer=source.buffer;sourceBarrier.size=size;
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&sourceBarrier,0,nullptr);
    VkBufferMemoryBarrier bridgeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bridgeBarrier.buffer=buffer->exported;bridgeBarrier.size=buffer->size;
    if(externalOwnership) {
      bridgeBarrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
      bridgeBarrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_EXTERNAL;bridgeBarrier.dstQueueFamilyIndex=producer.queueFamily;
      vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&bridgeBarrier,0,nullptr);
    }
    VkBufferCopy copy{0,0,size};
    vkCmdCopyBuffer(command,source.buffer,buffer->exported,1,&copy);
    bridgeBarrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;bridgeBarrier.dstAccessMask=0;
    bridgeBarrier.srcQueueFamilyIndex=producer.queueFamily;bridgeBarrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_EXTERNAL;
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,nullptr,1,&bridgeBarrier,0,nullptr);
    check(vkEndCommandBuffer(command),"End HDR bridge commands");
    check(vkResetFences(producer.device,1,&copied),"Reset HDR bridge copy fence");
    // External ownership operations are not assigned a pipeline stage without
    // maintenance8; ALL_COMMANDS orders the acquire after the external wait.
    const VkPipelineStageFlags waitStage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount=1;submit.pCommandBuffers=&command;
    submit.signalSemaphoreCount=1;submit.pSignalSemaphores=&producerReady;
    if(waitConsumed) {submit.waitSemaphoreCount=1;submit.pWaitSemaphores=&producerConsumed;submit.pWaitDstStageMask=&waitStage;}
    check(vkQueueSubmit(producer.queue,1,&submit,copied),"Submit HDR bridge copy");
    waitConsumed=false;externalOwnership=true;readyPending=true;
    acquireRecorded=false;releaseRecorded=false;consumerCommand=VK_NULL_HANDLE;
    // No borrowed renderer buffer remains in use once stage returns. Consumer
    // rendering can therefore outlive the next renderer state mutation safely.
    check(vkWaitForFences(producer.device,1,&copied,VK_TRUE,UINT64_MAX),"Finish HDR bridge copy");
    return {buffer->imported,source.width,source.height};
  }

  void acquire(VkCommandBuffer destination) {
    if(!destination || !readyPending || acquireRecorded) throw std::logic_error("Invalid HDR bridge consumer acquire");
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_EXTERNAL;barrier.dstQueueFamilyIndex=consumer.queueFamily;
    barrier.buffer=buffer->imported;barrier.size=buffer->size;
    vkCmdPipelineBarrier(destination,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    consumerCommand=destination;acquireRecorded=true;
  }
  void release(VkCommandBuffer destination) {
    if(!readyPending || !acquireRecorded || releaseRecorded || destination!=consumerCommand)
      throw std::logic_error("Invalid HDR bridge consumer release");
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask=VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex=consumer.queueFamily;barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_EXTERNAL;
    barrier.buffer=buffer->imported;barrier.size=buffer->size;
    vkCmdPipelineBarrier(destination,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    releaseRecorded=true;
  }
  void submitted() {
    if(!readyPending || !acquireRecorded || !releaseRecorded) throw std::logic_error("HDR bridge consumer submitted without ownership transfer");
    readyPending=false;waitConsumed=true;
  }

  void cleanup() noexcept {
    if(consumer.device) vkDeviceWaitIdle(consumer.device);
    if(producer.device && producer.device!=consumer.device) vkDeviceWaitIdle(producer.device);
    buffer.reset();
    // Imported payload references are released before their exporting objects.
    if(consumerReady) vkDestroySemaphore(consumer.device,consumerReady,nullptr);
    if(producerConsumed) vkDestroySemaphore(producer.device,producerConsumed,nullptr);
    if(consumerConsumed) vkDestroySemaphore(consumer.device,consumerConsumed,nullptr);
    if(producerReady) vkDestroySemaphore(producer.device,producerReady,nullptr);
    consumerReady=producerConsumed=consumerConsumed=producerReady=VK_NULL_HANDLE;
    if(copied) vkDestroyFence(producer.device,copied,nullptr);
    if(commandPool) vkDestroyCommandPool(producer.device,commandPool,nullptr);
    copied=VK_NULL_HANDLE;commandPool=VK_NULL_HANDLE;command=VK_NULL_HANDLE;
  }
};

VulkanFrameBridge::VulkanFrameBridge(VulkanContext producer,VulkanContext consumer):impl_(std::make_unique<Impl>(producer,consumer)) {}
VulkanFrameBridge::~VulkanFrameBridge()=default;
VulkanFrame VulkanFrameBridge::stage(VulkanFrame source) {return impl_->stage(source);}
void VulkanFrameBridge::acquire(VkCommandBuffer command) {impl_->acquire(command);}
void VulkanFrameBridge::release(VkCommandBuffer command) {impl_->release(command);}
VkSemaphore VulkanFrameBridge::ready() const {return impl_->consumerReady;}
VkSemaphore VulkanFrameBridge::consumed() const {return impl_->consumerConsumed;}
void VulkanFrameBridge::submitted() {impl_->submitted();}

} // namespace caustra::example
