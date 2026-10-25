# 查看器实际退出

查看器接受标题栏关闭、Esc 或帧数退出请求后停止渲染、释放鼠标捕获并禁用输入，保留显示 Closing 的窗口，直到 GPU 资源清理结束才销毁窗口。主线程继续分发原生消息；清理线程独占全部 GPU 对象，完成后 join。窗口隐藏或输出一条完成日志都不能代替实际进程退出。

GPU 同步、DLSS/NGX 释放以及 Vulkan 资源释放均保留。渲染核心没有增加窗口依赖或异步 API，也没有增加每帧回读。无法创建清理线程时使用安全的同步释放后备路径。

## 定位与改动

2026-10-25，RTX 5060、NVIDIA 610.62 上复现：同一逻辑设备执行 Path Trace/DLSS 与窗口呈现时，GPU 等待、交换链和 NGX 关闭均已结束，`vkDestroyDevice` 仍等待约 77 秒，另有 `nvgpucomp64.dll` 线程持续运行。上次隐藏窗口的处理没有解决该问题。

仅更换图像 blit、补齐 present fence、调整窗口隐藏或编译缓存，均未稳定通过严格的退出回归。离屏对照可以正常退出；进一步将窗口呈现放到第二逻辑设备，连续六次隔离对照正常退出。最终将该边界用于真实 HDR 上屏：

1. Renderer 设备负责路径追踪和 DLSS，不请求 swapchain 扩展、不调用窗口呈现。
2. 查看器在同一物理 GPU 上创建独立呈现设备，负责显示 compute、swapchain 和 present。
3. 私有 `VulkanFrameBridge` 在 GPU 上复制 HDR 到共享设备本地 Buffer，以 opaque Win32 内存句柄导入呈现设备。普通帧没有 CPU HDR 回读。
4. 双向外部二进制信号量与 EXTERNAL 所有权屏障保护共享 Buffer；外部信号量等待使用 ALL_COMMANDS。共享句柄在导入后关闭，内存按两端要求使用专用分配。
5. 退出时等待 GPU 和 present completion fences，释放桥和呈现设备，最后释放 Renderer/实例与窗口。资源释放和 GPU 同步完整保留。

原渲染内核、官方材质数学、资产格式和采样逻辑不变，Path Trace SPIR-V 的 SHA-256 保持 `43f0e58f8e6da242429ed47009063a214a1d07040c7705975371c5d7a8a3bbd6`。函数拆分实验未采用。Windows 查看器需要 opaque Win32 外部 Buffer 内存和二进制信号量的导入/导出支持，不支持时明确报错；离线核心没有新增扩展要求。[KHR present fence](https://docs.vulkan.org/refpages/latest/refpages/source/VkSwapchainPresentFenceInfoKHR.html) 用于确认呈现资源已可释放，不能仅以 device idle 推断呈现引擎已完成。

## 回归

```powershell
cmake --preset windows
cmake --build --preset windows-release
ctest --preset windows-release --parallel 1 -L Shutdown
```

测试需要启用 GPU、查看器及相应 DLSS 测试选项和桌面会话。十项场景覆盖标题栏关闭、Esc、帧数结束、暂停后最小化、暂停后两次缩放，以及旋转/平移/缩放和两轮 DLSS 开关后的关闭。`tests/viewer_shutdown_test.py` 使用标准库 Python 启动子进程；初始化/运行预算为 300 秒，收到 `Viewer cleanup starting` 后，必须在 5 秒内观察到子进程真正返回 0 和输出管道结束，包括 CRT/DLL 卸载。未见标记、非零返回或 Vulkan 验证输出均失败。超时只收尾本次测试子进程，不改变生产程序的释放规则。输出 `Viewer shutdown complete` 本身不能让测试通过。

真实 HDR 双设备版本已通过十项严格退出回归、连续五次 DLSS 交互关闭及额外三次带阶段跟踪的重复，Vulkan 错误为零，GPU 色调转换参考、有限非黑 HDR 和缓存重呈现检查通过。交互场景实测清理约 170ms，父进程确认实际退出约 280ms；整个进程约 4.4 秒，包含初始化与测试交互。该场景呈现阶段约 0.97ms，原同设备路径约 0.72ms，新增 GPU 复制有少量开销，这不是所有场景的性能承诺。

完整 San Miguel（846 网格、334 材质、374 纹理），24 次反弹、DLSS Quality：16 帧交互/缩放/两轮 DLSS 开关和图像检查通过，清理 311ms，父进程确认实际退出 422ms，Vulkan 错误为零。默认离线 Vulkan 与 DLSS 借用输出互操作回归也通过；其中首轮冷编译曾触及原 120 秒限时，相同二进制缓存就绪后两项分别为 1.23s、2.12s。


可选阶段跟踪默认关闭：

```powershell
$env:CAUSTRA_TRACE_SHUTDOWN='1'
.\build\bin\Release\caustra_viewer.exe --dlss quality --smoke-test --frames 16 --shutdown-test close --validation
Remove-Item Env:CAUSTRA_TRACE_SHUTDOWN
```

日志覆盖 GPU 等待、呈现完成、显示资源、桥、呈现设备、DLSS/计算设备及实例。关闭请求由主线程接受，正在执行的同步 `render()` 仍需先返回；首次冷管线编译不属于退出清理预算。
