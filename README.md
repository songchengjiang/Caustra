# Caustra：纯 Vulkan Path Tracer

基于本机 `vk_gltf_renderer` 的路径追踪实现与其使用的 NVIDIA `nvshaders` 拆分改造。提供独立 C++20 静态库、命令行出图、Windows 实时查看 Demo 和回归测试。原始工程保持原样；新工程不链接原工程或 `nvpro_core2`，运行时不需要任何模型文件。

保留 Vulkan Ray Query 硬件求交、每个网格的 BLAS、实例 TLAS、漫反射与 GGX 金属/粗糙度材质、三角形面积光、NEE、多重重要性采样、Russian Roulette、渐进累积和线性 HDR 回读；提供可选的 NVIDIA DLSS Ray Reconstruction 降噪与升采样。去掉 glTF/OBJ 加载、场景编辑器、光栅化、动画、复杂天空、AI 桥接和 OptiX/OIDN。材质第一阶段是双面不透明材质；基础颜色、金属度、粗糙度和自发光可由调用方直接提交。

## 渲染效果

以下图片均由本工程在 NVIDIA GeForce RTX 5060 上实际渲染，场景由代码生成。

Cornell Box：960×720、2048 samples per pixel、最多 8 次散射，包含金属球、漫反射箱体和三角形面积光。此图未使用 DLSS。

![Caustra 路径追踪渲染的 Cornell Box](docs/images/cornell.png)

DLSS 原生分辨率对照：960×720、64 个 1 spp 时域帧、RR 预设 E。左侧为同一批样本的原始累积结果，右侧为 DLSS Ray Reconstruction 输出。

![左侧为原始累积图，右侧为 DLSS Ray Reconstruction 降噪图](docs/images/dlss_comparison.png)

## 实时查看 Demo（Windows）

构建后运行：

```powershell
./build/bin/Release/caustra_viewer.exe
```

窗口中实时渲染程序生成的铜色环结模型、底座和开放式摄影棚，可从各方向旋转查看。默认每帧 1 spp，在支持的硬件上自动开启 DLSS Quality；不可用时状态栏会说明并使用原始 Path Trace。静止后继续渐进收敛，拖动时通过相机运动向量保留 DLSS 历史。模型、材质和窗口逻辑都在示例层，渲染库不依赖窗口系统或模型加载器。

![实时查看 Demo：环绕相机、DLSS 模式和渐进采样](docs/images/realtime_viewer.png)

| 操作 | 功能 |
| --- | --- |
| 左键拖动 | 围绕模型旋转视角 |
| 右键或中键拖动 | 平移观察目标 |
| 鼠标滚轮 | 缩放 |
| `R` / `Home` | 恢复初始视角并重置累积 |
| `Space` | 暂停 / 恢复渲染 |
| `A` | 自动旋转视角 |
| `D` | 切换 DLSS 开关 |
| `S` | 保存当前 PPM 与线性 HDR PFM 到 `renders/` |
| `Esc` | 关闭窗口 |

顶部控件可以选择 DLSS 模式、每次呈现的采样数，也可暂停、重置、保存和自动旋转。画面角落显示实际 FPS、渲染尺寸与当前视角累积样本数。初始化 DLSS 或改变分辨率时可能短暂等待；最小化后暂停 GPU 工作。大窗口的输出限制在约一百万像素，以控制回读和显示开销。

```powershell
# 明确指定模式；显式请求的 DLSS 不可用时会报错
./build/bin/Release/caustra_viewer.exe --dlss native --validation
# 连续旋转展示
./build/bin/Release/caustra_viewer.exe --auto-orbit --width 960 --height 640
# 自动交互验证：真实窗口消息驱动旋转、平移、缩放、尺寸变化和暂停/重置
./build/bin/Release/caustra_viewer.exe --dlss off --smoke-test --validation --screenshot renders/viewer.bmp
```

实时查看器目前使用 Win32/GDI，只在 Windows 构建。`-DCAUSTRA_BUILD_VIEWER=OFF` 可关闭这个目标；Linux 保留原有离线示例。可用 `-DCAUSTRA_VIEWER_TESTS=ON` 将需要桌面会话的窗口交互测试加入 CTest。

## 构建与运行

需要 CMake 3.24+、C++20 编译器、Vulkan SDK 1.2+、支持 `VK_KHR_ray_query` 与 `VK_KHR_acceleration_structure` 的显卡和驱动，以及能够生成 SPIR-V 的 Slang 编译器。当前 Windows 环境可直接复用原项目已下载的 Slang 与 DLSS SDK；也可以用 `-DCAUSTRA_SLANG_ROOT=...`、`-DCAUSTRA_SLANGC=...` 和 `-DCAUSTRA_DLSS_ROOT=...` 指定自己的 SDK。此工程配置时不会下载依赖。

在本目录的 PowerShell 中执行：

```powershell
cmake --preset windows
cmake --build --preset windows-release
ctest --preset windows-release
./build/bin/Release/caustra_demo.exe --samples 128 --output renders/cornell --validation --self-test
```

程序输出 `renders/cornell.ppm`（Reinhard + sRGB）与 `renders/cornell.pfm`（线性浮点 HDR）。SPIR-V 在构建时嵌入静态库，不需要 Slang DLL、原始仓库、模型资源或着色器文件。启用 DLSS 编译时，构建自动将 SDK 的 release 版 `nvngx_dlss.dll` 和 `nvngx_dlssd.dll` 复制到程序旁；运行时需要这些库与兼容的 NVIDIA 驱动。`--help` 查看示例参数。

Vulkan SDK 自动查找环境变量 `VULKAN_SDK`，Windows 也会查找 `C:/VulkanSDK`。自定义安装位置可传 `-DVulkan_ROOT=...`。Slang 自定义路径例如：

```powershell
cmake --preset windows -DCAUSTRA_SLANG_ROOT=C:/SDK/slang
```

Linux 提供 `linux` 配置及 `linux-release` 构建/测试预设，需要 Ninja；此轮交付仅实测 Windows。没有 Vulkan Validation Layer 时使用 `-DCAUSTRA_GPU_TESTS=OFF`，仍可手动运行 GPU 测试但不传 `--validation`。没有光追硬件时不能运行渲染，不提供 CPU 或光栅化回退。

## DLSS 降噪配置

DLSS 在构建时默认启用支持，运行时默认关闭。纯原生渲染构建可用 `cmake --preset windows -DCAUSTRA_ENABLE_DLSS=OFF`，不需要 NGX SDK 或 DLL。重新启用用 `-DCAUSTRA_ENABLE_DLSS=ON`；独立 SDK 路径用 `-DCAUSTRA_DLSS_ROOT=C:/SDK/DLSS`。显式请求 DLSS 而硬件、驱动、运行库不支持时会报错，避免把原始噪声图误认为降噪结果。

原生分辨率降噪示例：

```powershell
./build/bin/Release/caustra_demo.exe --width 960 --height 720 --samples 64 --dlss native --dlss-preset e --save-raw --output renders/dlss --validation --self-test
```

`--dlss` 接受 `off`、`native`、`quality`、`balanced`、`performance`。`native` 在输出分辨率做 RR；其他模式使用 NGX 查询的推荐内部分辨率，并降噪重建到 `--width` / `--height`。`--dlss-preset` 接受 `default`、`d`、`e`，对应本次接入 SDK 的 RR 模型预设。输出尺寸每边至少 32 像素。

DLSS 是时域降噪：每次采样提交一帧全新的 1 spp 噪声输入，`--samples 64` 在 DLSS 模式下代表 64 个时域帧。输入还包括法线/粗糙度、漫反射与高光反照率、线性深度、运动矢量和高光命中距离。DLSS 自己维护历史，原生路径的累积平均保留为独立输出。`--save-raw` 额外保存 `_raw.ppm` 与 `_raw.pfm`，其尺寸是实际内部分辨率。

库接口用 `setDlssSettings` 设置 `DlssSettings` 的开关、质量与模型预设；`dlssStatus` 返回当前状态。开启后 `readback` 返回降噪后的目标分辨率图，`readbackRaw` 返回内部原始累积图。设置失败保留此前可运行的配置。`setCamera` 表示相机切换，会重置累积和 DLSS 历史；连续移动时使用 `updateCamera`，它清空当前视角的原始累积，并通过当前/上一帧矩阵生成像素运动向量，保留 DLSS 历史。场景、分辨率和渲染配置变化仍完整重置。当前运动向量支持静态几何的相机移动；未来动态模型需要补充实例/顶点的上一帧变换。

当前开发用 NGX application ID 与原始示例一致，可通过 `CAUSTRA_DLSS_APPLICATION_ID` 覆盖。DLSS SDK 和运行库沿用 NVIDIA 自身的许可，不属于工程 Apache-2.0 代码。

## 接入自己的几何体与材质

公开 API 在 [include/caustra/renderer.hpp](include/caustra/renderer.hpp)。调用顺序：

```cpp
caustra::Renderer renderer;
renderer.setSettings(settings);
renderer.setScene(scene);       // 自己填充 vertices / indices / materials / instances
renderer.setCamera(camera);
renderer.render(64);            // 增加 64 samples per pixel
auto image = renderer.readback();
image.savePfm("result.pfm");
```

通过 `add_subdirectory` 引入此工程后链接 `caustra::caustra`。`examples/demo_scene.cpp` 演示如何构造三角网格、平滑法线、材质和实例变换；它属于示例，不属于渲染库。网格可以被多个实例共享。实例使用行主序 3×4 仿射矩阵，顶点索引属于各自网格。顶点位置以统一世界单位解释，颜色与发光均为线性 RGB。零法线会回退到三角形几何法线。

相机、场景或渲染设置变化会清空累积；连续调用 `render` 则续采样。`setScene` 是同步的完整场景上传，会重建 BLAS/TLAS，适合作为第一阶段明确的输入边界。后续增加动态更新、纹理或自定义材质时，参阅 [docs/architecture.md](docs/architecture.md)。

## 验证

CPU 测试检查非法索引、退化网格、矩阵、相机、材质、DLSS 配置、图像保存和环绕相机控制。GPU 集成测试检查渐进采样确定性、连续相机与相机切换、尺寸变化、非均匀缩放、无面积光场景和失败上传后的状态。`--validation` 开启 Vulkan Validation Layer，错误会使测试失败。在 NVIDIA RTX 环境配置 `-DCAUSTRA_DLSS_GPU_TESTS=ON` 可将 DLSS 实际执行、升采样、配置切换、运动历史和历史重置加入 CTest，并启用同步验证。

当前实现以清晰的核心接口和可运行性为优先：GPU 提交同步执行，输入/输出使用主机可见 coherent buffer，每次调用线程独占，不提供并发渲染。示例按米级空间构建，射线偏移包含固定绝对项，暂不支持微米级场景或极端巨大坐标。非常高亮的自发光和小面积强光会增加采样方差；没有降噪时低采样图像有 Monte Carlo 噪声。粗糙度在着色时设有有限下限，第一阶段不支持理想镜面、透射、体积、纹理、HDR 环境贴图或 glTF 扩展材质。

许可证与改造来源见 [NOTICE](NOTICE) 和 [third_party/nvshaders/ORIGIN.md](third_party/nvshaders/ORIGIN.md)。
