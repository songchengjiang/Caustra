# 官方 MaterialX 着色代码接入

六类表面使用 MaterialX **1.39.5** 的官方 Slang ShaderGen：Standard Surface、OpenPBR、Disney Principled、glTF PBR、USD Preview Surface 和 Simple Hair。`tools/generate_materialx_surfaces.py` 读取官方 NodeDef / NodeGraph，在副本中将 surface 终端改为 BSDF / EDF 输出，让官方生成器展开材质层叠和节点计算。原始库和示例文件不被改写。

生成的六个模型、分发入口及来源清单位于 `shaders/generated/`。清单记录全部源文件 SHA-256、产物 SHA-256、模型标记和每个输入的 ABI 表达式。普通 C++ 构建直接编译这些源码，无需安装 Python 或 MaterialX SDK；运行时也不需要这些依赖。

## 运行边界

已有资产继续兼容：参数和纹理由原有资产编译器转换为最多 128 条 GPU 指令；命中时执行该参数图，再调用官方生成的参数化表面函数。本次接入替换表面模型内部的手写图求值，未把全部参数图改为逐材质的专用 shader，也未增加任意 closure 支持。

- 普通表面反射和全方向毛发散射调用 `mxGeneratedReflection`；全部发光调用 `mxGeneratedEmission`。
- Disney 和 Simple Hair 的原始表面图没有 EDF；为保持通用 `Material.emission` 接口，它们显式接入官方 `uniform_edf` 原语。来源清单将此标为 Caustra 发光适配，未把它描述成原模型自带功能。
- 原始命中法线、coat 法线和切线单独保留。采样提议所需的旋转和框架修正不改变官方函数的输入。
- Standard Surface 默认由同一 authored tangent 驱动 base/coat。通用 IR 的显式 `CoatTangent` 输出仍可独立覆盖 coat；生成图只提升该旋转节点的输入，保留官方旋转和归一化。这项既有 API 扩展也记录在来源清单中。
- 核心继续提供采样分布与 PDF；非 delta 样本用当前求值结果重新计算 `f*cos/pdf`，与 NEE 使用同一接口。
- 真实折射、内部介质界面、薄片和 SSS 随机游走使用 Caustra 的物理传输适配。官方硬件生成器的环境折射、屏幕曲率 SSS 不承担这些积分；生成源码中的曲率 SSS helper 明确置零，具有内部传输的材质不会走该局部反射近似。
- DLSS 法线、粗糙度和反照率 guides 仍是核心对多层材质的压缩近似。

适配还包括命名空间/宏隔离、去除光栅入口和环境资源、使用解析方向反照率、GGX 粗糙度有限下限 0.02、OpenPBR 薄膜 nm→µm，以及已折叠发光强度的逆映射。USD Preview Surface 的切线法线已由参数图转换为世界法线，生成桥接不再重复转换。全部修改见生成清单和 [`ORIGIN.md`](../third_party/materialx/ORIGIN.md)。

Simple Hair 保留官方生成公式，包括上游 TRRT+ 方位密度的 `1 / 2 * pi` 表达式。此前手写求值器的 `1 / (2*pi)` 修正不应用于此后端。这是一项已知上游数学差异，不能把“使用官方代码”等同于“所有公式均已被独立证明正确”。

## 再生成与验证

```powershell
.\build\asset-tools-env\Scripts\python.exe tools/fetch_materialx_examples.py
.\build\asset-tools-env\Scripts\python.exe tools/generate_materialx_surfaces.py
# 只验证来源与产物，不修改文件
.\build\asset-tools-env\Scripts\python.exe tools/generate_materialx_surfaces.py --check
cmake --preset windows
cmake --build --preset windows-release
ctest --preset windows-release --output-on-failure
```

`--source-root` 可指定另一份相同版本的源码，`--output` 可生成到独立目录。没有源码或版本不一致时再生成明确失败；普通渲染构建使用已记录的产物。

`shadergen_source_cpu` 检查版本、来源/产物哈希、可复现生成和接口。它还在临时副本中修改官方 Disney 图的镜面常量，确认该模型的生成结果随源图变化，其他模型保持不变。

`generated_material_gpu` 用独立 compute 程序比较运行时闭包和直接调用生成函数的结果，覆盖六类模型、粗糙度、角度、各向异性、coat、不同法线和非零旋转，并验证非 delta 样本权重；193 组案例另包含独立 Disney Burley 和 coated EDF 解析参照。该测试证明路由和输入保真，不是另一个渲染器的独立外观参照。既有白炉、Beer、TIR、介质边界、完整画廊和 DLSS 回归继续保留。

倾斜 base/coat 白炉的积分参照也可重新计算：

```powershell
.\build\Release\caustra_generated_material_tests.exe --spirv build/generated/materialx_probe.spv --integrate-tilted --integration-count 262144 --validation
```

该模式均匀覆盖几何上半球，直接调用官方函数，绕过生产闭包和随机采样器。65,536 与 262,144 个方向的 RGB 积分每通道差小于 0.000005；新参考为 `(0.435242762593, 0.342283699616, 0.202845129575)`。这解释了旧手写材质参考的变化，同时保留原采样误差阈值。

当前仍未完成全部 66 个样本在独立渲染后端下的同条件线性 HDR 外观验收。

## 本次验证与性能

Windows Release 的完整 28 项 CTest 已通过。无资产读取库、窗口和 DLSS 的最小构建也已完成 CPU 检查与实际出图；两种构建使用相同 SHA-256 的 GPU 模块。README 的 Blender、综合特性和完整官方画廊三张图片已用该后端重新渲染，全部关闭 DLSS 并通过 GPU 自检。

RTX 5060、综合特性场景、960×640 输出、24 次散射、每帧 1 spp、固定旋转轨迹；剔除 3 帧预热，取 12 帧中位数。以下不启用验证层，不包含窗口绘制、初始化或资产上传：

| 模式 | 渲染/降噪 | 回读与显示转换后的整帧 | 等效 FPS |
| --- | ---: | ---: | ---: |
| 原生 Path Trace | 16.200 ms | 20.991 ms | 47.6 |
| DLSS Quality | 10.420 ms | 14.018 ms | 71.3 |
| DLSS Performance | 6.555 ms | 10.059 ms | 99.4 |

```powershell
.\build\bin\Release\caustra_benchmark.exe --asset renders/features.caustra --width 960 --height 640 --bounces 24 --frames 12 --warmup 3 --dlss all
```

本机首次运行该程序、启用验证层时初始化约 26.4 秒；随后同程序关闭验证层启动约 0.87 秒。首次管线编译、驱动缓存和验证层会影响启动耗时，这些时间没有计入上述帧时间。
