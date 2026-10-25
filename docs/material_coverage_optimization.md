# 透明覆盖求值优化（2026-10-25）

保留 San Miguel 的全部几何、材质图、纹理过滤、透明覆盖、真实介质和采样设置，将透明拒绝与普通阴影所需的覆盖率求值从完整着色中拆出。官方 MaterialX ShaderGen 数学及资产格式保持原样。

## 实测

RTX 5060、Windows Release、960×640 输出、DLSS Quality（640×427 追踪）、每帧 1 spp。各配置重启同一相机与随机序列，额外初始化一帧，再预热 4 帧，统计 20 帧中位数。上传、初始化、导出不计时；数据是包含 GPU 完成和 DLSS 的同步 API 墙钟时间。

| 最大反弹数 | 优化前 | 优化后 | 提速 |
| --- | ---: | ---: | ---: |
| 1 | 75.184ms | 23.660ms | 3.18× |
| 6 | 263.991ms | 69.528ms | 3.80× |
| 24 | 268.309ms | 69.234ms | 3.88× |

24 次反弹的渲染吞吐由约 3.73 提高到 14.44 帧/秒。这不包含窗口限帧和呈现开销，也不代表所有视角。

原始数据：[优化前](benchmarks/sanmiguel_coverage_before.csv)、[优化后](benchmarks/sanmiguel_coverage_after.csv)。此前[禁用透明覆盖的诊断](sanmiguel_performance.md)仅用于定位；本表始终使用完整原资产。

## 图像与语义

对 1、6、24 次反弹的同一末帧，分别保存 640×427 原始线性 HDR 和 960×640 DLSS 输出，共六张 PFM。前后全部逐像素一致：最大绝对差、RMSE、变化像素数均为零，见[比较结果](benchmarks/sanmiguel_coverage_comparison.json)。这验证该固定轨迹和序列，其他覆盖模式由程序化回归补充。

新增 `coverage_gpu` 对比优化开关，覆盖多层主射线/阴影透明度、UV/mip 过滤、法线依赖、glTF/USD 模式及图驱动阈值、复杂图、HexImage、真实体积和运动后的随机序列、样本计数。`material_coverage_cpu` 的 160 项检查覆盖依赖切片、最后写入、纹理/参数 ID、连续配置寄存器和容量边界。

## 实现

- 上传已验证的 Scene 时，从原 SSA 按最后一次透明度及原生模式/阈值写入反向提取依赖，保留原顺序、运算和 footprint。子图仅存于私有 GPU 缓冲，原 Material 和原指令不变。
- 子图使用 8/16/32 寄存器容量；超过容量或无法安全重映射时继续完整求值。完整着色图不超过 32 条时采用 32 容量，其余为 128。所有容量共用同一套运算代码。
- 主路径先求覆盖率，只有接受的表面再求完整材质。随机数消耗、接受规则、透明交点推进保持不变。
- 普通阴影只求覆盖率；只有初始化表证明无内部介质且未设置异构标志时才使用此路径。真实介质及异构介质保留完整求值和原传输逻辑。
- 保留数组初始化。取消清零的等价实验反而变慢（相同小图容量下约 136ms/帧），单一 32 容量方案约 105ms/帧，均未采用。具体硬件原因尚未测量，不能把源码操作减少直接当作性能收益。

## 验证与启动成本

Windows Release 构建、11 项 CPU 回归通过。最终库的 11 项 GPU 测试本体通过，覆盖材质、表面、介质、边界恢复、偏移、发光采样及 Vulkan/DLSS 互操作；本机使用临时单进程串行 runner 复用驱动编译缓存，测试正文保持原样。

`dlss_rr_gpu` 按原来的独立进程方式通过；它在临时聚合进程中曾遇到 NGX 重复初始化错误，没有通过修改渲染器绕过。`viewer_dlss_window` 的旋转、平移、缩放、尺寸变化、暂停、GPU 色调转换和 DLSS 两轮开关检查通过，热缓存重测 75.94 秒正常退出；首次冷测试的断言虽已完成，但整体触及原 180 秒限时，未计作通过。累计验证范围为 11 项 CPU 和 13 项 GPU/GUI 测试，未声称完整 CTest 套件全部重跑。

完整 San Miguel 查看器另以 960×640、DLSS Quality、24 次反弹自动旋转 60 帧，正常结束，Vulkan 验证错误为零，但末帧截图为黑色，不能将该次窗口 FPS 作为性能收益。窗口按墙钟时间旋转，与上方固定步长基准不同；上表只使用有原始 HDR 对照的固定轨迹结果。

随后对完整 San Miguel 执行带有限且非黑 HDR 断言的 `--smoke-test`，24 帧正常结束，旋转/平移/缩放/尺寸变化/暂停/两轮 DLSS 开关及 GPU 色调转换参考检查全部通过，Vulkan 错误为零。末帧截图为 `build/experiments/material-coverage/after/viewer-verified.bmp`。自动环绕末帧黑色的具体视角原因未在本轮追查，未围绕该视角增加代码补丁。

新版 Shader 的驱动冷编译在本次实验中约 111 秒；相同代码命中缓存后，最终复测初始化为 0.869 秒。两者均不计入帧耗时。首次管线编译成本仍有优化空间。

后续退出定位发现，同一逻辑设备同时负责复杂计算与窗口呈现时，`vkDestroyDevice` 可能等待驱动后台工作约 77 秒；只隐藏窗口没有解决进程退出。查看器现将计算和呈现拆到两个逻辑设备，通过 GPU 共享 Buffer 传递 HDR，原路径追踪 SPIR-V 保持不变。严格回归由父进程确认 5 秒内真正退出，具体证据见[查看器退出](viewer_shutdown.md)。

## 复现

```powershell
cmake --preset windows
cmake --build --preset windows-release
.\build\bin\Release\caustra_benchmark.exe --asset renders/sanmiguel.caustra --width 960 --height 640 --depths 1,6,24 --frames 20 --warmup 4 --dlss quality --variants original --skip-invalid-media --render-only --output build/experiments/coverage/image
ctest --preset windows-release --parallel 1 -R "material_coverage_cpu|coverage_gpu"
```

`--output` 仅在计时外保存末帧 PFM/PPM 和原始 HDR。普通查看器保持 GPU 直接呈现。`RendererOptions::optimizeMaterialCoverage` 默认开启，设为 false 可对照完整材质求值路径。
