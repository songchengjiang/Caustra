# GPU 计时与射线查询统计

`caustra_benchmark` 保留原有 stdout CSV，同时可以输出逐帧性能 CSV 和汇总 JSON。
GPU 计时与射线统计分别开关，正常查看器和离线渲染默认均不启用。
首批完整场景数据见 [San Miguel GPU 性能报告](sanmiguel_profile.md)。

## 采集方式

```powershell
# 只加时间戳，使用正常路径追踪 Shader。
.\build\bin\Release\caustra_benchmark.exe --asset renders/sanmiguel.caustra --width 960 --height 640 --depths 1,6,24 --frames 20 --warmup 4 --dlss all --variants original --skip-invalid-media --render-only --gpu-timing --profile-output build/experiments/profiling/timing

# 相同轨迹，启用独立统计 Shader；它会改变寄存器和原子操作开销。
.\build\bin\Release\caustra_benchmark.exe --asset renders/sanmiguel.caustra --width 960 --height 640 --depths 1,6,24 --frames 20 --warmup 4 --dlss all --variants original --skip-invalid-media --render-only --ray-stats --profile-output build/experiments/profiling/counters
```

每个配置排除一帧资源初始化，重启相机和随机序列，然后预热指定帧数。
默认每帧向轨道相机传入 `dx=1.5`、`dy=0.2` 像素的固定步长，
按当前灵敏度对应 yaw -0.009 rad、pitch -0.0012 rad（俯仰角仍受范围限制）；
`--static` 可测固定视角。
`--dlss all` 依次测 Off、Quality、Performance，需要可用的 DLSS。
上述 `--render-only` 命令的显式 HDR 回读、有限且非黑图像检查和报告写盘
均在测量窗口外。省略 `--render-only` 时，原基准仍会逐帧测量 HDR 回读和
CPU 色调转换；最后的图像检查与写盘仍在计时外。

`--profile-output PREFIX` 生成 `PREFIX_frames.csv` 和 `PREFIX_summary.json`。
JSON 包含配置、相机、实际追踪/输出尺寸、场景规模、逐项分布、计数有效性及图像检查。
关闭统计时相应值为空/null，不能将其解释成零次查询；溢出的统计不进入汇总分布。

## 指标口径

- CPU render 时间包含同步提交、等待 GPU 完成、诊断和可选性能数据读取。
- GPU 时间戳记录初始化、路径追踪、DLSS 和总时间；DLSS 区间包括后端输入准备、
  NGX 执行和输出转换。无初始化或未开启 DLSS 时对应时间为零。
- 路径追踪仍是一个计算 dispatch；其时间同时包含硬件遍历、纹理、材质、采样及
  介质传输。时间戳不能给出独立的 BVH/RT Core、ALU 或显存瓶颈占比。
- `primary_ray_queries` / `indirect_ray_queries` / `shadow_ray_queries` /
  `auxiliary_ray_queries` 是 `TraceRayInline` 调用次数，透明拒绝后重试也计一次。
  四类查询互斥，但包含同一射线的重复查询，不等同于物理路径条数；
  `path_samples` 才是像素路径样本数。
- `boundary_continuation_queries` 是查询分类中的子集，不能再次加到总查询次数。
- `candidate_visits` 只统计 `Proceed()` 暴露给 Shader 的候选事件，不是硬件
  BVH 节点访问或三角形测试数量。强制 opaque 查询可以命中三角形而没有此类事件。
- `coverage_evaluations` 与 `full_material_evaluations` 记录积分器/介质通路发起的
  求值调用；前者回退完整图时也增加后者。它们不是 IR 指令或纹理采样次数。
- `alpha_rejected_hits` 记录主/间接路径拒绝的覆盖交点；`shadow_transparent_hits`
  记录阴影通过 opacity 小于 1 的普通表面，真实体积边界不计作普通透明覆盖。
- `surface_scatters`、`medium_scatters`、RR 终止及最大深度描述路径工作量。
  最大深度取 max，其他计数求和。
- 汇总中的 `invalid_boundaries` / `near_skips` 包含初始化首帧和预热，射线分布
  只包含测量帧，两个窗口不能直接相减。

## 核心接口与实现

`RendererOptions::enableGpuTiming` 和 `enableRayStatistics` 默认 `false`。
`Renderer::profile()` 返回最近一次成功非空 `render(count)` 的聚合数据；
`samples` 为本次 count，不是渐进累计数。`render(0)` 与显式回读保留结果，
场景、相机、设置或历史变更使结果失效。

GPU 时间戳使用同一计算队列，检查 `timestampValidBits` 和 `timestampPeriod`，
支持有效位回绕；单次测量必须短于一次计数周期。不可用设备报告无效状态。
时间戳方案依据 [Khronos 的说明](https://docs.vulkan.org/samples/latest/samples/api/timestamp_queries/README.html)。

射线统计使用独立的 `pathtrace_profile.spv`（`CAUSTRA_PROFILE_RAYS=1`）。
正常 Shader 完全预处理掉计数代码。Binding 12 前 20 个诊断 word 保持原位；
统计模式追加 1024 个分片，每片 18 个 uint 计数与一个溢出标记，每像素局部
累计后归并。CPU 使用 uint64 累计并检查溢出，不要求 GPU 64 位原子扩展。
计数只归并主路径追踪 dispatch，独立的初始化 dispatch 不计入射线统计。
默认仍只分配 80 字节诊断缓冲，资产格式、材质布局、push constants 不变。

统计会扰动吞吐和占用率，所以应使用时间戳模式判断正常帧耗时，使用统计模式
解释工作量，再单独比较插桩开销。要确定硬件遍历、指令或带宽限制，还需要
Nsight 等硬件性能分析器；本工具没有提供这些硬件计数器。

GPU 回归 `profiling_gpu` 使用全屏发光平面和两层完全透明平面验证精确查询/
覆盖计数、批次聚合、失效语义及默认/计时/插桩的 raw 像素一致性。
可选 `profiling_dlss_gpu` 另外验证 DLSS 分段和实际输入分辨率。
