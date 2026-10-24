# 渲染核心与扩展边界

`caustra` 静态库拥有 Vulkan 实例、硬件选择、设备、队列、内存、BLAS/TLAS、描述符、compute pipeline 和累积缓冲；可选 `src/dlss_backend.*` 封装 NVIDIA NGX Ray Reconstruction。示例只负责生成场景、设置相机、调度样本与写出图像。核心没有文件格式或应用 UI 的概念。

`caustra_viewer` 是独立 Win32 示例，程序化场景和环绕相机位于 `examples/viewer_scene.*`、`examples/orbit_camera.*`。窗口消息只在应用线程处理；渲染循环在同一线程调用同步 API。示例回读 HDR，通过查找表执行 Reinhard/sRGB 显示转换；图像、背景与文字先在 GDI 后台位图中合成，再一次性复制到窗口，避免逐帧清屏闪烁。后台 DC/位图复用，仅在窗口尺寸变化时重建。它没有 Vulkan swapchain 或 GPU 直连显示接口，因此窗口大小和 CPU 回读/显示开销会影响帧率。最小化后停止渲染，尺寸变化合并后重建输出，绘制消息只读取最近一帧缓存。

查看器输出尺寸与视口物理像素一致，不再用固定像素预算缩小大窗口。`WM_SIZE` 合并后调用 `setSettings`，按实际尺寸重建累积、guides、DLSS 输出和回读资源，并清除历史。DLSS 只影响追踪输入尺寸；界面及日志分别显示输入和输出。调整窗口过程中暂时拉伸旧帧，尺寸稳定并完成下一帧后按原生输出呈现。

输入 `Scene` 的网格顶点和三角形索引在上传时拼接到连续缓冲。顶点包含位置、法线、UV 和可选切线/手性；未给切线时按三角形 UV 导数建立 TBN。每个网格创建一个 BLAS，每个实例的 TLAS 自定义索引指向实例元数据；着色器通过元数据找到局部索引、顶点和材质。实例的非均匀缩放通过逆转置变换插值法线，镜像变换同时修正切线手性，几何法线从世界空间边向量构造。

发光网格的所有实例三角形会展开成世界空间面积光表。常量发光或材质程序包含发光输出都会加入候选光表；面积光采样点重新执行同一材质图，使用相同位置、UV、法线、切线和滤波足迹，支持纹理/程序发光。按照三角形面积 CDF 选择光源、均匀采样三角形表面，结合 BSDF 采样的 PDF 使用 balance MIS。环境为恒定 RGB 辐射，仅由 BSDF 路径采样。表面双面着色；发光可显式选择单面（USD 区域灯）或双面。有限路径深度后仍执行一次只收集发光/环境的终点射线，使直接光与 BSDF 两种采样配对。

`src/shader_types.hpp` 与 `shaders/pathtrace.slang`、`shaders/material_program.slang` 的数据布局逐项对应；GPU 数据以四分量 float/uint 对齐。Vertex 为 64 字节、Material 为 384 字节（23 个 float4 与 program uint4）、材质指令为 32 字节，主机静态断言布局。Binding 0 为 TLAS，1–7 为几何/材质/光表/输出与 DLSS guides，8 为相机 uniform，9–11 为材质指令、纹理元数据与线性 mip texel atlas，12 为 GPU 传输诊断标记，13 为相机介质缓存，14 为预计算介质表。修改材质或描述符时必须同时更新主机和着色器。构建重新编译并嵌入 SPIR-V；完整 NVIDIA BSDF 辅助源码仍保留，但主积分器现在通过独立表面闭包进行求值和采样。资产包写入版本为 2，C++/Python 读取器同时接受版本 1 并补齐新参数默认值。

`tools/caustra_assets` 在库外把 USD/MaterialX 转成可移植 `.caustra` 包；可选 `caustra_assets` 仅用标准 C++ 读包并调用 CPU 校验，输出 `Scene` 与相机。渲染库不链接 USD/MaterialX SDK、Python、图片解码器或文件解析器。添加逐帧更新时再引入局部 buffer 更新、BLAS/TLAS update 和显式同步接口。当前 `setScene` 完整重建，`setCamera` / `setSettings` / `reset` 清除采样计数，下一次 dispatch 直接覆盖累积缓冲。

所有公开方法由单个调用线程拥有。当前提交通过 fence 等待结束，上传失败时保留旧场景；这是简单的同步 API，没有隐含的资源跨帧生命周期。后续异步化时需要把上传资源、scratch buffer、AS 和描述符更新纳入调用方可观察的生命周期。

静态几何、材质、指令、纹理与光表通过 staging 批量上传到 device-local buffer，累积图和 DLSS guides 同样驻留显存。CPU 上传区优先 coherent，回读区优先 host-cached；非 coherent 后备类型显式 flush/invalidate。原始 HDR 只在 `readbackRaw` 时复制到回读区，同一次渲染的重复回读复用该副本；每次渲染、相机、场景或输出设置更新均使副本失效。DLSS 结果直接复制到缓存回读区。命令缓冲复用，公开同步契约保持不变。

## MaterialX 图与表面闭包

Blender USD 的 `mtlx` 材质上下文是主要资产路径。外围编译器通过官方 OpenUSD API 解析组合层、实例、材质子集、primvars、变换与相机，重建原生 MaterialX Document 并校验连接/类型，按标准库 NodeDef 解析默认值。六类表面（Standard Surface、OpenPBR、Disney Principled、glTF PBR、USD Preview Surface、Simple Hair）有独立模型标记；未知节点和未支持的有效表面功能明确失败，不回退到 Preview Surface 或固定 PBR 常量。具体支持范围见 `tools/caustra_assets/README.md`。

外围编译器把受支持的参数图转成最多 128 条 float4 GPU 指令，命中位置执行算术、空间输入、图像、法线贴图和程序噪声，再构建表面闭包。这是 Caustra 的受限图执行后端，未声称使用官方 Slang 生成器或支持任意 MaterialX closure。核心只认识通用指令/资源，调用方仍可直接提供常量材质。

Blender 常见的三指令默认切线图有独立快速路径：精确匹配读取切线、三分量归一化和写回切线的指令及引用，按原顺序执行两次归一化后返回，跳过通用寄存器与 footprint 数组。现有资产包无需重编译；其他图继续执行完整解释器。回归将其与加入无用常量的通用路径逐像素比较，并覆盖各向异性及相似指令变体。

`shaders/surface_bsdf.slang` 分离 `makeSurfaceClosure / configureSurfaceInterface / evaluateSurface / sampleSurface / surfaceGuides / surfaceEmission`。所有 lobe 共用确定的完整混合 PDF，采样权重为总 f*cos/pdf，NEE 与随机路径使用同一求值接口。Standard Surface 数学源自 Aurora 的 MaterialX 移植；OpenPBR 不透明分支按 MaterialX 原生图实现其 diffuse、F0 调制、色 82 金属 Fresnel、各向异性、coat-relative IOR、roughness 展宽与 coat darkening。粗糙 BTDF 使用可见法线 GGX、折射 Jacobian、全内反射和辐射传输的 eta 修正。薄膜与 OpenPBR Zeltner fuzz 移植 MaterialX 参考函数。`official_models.slang` 独立实现另外四类官方节点图；Chiang 毛发采用全方向散射闭包及匹配的归一化采样 PDF。glTF 的 OPAQUE/MASK/BLEND 与 USD 的 presence/transparent 模式由 `materialOpacity` 在求交、阴影和发光中统一解释，透明 BSDF 与随机覆盖率分开处理。实体介质边界要求完全覆盖，避免 alpha 洞与介质体积定义冲突。许可和改动记录分别位于 `third_party/aurora`、`third_party/materialx`。

`shaders/medium.slang` 定义均匀介质系数、RGB 通道混合自由程采样和 Henyey–Greenstein 相函数。通道提议与当前路径 RGB throughput 成比例，使用对应的完整混合碰撞/逃逸 PDF，减少强色差下的重尾方差。积分器按实例标识维护最多 16 层介质栈，反射保持介质、穿越实体边界时更新栈，薄壁表面不创建光学内部介质；与之独立的 volume 终端仍可创建无 IOR 跳变的内部体积。相机初始位于体积内部时先探测包含关系，嵌套区域以内层系数为准。仅有吸收时使用精确 Beer–Lambert 透射；有散射时采样真实内部碰撞，并执行相函数 NEE/MIS。体积发光沿线段精确积分，通过路径传播照明；NEE 光表仍只采样三角形光源。实体折射界面阻挡直线阴影连接，光线通过实际 BTDF 路径进入或离开，避免用透明阴影冒充折射。

无表面体积的穿越保持原始射线路径，并记录已处理的交点距离。续查过滤已经消费的同实例交点，保留只相差一个浮点间隔的真实出口。共享边上的相邻三角形可能重复报告同一朝向边界；积分器对这些相邻报告去重。同一实例在精确相同的 t 同时报告入口和出口时，视为没有内部距离的擦碰。若相反朝向的报告存在浮点重建偏差，只有确认两个非平行面共享对象空间的边或顶点、两个世界空间交点均在该接触处的重建误差范围内，才将其视为棱边擦碰，不改变介质栈。此判定同时覆盖无表面体积、实体玻璃和 SSS；实体接触在 IOR/BSDF 求值前处理，相机包含探测与阴影可见性使用相同规则。每次仅消费当前交点，后续报告会重新探测包含已消费面的邻域，避免跨过其他实例的真实边界；平行薄体积的入口和出口不按宽容差合并。散射位置从较近的线段端点计算，并以浮点间隔修正真实边界的内侧约束，避免大入射距离吞掉很短的自由程。介质入口、出口直接使用 Vulkan 对象空间的硬件命中朝向；TLAS 镜像无需再次翻转该标志，切线手性和单面光源的朝外法线仍按实例变换处理。

散射点的安全距离同时计入三角形重建、对象到世界及世界到对象变换的浮点误差，再投影到几何法线，参考 [NVIDIA 的自相交误差分析](https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/)。仅修正用于发射相位/阴影射线的位置：沿已求交线段向内部限幅，入口、出口两侧各最多占线段四分之一；自由程、PDF、透射权重和真实边界交点保持原值。世界空间内侧的一个浮点单位可能在逆变换中被平移抵消，不能作为充分的硬件求交安全距离。小随机数的指数自由程使用稳定展开，并将逆 CDF 输入限制在开区间，避免人为生成零自由程。只有一个浮点间隔厚度的纯吸收体仍保留真实光学厚度；厚度小于变换误差包络的散射体不保证存在可稳健表示的内部点。

每个场景先以 `counts.w` 的 bit 30 调度介质表初始化：执行材质图，将均匀内部系数与 RGB IOR 写入 Binding 14，每材质占 128 字节；二维 dispatch 覆盖超过单个工作组的材料列表。表面参数仍按命中求值，薄壁表面的空间变化 IOR 也不放进内部介质表。每条路径的介质栈只存 16 个实例 ID 与计数，逻辑大小从 1412 字节降至 68 字节；同材质的不同实例仍保留各自边界身份。

针孔相机的所有路径共享射线起点。Binding 13 的 80 字节缓存保存该起点的包含栈 ID；场景或相机位置改变后，以 `counts.w` 的 bit 31 调度一次单 invocation 探测。材质表、相机缓存和整帧路径之间均有 compute 屏障。相机原地旋转、FOV 和渲染设置变化复用包含缓存。bit 0 仍表示相机历史有效。色散英雄通道不缓存，仍按每条路径原有的随机序列独立选择。直接提供 IR 时，薄壁标记的最后一个字面量布尔输出可以确定最终状态，其他控制图保守按可能实体校验；前端可先将该常量标记折叠到 POD，以保留薄壁表面的空间参数。

SSS 使用内部随机游走：radius×scale 定义 RGB 平均自由程，消光系数为其倒数；subsurface_color 表示观测反照率，通过 [OpenPBR 规范](https://academysoftwarefoundation.github.io/OpenPBR/) 的 van de Hulst 近似反解单次散射反照率，并保留 HG 各向异性。该颜色标定近似忽略折射界面对多次散射反照率的影响，有限路径深度也会截断长路径，因此不声称与其他渲染器逐像素一致。传输与 SSS 权重混合内部系数，显式 volume 的系数再叠加。此实现保留几何厚度、多次散射和实际出射点。OpenPBR transmission_scatter 按 attenuation depth 换算散射，按规范将吸收的负分量整体平移到非负；depth 为零时不创建该内部消光，而在边界使用 transmission_color 着色。均匀体积系数在编译时验证为空间常量；空间 SSS/透射参数按下面的网格路径处理。体积必须闭合且方向一致，彼此嵌套或分离；栈溢出或不匹配会通过 binding 12 传回 CPU 并抛出异常，重置累计历史。

空间变化的 SSS/透射参数设置 `MaterialProgramFlag::HeterogeneousInterior`。每个实例按物体包围盒预计算 32³ 个单元的内部系数，追加在 Binding 14；`counts.w` bit 29 调度网格初始化。内部 `GpuInstance` 为 144 字节，包含网格基址、双向仿射变换和物体包围盒；公开 Material 保持 384 字节。路径与阴影用三维 DDA 逐格积累 RGB 光学厚度，碰撞/逃逸使用完整通道混合 PDF。对分段常量场积分精确，对原图是网格离散近似，场景修改时重建；不会冻结入口参数。体内 UV 默认为包围盒归一化 XY，`SphericalInteriorUv` 选择与生成球面一致的径向延拓。体内法线与切线由映射局部框架经实例变换得到。实体 IOR、色散和 thin-walled 状态仍须均匀。任意曲面 UV 到体内的延拓需调用方明确映射；资产编译器目前仍要求显式 volume 终端系数均匀。

位移位于资产工具层：通过同一 MaterialX 参数图编译路径生成标量/向量位移指令，在细分后的对象空间顶点上 CPU 求值，修改几何并重算法线，再写入资产包。标量沿法线位移，向量按 MaterialX 的 (dPdu,dPdv,N) 基展开，保留 UV 导数的尺度与手性；向量位移要求有效 UV。它会改变 BLAS、轮廓与阴影；不在渲染核心中引入细分器、USD 或文件解析器。

纹理由外围工具解码、按声明颜色空间转换成线性 RGBA 后嵌入包；核心对所有图像构建完整 mip 链并放入只读 buffer atlas。Shader 提供独立 U/V 寻址、nearest/bilinear/trilinear 和显式 mip LOD；主射线及延续用 ray cone 估算足迹，UV 算术传播保守宽度，避免 compute shader 使用隐式屏幕导数。位置驱动的纹理坐标暂没有位置 footprint，复杂空间映射的过滤是当前近似边界；checker/noise 没有解析抗锯齿。该 atlas 路径优先保证布局简单和可移植性，大型纹理集未来可替换成硬件纹理/描述符数组而保留材质前端。

DLSS guides 来自求值后的闭包，包含法线贴图和表面参数。各向异性及多层 coat 仍需压缩成单个 normal/roughness，这是 NGX guide 契约下的近似；specular-albedo 使用相同粗糙度约定。材质、纹理或图变化经 `setScene` 完整清空原始累积与 DLSS 历史。

## 可选 DLSS Ray Reconstruction

编译启用 NGX 时，设备创建前查询需要的 instance/device extensions，并启用 NGX 内部计算着色器需要的 storage-image 无格式读写能力。运行时首次显式开启 DLSS 才初始化 NGX、查询硬件支持、创建 RR feature 与目标图片；不支持时抛出明确错误。NGX 设备会话保持到 Renderer 销毁；关闭 DLSS 只释放 feature、参数与图片，重新开启时复用会话，避免在同一设备上反复关闭/初始化 NGX。调整配置时先构造新 feature，成功后替换旧状态。

着色器在保留原始渐进平均的同时输出独立当前帧颜色和辅助信息到 planar buffer。后端将其复制到格式明确的 Vulkan 图片，执行 NGX RR，然后回读目标分辨率结果。面积光、求交和 BSDF 算法保持在原路径中。图片的布局和 buffer/image 屏障由 backend 管理，NGX 调用后不假设原 pipeline 或描述符绑定仍有效。

相机采用跨像素共享的 Halton 抖动，每次调用推进一个噪声帧。NGX 接收相反符号的实际像素偏移、正线性 ViewZ、世界空间法线和行向量矩阵；相机侧常规列向量变换在 backend 转置。`sampleCount` 记录当前视角的原始累积，`frameCount` 记录自完整重置以来的时序帧，分别驱动平均与随机序列/抖动。

`setCamera` 和 `reset` 表示相机切换或显式清空，会丢弃时域历史；`updateCamera` 仅在相机实际变化时清空原始累积，保留 DLSS 历史。Binding 8 的 uniform buffer 同时提供当前和上一帧无抖动世界到裁剪变换；着色器在当前主命中位置计算当前到上一帧的输入像素位移，背景按方向重投影。上一帧相机仅在 GPU 提交成功后推进。场景、分辨率和 DLSS 配置更新继续完整重置。当前几何静态，动态实例/顶点扩展仍需补充上一帧模型数据。
