# 渲染核心与扩展边界

`caustra` 静态库拥有 Vulkan 实例、硬件选择、设备、队列、内存、BLAS/TLAS、描述符、compute pipeline 和累积缓冲；可选 `src/dlss_backend.*` 封装 NVIDIA NGX Ray Reconstruction。示例只负责生成场景、设置相机、调度样本与写出图像。核心没有文件格式或应用 UI 的概念。

输入 `Scene` 的网格顶点和三角形索引在上传时拼接到连续缓冲。每个网格创建一个 BLAS，每个实例的 TLAS 自定义索引指向实例元数据；着色器通过元数据找到局部索引、顶点和材质。实例的非均匀缩放通过逆转置变换插值法线，几何法线从世界空间边向量构造。

发光网格的所有实例三角形会展开成世界空间面积光表。按照三角形面积 CDF 选择光源、均匀采样三角形表面，结合 BSDF 采样的 PDF 使用 balance MIS。环境为恒定 RGB 辐射，仅由 BSDF 路径采样。面积光与表面目前都是双面。有限路径深度后仍执行一次只收集发光/环境的终点射线，使直接光与 BSDF 两种采样配对。

`src/shader_types.hpp` 与 `shaders/pathtrace.slang` 的数据布局逐项对应；GPU 数据仅由四分量 float/uint 组成。修改材质或描述符时必须同时更新主机和着色器。构建会重新编译并嵌入 SPIR-V。材质构造与 `bsdfSampleSimple` / `bsdfEvaluateSimple` 调用是第一阶段替换材质模型的入口；完整的 NVIDIA BSDF 辅助代码已保留在 vendored 文件中，但没有对外暴露 glTF 的导入结构。

后续添加模型加载时，在库外把文件转成 `Scene`；添加纹理时扩展材质参数、顶点 UV、描述符与命中后的材质求值；添加逐帧更新时再引入局部 buffer 更新、BLAS/TLAS update 和显式同步接口。当前 `setScene` 完整重建，`setCamera` / `setSettings` / `reset` 清除采样计数，下一次 dispatch 直接覆盖累积缓冲。

所有公开方法由单个调用线程拥有。当前提交通过 fence 等待结束，上传失败时保留旧场景；这是简单的同步 API，没有隐含的资源跨帧生命周期。后续异步化时需要把上传资源、scratch buffer、AS 和描述符更新纳入调用方可观察的生命周期。

## 可选 DLSS Ray Reconstruction

编译启用 NGX 时，设备创建前查询需要的 instance/device extensions，并启用 NGX 内部计算着色器需要的 storage-image 无格式读写能力。运行时显式开启 DLSS 才初始化 NGX、查询硬件支持、创建 RR feature 与目标图片；不支持时抛出明确错误。NGX 生命周期按设备计数，允许调整配置时先构造新 feature，成功后替换旧状态。

着色器在保留原始渐进平均的同时输出独立当前帧颜色和辅助信息到 planar buffer。后端将其复制到格式明确的 Vulkan 图片，执行 NGX RR，然后回读目标分辨率结果。面积光、求交和 BSDF 算法保持在原路径中。图片的布局和 buffer/image 屏障由 backend 管理，NGX 调用后不假设原 pipeline 或描述符绑定仍有效。

相机采用跨像素共享的 Halton 抖动，每次调用推进一个噪声帧。NGX 接收相反符号的实际像素偏移、正线性 ViewZ、世界空间法线和行向量矩阵；相机侧常规列向量变换在 backend 转置。任何相机、场景和设置变化完整重置历史，因此零运动矢量与当前静态接口一致。动态实例/相机扩展需要补充上一帧数据，不能沿用静态零运动约定而保留历史。
