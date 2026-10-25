# San Miguel：PBRT → USD / MaterialX → Caustra

源模型由 **Guillermo M. Leal Llaguno** 创建，取自 [PBRT 官方场景库](https://github.com/mmp/pbrt-v4-scenes/tree/30cf4a0346ae5a80a2d7a530a3ef7d0fa4f70572/sanmiguel)。下载固定到提交 `30cf4a0346ae5a80a2d7a530a3ef7d0fa4f70572`，只获取 San Miguel 及上游 README，不下载其他大型场景。每个文件都校验 Git blob hash，另写入 SHA-256 清单。源模型/纹理沿用上游分发条件，不以转换工具的代码许可重新授权。

转换器属于外围资产工具。渲染核心没有新增 PBRT、USD、MaterialX、OpenEXR 或 Python 依赖。

## 重建与运行

```powershell
# 如尚未创建资产工具环境：
python -m venv build/asset-tools-env
.\build\asset-tools-env\Scripts\python.exe -m pip install -r requirements-sanmiguel.txt
.\build\asset-tools-env\Scripts\python.exe tools/fetch_sanmiguel.py
.\build\asset-tools-env\Scripts\python.exe tools/convert_sanmiguel.py
.\build\asset-tools-env\Scripts\python.exe tools/compile_asset.py renders/sanmiguel/scene.usdc renders/sanmiguel.caustra --texture-max-size 512
.\build\bin\Release\caustra_viewer.exe --asset renders/sanmiguel.caustra --bounces 24 --dlss quality
```

离线出图：

```powershell
.\build\bin\Release\caustra_demo.exe --asset renders/sanmiguel.caustra --width 960 --height 660 --samples 256 --bounces 24 --output renders/sanmiguel_preview --validation --self-test
```

首次下载约 435 MiB；转换生成一个包含约 16 万个原生实例的完整场景，USD 写出和运行包编译需要一些时间。源数据与生成资产位于忽略版本控制的 `build/` 和 `renders/` 中，可由这些命令重建。

输出文件：

- `renders/sanmiguel/scene.usdc`：几何、原生 USD 实例、相机、内嵌 `mtlx` 材质网络、天空网格。
- `renders/sanmiguel/materials.mtlx`：可编辑的完整 MaterialX 图。
- `renders/sanmiguel/textures/`：源纹理副本，保留原分辨率；`derived_normals/` 为离线凹凸转换产物。
- `renders/sanmiguel/conversion_report.json`：源文件 hash、几何统计、参数、逐材质映射、拓扑检查和所有已知近似。
- `renders/sanmiguel.caustra`：自包含运行包，运行时不再依赖 USD、MaterialX 文件或外部图片。
- `build/deps/pbrt-v4-scenes/sanmiguel-source.json`：固定版本的下载及校验清单。

`--texture-max-size 512` 只限制运行包纹理，使用线性空间平均过滤，不修改 USD 的源纹理。设为 `1024` 可提高纹理细节，显存需求也会增长；省略或设 `0` 保留原尺寸。

## 保留内容与适配边界

主入口为 `sanmiguel-courtyard-second.pbrt`。读取 30 个 PBRT 文件、808 个 PLY，保留 372 个直接形状、436 个对象定义和 159,924 个对象实例。原始 2,503,044 个三角形中有 6,936 个精确零面积三角形，转换时明确剔除并按文件记录。完整位置、法线和 UV 的逐位相同顶点合并为 1,797,495 个，保留法线和 UV 接缝。USD 导入和 C++ 场景上传都复用几何及局部包围盒，避免按实例展开顶点。

| 原 PBRT 材质 | 转换方式 |
| --- | --- |
| `diffuse` | MaterialX Standard Surface 漫反射，关闭额外高光 |
| `coateddiffuse` | Standard Surface 的介质高光和漫反射；保留 IOR、粗糙度，记录层模型差异 |
| `diffusetransmission` | 薄壁双向 Lambert 反射/透射，保留反射和透射纹理权重 |
| `dielectric` | 闭合、无已检出自交且空间关系有效的几何使用实体折射；开放/非流形边界、自交原型及确认穿插的具体实例明确转为薄壁透射 |

开放玻璃/水面不能作为 Caustra 的闭合实体介质边界：`Vidrio`、`Agua`、`Fake Foco`、`Plastico Cubiertos`、`VidrioSalero` 使用薄壁适配，原始 PBRT 类型和具体拓扑原因保留在报告中。全部 6 个拓扑闭合的源玻璃网格另做非邻接三角形自交检查，确认 `Candil Foco` 的底盖自交；20 个该原型实例使用明确的薄壁近似。报告保留通用检测产生的三角形交叉证据。

剩余 200 个闭合玻璃放置经过 63 对 AABB 候选筛选，严格三角形检测确认 16 对、共 32 个大/小酒杯实例的杯壁互相穿插。转换器仅对这 32 个实例绑定派生薄壁材质，其余 168 个闭合玻璃放置保留实体折射；不因包围盒相交或有效嵌套而修改材质。薄壁水面不表示有深度的吸收水体。

PBRT 的 `displacement` 在这些材质中表示着色法线凹凸，而非真实几何位移。转换器保留原始高度图，使用每材质面积加权的几何 UV 导数尺度烘焙切线法线；它是近似，不能精确表达同一材质在不同曲面、UV 尺度及实例缩放下的全部导数。报告包含各材质导数范围、差分和贴图过滤适配。未知活动节点/参数会报错，不会静默替换为固定 PBR 常量。

## HDR 天空

官方 `sky.exr` 为 512×512 的 PBRT v4 等面积方形映射，带 ACES AP0 色度坐标。离线工具按源白点作 Bradford 适应并转换到线性 Rec.709，保留源灯光 `scale=13` 及 Z 轴旋转。

当前核心没有原生 HDR 无限环境灯，因此转换为有限距离、向内发光的多面体天空。基础网格为 64×64，高亮区域细分到原像素；颜色分组在等面积参数域保持 RGB 积分，天空本身的角向分段、有限距离和多面体形状仍是近似。原始 EXR 一同保存。天空可见并参与照明，但不撑大查看器的模型包围盒和旋转中心。

核心面积光分布按面积和基础发光亮度构造，使用精确 24 位整数 CDF；每个光源三角形至少有一个采样 ticket。直接光与 BSDF 命中光源使用相同的实际概率和 MIS，太阳与微弱发光面都保留采样机会。图驱动发光没有正的基础值时使用正权重提议，实际发光仍在采样位置求值。

相机保留完整胶片视角。原 PBRT crop window、ISO 和无限远景深焦距写入元数据；当前运行时展示完整胶片，不应用非对称裁剪，也不把该焦距当作旋转半径。源 PBRT 与转换后的 RGB 材质、照明及相机处理存在上述差异，因此生成图不是逐像素一致性证明。

## 验证

收敛清理后的 Windows Release 构建通过。重跑资产编译及 PBRT/实例的 5 项 CPU 测试、面积光采样/表面误差界/基础渲染/介质的 4 项 GPU 测试，全部通过；480×330、32 spp、24 次反弹再次实际出图。预览已附在 README，线性 HDR 保存为 `renders/sanmiguel_initial.pfm`。这些结果不表示完整场景在所有视角均稳定。

`pbrt_scene_cpu`、`pbrt_material_cpu`、`pbrt_environment_cpu` 和 `usd_instance_cpu` 覆盖解析、变换、PLY 属性接缝、重名定义、材质语义、纹理颜色空间/透明度、天空能量和原生实例共享。`emitter_distribution_cpu` 检查极端明暗光源不会从 CDF 消失；`emitter_sampling_gpu` 通过独立 Lambert 面积积分及光源拆分/重排检查采样和 MIS。

实际场景还暴露了渲染器旧表面偏移跨过合法薄杯壁的问题：约 441 微米的偏移越过约 321 微米厚的玻璃。核心已改用三角形和实例变换的浮点误差界，保留这些闭合玻璃的实体折射。`surface_offset_gpu` 对合法空心玻璃、平移/旋转、大坐标和 NEE 端点附近的真实遮挡做回归验证。

### 尚未解决的渲染限制

资产转换已完成，但完整场景的实体玻璃尚未通过所有分辨率、采样和旋转视角的稳定性验证。已有 480×330、32 spp、24 次反弹的实际出图；此前严格模式的 960×660 测试仍会触发 `Invalid medium boundary nesting`。这不能归因于所有源玻璃模型都无效。

已确认的通用数值问题包括：约 45.7 微米的真实玻璃段可能被约 69.6 微米的接触窗口误判为擦边；杯底厚度自然收敛到零的合法锐边上，约 3 微米的真实路径可能被约 8 微米的数值偏移越过。进一步尝试的交点排序、起点回退和邻面过滤尚未形成完整通用方案，已从核心撤回；不按实例 ID 放宽边界检查。

这些限制需要后续统一设计射线起点、边界事件排序与最小可分辨特征约定。当前保留源实体材质，并在此记录问题，不继续围绕个别失败射线追加处理。

查看器现默认使用通用介质容错，`--medium-tolerance 0.001` 表示 1 毫米：仅对已经不匹配的近边界按零距离接触跳过；其余不匹配终止当前路径或阴影连接，窗口继续渲染并显示诊断计数。正常匹配的薄玻璃不会因该阈值被过滤。使用 `--strict-media` 可恢复严格检查；离线示例和核心 API 仍默认严格。这一策略可能丢失局部辐射贡献，没有修复上述玻璃传输的正确性问题，也不代表本场景所有视角已验证稳定。

通用容错改动的 Windows Release 构建及 12 项相关回归通过，包括介质恢复、合法薄壳、原有严格介质检查、DLSS 和窗口开关。完整 San Miguel 在 RTX 5060 上以 960×640 输出、24 次反弹、DLSS Quality、默认 1 毫米容差自动旋转 240 帧并正常结束：近边界跳过 257 次，其余异常边界 0 次，Vulkan 验证错误 0。测试截图保存于 `build/experiments/medium-recovery/sanmiguel.bmp`；这是该次运行的稳定性验证，不是所有视角或物理正确性的证明。
