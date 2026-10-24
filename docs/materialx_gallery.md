# MaterialX 官方示例画廊

画廊使用 MaterialX **v1.39.5** 的 `resources/Materials/Examples`：50 个原始 `.mtlx` 文件、62 个独立材质，加上两个 Look 文件中的 4 次材质复用，共 **66 个展示样本**。棋盘示例里的 15 个材质分别展示。六个系列为 Standard Surface、OpenPBR、USD Preview Surface、glTF PBR、Disney Principled 和 Simple Hair。

场景内的 UV 球、基座、字体几何、毛发薄带、面光源和相机都由脚本生成。默认按材质系列顺序连续排列为 **11 列 × 6 行**，没有空位。每个基座前方带有朝斜上方的暗底白字标签，分两行显示编号/系列缩写和短名称，JSON 清单提供完整源文件名、节点名、材质索引与 SHA-256。构建要求全部材质成功；任何活动节点不受支持都会使整个构建失败，不会静默跳过或替换成常量 PBR。

## 构建和运行

安装 [README](../README.md) 所述的独立资产工具环境后，在项目根目录运行：

```powershell
# 下载固定版本，验证归档 SHA-256，并保留上游许可证。
.\build\asset-tools-env\Scripts\python.exe tools/fetch_materialx_examples.py

# 生成完整画廊：.caustra、.usda、.json 和原始纹理副本。
.\build\asset-tools-env\Scripts\python.exe tools/build_materialx_gallery.py --source-root build/deps/materialx-official/MaterialX-1.39.5 --output renders/materialx_official

cmake --preset windows
cmake --build --preset windows-release
.\build\bin\Release\caustra_viewer.exe --asset renders/materialx_official.caustra --bounces 24

# 离线渲染使用资产中的相机；不加 --dlss 则输出原生路径追踪结果。
.\build\bin\Release\caustra_demo.exe --asset renders/materialx_official.caustra --width 2160 --height 1440 --samples 256 --bounces 24 --output renders/materialx_official_gallery --validation --self-test
```

查看器可以旋转、平移和缩放，操作与其他场景相同。单个窗口中的总体预览包含全部 66 个样本；放大或离线高分辨率出图可检查细节与名称。

## 输出与可编辑源文件

| 文件 | 内容 |
| --- | --- |
| `renders/materialx_official.caustra` | 自包含可运行资产包，含全部几何、材质程序和线性纹理；运行时不需要 USD/MaterialX SDK。 |
| `renders/materialx_official.usda` | 可编辑的 USD 场景，保留实际 MaterialX NodeDef 网络、输入、连接与绑定。 |
| `renders/materialx_official_resources/` | USD 使用的原始分辨率纹理，以及 Simple Hair 的本地 MaterialX NodeDef/实现图；这些运行资源按内容哈希命名。 |
| `renders/materialx_official_resources/sources/` | 原始 50 个 `.mtlx` 与引用的原始纹理，保留上游相对目录，便于检查原图与 XInclude。 |
| `renders/materialx_official_resources/LICENSES/` | 上游 LICENSE、THIRD-PARTY 记录与可用的资源 README/许可证；`SOURCE.txt` 记录固定版本与来源。 |
| `renders/materialx_official.json` | 66 个样本的完整索引、来源、校验值、程序与纹理尺寸记录。 |

USD 的本地 MaterialX 定义通过标准 `sourceAsset` / `sourceAssetSubIdentifier` 记录。资产编译器读取这些真实定义；未知且未提供定义的着色器仍明确报错。导入工具和 SDK 依赖均留在渲染器之外。

## 测试几何与精度约定

包内纹理默认将最长边限制到 **512 像素**，在线性颜色空间缩小，以控制完整画廊的 CPU 内存和 GPU 显存。材质参数、节点连接与程序不因此删减，USD 资源副本保留原始分辨率。清单逐张记录原始尺寸和包内尺寸。需要原分辨率包时，构建命令增加 `--texture-max-size 0`；也可指定 `1024` 等预算。

一般材质使用半径 0.84 的封闭 UV 球，物体坐标原点在球心。OpenPBR Lightbulb 保留上游发光参数，展示实例均匀缩放到 0.15（世界半径 0.126），由不发光的细灯杆支撑，避免把高辐亮度灯泡变成巨大的区域光源。实例缩放、标签和灯杆参数记录在 JSON 清单内。

球体的 `u = wrap(atan2(z,x)/(2π))`，`v = 1-acos(y/length(position))/π`，纵向轴为 Y。空间变化的 SSS 材质保留原始材质图，并通过 **32³ 个物体空间网格单元**预计算内部参数；传输逐格经过这个分段常量场。这是离散近似。球面 UV 沿径向延拓到内部，不把入口采样值冻结为整块体积的值。USD 中的 `caustra:sphericalInteriorUV` 明确记录这一约定；同一材质绑定混用不同内部 UV 映射会报错。

Simple Hair 使用 45 根不同朝向、带弯曲的窄薄带，法线表示纤维径向，切线和 UV 的 U 方向沿毛发纵向。Chiang 毛发模型本身已积分纤维横截面的 R、TT 和 TRT 散射，因此使用单层几何，避免在封闭圆柱的背面再次计算同一次穿透事件。薄带不是体积边界，不进入实体介质栈。

生成器可用 `--columns` 调整布局、`--segments` / `--rings` 调整球网格，`--displacement-subdivisions` 调整离线位移细分。具体参数见 `--help`。这些是官方示例在 Caustra 后端中的兼容性与视觉检查样本，不声明与其他 MaterialX 后端逐像素相同。

## 来源与许可证

原始资源来自 [MaterialX v1.39.5](https://github.com/AcademySoftwareFoundation/MaterialX/tree/v1.39.5/resources/Materials/Examples)，归档下载脚本固定版本并校验完整 SHA-256。上游文件及资源各自的许可证和贡献者记录保留在下载目录；现有 MaterialX 数学移植记录保存在 [`third_party/materialx`](../third_party/materialx/)。原始 `.mtlx` 文件不被改写，输出 USD 记录了对应源文件，生成的材质程序来自真实节点图。

## 验证

固定清单 [`tests/assets/materialx_official_inventory.json`](../tests/assets/materialx_official_inventory.json) 独立记录 50 个原文件的 SHA-256 和全部 66 个材质入口，用于检查遗漏。`gallery_tests.py` 还验证球面封闭性、法线与切线、UV、毛发薄带、标签和灯泡几何、本地 NodeDef 的 USD 往返和内部 UV 映射冲突。资产编译器 CPU 回归 40 项、画廊 CPU 回归 13 项均通过。

完整构建的 26 项 CTest 回归已通过，包括新增材质的解析白炉参照、HSV/HexImage、异质 Beer/发光/SSS、官方场景出图、512 帧官方画廊 DLSS 旋转与原综合场景 2048 帧旋转。关闭资产读取库、窗口和 DLSS 的最小配置也通过 7 项测试。全部 GPU 验证使用 NVIDIA GeForce RTX 5060 和 Vulkan 验证层。

最终 512 像素纹理包另行完成 512 帧 DLSS Quality 自动旋转：960×640 输出、640×427 内部分辨率、24 次散射，Vulkan 错误为 0。该次运行约 45 FPS，帧工作中位数 21.77 ms；这是这台 RTX 5060 上的实测结果。

完整 USD 重新编译后，全部 66 个材质的参数与 632 条图指令一致；图像按内容去重后，线性纹理像素逐字节一致。图像数量可因重复资源去重而改变，比较时使用内容而非纹理索引。

```powershell
ctest --preset windows-release --output-on-failure
```

下载官方资源后重新执行 CMake 配置，完整画廊测试会自动加入 CTest；配置过程和测试本身不会联网下载。可通过 `CAUSTRA_MATERIALX_EXAMPLES` 指定另一份同版本源码。没有官方源码时，普通核心与小型资产回归仍可运行。
