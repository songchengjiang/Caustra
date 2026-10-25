# Einar 官方角色兼容性测试

测试日期：2026-10-25。结论：**原始材质的完整转换未通过，没有交付可运行的
Einar `.caustra` 包，也没有运行 Caustra 出图或帧率测试。**

## 来源与许可

- 官方资源：[Einar v1](https://studio.blender.org/characters/einar/v1/)，2022-12-14，
  发布者 Demeter Dzadik，角色来自 Blender Studio 的《Charge》。官方要求 Blender 3.5 或更新版本。
- [官方公开下载](https://studio.blender.org/download-source/files/41/419e83fb75f30b989adf7920dec7a21c/419e83fb75f30b989adf7920dec7a21c.zip)。
- 许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
- 署名：`Einar Rig (CC-BY) Blender Foundation | studio.blender.org`；保留 Blender Studio 署名。
- 下载归档 579,834,421 字节，SHA256：
  `3cbb5a1dd9ffd3ca39edab420a83ea7f7db337b435a78ef27a0c87ff3e6977f7`。
- 原始文件：`build/deps/einar-official/source/einar_release_v1.blend`。
  官方归档、解压目录与实验输出均位于 Git 忽略目录。原始 `.blend` 未修改。

## 方法与实际结果

使用本机 Blender 4.5.3 LTS，启动参数包括 `--background --factory-startup
--disable-autoexec --python-exit-code 1`。未执行资源内嵌的 `cloudrig.py`。
因此该测试不证明原始绑定、驱动器或参考姿势已经正确执行；日志中确实存在受限制的
场景驱动器以及旧文件迁移警告。

| 检查 | 结果 |
| --- | --- |
| 原始文件读取与静态库存 | 成功：261 Mesh、46 Curve、17 Curves、95 材质、64 图片数据块；这些是文件库存，含隐藏/未使用内容 |
| 渲染集合与对象筛选 | 300 对象，其中 211 Mesh、44 Curve、17 Curves；88 个使用材质 |
| 连接图结构回溯 | 从活动 Cycles/All 输出回溯，正确映射嵌套组接口；尚未裁剪常量零权重分支 |
| 可达贴图 | 44 张图片数据块，无缺失；归档内五张旧道具外链缺图未进入可达链 |
| 原始材质 USD/MaterialX 导出 | 两次失败，Blender 进程退出码 11；普通桌面权限重跑仍复现，排除仅由沙箱权限引起 |
| 无材质几何对照导出 | 成功，导出阶段约 8.3 秒；此文件只供几何诊断，不能作为完整角色资产 |

原始导出启用 MaterialX、禁用 Preview Surface，保留贴图导出与曲线，使用
`export_subdivision='TESSELLATE'`、`evaluation_mode='RENDER'`，关闭动画、骨架与
shape-key 动画导出。Blender 报告 `Unsupported node: Voronoi Texture`、旧
`Separate RGB`，随后在 `USDGenericMeshWriter::do_write` 调用链抛出
`UNKNOWN EXCEPTION`。这是导出进程的实际失败；尚未定位该异常的底层原因，
不能把它直接归因于 Caustra 的 Vulkan 渲染器。

对照 USD 包含 221 个 Mesh、约 221 万个三角形，以及 66 个 BasisCurves prim，
约 16.2 万根曲线、172 万个曲线控制点。Blender 同时警告无法导出同一个 Curves
对象中的混合闭合/非闭合曲线，所以即使对照文件写出，也不能据此证明所有毛发完整保留。

## 已证实的通用能力缺口

1. **UDIM 和多套 UV**：20 个材质的可达链引用两组 UDIM，7 张 tile 均存在；
   31 个材质读取多个命名 UV。当前资产编译器明确拒绝 UDIM，单个材质只接受一套 UV。
2. **曲线几何与毛发语义**：当前 USD 导入器只消费 Mesh；BasisCurves 会被跳过。
   头发、胡须、睫毛、绒毛等因此不能作为已经支持的内容交付。
   SimpleHair 表面模型支持不等于曲线几何支持。
3. **完整材质图和几何属性**：可达链含 Mix Shader、Add Shader、HairInfo、
   自定义 Attribute，以及 Voronoi、WhiteNoise、ColorRamp、FloatCurve、Bump 等。
   36 个材质使用自定义属性，当前通路不足以保留这些语义。
   Blender 本身也已明确报告部分节点不支持 MaterialX 导出。

活动 LightPath 节点为零；没有连接的材质 Volume/Displacement 终端。
这两项没有作为本次实际阻塞报告。可达性统计没有证明每个分支在当前参数下均有非零贡献。

本次没有改动渲染器或加入 Einar 特判，没有把材质烘焙/替换成常量 PBR，
也没有把缺少毛发的网格导入当作完整渲染通过。后续若继续支持 Einar，应先明确
通用的材质导出、UDIM/多 UV、属性与毛发方案，再进行外观和性能验证。

## 本机复现记录

实验文件位于 `build/experiments/einar/`：

- `inspect_blend.py` / `source_inventory.json`：原文件库存。
- `audit_active_materials.py` / `active_material_inventory.json`：输出连线回溯。
- `export_original.py`：原始材质导出脚本。
- `export.log` / `export_unsandboxed.log` / `einar_release_v1.crash.txt`：失败日志。
- `export_geometry_control.py` / `geometry_control.log` / `geometry_control.usdc`：几何对照。
- `usd_geometry_inventory.json`：对照 USD 的几何统计。

例如，在仓库根目录运行下列命令可重做库存；更换 `--python` 为上述导出脚本，
并将 `--` 后参数替换为目标 `.usdc` 路径，可重做对应导出。

```powershell
.\build\deps\blender-4.5.3-windows-x64\blender.exe --background --factory-startup --disable-autoexec --python-exit-code 1 build/deps/einar-official/source/einar_release_v1.blend --python build/experiments/einar/inspect_blend.py -- build/experiments/einar/source_inventory.json
```
