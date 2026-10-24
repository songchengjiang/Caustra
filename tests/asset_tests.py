"""External asset compiler regressions; run with the requirements-assets environment."""
import math
import shutil
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import MaterialX as mx
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdUtils
from caustra_assets.model import Asset, AssetError, Material, Op, Parameter
from caustra_assets.materialx_compiler import MaterialCompiler, TextureStore, load_materialx, standard_library
from caustra_assets.package import read_asset, write_asset
from caustra_assets.usd_import import import_usd, materialx_from_usd, triangulate


class AssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.libraries=standard_library()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def document(self):
        doc=mx.createDocument();doc.setDataLibrary(self.libraries)
        surface=doc.addNodeInstance(self.libraries.getNodeDef("ND_standard_surface_surfaceshader"),"surface")
        return doc,surface

    def node(self,doc,definition,name):
        return doc.addNodeInstance(self.libraries.getNodeDef(definition),name)

    @staticmethod
    def set_input(node,name,kind,value):
        port=node.addInput(name,kind);port.setValueString(value);return port

    def usd_material(self,stage,path,identifier="ND_standard_surface_surfaceshader"):
        material=UsdShade.Material.Define(stage,path)
        shader=UsdShade.Shader.Define(stage,path+"/surface");shader.CreateIdAttr(identifier)
        shader.CreateOutput("out",Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput("mtlx").ConnectToSource(shader.ConnectableAPI(),"out")
        return material,shader

    def usd_mesh(self,stage,path="/mesh"):
        mesh=UsdGeom.Mesh.Define(stage,path)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreatePointsAttr([(0,0,0),(1,0,0),(1,1,0),(0,1,0)])
        mesh.CreateFaceVertexCountsAttr([3,3]);mesh.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
        return mesh

    def test_real_nodedef_defaults_and_connected_patterns(self):
        doc,surface=self.document();mix=self.node(doc,"ND_mix_color3","mix")
        self.set_input(mix,"fg","color3","1, 0, 0");self.set_input(mix,"bg","color3","0, 0, 1")
        self.set_input(mix,"mix","float","0.25")
        surface.addInput("base_color","color3").setConnectedNode(mix)
        asset=Asset();index=MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertAlmostEqual(asset.materials[index].values[2][1],0.2,places=6)
        self.assertEqual(asset.materials[index].values[9][0],1)
        code=[ins for ins in asset.instructions if ins[0][0]==Op.Mix][0][0]
        self.assertEqual(asset.instructions[code[1]][1][:3],[0,0,1]) # bg before fg
        self.assertEqual(asset.instructions[code[2]][1][:3],[1,0,0])
        self.assertEqual(asset.instructions[code[3]][1],[0.25]*4)
        self.assertEqual(asset.instructions[-1][0][0],Op.Output)
        self.assertEqual(asset.instructions[-1][0][2],Parameter.BaseColor)

    def test_unknown_nodes_and_unsupported_surface_inputs_fail(self):
        doc,surface=self.document();unknown=doc.addNode("unknown_caustra_node","unknown","color3")
        surface.addInput("base_color","color3").setConnectedNode(unknown)
        with self.assertRaisesRegex(AssetError,"Unknown MaterialX NodeDef"): MaterialCompiler(Asset(),self.libraries).compile(doc,surface)
        doc,surface=self.document();noise=self.node(doc,"ND_noise3d_color3","color_noise")
        surface.addInput("base_color","color3").setConnectedNode(noise)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertTrue(any(code[0]==Op.Noise3D and value[0]==3 for code,value in asset.instructions))

    def test_scalar_opacity_and_colored_rejection(self):
        doc,surface=self.document();self.set_input(surface,"opacity","color3","0.4, 0.4, 0.4")
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertAlmostEqual(asset.materials[0].values[9][2],0.4,places=6)
        doc,surface=self.document();self.set_input(surface,"opacity","color3","0.4, 0.2, 0.4")
        with self.assertRaisesRegex(AssetError,"Colored opacity"): MaterialCompiler(Asset(),self.libraries).compile(doc,surface)
        doc,surface=self.document();convert=self.node(doc,"ND_convert_float_color3","coverage")
        self.set_input(convert,"in","float","0.6");surface.addInput("opacity","color3").setConnectedNode(convert)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertEqual(asset.instructions[-1][0][2],Parameter.Opacity)

    def test_native_openpbr_defaults_and_inactive_lobes(self):
        doc=mx.createDocument();doc.setDataLibrary(self.libraries)
        surface=self.node(doc,"ND_open_pbr_surface_surfaceshader","openpbr")
        # The unused transmission color graph must not turn into active
        # reflection work, or fail because color noise is not yet implemented.
        noise=self.node(doc,"ND_noise3d_color3","unused_color_noise")
        surface.addInput("transmission_color","color3").setConnectedNode(noise)
        self.set_input(surface,"emission_luminance","float","2")
        self.set_input(surface,"emission_color","color3","1, 2, 3")
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        material=asset.materials[0]
        self.assertEqual(material.program[2],1)
        self.assertAlmostEqual(material.values[2][1],.3,places=6)
        self.assertAlmostEqual(material.values[5][2],1.6,places=6)
        self.assertEqual(material.values[9][3],1)
        self.assertEqual(material.values[1][:3],[2,4,6])
        self.assertEqual(len(asset.instructions),0)
        self.set_input(surface,"fuzz_weight","float","0.2")
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertAlmostEqual(asset.materials[0].values[17][0],.2)

    def test_rotate3d_uses_materialx_degrees_and_row_vectors(self):
        doc,surface=self.document();rotate=self.node(doc,"ND_rotate3d_vector3","rotate")
        self.set_input(rotate,"in","vector3","1, 0, 0")
        self.set_input(rotate,"axis","vector3","0, 0, 1")
        self.set_input(rotate,"amount","float","90")
        surface.addInput("tangent","vector3").setConnectedNode(rotate)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        registers=[];result=None
        for code,value in asset.instructions:
            op,a,b,c=code;v=np.asarray(value,dtype=float)
            if op==Op.Constant: pass
            elif op==Op.Multiply: v=registers[a]*registers[b]
            elif op==Op.Add: v=registers[a]+registers[b]
            elif op==Op.Normalize: v=registers[a]/np.linalg.norm(registers[a][:3])
            elif op==Op.Sin: v=np.sin(registers[a])
            elif op==Op.Cos: v=np.cos(registers[a])
            elif op==Op.OneMinus: v=1-registers[a]
            elif op==Op.Cross: v=np.append(np.cross(registers[a][:3],registers[b][:3]),0)
            elif op==Op.Dot: v=np.repeat(np.dot(registers[a][:3],registers[b][:3]),4)
            elif op==Op.Output: result=registers[a]
            else: self.fail("Unexpected rotation instruction")
            registers.append(v)
        np.testing.assert_allclose(result[:3],[0,-1,0],atol=1e-6)

    def test_texture_color_space_alpha_and_top_row(self):
        path=self.directory/"texture.png"
        image=Image.new("RGBA",(1,2));image.putdata([(128,64,32,128),(0,255,0,255)]);image.save(path)
        asset=Asset();store=TextureStore(asset)
        linear=store.add(path,"srgb_texture");raw=store.add(path,"raw")
        self.assertAlmostEqual(asset.textures[linear].pixels[0][0],0.2158605,places=5)
        self.assertAlmostEqual(asset.textures[linear].pixels[0][3],128/255,places=6)
        self.assertAlmostEqual(asset.textures[raw].pixels[0][0],128/255,places=6)
        self.assertEqual(asset.textures[linear].pixels[1],[0,1,0,1])
        self.assertEqual(store.add(path,"srgb"),linear)
        with self.assertRaisesRegex(AssetError,"color space"): store.add(path,"acescg")

    def test_image_uv_wrap_and_filter_compile(self):
        path=self.directory/"image.png";Image.new("RGB",(2,2),(255,0,0)).save(path)
        doc,surface=self.document();image=self.node(doc,"ND_image_color3","image")
        self.set_input(image,"file","filename",path.name).setColorSpace("srgb_texture")
        self.set_input(image,"uaddressmode","string","clamp");self.set_input(image,"vaddressmode","string","mirror")
        self.set_input(image,"filtertype","string","closest")
        surface.addInput("base_color","color3").setConnectedNode(image)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface,self.directory)
        instruction=[item for item in asset.instructions if item[0][0]==Op.Image][0]
        self.assertEqual(instruction[0][3],1|(2<<2)|(1<<4))
        self.assertEqual(asset.instructions[instruction[0][1]][0][0],Op.Texcoord)
        self.assertFalse(any(item[0][0]==Op.OneMinus for item in asset.instructions)) # shader flips V once

    def test_usd_facevarying_subsets_axis_units_transform_camera(self):
        path=self.directory/"scene.usda";stage=Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage,"Z");UsdGeom.SetStageMetersPerUnit(stage,0.01)
        root=UsdGeom.Xform.Define(stage,"/root");root.AddTranslateOp().Set(Gf.Vec3d(10,20,30))
        mesh=self.usd_mesh(stage,"/root/mesh")
        mesh.CreateNormalsAttr([(0,0,1),(0,0,-1)]);mesh.SetNormalsInterpolation("uniform")
        uv=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,"faceVarying")
        uv.Set([(0,0),(1,0),(1,1),(.2,.3),(.8,.9),(0,1)])
        first,_=self.usd_material(stage,"/first");second,surface=self.usd_material(stage,"/second")
        surface.CreateInput("metalness",Sdf.ValueTypeNames.Float).Set(1)
        binding=UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim());binding.Bind(first)
        subset=binding.CreateMaterialBindSubset("part",[1]);UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(second)
        camera=UsdGeom.Camera.Define(stage,"/root/camera");camera.AddTranslateOp().Set(Gf.Vec3d(0,-5,2))
        stage.GetRootLayer().Save();asset=import_usd(path)
        self.assertEqual(len(asset.meshes),2);self.assertEqual(len(asset.materials),2)
        self.assertEqual({m.material for m in asset.meshes},{0,1})
        self.assertAlmostEqual(asset.meshes[1].vertices[0][8],.2,places=6)
        self.assertAlmostEqual(asset.meshes[1].vertices[0][9],.3,places=6)
        self.assertEqual(asset.meshes[1].vertices[0][4:7],[0,0,-1])
        transform=asset.instances[0][1]
        self.assertAlmostEqual(transform[3],.1);self.assertAlmostEqual(transform[7],.3);self.assertAlmostEqual(transform[11],-.2)
        self.assertAlmostEqual(asset.bounds_min[0],.1);self.assertAlmostEqual(asset.bounds_max[1],.3)
        self.assertTrue(asset.has_camera)
        self.assertAlmostEqual(asset.camera[0],.1);self.assertAlmostEqual(asset.camera[1],.32);self.assertAlmostEqual(asset.camera[2],-.15)

    def test_usd_preserves_graph_connections_and_relative_texture(self):
        path=self.directory/"scene.usda";Image.new("RGB",(2,2),(128,128,128)).save(self.directory/"texture.png")
        stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1)
        mesh=self.usd_mesh(stage);material,surface=self.usd_material(stage,"/material")
        image=UsdShade.Shader.Define(stage,"/material/image");image.CreateIdAttr("ND_image_color3")
        file=image.CreateInput("file",Sdf.ValueTypeNames.Asset);file.Set(Sdf.AssetPath("texture.png"));file.GetAttr().SetMetadata("colorSpace","srgb_texture")
        image.CreateOutput("out",Sdf.ValueTypeNames.Color3f)
        surface.CreateInput("base_color",Sdf.ValueTypeNames.Color3f).ConnectToSource(image.ConnectableAPI(),"out")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material);stage.GetRootLayer().Save()
        doc,mx_surface=materialx_from_usd(material,self.libraries)
        self.assertTrue(mx_surface.getInput("base_color").getConnectedNode())
        asset=import_usd(path);self.assertEqual(len(asset.textures),1)
        self.assertAlmostEqual(asset.textures[0].pixels[0][0],.2158605,places=5)
        package=self.directory/"compiled.caustra";write_asset(asset,package)
        relocated=self.directory/"relocated";relocated.mkdir();shutil.copyfile(package,relocated/package.name)
        (self.directory/"texture.png").unlink();path.unlink()
        restored=read_asset(relocated/package.name)
        self.assertEqual(len(restored.textures),1);self.assertEqual(len(restored.meshes),1)
        self.assertNotIn(b"texture.png",(relocated/package.name).read_bytes())

    def test_usdz_embedded_material_and_png_survive_source_removal(self):
        source=self.directory/"source";source.mkdir();textures=source/"textures";textures.mkdir()
        image_path=textures/"albedo.png";Image.new("RGB",(2,2),(128,64,32)).save(image_path)
        scene_path=source/"scene.usda";stage=Usd.Stage.CreateNew(str(scene_path))
        UsdGeom.SetStageMetersPerUnit(stage,1)
        mesh=self.usd_mesh(stage);material,surface=self.usd_material(stage,"/material")
        image=UsdShade.Shader.Define(stage,"/material/image");image.CreateIdAttr("ND_image_color3")
        file=image.CreateInput("file",Sdf.ValueTypeNames.Asset);file.Set(Sdf.AssetPath("textures/albedo.png"));file.GetAttr().SetMetadata("colorSpace","srgb_texture")
        image.CreateOutput("out",Sdf.ValueTypeNames.Color3f)
        surface.CreateInput("base_color",Sdf.ValueTypeNames.Color3f).ConnectToSource(image.ConnectableAPI(),"out")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material);stage.GetRootLayer().Save()
        del stage,mesh,material,surface,image,file
        archive=self.directory/"packed.usdz"
        self.assertTrue(UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(scene_path)),str(archive)))
        with zipfile.ZipFile(archive) as package:
            self.assertTrue(any(name.endswith(".png") for name in package.namelist()))
        # The importer must use the packaged PNG, not a source-file fallback.
        image_path.unlink();scene_path.unlink();textures.rmdir();source.rmdir()
        asset=import_usd(archive)
        self.assertEqual(len(asset.meshes),1);self.assertEqual(len(asset.textures),1)
        self.assertAlmostEqual(asset.textures[0].pixels[0][0],.2158605,places=5)
        self.assertIn(Op.Image,{code[0] for code,_ in asset.instructions})
        result=self.directory/"relocated.caustra";write_asset(asset,result);archive.unlink()
        self.assertEqual(len(read_asset(result).textures),1)

    def test_usd_nodegraph_output_and_interface_input(self):
        stage=Usd.Stage.CreateInMemory();material,surface=self.usd_material(stage,"/material")
        graph=UsdShade.NodeGraph.Define(stage,"/material/graph")
        tint=graph.CreateInput("tint",Sdf.ValueTypeNames.Color3f);tint.Set(Gf.Vec3f(.2,.4,.6))
        constant=UsdShade.Shader.Define(stage,"/material/graph/constant");constant.CreateIdAttr("ND_constant_color3")
        constant.CreateInput("value",Sdf.ValueTypeNames.Color3f).ConnectToSource(tint)
        constant.CreateOutput("out",Sdf.ValueTypeNames.Color3f)
        output=graph.CreateOutput("color",Sdf.ValueTypeNames.Color3f);output.ConnectToSource(constant.ConnectableAPI(),"out")
        surface.CreateInput("base_color",Sdf.ValueTypeNames.Color3f).ConnectToSource(output)
        document,root=materialx_from_usd(material,self.libraries)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(document,root)
        constants=[value for code,value in asset.instructions if code[0]==Op.Constant]
        self.assertAlmostEqual(constants[0][0],.2,places=6)
        self.assertAlmostEqual(constants[0][2],.6,places=6)

    def test_camera_focus_or_bounds_target_preserves_view_direction(self):
        path=self.directory/"camera.usda";stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,"Y")
        self.usd_mesh(stage)
        camera=UsdGeom.Camera.Define(stage,"/camera");camera.AddTranslateOp().Set(Gf.Vec3d(.5,.5,5))
        stage.GetRootLayer().Save();asset=import_usd(path)
        self.assertAlmostEqual(asset.camera[6],0,places=6) # bbox center projected onto camera's -Z ray
        self.assertAlmostEqual(asset.camera[4],.5);self.assertAlmostEqual(asset.camera[5],.5)
        camera.GetFocusDistanceAttr().Set(2);stage.GetRootLayer().Save();asset=import_usd(path)
        self.assertAlmostEqual(asset.camera[6],3,places=6)

    def test_area_light_normalization_transform_and_one_sided_emission(self):
        path=self.directory/"light.usda";stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,.1)
        light=UsdLux.RectLight.Define(stage,"/area");light.GetWidthAttr().Set(2);light.GetHeightAttr().Set(4)
        light.GetIntensityAttr().Set(6);light.GetExposureAttr().Set(1);light.GetColorAttr().Set(Gf.Vec3f(1,.5,.25));light.GetNormalizeAttr().Set(True)
        UsdGeom.Xformable(light).AddScaleOp().Set(Gf.Vec3f(2,3,1));stage.GetRootLayer().Save()
        asset=import_usd(path);material=asset.materials[0]
        self.assertEqual(material.values[2][2],1)
        self.assertAlmostEqual(material.values[1][0],25,places=5)
        self.assertAlmostEqual(material.values[1][1],12.5,places=5)
        p,q,r=[np.asarray(asset.meshes[0].vertices[i][:3]) for i in asset.meshes[0].indices[:3]]
        self.assertLess(np.cross(q-p,r-p)[2],0) # USD lights emit towards local -Z

    def test_disk_light_temperature_and_unsupported_lights(self):
        path=self.directory/"disk.usda";stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1)
        light=UsdLux.DiskLight.Define(stage,"/disk");light.GetRadiusAttr().Set(2);light.GetIntensityAttr().Set(3)
        light.GetEnableColorTemperatureAttr().Set(True);light.GetColorTemperatureAttr().Set(3000)
        stage.GetRootLayer().Save();asset=import_usd(path)
        expected=UsdLux.BlackbodyTemperatureAsRgb(3000)
        np.testing.assert_allclose(asset.materials[0].values[1][:3],np.asarray(expected)*3,rtol=1e-6)
        self.assertEqual(len(asset.meshes[0].indices),96*3)
        stage=Usd.Stage.CreateNew(str(self.directory/"sphere.usda"));UsdLux.SphereLight.Define(stage,"/sphere");stage.GetRootLayer().Save()
        with self.assertRaisesRegex(AssetError,"Unsupported USD light type"): import_usd(self.directory/"sphere.usda")

    def test_preview_surface_is_not_silently_substituted(self):
        path=self.directory/"preview.usda";stage=Usd.Stage.CreateNew(str(path));mesh=self.usd_mesh(stage)
        material,surface=self.usd_material(stage,"/material","UsdPreviewSurface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material);stage.GetRootLayer().Save()
        with self.assertRaisesRegex(AssetError,"Unknown MaterialX NodeDef"): import_usd(path)

    def test_usd_displacement_and_volume_terminals_are_preserved(self):
        for terminal in ("displacement","volume"):
            for context in ("mtlx",""):
                stage=Usd.Stage.CreateInMemory();material,_=self.usd_material(stage,"/material")
                shader=UsdShade.Shader.Define(stage,"/material/unsupported")
                shader.CreateIdAttr("ND_displacement_float" if terminal=="displacement" else "ND_volume")
                shader.CreateOutput("out",Sdf.ValueTypeNames.Token)
                if terminal=="displacement": shader.CreateInput("displacement",Sdf.ValueTypeNames.Float).Set(.1)
                else:
                    vdf=UsdShade.Shader.Define(stage,"/material/absorption");vdf.CreateIdAttr("ND_absorption_vdf")
                    vdf.CreateInput("absorption",Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(.1))
                    vdf.CreateOutput("out",Sdf.ValueTypeNames.Token)
                    shader.CreateInput("vdf",Sdf.ValueTypeNames.Token).ConnectToSource(vdf.ConnectableAPI(),"out")
                output=(material.CreateDisplacementOutput(context) if terminal=="displacement" else material.CreateVolumeOutput(context))
                output.ConnectToSource(shader.ConnectableAPI(),"out")
                doc,surface=materialx_from_usd(material,self.libraries)
                asset=Asset();compiler=MaterialCompiler(asset,self.libraries);index=compiler.compile(doc,surface)
                if terminal=="displacement": self.assertIn(index,compiler.material_displacements)
                else: np.testing.assert_allclose(asset.materials[index].values[20][:3],[.1]*3)

    def test_standalone_materialx_displacement_and_volume_are_compiled(self):
        for terminal in ("displacement","volume"):
            doc,surface=self.document()
            wrapper=doc.addNode("surfacematerial","material","material")
            wrapper.addInput("surfaceshader","surfaceshader").setConnectedNode(surface)
            if terminal=="displacement":
                displacement=self.node(doc,"ND_displacement_float","offset")
                displacement.addInput("displacement","float").setValueString("0.1")
                wrapper.addInput("displacementshader","displacementshader").setConnectedNode(displacement)
            else:
                volume=self.node(doc,"ND_volume","volume")
                vdf=self.node(doc,"ND_absorption_vdf","absorption")
                self.set_input(vdf,"absorption","vector3","0.1, 0.1, 0.1")
                volume.addInput("vdf","VDF").setConnectedNode(vdf)
                volume_wrapper=self.node(doc,"ND_volumematerial","volume_material")
                volume_wrapper.addInput("volumeshader","volumeshader").setConnectedNode(volume)
            self.assertTrue(doc.validate()[0],doc.validate()[1])
            path=self.directory/(terminal+".mtlx");mx.writeToXmlFile(doc,str(path))
            loaded,root=load_materialx(path,self.libraries)
            asset=Asset();compiler=MaterialCompiler(asset,self.libraries);index=compiler.compile(loaded,root)
            if terminal=="displacement":
                self.assertIn(index,compiler.material_displacements)
                self.assertEqual(compiler.material_displacements[index][2],"float")
            else: np.testing.assert_allclose(asset.materials[index].values[20][:3],[.1]*3)

    def test_concave_polygon_triangulation(self):
        points=[(0,0,0),(2,0,0),(2,2,0),(1,1,0),(0,2,0)]
        triangles=triangulate(points);self.assertEqual(len(triangles),3)
        area=0
        for a,b,c in triangles:
            p,q,r=points[a],points[b],points[c]
            area+=abs((q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0]))*.5
        self.assertAlmostEqual(area,3)

    def test_package_rejects_truncated_and_wrong_version(self):
        path=self.directory/"empty.caustra";write_asset(Asset(),path)
        raw=path.read_bytes();self.assertEqual(len(raw),120)
        path.write_bytes(raw[:-1])
        with self.assertRaisesRegex(AssetError,"Truncated"): read_asset(path)
        changed=bytearray(raw);struct.pack_into("<I",changed,8,999);path.write_bytes(changed)
        with self.assertRaisesRegex(AssetError,"Unsupported"): read_asset(path)

    def test_writer_rejects_wrong_instance_transform_layout(self):
        for length in (11,13):
            asset=Asset(instances=[(0,[0]*length)])
            with self.assertRaisesRegex(AssetError,"12 affine values"):
                write_asset(asset,self.directory/"invalid.caustra")

    def test_version2_material_rows_and_legacy_version1_defaults(self):
        path=self.directory/"material.caustra";material=Material();material.values[10]=[.7,.4,.1,.2]
        write_asset(Asset(materials=[material]),path);raw=path.read_bytes()
        self.assertEqual(struct.unpack_from("<I",raw,8)[0],2)
        self.assertEqual(len(raw),120+384)
        np.testing.assert_allclose(read_asset(path).materials[0].values[10],[.7,.4,.1,.2])
        legacy=bytearray(raw[:120]+raw[120:280]+raw[488:504]);struct.pack_into("<I",legacy,8,1)
        path.write_bytes(legacy);restored=read_asset(path).materials[0]
        self.assertEqual(len(restored.values),23);self.assertEqual(restored.values[10],[0,0,0,0])
        self.assertEqual(restored.values[16],[1,0,1.5,0])

    def test_standard_and_openpbr_extended_lobes_units_and_dispersion(self):
        doc,surface=self.document()
        for name,value in (("transmission",".8"),("transmission_depth","12"),("transmission_dispersion","20"),
                           ("transmission_extra_roughness","-.2"),("subsurface",".3"),("subsurface_scale","4"),
                           ("thin_film_thickness","450"),("coat_anisotropy",".4"),("coat_rotation",".25"),
                           ("coat_affect_color",".2"),("coat_affect_roughness",".3")):
            self.set_input(surface,name,"float",value)
        asset=Asset();MaterialCompiler(asset,self.libraries,.01).compile(doc,surface);m=asset.materials[0]
        np.testing.assert_allclose(m.values[10],[.8,.12,0,-.2]);self.assertAlmostEqual(m.values[13][1],.04)
        self.assertEqual(m.values[16],[1,450,1.5,.05]);np.testing.assert_allclose(m.values[17][2:],[.25,.2])
        self.assertAlmostEqual(m.values[19][3],.3)
        doc=mx.createDocument();doc.setDataLibrary(self.libraries);surface=self.node(doc,"ND_open_pbr_surface_surfaceshader","open")
        for name,value in (("transmission_weight","1"),("transmission_dispersion_scale",".5"),
                           ("transmission_dispersion_abbe_number","25"),("subsurface_weight",".2"),
                           ("subsurface_radius","3"),("fuzz_weight",".7"),("thin_film_weight","1"),("thin_film_thickness",".45")):
            self.set_input(surface,name,"float",value)
        self.set_input(surface,"geometry_thin_walled","boolean","true")
        asset=Asset();MaterialCompiler(asset,self.libraries,.1).compile(doc,surface);m=asset.materials[0]
        self.assertEqual(m.values[13][3],1);self.assertAlmostEqual(m.values[13][1],.3)
        np.testing.assert_allclose(m.values[16],[1,450,1.4,.02]);self.assertAlmostEqual(m.values[17][0],.7)

    def test_homogeneous_volume_coefficients_emission_and_boundary_only(self):
        doc=mx.createDocument();doc.setDataLibrary(self.libraries)
        volume=self.node(doc,"ND_volume","vol");vdf=self.node(doc,"ND_anisotropic_vdf","scatter")
        self.set_input(vdf,"absorption","vector3",".1, .2, .3");self.set_input(vdf,"scattering","vector3",".4, .5, .6")
        self.set_input(vdf,"anisotropy","float",".35");volume.addInput("vdf","VDF").setConnectedNode(vdf)
        edf=self.node(doc,"ND_uniform_edf","emission");self.set_input(edf,"color","color3",".01, .02, .03")
        volume.addInput("edf","EDF").setConnectedNode(edf)
        wrapper=self.node(doc,"ND_volumematerial","wrapper");wrapper.addInput("volumeshader","volumeshader").setConnectedNode(volume)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,None);m=asset.materials[0]
        np.testing.assert_allclose(m.values[19][:3],[1,.35,1]);np.testing.assert_allclose(m.values[20][:3],[.1,.2,.3])
        np.testing.assert_allclose(m.values[21][:3],[.4,.5,.6]);np.testing.assert_allclose(m.values[22][:3],[.01,.02,.03])
        position=self.node(doc,"ND_position_vector3","varying")
        vdf.getInput("absorption").setConnectedNode(position)
        with self.assertRaisesRegex(AssetError,"spatially constant"): MaterialCompiler(Asset(),self.libraries).compile(doc,None)

    def test_solid_interiors_fold_constant_graphs_flag_heterogeneous_and_reject_alpha(self):
        doc,surface=self.document();self.set_input(surface,"transmission","float","1")
        tint=self.node(doc,"ND_multiply_color3","constant_tint")
        self.set_input(tint,"in1","color3",".2, .3, .4");self.set_input(tint,"in2","color3",".5, .5, .5")
        surface.addInput("transmission_color","color3").setConnectedNode(tint)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        np.testing.assert_allclose(asset.materials[0].values[11][:3],[.1,.15,.2]);self.assertEqual(len(asset.instructions),0)
        self.set_input(surface,"opacity","color3",".5, .5, .5")
        with self.assertRaisesRegex(AssetError,"constant opacity 1"): MaterialCompiler(Asset(),self.libraries).compile(doc,surface)
        surface.getInput("opacity").setValueString("1, 1, 1")
        noise=self.node(doc,"ND_noise3d_color3","varying_tint");surface.getInput("transmission_color").setConnectedNode(noise)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertEqual(asset.materials[0].program[3],1)
        self.assertIn(Parameter.TransmissionColor,[code[2] for code,_ in asset.instructions if code[0]==Op.Output])
        self.set_input(surface,"thin_walled","boolean","true")
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertEqual(asset.materials[0].program[3],0)
        self.assertIn(Parameter.TransmissionColor,[code[2] for code,_ in asset.instructions if code[0]==Op.Output])

    def test_true_scalar_and_vector_displacement_tessellation_bounds_and_normals(self):
        for kind in ("float","vector3"):
            path=self.directory/(kind+".usda");stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,.1)
            mesh=self.usd_mesh(stage);material,_=self.usd_material(stage,"/material")
            uv=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,"vertex")
            uv.Set([(0,0),(1,0),(1,1),(0,1)])
            displacement=UsdShade.Shader.Define(stage,"/material/disp");displacement.CreateIdAttr("ND_displacement_"+kind)
            displacement.CreateOutput("out",Sdf.ValueTypeNames.Token)
            if kind=="float":
                noise=UsdShade.Shader.Define(stage,"/material/noise");noise.CreateIdAttr("ND_noise3d_float")
                noise.CreateInput("amplitude",Sdf.ValueTypeNames.Float).Set(.2);noise.CreateInput("pivot",Sdf.ValueTypeNames.Float).Set(.15)
                noise.CreateOutput("out",Sdf.ValueTypeNames.Float)
                displacement.CreateInput("displacement",Sdf.ValueTypeNames.Float).ConnectToSource(noise.ConnectableAPI(),"out")
            else: displacement.CreateInput("displacement",Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(.2,.3,.4))
            material.CreateDisplacementOutput("mtlx").ConnectToSource(displacement.ConnectableAPI(),"out")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material);stage.GetRootLayer().Save()
            asset=import_usd(path,displacement_subdivisions=2);result=asset.meshes[0]
            self.assertEqual(len(result.indices)//3,32)
            positions=np.asarray(result.vertices)[:,:3];normals=np.asarray(result.vertices)[:,4:7]
            np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=1e-6)
            if kind=="float":
                self.assertGreater(np.ptp(positions[:,2]),.01);self.assertGreater(np.ptp(normals[:,0]),.01)
            else:
                np.testing.assert_allclose(asset.bounds_min[:3],[.02,.03,.04],atol=1e-7)
                np.testing.assert_allclose(asset.bounds_max[:3],[.12,.13,.04],atol=1e-7)
            with self.assertRaisesRegex(AssetError,"between 0 and 7"): import_usd(path,displacement_subdivisions=8)

    def test_standard_surface_sss_tracks_spatial_coat_color_controls(self):
        doc,surface=self.document();self.set_input(surface,"subsurface","float",".7")
        self.set_input(surface,"coat_affect_color","float",".5")
        for name,expected_parameter,row,lane in (("coat",Parameter.CoatWeight,5,0),
                                                 ("coat_affect_color",Parameter.CoatAffectColor,17,3)):
            with self.subTest(input=name):
                graph=self.node(doc,"ND_multiply_float","constant_"+name)
                self.set_input(graph,"in1","float",".4");self.set_input(graph,"in2","float",".5")
                port=surface.getInput(name) or surface.addInput(name,"float");port.setConnectedNode(graph)
                asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
                self.assertAlmostEqual(asset.materials[0].values[row][lane],.2,places=6)
                self.assertNotIn(expected_parameter,[code[2] for code,_ in asset.instructions if code[0]==Op.Output])
                noise=self.node(doc,"ND_noise3d_float","varying_"+name);port.setConnectedNode(noise)
                asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
                self.assertEqual(asset.materials[0].program[3],1)
                self.assertIn(expected_parameter,[code[2] for code,_ in asset.instructions if code[0]==Op.Output])
                port.setConnectedNode(graph)
        # The constraint follows SSS medium coefficients; glass alone still
        # evaluates textured/procedural coat at each shading point.
        surface.getInput("subsurface").setValueString("0")
        self.set_input(surface,"transmission","float","1")
        surface.getInput("coat").setConnectedNode(doc.getNode("varying_coat"))
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertIn(Parameter.CoatWeight,[code[2] for code,_ in asset.instructions if code[0]==Op.Output])

    def test_native_openpbr_solid_requires_uniform_specular_weight(self):
        doc=mx.createDocument();doc.setDataLibrary(self.libraries)
        surface=self.node(doc,"ND_open_pbr_surface_surfaceshader","open")
        self.set_input(surface,"transmission_weight","float","1")
        constant=self.node(doc,"ND_multiply_float","constant_specular")
        self.set_input(constant,"in1","float",".8");self.set_input(constant,"in2","float",".5")
        port=surface.addInput("specular_weight","float");port.setConnectedNode(constant)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertAlmostEqual(asset.materials[0].values[3][0],.4,places=6)
        self.assertNotIn(Parameter.SpecularWeight,[code[2] for code,_ in asset.instructions if code[0]==Op.Output])
        noise=self.node(doc,"ND_noise3d_float","varying_specular");port.setConnectedNode(noise)
        with self.assertRaisesRegex(AssetError,"spatially constant"):
            MaterialCompiler(Asset(),self.libraries).compile(doc,surface)
        thin_port=surface.addInput("geometry_thin_walled","boolean")
        for thin,transmission in ((True,1),(False,0)):
            with self.subTest(thin=thin,transmission=transmission):
                thin_port.setValueString("true" if thin else "false")
                surface.getInput("transmission_weight").setValueString(str(transmission))
                asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
                self.assertIn(Parameter.SpecularWeight,[code[2] for code,_ in asset.instructions if code[0]==Op.Output])

    def test_native_official_surface_families_preserve_graphs_and_defaults(self):
        for definition,model,name in (("ND_disney_principled",2,"baseColor"),
                                      ("ND_gltf_pbr_surfaceshader",3,"base_color"),
                                      ("ND_UsdPreviewSurface_surfaceshader",4,"diffuseColor")):
            with self.subTest(model=model):
                doc=mx.createDocument();doc.setDataLibrary(self.libraries)
                surface=self.node(doc,definition,"surface");pattern=self.node(doc,"ND_checkerboard_color3","pattern")
                surface.addInput(name,"color3").setConnectedNode(pattern)
                asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
                self.assertEqual(asset.materials[0].program[2],model)
                self.assertTrue(any(code[0]==Op.Checker for code,_ in asset.instructions))
                self.assertIn(Parameter.BaseColor,[code[2] for code,_ in asset.instructions if code[0]==Op.Output])
                if model==2: self.assertEqual(asset.materials[0].values[5][1],1) # native clearcoat gloss
                if model==3: self.assertEqual(asset.materials[0].values[13][3],1) # official surface BTDF
        doc=mx.createDocument();doc.setDataLibrary(self.libraries)
        surface=self.node(doc,"ND_UsdPreviewSurface_surfaceshader","surface")
        self.set_input(surface,"opacity","float",".25")
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertAlmostEqual(asset.materials[0].values[10][0],.75)
        self.assertEqual(asset.materials[0].values[9][2],.25)

    def test_thin_plane_spatial_transmission_preserves_graph_without_bulk_grid(self):
        for definition,transmission,tint,thin in (("ND_gltf_pbr_surfaceshader","transmission","attenuation_color",None),
                                                 ("ND_standard_surface_surfaceshader","transmission","transmission_color","thin_walled")):
            with self.subTest(definition=definition):
                path=self.directory/(definition+".usda");stage=Usd.Stage.CreateNew(str(path))
                UsdGeom.SetStageMetersPerUnit(stage,1);mesh=self.usd_mesh(stage)
                material,surface=self.usd_material(stage,"/material",definition)
                if thin: surface.CreateInput(thin,Sdf.ValueTypeNames.Bool).Set(True)
                for name,node_definition,kind in ((transmission,"ND_noise3d_float",Sdf.ValueTypeNames.Float),
                                                  (tint,"ND_noise3d_color3",Sdf.ValueTypeNames.Color3f)):
                    noise=UsdShade.Shader.Define(stage,"/material/"+name);noise.CreateIdAttr(node_definition)
                    noise.CreateOutput("out",kind)
                    surface.CreateInput(name,kind).ConnectToSource(noise.ConnectableAPI(),"out")
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material);stage.GetRootLayer().Save()
                asset=import_usd(path)
                self.assertEqual(asset.bounds_min[2],asset.bounds_max[2]) # An actual zero-thickness plane.
                self.assertEqual(asset.materials[0].values[13][3],1)
                self.assertEqual(asset.materials[0].program[3],0)
                outputs={code[2] for code,_ in asset.instructions if code[0]==Op.Output}
                self.assertIn(Parameter.TransmissionWeight,outputs)
                self.assertIn(Parameter.TransmissionColor,outputs)
                self.assertTrue(any(code[0]==Op.Noise3D for code,_ in asset.instructions))

    def test_inactive_native_spatial_sss_inputs_do_not_create_a_bulk_grid(self):
        doc=mx.createDocument();doc.setDataLibrary(self.libraries)
        surface=self.node(doc,"ND_disney_principled","surface")
        noise=self.node(doc,"ND_noise3d_color3","radius")
        surface.addInput("subsurfaceDistance","color3").setConnectedNode(noise)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertEqual(asset.materials[0].program[3],0)
        self.assertIn(Parameter.SubsurfaceRadius,{code[2] for code,_ in asset.instructions if code[0]==Op.Output})

    def test_official_image_fileprefix_and_gltf_color_alpha_subgraph(self):
        from caustra_assets.pattern_cpu import evaluate_program
        textures=self.directory/"textures";textures.mkdir()
        Image.new("RGBA",(1,1),(51,102,204,128)).save(textures/"pixel.png")
        doc=mx.createDocument();doc.setDataLibrary(self.libraries);doc.setFilePrefix("textures/")
        surface=self.node(doc,"ND_gltf_pbr_surfaceshader","surface")
        image=self.node(doc,"ND_gltf_colorimage","image")
        self.set_input(image,"file","filename","pixel.png")
        self.set_input(image,"color","color4",".5, 1, .25, .5")
        color=surface.addInput("base_color","color3");color.setConnectedNode(image);color.setOutputString("outcolor")
        alpha=surface.addInput("alpha","float");alpha.setConnectedNode(image);alpha.setOutputString("outa")
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface,self.directory)
        values=evaluate_program(asset.instructions,asset.textures,uv=np.asarray([[.3,.7]]))
        outputs={code[2]:values[index][0] for index,(code,_) in enumerate(asset.instructions) if code[0]==Op.Output}
        np.testing.assert_allclose(outputs[Parameter.BaseColor][:3],[.1,.4,.2],atol=1e-7)
        self.assertAlmostEqual(outputs[Parameter.Opacity][0],128/255*.5)
        self.assertEqual(len(asset.textures),1)

    def test_official_place2d_row_rotation_and_operation_order(self):
        from caustra_assets.pattern_cpu import evaluate_program
        for order,expected in ((0,1/6-.1),(1,.15)):
            with self.subTest(order=order):
                doc,surface=self.document();uv=self.node(doc,"ND_texcoord_vector2","uv")
                place=self.node(doc,"ND_place2d_vector2","place");place.addInput("texcoord","vector2").setConnectedNode(uv)
                self.set_input(place,"scale","vector2","2, 3");self.set_input(place,"rotate","float","90")
                self.set_input(place,"offset","vector2",".1, .2");self.set_input(place,"operationorder","integer",str(order))
                channel=self.node(doc,"ND_extract_vector2","x");channel.addInput("in","vector2").setConnectedNode(place)
                surface.addInput("specular_roughness","float").setConnectedNode(channel)
                asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
                values=evaluate_program(asset.instructions,[],uv=np.asarray([[.3,.5]]))
                result=next(values[i][0,0] for i,(code,_) in enumerate(asset.instructions) if code[0]==Op.Output and code[2]==Parameter.Roughness)
                self.assertAlmostEqual(result,expected,places=7)

    def test_fractal_noise_and_hsv_reference_math(self):
        from caustra_assets.pattern_cpu import evaluate_program,perlin
        doc,surface=self.document();noise=self.node(doc,"ND_fractal3d_float","fractal")
        self.set_input(noise,"amplitude","float",".7");self.set_input(noise,"octaves","integer","3")
        self.set_input(noise,"lacunarity","float","2.1");self.set_input(noise,"diminish","float",".4")
        surface.addInput("specular_roughness","float").setConnectedNode(noise)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        positions=np.asarray([[.3,.7,1.1],[-1.2,2.3,.4]])
        values=evaluate_program(asset.instructions,[],object_position=positions)
        result=next(values[i][:,0] for i,(code,_) in enumerate(asset.instructions) if code[0]==Op.Output and code[2]==Parameter.Roughness)
        expected=sum(.7*.4**i*perlin(positions*2.1**i)[:,0] for i in range(3))
        np.testing.assert_allclose(result,expected,atol=1e-7)
        program=[([int(Op.Constant),0,0,0],[.2,.4,.8,.3]),([int(Op.RgbToHsv),0,0,0],[0]*4),
                 ([int(Op.HsvToRgb),1,0,0],[0]*4)]
        values=evaluate_program(program,[])
        np.testing.assert_allclose(values[1],[11/18,.75,.8,.3])
        np.testing.assert_allclose(values[2],[.2,.4,.8,.3],atol=1e-12)

    def test_solid_ior_remains_uniform_with_spatial_sss(self):
        doc,surface=self.document();self.set_input(surface,"subsurface","float",".4")
        noise=self.node(doc,"ND_noise3d_color3","sss_color")
        surface.addInput("subsurface_color","color3").setConnectedNode(noise)
        asset=Asset();MaterialCompiler(asset,self.libraries).compile(doc,surface)
        self.assertEqual(asset.materials[0].program[3],1)
        ior=self.node(doc,"ND_noise3d_float","ior")
        surface.addInput("specular_IOR","float").setConnectedNode(ior)
        with self.assertRaisesRegex(AssetError,"spatially constant"):
            MaterialCompiler(Asset(),self.libraries).compile(doc,surface)

    def test_vector_displacement_uses_actual_uv_derivatives_and_normal_basis(self):
        from caustra_assets.displacement import displace_mesh
        from caustra_assets.model import Mesh
        points=[(0,0,0),(0,2,0),(0,2,3),(0,0,3)];uvs=[(0,0),(1,0),(1,1),(0,1)]
        vertices=[list(p)+[0,1,0,0,0]+list(uv)+[0]*6 for p,uv in zip(points,uvs)]
        mesh=Mesh(0,vertices,[0,1,2,0,2,3]);program=[([int(Op.Constant),0,0,0],[.2,.3,.4,0])]
        identity=[1,0,0,0,0,1,0,0,0,0,1,0]
        result=displace_mesh(mesh,(program,0,"vector3"),[],identity,1)
        positions=np.asarray(result.vertices)[:,:3]
        # dPdu=(0,2,0), dPdv=(0,0,3), N=(1,0,0).
        np.testing.assert_allclose(positions.min(axis=0),[.4,.4,.9],atol=1e-12)
        np.testing.assert_allclose(positions.max(axis=0),[.4,2.4,3.9],atol=1e-12)
        for vertex in mesh.vertices: vertex[8:10]=[0,0]
        with self.assertRaisesRegex(AssetError,"nondegenerate UV derivatives"):
            displace_mesh(mesh,(program,0,"vector3"),[],identity,1)

    def test_cpu_pattern_power_and_divide_match_gpu_valid_domain(self):
        from caustra_assets.pattern_cpu import evaluate_program
        program=[([int(Op.Constant),0,0,0],[-2]*4),([int(Op.Constant),0,0,0],[2]*4),
                 ([int(Op.Power),0,1,0],[0]*4),([int(Op.Constant),0,0,0],[1e-9,-1e-9,0,1e-21]),
                 ([int(Op.Divide),1,3,0],[0]*4)]
        result=evaluate_program(program,[])
        np.testing.assert_allclose(result[2],[4]*4)
        np.testing.assert_allclose(result[4],[2e9,-2e9,0,0])
        program[1]=([int(Op.Constant),0,0,0],[.5]*4)
        with self.assertRaisesRegex(AssetError,"Nonfinite"):
            evaluate_program(program,[])

    def test_displacement_rejects_cracked_uv_seams_and_excessive_mesh_size(self):
        from caustra_assets.displacement import displace_mesh
        from caustra_assets.model import Mesh
        vertices=[[0,0,0,0,0,0,1,0,u,0,0,0,0,0,0,0] for u in (0,.5)]
        vertices+=[[1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],[0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0]]
        mesh=Mesh(0,vertices,[0,2,3,1,2,3]);program=[([int(Op.Texcoord),0,0,0],[0]*4),([int(Op.Swizzle),0,0,0],[0]*4)]
        identity=[1,0,0,0,0,1,0,0,0,0,1,0]
        with self.assertRaisesRegex(AssetError,"discontinuous"): displace_mesh(mesh,(program,1,"float"),[],identity,0)
        mesh.indices=[0,2,3]*100
        with self.assertRaisesRegex(AssetError,"million-triangle"): displace_mesh(mesh,(program,1,"float"),[],identity,7)

    def test_cpu_materialx_noise_reference_and_texture_v_orientation(self):
        from caustra_assets.pattern_cpu import perlin,sample_texture
        from caustra_assets.model import Texture
        result=perlin(np.asarray([[.25,.5,.75],[.1,1.2,-2.3]]),3)
        np.testing.assert_allclose(result,[[.20869569683074954,-.585706262588501,.1530549364089966],
                                           [.2782254037659447,-.43276016022534375,.037631331459797315]],atol=1e-12)
        texture=Texture(1,2,[[1,0,0,1],[0,0,1,1]])
        result=sample_texture(texture,np.asarray([[.5,.25],[.5,.75]]),1|(1<<2)|16)
        np.testing.assert_allclose(result,[[0,0,1,1],[1,0,0,1]])

    def test_real_feature_gallery_has_renderable_lobes_volume_and_displaced_geometry(self):
        path=Path(__file__).resolve().parent/"assets"/"features"/"scene.usda"
        asset=import_usd(path,displacement_subdivisions=2)
        self.assertTrue(asset.has_camera);self.assertEqual(len(asset.meshes),12)
        self.assertTrue(any(m.values[10][0]>.9 for m in asset.materials))
        self.assertTrue(any(m.values[13][0]>.8 for m in asset.materials))
        self.assertTrue(any(m.values[13][3]==1 for m in asset.materials))
        self.assertTrue(any(m.values[16][0]>0 and 440<m.values[16][1]<460 for m in asset.materials))
        self.assertTrue(any(m.values[17][0]>.8 for m in asset.materials))
        self.assertTrue(any(m.values[19][2]==1 and sum(m.values[21][:3])>1 for m in asset.materials))
        self.assertTrue(any(len(mesh.indices)//3==800 and np.ptp(np.asarray(mesh.vertices)[:,2])>.1 for mesh in asset.meshes))

    def test_real_blender_usdc_keeps_native_material_graphs(self):
        path=Path(__file__).resolve().parent/"assets"/"blender"/"scene.usdc"
        asset=import_usd(path)
        self.assertTrue(asset.has_camera)
        self.assertGreater(math.dist(asset.camera[:3],asset.camera[4:7]),1)
        self.assertGreaterEqual(len(asset.meshes),5)
        self.assertTrue(all(material.program[2]==1 for material in asset.materials))
        self.assertEqual(len(asset.textures),2)
        self.assertTrue(all(any(max(pixel[:3])>0 for pixel in texture.pixels) for texture in asset.textures))
        self.assertTrue(any(len({tuple(pixel[:3]) for pixel in texture.pixels})>1 for texture in asset.textures))
        ops={code[0] for code,_ in asset.instructions}
        self.assertIn(Op.Image,ops);self.assertIn(Op.Normalmap,ops)


if __name__=="__main__": unittest.main()
