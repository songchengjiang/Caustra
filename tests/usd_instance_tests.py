"""Native USD instance sharing, including inherited material and UV opinions."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from caustra_assets.package import read_asset, write_asset
from caustra_assets.usd_import import import_usd, triangulate


class UsdInstanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    @staticmethod
    def material(stage,path,color):
        material=UsdShade.Material.Define(stage,path)
        surface=UsdShade.Shader.Define(stage,path+"/surface")
        surface.CreateIdAttr("ND_standard_surface_surfaceshader")
        surface.CreateInput("base_color",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        surface.CreateOutput("out",Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput("mtlx").ConnectToSource(surface.ConnectableAPI(),"out")
        return material,surface

    def prototype(self,internal_material=True,subsets=False,root_input=False):
        path=self.directory/"prototype.usda";stage=Usd.Stage.CreateNew(str(path))
        root=UsdGeom.Xform.Define(stage,"/Model");stage.SetDefaultPrim(root.GetPrim())
        mesh=UsdGeom.Mesh.Define(stage,"/Model/mesh")
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreatePointsAttr([(0,0,0),(1,0,0),(1,1,0),(0,1,0)])
        mesh.CreateFaceVertexCountsAttr([3,3]);mesh.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
        if internal_material:
            material,surface=self.material(stage,"/Model/red",(1,0,0))
            binding=UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim());binding.Bind(material)
            if root_input:
                interface=UsdShade.ConnectableAPI(root.GetPrim())
                interface.CreateInput("tint",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1,0,0))
                surface.GetInput("base_color").ConnectToSource(interface,"tint",UsdShade.AttributeType.Input)
            if subsets:
                blue,_=self.material(stage,"/Model/blue",(0,0,1))
                subset=binding.CreateMaterialBindSubset("blue_face",[1])
                UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(blue)
        stage.GetRootLayer().Save()
        return path

    def scene(self,prototype,count=3):
        path=self.directory/"scene.usda";stage=Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.y)
        roots=[]
        for i in range(count):
            root=UsdGeom.Xform.Define(stage,"/instance_%d" % i)
            root.GetPrim().GetReferences().AddReference(str(prototype))
            root.GetPrim().SetInstanceable(True);root.AddTranslateOp().Set(Gf.Vec3d(i*2,0,0))
            roots.append(root)
        return path,stage,roots

    def test_many_native_instances_share_internal_material_and_geometry(self):
        path,stage,roots=self.scene(self.prototype(),40);stage.GetRootLayer().Save()
        self.assertTrue(all(root.GetPrim().IsInstance() for root in roots))
        with patch("caustra_assets.usd_import.triangulate",wraps=triangulate) as triangulation:
            asset=import_usd(path)
        self.assertEqual(len(asset.instances),40)
        self.assertEqual(len(asset.meshes),1);self.assertEqual(len(asset.materials),1)
        self.assertEqual(triangulation.call_count,2)
        self.assertEqual({index for index,_ in asset.instances},{0})
        self.assertEqual([matrix[3] for _,matrix in asset.instances],list(range(0,80,2)))
        self.assertEqual(asset.bounds_min[:3],[0,0,0]);self.assertEqual(asset.bounds_max[:3],[79,1,0])
        package=self.directory/"instances.caustra";write_asset(asset,package);loaded=read_asset(package)
        self.assertEqual(len(loaded.meshes),1);self.assertEqual(len(loaded.instances),40)
        self.assertEqual(loaded.instances,asset.instances)

    def test_ancestor_material_override_preserves_subsets_and_instance_count(self):
        path,stage,roots=self.scene(self.prototype(subsets=True),3)
        green,_=self.material(stage,"/green",(0,1,0))
        UsdShade.MaterialBindingAPI.Apply(roots[2].GetPrim()).Bind(green,UsdShade.Tokens.strongerThanDescendants)
        stage.GetRootLayer().Save();asset=import_usd(path)
        # Two ordinary instances each have red/blue subsets; the root override
        # coalesces the third instance's two faces into one green mesh.
        self.assertEqual(len(asset.instances),5);self.assertEqual(len(asset.meshes),3)
        self.assertEqual([index for index,_ in asset.instances[:4]],[0,1,0,1])
        green_mesh=asset.meshes[asset.instances[-1][0]]
        self.assertEqual(asset.materials[green_mesh.material].values[0][:3],[0,1,0])
        self.assertEqual(len(green_mesh.indices),6)

    def test_inherited_uvs_do_not_share_when_instance_root_values_differ(self):
        path,stage,roots=self.scene(self.prototype(),2)
        for root,uv in zip(roots,((.1,.2),(.7,.8))):
            UsdGeom.PrimvarsAPI(root).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,"constant").Set([uv])
        stage.GetRootLayer().Save();asset=import_usd(path)
        self.assertEqual(len(asset.instances),2);self.assertEqual(len(asset.meshes),2)
        for (index,_),expected in zip(asset.instances,((.1,.2),(.7,.8))):
            for vertex in asset.meshes[index].vertices:
                for actual,value in zip(vertex[8:10],expected): self.assertAlmostEqual(actual,value,places=6)

    def test_internal_graph_connected_to_instance_root_input_retains_override(self):
        path,stage,roots=self.scene(self.prototype(root_input=True),3)
        UsdShade.ConnectableAPI(roots[1].GetPrim()).GetInput("tint").Set(Gf.Vec3f(0,0,1))
        stage.GetRootLayer().Save();asset=import_usd(path)
        self.assertEqual(len(asset.instances),3);self.assertEqual(len(asset.meshes),2)
        colors=[asset.materials[asset.meshes[index].material].values[0][:3] for index,_ in asset.instances]
        self.assertEqual(colors,[[1,0,0],[0,0,1],[1,0,0]])
        self.assertEqual(asset.instances[0][0],asset.instances[2][0])

    def test_displacement_shares_local_fields_but_respects_world_fields(self):
        prototype=self.prototype(internal_material=False)
        for world in (False,True):
            with self.subTest(world=world):
                path,stage,roots=self.scene(prototype,3)
                # The first and third instance have the same full transform.
                roots[2].GetOrderedXformOps()[0].Set(Gf.Vec3d(0,0,0))
                material,_=self.material(stage,"/displaced",(1,1,1))
                displacement=UsdShade.Shader.Define(stage,"/displaced/offset")
                displacement.CreateIdAttr("ND_displacement_float")
                displacement.CreateOutput("out",Sdf.ValueTypeNames.Token)
                height=displacement.CreateInput("displacement",Sdf.ValueTypeNames.Float)
                if world:
                    position=UsdShade.Shader.Define(stage,"/displaced/position");position.CreateIdAttr("ND_position_vector3")
                    position.CreateInput("space",Sdf.ValueTypeNames.String).Set("world")
                    position.CreateOutput("out",Sdf.ValueTypeNames.Vector3f)
                    extract=UsdShade.Shader.Define(stage,"/displaced/x");extract.CreateIdAttr("ND_extract_vector3")
                    extract.CreateInput("in",Sdf.ValueTypeNames.Vector3f).ConnectToSource(position.ConnectableAPI(),"out")
                    extract.CreateInput("index",Sdf.ValueTypeNames.Int).Set(0)
                    extract.CreateOutput("out",Sdf.ValueTypeNames.Float)
                    height.ConnectToSource(extract.ConnectableAPI(),"out")
                else: height.Set(.25)
                material.CreateDisplacementOutput("mtlx").ConnectToSource(displacement.ConnectableAPI(),"out")
                for root in roots: UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(material)
                stage.GetRootLayer().Save();asset=import_usd(path,displacement_subdivisions=0)
                self.assertEqual(len(asset.instances),3);self.assertEqual(len(asset.meshes),2 if world else 1)
                self.assertEqual(asset.instances[0][0],asset.instances[2][0])
                for instance,(index,matrix) in enumerate(asset.instances):
                    for vertex in asset.meshes[index].vertices:
                        expected=vertex[0]+matrix[3] if world else .25
                        self.assertAlmostEqual(vertex[2],expected,places=6)
                self.assertAlmostEqual(asset.bounds_max[2],3 if world else .25,places=6)
                del stage

    def test_transformed_bounds_contain_all_shared_vertices(self):
        path,stage,roots=self.scene(self.prototype(),2)
        roots[1].AddRotateXYZOp().Set(Gf.Vec3f(25,35,45))
        roots[1].AddScaleOp().Set(Gf.Vec3f(-2,3,.5))
        stage.GetRootLayer().Save();asset=import_usd(path)
        self.assertEqual(len(asset.meshes),1)
        for index,matrix in asset.instances:
            for vertex in asset.meshes[index].vertices:
                for r in range(3):
                    value=sum(matrix[r*4+c]*vertex[c] for c in range(3))+matrix[r*4+3]
                    self.assertGreaterEqual(value,asset.bounds_min[r]-1e-6)
                    self.assertLessEqual(value,asset.bounds_max[r]+1e-6)


if __name__=="__main__": unittest.main()
