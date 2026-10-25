"""Small independent PBRT parser/transform/PLY regressions, without downloads."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from caustra_assets.model import AssetError
from caustra_assets.pbrt_scene import read_commands,load_scene,read_ply


class PbrtSceneTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)

    def tearDown(self): self.temporary.cleanup()

    def write(self,name,text):
        path=self.root/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_text(text,encoding="utf-8");return path

    def test_include_comments_quotes_and_source_lines(self):
        path=self.write("scene.pbrt",'WorldBegin\nInclude "geometry/mesh.pbrt" # comment\n')
        self.write("geometry/mesh.pbrt",'\nMakeNamedMaterial "a # quoted" "string type" "diffuse"\nNamedMaterial "a # quoted"\nShape "plymesh" "string filename" ["geometry/a.ply"]\n')
        scene=load_scene(path)
        self.assertEqual(len(scene.source_files),2)
        self.assertEqual(scene.shapes[0].command.line,4)
        self.assertEqual(scene.shapes[0].material,"a # quoted")
        self.assertEqual(scene.shapes[0].command.params["filename"].values,("geometry/a.ply",))

    def test_camera_handedness_world_reset_and_column_major_matrix(self):
        path=self.write("scene.pbrt",'''Scale -1 1 1
LookAt 2 3 4 2 4 4 0 0 1
Camera "perspective" "float fov" 60
WorldBegin
MakeNamedMaterial "m" "string type" "diffuse"
NamedMaterial "m"
Shape "plymesh" "string filename" "a.ply"
AttributeBegin
Transform [1 0 0 0 0 2 0 0 0 0 3 0 5 6 7 1]
Shape "plymesh" "string filename" "a.ply"
AttributeEnd
Shape "plymesh" "string filename" "a.ply"
''')
        scene=load_scene(path)
        self.assertTrue(np.array_equal(scene.shapes[0].transform,np.eye(4)))
        self.assertTrue(np.array_equal(scene.shapes[2].transform,np.eye(4)))
        self.assertTrue(np.array_equal(scene.shapes[1].transform[:3,3],(5,6,7)))
        self.assertTrue(np.array_equal(scene.shapes[1].transform[:3,:3],np.diag((1,2,3))))
        camera=scene.camera.world_from_camera
        self.assertTrue(np.allclose(camera[:3,3],(2,3,4)))
        self.assertTrue(np.allclose(camera[:3,2],(0,1,0)))
        usd_camera=camera@np.diag((1,1,-1,1))
        self.assertAlmostEqual(np.linalg.det(usd_camera[:3,:3]),1)
        self.assertTrue(np.allclose(usd_camera[:3,0],(1,0,0)))

    def test_redefined_texture_preserves_original_material_binding(self):
        path=self.write("scene.pbrt",'''WorldBegin
Texture "map" "spectrum" "imagemap" "string filename" "first.png"
MakeNamedMaterial "first" "string type" "diffuse" "texture reflectance" "map"
Texture "map" "spectrum" "imagemap" "string filename" "second.png"
MakeNamedMaterial "second" "string type" "diffuse" "texture reflectance" "map"
NamedMaterial "first"
Shape "plymesh" "string filename" "a.ply"
NamedMaterial "second"
Shape "plymesh" "string filename" "b.ply"
''')
        scene=load_scene(path)
        first=scene.materials["first"].params["reflectance"].value
        second=scene.materials["second"].params["reflectance"].value
        self.assertNotEqual(first,second)
        self.assertEqual(scene.textures[first].params["filename"].value,"first.png")
        self.assertEqual(scene.textures[second].params["filename"].value,"second.png")
        self.assertEqual(len(scene.renamed_definitions),1)

    def test_object_instances_and_alpha_material_identity(self):
        path=self.write("scene.pbrt",'''WorldBegin
Texture "cutout" "float" "imagemap" "string filename" "alpha.png"
MakeNamedMaterial "leaf" "string type" "diffuse"
NamedMaterial "leaf"
ObjectBegin "plant"
Shape "plymesh" "string filename" "a.ply" "texture alpha" "cutout"
ObjectEnd
Translate 1 2 3
ObjectInstance "plant"
Translate 1 0 0
ObjectInstance "plant"
''')
        scene=load_scene(path)
        self.assertEqual(len(scene.shapes),0)
        self.assertEqual(len(scene.objects["plant"]),1)
        self.assertEqual(len(scene.instances),2)
        self.assertEqual(scene.objects["plant"][0].material,"leaf__alpha_cutout")
        self.assertEqual(scene.materials["leaf__alpha_cutout"].params["alpha"].type,"texture")
        self.assertTrue(np.array_equal(scene.instances[0].transform[:3,3],(1,2,3)))
        self.assertTrue(np.array_equal(scene.instances[1].transform[:3,3],(2,2,3)))

    def test_unknown_features_and_invalid_scopes_fail(self):
        bad=['WorldBegin\nShape "sphere"','WorldBegin\nAttributeEnd',
             'WorldBegin\nActiveTransform StartTime','WorldBegin\nMediumInterface "a" "b"',
             'WorldBegin\nObjectInstance "missing"', 'WorldBegin\nInclude "../escape.pbrt"']
        for text in bad:
            with self.subTest(text=text),self.assertRaises(AssetError): load_scene(self.write("bad.pbrt",text))

    def test_include_cycle_fails(self):
        path=self.write("a.pbrt",'WorldBegin\nInclude "b.pbrt"')
        self.write("b.pbrt",'Include "a.pbrt"')
        with self.assertRaisesRegex(AssetError,"Cyclic"): load_scene(path)

    def test_binary_ply_keeps_uv_normals_and_rejects_nontriangles(self):
        path=self.root/"triangle.ply"
        header=('ply\nformat binary_little_endian 1.0\nelement vertex 3\n'+''.join('property float '+s+'\n' for s in ('x','y','z','nx','ny','nz','u','v'))+'element face 1\nproperty list uint8 int vertex_indices\nend_header\n').encode()
        vertices=np.array([[0,0,0,0,0,1,0,0],[2,0,0,0,0,1,1,0],[0,3,0,0,0,1,0,1]],dtype='<f4')
        path.write_bytes(header+vertices.tobytes()+struct.pack('<Biii',3,0,1,2))
        points,indices,normals,uv=read_ply(path)
        self.assertTrue(np.array_equal(points,vertices[:,:3]))
        self.assertTrue(np.array_equal(normals,vertices[:,3:6]))
        self.assertTrue(np.array_equal(uv,vertices[:,6:8]))
        self.assertTrue(np.array_equal(indices,[[0,1,2]]))
        path.write_bytes(header+vertices.tobytes()+struct.pack('<Biii',4,0,1,2))
        with self.assertRaisesRegex(AssetError,"triangles"): read_ply(path)

    def test_usd_native_instances_do_not_render_prototype_source(self):
        from pxr import Usd,UsdGeom
        stage=Usd.Stage.CreateInMemory();stage.CreateClassPrim('/__Prototypes')
        UsdGeom.Xform.Define(stage,'/__Prototypes/Leaf')
        UsdGeom.Mesh.Define(stage,'/__Prototypes/Leaf/Mesh')
        for i in range(2):
            prim=UsdGeom.Xform.Define(stage,f'/World/Leaf{i}').GetPrim()
            prim.GetReferences().AddInternalReference('/__Prototypes/Leaf');prim.SetInstanceable(True)
        paths=[str(p.GetPath()) for p in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies())]
        self.assertEqual([p for p in paths if p.endswith('/Mesh')],['/World/Leaf0/Mesh','/World/Leaf1/Mesh'])
        self.assertEqual(len(stage.GetPrototypes()),1)

    def test_vertex_deduplication_preserves_uv_and_normal_seams(self):
        from convert_sanmiguel import _deduplicate_vertices
        points=np.array([[0,0,0]]*4,dtype=np.float32)
        normals=np.array([[0,0,1],[0,0,1],[0,0,1],[0,1,0]],dtype=np.float32)
        uv=np.array([[0,0],[0,0],[1,0],[0,0]],dtype=np.float32)
        original=np.array([[0,2,3],[1,2,3]],dtype=np.int32)
        p,i,n,t=_deduplicate_vertices((points,original,normals,uv))
        self.assertEqual(len(p),3)
        self.assertTrue(np.array_equal(i[0],i[1]))
        self.assertTrue(np.array_equal(p[i],points[original]))
        self.assertTrue(np.array_equal(n[i],normals[original]))
        self.assertTrue(np.array_equal(t[i],uv[original]))

    def test_closed_and_open_dielectric_topology_selects_explicit_override(self):
        from convert_sanmiguel import _dielectric_topology
        path=self.write('scene.pbrt','''WorldBegin
MakeNamedMaterial "solid" "string type" "dielectric"
MakeNamedMaterial "sheet" "string type" "dielectric"
NamedMaterial "solid"
Shape "plymesh" "string filename" "solid.ply"
NamedMaterial "sheet"
Shape "plymesh" "string filename" "sheet.ply"
''')
        scene=load_scene(path)
        points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        triangles=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=np.int32)
        data={self.root/'solid.ply':(points,triangles,None,None),self.root/'sheet.ply':(points,triangles[:1],None,None)}
        overrides,records=_dielectric_topology(scene,data)
        self.assertEqual(set(overrides),{'sheet'})
        self.assertTrue(records[0]['closed_consistent_positive_volume'])
        self.assertEqual(records[1]['boundary_edges'],3)

    def test_triangle_crossing_requires_actual_interpenetration(self):
        from convert_sanmiguel import _strict_triangle_crossings
        points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)
        indices=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=np.int32)
        triangles=points[indices]
        self.assertIsNotNone(_strict_triangle_crossings(triangles,(points+(.2,.1,.15))[indices]))
        self.assertIsNone(_strict_triangle_crossings(triangles,(points*.1+(.1,.1,.1))[indices]))
        self.assertIsNone(_strict_triangle_crossings(triangles,(points+(.6,.6,.6))[indices]))
        self.assertIsNone(_strict_triangle_crossings(triangles,(points+(1,0,0))[indices]))

    def test_self_intersection_evidence_excludes_shared_vertex_adjacency(self):
        from convert_sanmiguel import _strict_triangle_crossings
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[.2,.2,-1],[.2,.2,1],[.8,.2,0]],dtype=np.float64)
        indices=np.array([[0,1,2],[3,4,5]],dtype=np.int32);triangles=vertices[indices]
        witness=_strict_triangle_crossings(triangles,triangles,indices,indices,True)
        self.assertIsNotNone(witness)
        self.assertNotEqual(witness['first_triangle'],witness['second_triangle'])
        # The same geometric test excludes authored adjacency when requested;
        # touching edges/vertices cannot count as proof of a bad closed volume.
        adjacent=indices.copy();adjacent[1,0]=indices[0,0]
        self.assertIsNone(_strict_triangle_crossings(triangles,triangles,adjacent,adjacent,True))

    def test_overlap_material_override_changes_only_proven_conflicting_placements(self):
        from convert_sanmiguel import _overlapping_dielectrics
        path=self.write('scene.pbrt','''WorldBegin
MakeNamedMaterial "glass" "string type" "dielectric"
NamedMaterial "glass"
ObjectBegin "object"
Shape "plymesh" "string filename" "solid.ply"
ObjectEnd
ObjectInstance "object"
Translate .2 .1 .15
ObjectInstance "object"
Translate 2 0 0
ObjectInstance "object"
''')
        scene=load_scene(path);points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        indices=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=np.int32)
        overrides={};mapping,report=_overlapping_dielectrics(scene,{self.root/'solid.ply':(points,indices,None,None)},overrides)
        self.assertEqual(set(mapping),{('object_instance',0),('object_instance',1)})
        self.assertEqual(report['confirmed_intersecting_pairs'],1)
        self.assertEqual(scene.materials['glass'].params['type'].value,'dielectric')
        self.assertIn('glass__interpenetrating_instance_thin',overrides)


if __name__=='__main__': unittest.main()
