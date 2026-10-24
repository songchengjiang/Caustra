"""Official inventory, offline fetch, and generated gallery CPU regressions."""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import MaterialX as mx
from pxr import Usd, UsdGeom

TOOLS=Path(__file__).resolve().parents[1]/"tools"
sys.path.insert(0,str(TOOLS))
from caustra_assets.materialx_compiler import standard_library
from caustra_assets.model import AssetError, Material
from caustra_assets.usd_import import import_usd
SPEC=importlib.util.spec_from_file_location("gallery_builder",TOOLS/"build_materialx_gallery.py")
GALLERY=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(GALLERY)
FETCH_SPEC=importlib.util.spec_from_file_location("gallery_fetch",TOOLS/"fetch_materialx_examples.py")
FETCH=importlib.util.module_from_spec(FETCH_SPEC); FETCH_SPEC.loader.exec_module(FETCH)
INVENTORY_PATH=Path(__file__).resolve().parent/"assets/materialx_official_inventory.json"


class OfficialInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected=json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))

    def source_root(self):
        override=os.environ.get("CAUSTRA_MATERIALX_SOURCE")
        candidate=Path(override) if override else TOOLS.parent/"build/deps/materialx-official/MaterialX-1.39.5"
        if not candidate.exists():
            if override: self.fail("CAUSTRA_MATERIALX_SOURCE does not exist: "+str(candidate))
            self.skipTest("Pinned official sources are absent; set CAUSTRA_MATERIALX_SOURCE to verify them")
        return GALLERY.examples_root(candidate)

    def test_checked_in_inventory_is_complete_and_unambiguous(self):
        expected=self.expected
        self.assertEqual(expected["schema_version"],1)
        self.assertEqual(expected["upstream"]["tag"],"v1.39.5")
        self.assertEqual(expected["upstream"]["archive_sha256"],FETCH.SHA256)
        self.assertEqual(expected["counts"],{"files":50,"entries":66,"unique_source_identities":62})
        files=expected["files"]; entries=expected["entries"]
        paths={item["path"] for item in files}
        self.assertEqual(len(paths),50); self.assertEqual(len(files),50); self.assertEqual(len(entries),66)
        self.assertEqual(len({entry["source_identity"] for entry in entries}),62)
        self.assertEqual(len({(e["source"],e["surface"]) for e in entries}),66)
        for item in files:
            path=Path(item["path"])
            self.assertFalse(path.is_absolute()); self.assertNotIn("..",path.parts)
            self.assertRegex(item["sha256"],r"^[0-9a-f]{64}$")
        for entry in entries:
            self.assertIn(entry["source"],paths)
            origin,surface=entry["source_identity"].split("#")
            self.assertIn(origin,paths); self.assertEqual(surface,entry["surface"])
        family_counts=Counter(Path(entry["source"]).parts[0] for entry in entries)
        self.assertEqual(family_counts,{"DisneyPrincipled":5,"GltfPbr":6,"OpenPbr":10,
                                        "SimpleHair":1,"StandardSurface":38,"UsdPreviewSurface":6})

    def test_real_pinned_sources_match_every_file_digest(self):
        root=self.source_root()
        expected={entry["path"]:entry["sha256"] for entry in self.expected["files"]}
        actual={path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*.mtlx")}
        self.assertEqual(actual,expected)
        self.assertEqual({path.relative_to(root).as_posix() for path in GALLERY.inventory(root)},set(expected))

    def test_real_generator_enumerates_all_recorded_material_roots(self):
        root=self.source_root(); actual=[]; libraries=standard_library()
        # Exercise the real build loop without loading large textures or
        # constructing geometry. Expected entries come from the checked-in
        # inventory, authored independently from the source XML and XIncludes.
        class EnumeratingCompiler:
            def __init__(self,asset):
                self.asset=asset; self.libraries=libraries

            def compile(self,document,surface,base_directory):
                source=Path(document.getSourceUri()).resolve()
                origin=Path(surface.getSourceUri()) if surface.getSourceUri() else source
                if not origin.is_absolute(): origin=Path(base_directory)/origin
                actual.append({"source":source.relative_to(root).as_posix(),
                               "surface":surface.getName(),
                               "source_identity":origin.resolve().relative_to(root).as_posix()+"#"+surface.getName(),
                               "surface_category":surface.getCategory()})
                index=len(self.asset.materials);self.asset.materials.append(Material());return index
        with tempfile.TemporaryDirectory() as temp:
            options=SimpleNamespace(source_root=root,output=Path(temp)/"gallery",texture_max_size=512,validate_only=True)
            with mock.patch.object(GALLERY,"MaterialCompiler",EnumeratingCompiler),contextlib.redirect_stdout(io.StringIO()):
                GALLERY.build(options)
        key=lambda entry:(entry["source"],entry["surface"])
        self.assertEqual(sorted(actual,key=key),sorted(self.expected["entries"],key=key))
        self.assertEqual(len({entry["source_identity"] for entry in actual}),62)


class OfflineFetchTests(unittest.TestCase):
    def archive(self,destination,members):
        path=destination/("MaterialX-"+FETCH.VERSION+".zip")
        with zipfile.ZipFile(path,"w") as bundle:
            for name,contents in members: bundle.writestr(name,contents)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_fetch_rejects_checksum_before_extracting_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            destination=Path(temp)
            self.archive(destination,[("must_not_extract.txt","content")])
            with mock.patch.object(FETCH.urllib.request,"urlopen",side_effect=AssertionError("Network must not be used")):
                with self.assertRaisesRegex(RuntimeError,"checksum mismatch"): FETCH.fetch(destination)
            self.assertFalse((destination/"must_not_extract.txt").exists())

    def test_fetch_validates_every_zip_path_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            destination=Path(temp)/"cache";destination.mkdir()
            digest=self.archive(destination,[("safe.txt","content"),("../escaped.txt","unsafe")])
            with mock.patch.object(FETCH,"SHA256",digest),mock.patch.object(FETCH.urllib.request,"urlopen",side_effect=AssertionError("Network must not be used")):
                with self.assertRaisesRegex(RuntimeError,"Unsafe archive member"): FETCH.fetch(destination)
            self.assertFalse((destination/"safe.txt").exists())
            self.assertFalse((Path(temp)/"escaped.txt").exists())

    def test_verified_cached_zip_extracts_and_reuses_pinned_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            destination=Path(temp)
            relative="MaterialX-"+FETCH.VERSION+"/resources/Materials/Examples/StandardSurface/standard_surface_chess_set.mtlx"
            digest=self.archive(destination,[(relative,"<materialx/>"),("MaterialX-"+FETCH.VERSION+"/LICENSE","license")])
            with mock.patch.object(FETCH,"SHA256",digest),mock.patch.object(FETCH.urllib.request,"urlopen",side_effect=AssertionError("Network must not be used")),contextlib.redirect_stdout(io.StringIO()):
                source=FETCH.fetch(destination)
                self.assertEqual((destination/relative).read_text(),"<materialx/>")
                self.assertEqual((source/"LICENSE").read_text(),"license")
                self.assertEqual(FETCH.fetch(destination),source)


class GeneratedGalleryTests(unittest.TestCase):
    def test_raised_labels_face_overview_and_clear_sample_geometry(self):
        backing=GALLERY.Mesh(0); lettering=GALLERY.Mesh(1)
        entry={"id":45,"family":"OpenPbr","name":"open_pbr_lightbulb"}
        result=GALLERY.label_plate(backing,lettering,entry,0,0,0)
        plate=np.asarray(backing.vertices); text=np.asarray(lettering.vertices)
        direction=np.array((0,math.sin(math.radians(60)),math.cos(math.radians(60))))
        self.assertGreater(plate[:,1].min(),.29)
        self.assertGreater(plate[:,2].min(),.84)
        self.assertLess(plate[:,2].max(),1.6)
        self.assertTrue(np.allclose(text[:,4:7]@direction,1,atol=1e-8))
        up=np.array((0,.5,-math.sin(math.radians(60))))
        local_height=(text[:,:3]-np.array((0,.47,1.35)))@up
        self.assertGreater(local_height.max(),.20)
        self.assertLess(local_height.min(),-.20)
        self.assertEqual(result["display_name"],"lightbulb")
        self.assertEqual(result["line_height"],.20)

    def test_emissive_bulb_uses_small_geometry_without_scaling_other_samples(self):
        bulb=GALLERY.sample_scale({"source":"OpenPbr/open_pbr_lightbulb.mtlx"})
        self.assertAlmostEqual(.84*bulb,.126)
        self.assertGreater(.84*bulb,.10);self.assertLess(.84*bulb,.15)
        self.assertEqual(GALLERY.sample_scale({"source":"OpenPbr/open_pbr_glass.mtlx"}),1)
        self.assertEqual(GALLERY.sample_scale({"source":"StandardSurface/standard_surface_glass.mtlx"}),1)

    def check_closed(self, mesh):
        vertices=np.asarray(mesh.vertices,dtype=float)
        self.assertTrue(np.isfinite(vertices).all())
        self.assertTrue(np.allclose(np.linalg.norm(vertices[:,4:7],axis=1),1,atol=1e-7))
        positions=[tuple(np.round(v[:3],10)) for v in vertices]
        edges=Counter()
        for a,b,c in np.asarray(mesh.indices).reshape(-1,3):
            p,q,r=vertices[[a,b,c],:3]
            n=np.cross(q-p,r-p)
            self.assertGreater(np.linalg.norm(n),1e-12)
            self.assertGreater(np.dot(n,vertices[[a,b,c],4:7].sum(axis=0)),0)
            for i,j in ((a,b),(b,c),(c,a)):
                edges[(positions[i],positions[j])]+=1
        for (a,b),count in edges.items():
            self.assertEqual(count,1)
            self.assertEqual(edges[(b,a)],1)

    def test_sphere_is_closed_with_exact_seam_and_outward_winding(self):
        mesh=GALLERY.sphere(0,16,8)
        self.check_closed(mesh)
        for ring in range(9):
            first=mesh.vertices[ring*17]; last=mesh.vertices[ring*17+16]
            self.assertEqual(first[:7],last[:7])
            self.assertEqual(first[8],0); self.assertEqual(last[8],1)

    def test_sphere_uv_matches_radial_extension(self):
        mesh=GALLERY.sphere(0,16,8)
        for vertex in mesh.vertices:
            x,y,z=vertex[:3]
            if x*x+z*z<1e-20:
                continue
            u=(math.atan2(z,x)/(2*math.pi))%1
            v=1-math.acos(y/.84)/math.pi
            difference=abs(vertex[8]-u)%1
            self.assertLess(min(difference,1-difference),1e-6)
            self.assertAlmostEqual(vertex[9],v,places=6)

    def test_hair_ribbons_are_open_with_longitudinal_tangents(self):
        mesh=GALLERY.hair_bundle(0)
        vertices=np.asarray(mesh.vertices)
        edges=Counter()
        for a,b,c in np.asarray(mesh.indices).reshape(-1,3):
            p,q,r=vertices[[a,b,c],:3]
            self.assertGreater(np.dot(np.cross(q-p,r-p),vertices[[a,b,c],4:7].sum(axis=0)),0)
            for i,j in ((a,b),(b,c),(c,a)):
                edges[tuple(sorted((i,j)))]+=1
        self.assertGreater(sum(count==1 for count in edges.values()),0)
        self.assertTrue(all(count in (1,2) for count in edges.values()))
        self.assertTrue(np.allclose(np.linalg.norm(vertices[:,12:15],axis=1),1,atol=1e-7))
        self.assertTrue(np.allclose(np.sum(vertices[:,4:7]*vertices[:,12:15],axis=1),0,atol=1e-7))
        # At the broad fiber sides, positive u follows the longitudinal axis.
        self.assertGreater(np.median(vertices[:,13]),.9)

    def test_usd_roundtrip_preserves_local_nodedef_defaults(self):
        libraries=standard_library(); document=mx.createDocument(); document.setDataLibrary(libraries)
        definition=document.addNodeDef("ND_gallery_local_surface")
        base=libraries.getNodeDef("ND_standard_surface_surfaceshader")
        definition.setNodeString("standard_surface")
        for port in base.getActiveInputs():
            definition.addInput(port.getName(),port.getType()).copyContentFrom(port)
        for port in base.getActiveOutputs():
            target=definition.getOutput(port.getName()) or definition.addOutput(port.getName(),port.getType())
            target.copyContentFrom(port)
        definition.getInput("base_color").setValueString("0.12, 0.34, 0.56")
        surface=document.addNodeInstance(definition,"local_surface")
        surface.addInput("specular_roughness","float").setValue(.72)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"scene.usda"; stage=Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage,1); UsdGeom.SetStageUpAxis(stage,"Y")
            writer=GALLERY.UsdMaterialWriter(stage,Path(temp)/"resources")
            material=writer.material("/Material",document,surface,Path(temp))
            GALLERY.usd_mesh(stage,"/Sphere",GALLERY.sphere(0,8,4),GALLERY.IDENTITY,material)
            stage.GetRootLayer().Save()
            compiled=import_usd(path)
            self.assertTrue(np.allclose(compiled.materials[0].values[0][:3],[.12,.34,.56]))
            self.assertAlmostEqual(compiled.materials[0].values[2][1],.72,places=6)

    def test_conflicting_shared_interior_uv_mappings_fail_explicitly(self):
        libraries=standard_library(); document=mx.createDocument(); document.setDataLibrary(libraries)
        surface=document.addNodeInstance(libraries.getNodeDef("ND_standard_surface_surfaceshader"),"surface")
        surface.addInput("subsurface","float").setValue(.5)
        uv=document.addNodeInstance(libraries.getNodeDef("ND_noise3d_float"),"noise")
        combine=document.addNodeInstance(libraries.getNodeDef("ND_convert_float_color3"),"color")
        combine.addInput("in","float").setConnectedNode(uv)
        surface.addInput("subsurface_color","color3").setConnectedNode(combine)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"scene.usda"; stage=Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage,1); UsdGeom.SetStageUpAxis(stage,"Y")
            writer=GALLERY.UsdMaterialWriter(stage,Path(temp)/"resources")
            material=writer.material("/Material",document,surface,Path(temp))
            first=GALLERY.usd_mesh(stage,"/SphereA",GALLERY.sphere(0,8,4),GALLERY.IDENTITY,material)
            first.GetPrim().SetCustomDataByKey("caustra:sphericalInteriorUV",True)
            transform=GALLERY.IDENTITY.copy(); transform[3]=3
            GALLERY.usd_mesh(stage,"/SphereB",GALLERY.sphere(0,8,4),transform,material)
            stage.GetRootLayer().Save()
            with self.assertRaisesRegex(AssetError,"cannot mix spherical and ordinary"):
                import_usd(path)


if __name__=="__main__":
    unittest.main()
