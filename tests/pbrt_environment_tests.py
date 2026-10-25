"""Angular HDR conversion, linear texture limits and backdrop bounds contracts."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from caustra_assets.pbrt_environment import equal_area_direction, convert_to_rec709, sky_patches, palette_patches
from caustra_assets.materialx_compiler import TextureStore
from caustra_assets.model import Asset
from caustra_assets.package import read_asset, write_asset
from caustra_assets.usd_import import import_usd


class EnvironmentTests(unittest.TestCase):
    def test_mapping_poles_seams_and_unit_length(self):
        np.testing.assert_allclose(equal_area_direction(.5,.5),(0,0,1),atol=1e-15)
        np.testing.assert_allclose(equal_area_direction(0,0),(0,0,-1),atol=1e-15)
        for u in np.linspace(0,1,65):
            for v in np.linspace(0,1,65):
                self.assertAlmostEqual(float(np.linalg.norm(equal_area_direction(u,v))),1,places=13)
        for t in np.linspace(0,1,17):
            np.testing.assert_allclose(equal_area_direction(0,t),equal_area_direction(0,1-t),atol=1e-14)

    def test_equal_area_z_distribution(self):
        # Uniform square samples must have uniform z (mean z^2 = 1/3).
        points=np.array([equal_area_direction((x+.5)/128,(y+.5)/128) for y in range(128) for x in range(128)])
        self.assertAlmostEqual(float(points[:,2].mean()),0,places=12)
        self.assertAlmostEqual(float((points[:,2]**2).mean()),1/3,delta=1e-4)

    def test_white_adaptation_and_identity_primaries(self):
        white=np.ones((2,2,3))
        ap0=(.7347,.2653,0.,1.,.0001,-.077,.32168,.33767)
        rgb,report=convert_to_rec709(white,ap0)
        np.testing.assert_allclose(rgb,white,atol=1e-12)
        rng=np.random.default_rng(5);p=rng.random((4,4,3))
        rgb,_=convert_to_rec709(p,(.64,.33,.30,.60,.15,.06,.3127,.3290))
        np.testing.assert_allclose(rgb,p,atol=1e-14)

    def test_sun_refinement_and_rgb_integral(self):
        rng=np.random.default_rng(7)
        sky=rng.random((128,128,3))*.1
        sky[33,51]=(12000,10000,8000)
        patches=sky_patches(sky)
        self.assertTrue(any(p["pixel_rect"]==(51,33,1) for p in patches))
        palette=palette_patches(patches)
        self.assertLessEqual(len(palette),256)
        total=sum((c*sum(p["weight"] for p in members) for c,members in palette),np.zeros(3))
        np.testing.assert_allclose(total,sky.sum(axis=(0,1)),rtol=2e-13)
        self.assertGreater(max(c[0] for c,members in palette),10000)

    def test_runtime_texture_limit_averages_linear_samples_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"image.png"
            Image.fromarray(np.array([[[0,0,0,255],[255,255,255,255]]],dtype=np.uint8)).save(path)
            digest=hashlib.sha256(path.read_bytes()).digest()
            asset=Asset();store=TextureStore(asset);store.max_size=1
            store.add(path,"srgb")
            self.assertEqual((asset.textures[0].width,asset.textures[0].height),(1,1))
            np.testing.assert_allclose(asset.textures[0].pixels[0],(.5,.5,.5,1),atol=1e-7)
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(),digest)
            write_asset(asset,Path(temp)/"test.caustra")
            np.testing.assert_allclose(read_asset(Path(temp)/"test.caustra").textures[0].pixels,[[.5,.5,.5,1]])

    def test_backdrop_is_rendered_but_does_not_expand_orbit_bounds(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"scene.usda";stage=Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage,1.)
            UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.y)
            material=UsdShade.Material.Define(stage,"/Material")
            shader=UsdShade.Shader.Define(stage,"/Material/surface")
            shader.CreateIdAttr("ND_standard_surface_surfaceshader");shader.CreateOutput("out",Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput("mtlx").ConnectToSource(shader.ConnectableAPI(),"out")
            for name,scale,backdrop in (("Object",1,False),("Sky",10000,True)):
                mesh=UsdGeom.Mesh.Define(stage,"/"+name)
                mesh.CreatePointsAttr([(0,0,0),(scale,0,0),(0,scale,0)])
                mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2]);mesh.CreateSubdivisionSchemeAttr("none")
                mesh.GetPrim().SetCustomDataByKey("caustra:environmentBackdrop",backdrop)
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            stage.GetRootLayer().Save();asset=import_usd(path)
            self.assertEqual(len(asset.meshes),2);self.assertEqual(len(asset.instances),2)
            np.testing.assert_allclose(asset.bounds_max,(1,1,0,0))
            self.assertEqual(max(v[0] for v in asset.meshes[1].vertices),10000)


if __name__=="__main__": unittest.main()
