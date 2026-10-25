"""Offline PBRT material adapter regressions; no GPU or downloaded assets needed."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from pxr import Usd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from caustra_assets.materialx_compiler import MaterialCompiler, srgb_to_linear
from caustra_assets.model import Asset, AssetError, Op, Parameter
from caustra_assets.pattern_cpu import evaluate_program
from caustra_assets.pbrt_materials import MaterialLibrary
from caustra_assets.usd_import import materialx_from_usd


def definition(kind, **params):
    """Each test supplies (PBRT type, values) so intent is unambiguous."""
    return {"type": kind,
            "params": {key:tuple(value[1]) for key, value in params.items()},
            "param_types": {key:value[0] for key, value in params.items()}}


class PbrtMaterialsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root/"source"
        self.output = self.root/"converted"
        self.compile_count = 0
        (self.source/"textures").mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def image(self, name, pixels):
        filename = "textures/"+name+".png"
        Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(self.source/filename)
        return filename

    def library(self, materials, textures=None, **options):
        result = MaterialLibrary(materials, textures or {}, self.source, self.output, **options)
        result.build()
        return result

    def compile(self, library, name, usd=False):
        asset = Asset()
        compiler = MaterialCompiler(asset)
        if usd:
            self.compile_count += 1
            usd_path = self.output/("materials%d.usda" % self.compile_count)
            stage = Usd.Stage.CreateNew(str(usd_path))
            materials = library.write(stage)
            stage.GetRootLayer().Save()
            # Re-open from disk so asset path resolution is exercised.
            stage = Usd.Stage.Open(str(usd_path))
            material = type(materials[name])(stage.GetPrimAtPath(materials[name].GetPath()))
            document, surface = materialx_from_usd(material, compiler.libraries)
        else:
            document, surface = library.document, library.surfaces[name]
        compiler.compile(document, surface, self.output)
        return asset

    def parameters(self, asset, uv=((.5, .5),), normal=((0, 0, 1),), tangent=((1, 0, 0),)):
        registers = evaluate_program(asset.instructions, asset.textures, uv=uv, normal=normal, tangent=tangent)
        return {code[2]: registers[index] for index, (code, _) in enumerate(asset.instructions) if code[0] == Op.Output}

    def test_diffuse_texture_srgb_and_original_bytes_survive_usd(self):
        filename = self.image("albedo", [[[128, 64, 32]]])
        texture = definition("imagemap", filename=("string", [filename]))
        texture["value_type"] = "spectrum"
        material = definition("diffuse", reflectance=("texture", ["albedo"]))
        library = self.library({"paint":material}, {"albedo":texture})
        asset = self.compile(library, "paint", usd=True)
        color = self.parameters(asset)[Parameter.BaseColor][0, :3]
        np.testing.assert_allclose(color, srgb_to_linear(np.array([128, 64, 32])/255), atol=1e-7)
        self.assertEqual(asset.materials[0].values[3][0], 0)
        self.assertEqual((self.source/filename).read_bytes(), (self.output/filename).read_bytes())
        self.assertEqual(library.report["textures"][0]["color_space"], "srgb_texture")

    def test_uv_scale_and_offset_preserve_lower_left_convention(self):
        filename = self.image("quadrants", [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])
        texture = definition("imagemap", filename=("string", [filename]), encoding=("string", ["linear"]),
                             uscale=("float", [2]), vscale=("float", [2]), udelta=("float", [.25]), vdelta=("float", [.25]))
        texture["value_type"] = "spectrum"
        library = self.library({"test":definition("diffuse", reflectance=("texture", ["quadrants"]))}, {"quadrants":texture})
        values = self.parameters(self.compile(library, "test"), uv=((0, 0), (.25, .25)))[Parameter.BaseColor]
        np.testing.assert_allclose(values[:, :3], [[0, 0, 1], [0, 1, 0]], atol=1e-7)

    def test_scalar_rgb_average_and_rgba_alpha_are_distinct(self):
        for alpha in (None, 1, 64, 255):
            with self.subTest(alpha=alpha):
                pixel = [128, 64, 0]+([] if alpha is None else [alpha])
                filename = self.image("scalar"+str(alpha), [[pixel]])
                texture = definition("imagemap", filename=("string", [filename]))
                texture["value_type"] = "float"
                library = self.library({"cutout":definition("diffuse", alpha=("texture", ["scalar"]))}, {"scalar":texture})
                opacity = self.parameters(self.compile(library, "cutout", usd=True))[Parameter.Opacity][0, 0]
                # PBRT's PNG reader decodes the A channel with the requested
                # color encoding too; this differs from MaterialX alpha rules.
                expected = float(srgb_to_linear(np.array([128, 64, 0])/255).mean()) if alpha in (None, 255) else float(srgb_to_linear(alpha/255))
                self.assertAlmostEqual(opacity, expected, places=6)

    def test_point_scalar_uses_first_color_channel_even_with_alpha(self):
        filename = self.image("point", [[[128, 64, 0, 32]]])
        texture = definition("imagemap", filename=("string", [filename]), filter=("string", ["point"]))
        texture["value_type"] = "float"
        library = self.library({"cutout":definition("diffuse", alpha=("texture", ["point"]))}, {"point":texture})
        opacity = self.parameters(self.compile(library, "cutout", usd=True))[Parameter.Opacity][0, 0]
        self.assertAlmostEqual(opacity, float(srgb_to_linear(128/255)), places=6)
        self.assertEqual(library.report["textures"][0]["scalar_channel"], "red")

    def test_explicit_gamma_decoding_is_a_live_graph(self):
        filename = self.image("gamma", [[[128, 128, 128]]])
        texture = definition("imagemap", filename=("string", [filename]), encoding=("string", ["gamma 2"]))
        texture["value_type"] = "spectrum"
        library = self.library({"test":definition("diffuse", reflectance=("texture", ["gamma"]))}, {"gamma":texture})
        color = self.parameters(self.compile(library, "test"))[Parameter.BaseColor][0, :3]
        np.testing.assert_allclose(color, (128/255)**2, atol=1e-7)

    def test_coated_roughness_matches_pbrt_alpha_mapping(self):
        library = self.library({"coated":definition("coateddiffuse", roughness=("float", [.0625]))})
        asset = self.compile(library, "coated", usd=True)
        roughness = self.parameters(asset)[Parameter.Roughness][0, 0]
        self.assertAlmostEqual(roughness, .5, places=7)
        self.assertAlmostEqual(roughness*roughness, np.sqrt(.0625), places=7)
        self.assertEqual(asset.materials[0].values[3][0], 1)
        self.assertIn("finite-thickness", library.report["materials"][0]["approximation"])

    def test_diffuse_transmission_defaults_are_quarter_reflectance_and_transmittance(self):
        library = self.library({"default_leaf":definition("diffusetransmission")})
        asset = self.compile(library, "default_leaf")
        outputs = self.parameters(asset)
        np.testing.assert_allclose(outputs[Parameter.BaseColor][0, :3]*.5, .25)
        np.testing.assert_allclose(outputs[Parameter.SubsurfaceColor][0, :3]*.5, .25)

    def test_open_glass_override_is_explicit_and_keeps_original_source_type(self):
        library = self.library({"pane":definition("dielectric", eta=("float", [1.5]))},
                               thin_walled_overrides={"pane":"Exact position weld found 126 boundary edges"})
        asset = self.compile(library, "pane", usd=True)
        self.assertEqual(asset.materials[0].values[13][3], 1)
        self.assertEqual(asset.materials[0].values[3][1], 1.5)
        record = library.report["materials"][0]
        self.assertEqual(record["pbrt_type"], "dielectric")
        self.assertIn("126 boundary", record["thin_walled_override"]["reason"])
        self.assertTrue(any(a["feature"] == "thin_walled_override" for a in library.report["approximations"]))

    def test_thin_lambert_transmission_preserves_both_colors(self):
        material = definition("diffusetransmission", reflectance=("rgb", [.1, .2, .3]),
                              transmittance=("rgb", [.6, .4, .2]), scale=("float", [.5]))
        library = self.library({"leaf":material})
        asset = self.compile(library, "leaf", usd=True)
        evaluated = self.parameters(asset)
        r = evaluated[Parameter.BaseColor][0, :3]
        t = evaluated[Parameter.SubsurfaceColor][0, :3]
        s = asset.materials[0].values[13][0]
        self.assertEqual(s, .5)
        self.assertEqual(asset.materials[0].values[13][3], 1)
        self.assertEqual(asset.materials[0].values[3][0], 0)
        np.testing.assert_allclose((1-s)*r, [.05, .1, .15], atol=1e-7)
        np.testing.assert_allclose(s*t, [.3, .2, .1], atol=1e-7)

    def test_bump_bake_has_correct_sign_and_geometric_scale(self):
        # Clamp borders and a 4x4 ramp give a constant positive du at interior.
        ramp = np.tile(np.array([0, 64, 128, 192], dtype=np.uint8), (4, 1))
        filename = self.image("height", np.stack([ramp]*3, axis=-1))
        height = definition("imagemap", filename=("string", [filename]), encoding=("string", ["linear"]), wrap=("string", ["clamp"]))
        height["value_type"] = "float"
        material = definition("diffuse", displacement=("texture", ["height"]))
        library = self.library({"wall":material}, {"height":height}, bump_metrics={"wall":(10, 10)})
        record = library.report["materials"][0]["bump"]
        with Image.open(self.output/record["normal_file"]) as normal:
            x = np.asarray(normal)[1, 1, 0]/255*2-1
        self.assertLess(x, 0)
        self.assertAlmostEqual(x, -(4*64/255/10)/np.sqrt(1+(4*64/255/10)**2), delta=.005)
        self.assertIn("height", library.texture_nodes)
        self.assertEqual(record["metric_dPdu_dPdv"], [10, 10])
        self.assertIn("varying metric", record["approximation"])
        self.compile(library, "wall", usd=True)

    def test_bump_vertical_derivative_accounts_for_image_origin_and_uv_mirroring(self):
        ramp = np.tile(np.array([0, 64, 128, 192], dtype=np.uint8)[:, None], (1, 4))
        filename = self.image("vertical", np.stack([ramp]*3, axis=-1))
        for scale in (1, -1):
            with self.subTest(vscale=scale):
                height = definition("imagemap", filename=("string", [filename]), encoding=("string", ["linear"]),
                                    wrap=("string", ["clamp"]), vscale=("float", [scale]))
                height["value_type"] = "float"
                library = self.library({"wall":definition("diffuse", displacement=("texture", ["height"]))},
                                       {"height":height}, bump_metrics={"wall":(10, 10)})
                record = library.report["materials"][0]["bump"]
                with Image.open(self.output/record["normal_file"]) as normal:
                    y = np.asarray(normal)[1, 1, 1]/255*2-1
                self.assertGreater(y*scale, 0)

    def test_unknown_active_inputs_fail_explicitly(self):
        for material in (definition("unknown"), definition("diffuse", invented=("float", [1]))):
            with self.subTest(material=material), self.assertRaisesRegex(AssetError, "Unsupported PBRT"):
                self.library({"broken":material})
        filename = self.image("bad", [[[255, 255, 255]]])
        texture = definition("imagemap", filename=("string", [filename]), unknown=("float", [1]))
        texture["value_type"] = "spectrum"
        with self.assertRaisesRegex(AssetError, "Unsupported PBRT.*unknown"):
            self.library({"bad":definition("diffuse", reflectance=("texture", ["bad"]))}, {"bad":texture})

    def test_source_parameters_and_bake_record_are_json_serializable(self):
        material = definition("diffuse", sigma=("float", [30]))
        library = self.library({"legacy":material})
        library.write_mtlx(self.output/"materials.mtlx")
        self.assertIn("unused_legacy_sigma", library.report["materials"][0])
        self.assertIn('"sigma": [30]', json.dumps(library.report))
        self.assertTrue((self.output/"materials.mtlx").is_file())


if __name__ == "__main__":
    unittest.main()
