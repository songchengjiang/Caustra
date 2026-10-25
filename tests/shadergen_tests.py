"""CPU checks for the official MaterialX-to-path-tracer source generation.

These verify provenance, reproducibility and dependence on upstream graphs.
They do not replace numeric BSDF or rendered-image validation on a GPU.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools/generate_materialx_surfaces.py"
CHECKED_IN = ROOT / "shaders/generated"
MODELS = {
    "standard_surface": 0,
    "open_pbr_surface": 1,
    "disney_principled": 2,
    "gltf_pbr": 3,
    "usd_preview_surface": 4,
    "simple_hair": 5,
}


def without_comments(source):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)


class OfficialShaderGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        override = os.environ.get("CAUSTRA_MATERIALX_SOURCE")
        cls.source = Path(override) if override else ROOT / "build/deps/materialx-official/MaterialX-1.39.5"
        if not cls.source.is_dir():
            if override:
                raise AssertionError("CAUSTRA_MATERIALX_SOURCE does not exist: " + str(cls.source))
            raise unittest.SkipTest("Pinned sources absent; set CAUSTRA_MATERIALX_SOURCE to regenerate official shaders")
        cls.temp = tempfile.TemporaryDirectory(prefix="caustra-shadergen-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.output = Path(cls.temp.name) / "generated"
        cls.run_generator(cls.source, cls.output)
        cls.manifest = json.loads((cls.output / "materialx_surfaces.json").read_text(encoding="utf-8"))

    @classmethod
    def run_generator(cls, source, output, *extra):
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--source-root", str(source), "--output", str(output), *extra],
            cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True,
        )
        if result.returncode:
            raise AssertionError("Official ShaderGen failed:\n" + result.stdout + result.stderr)
        return result

    @staticmethod
    def artifact_bytes(directory):
        return {path.relative_to(directory).as_posix(): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    def test_manifest_pins_version_and_all_six_distinct_models(self):
        self.assertEqual(self.manifest["materialx_version"], "1.39.5")
        self.assertIn("Slang", self.manifest["generator"])
        models = self.manifest["models"]
        self.assertEqual(len(models), 6)
        self.assertEqual({model["name"]: model["id"] for model in models}, MODELS)
        self.assertEqual(len({model["artifact"] for model in models}), 6)
        self.assertEqual(len({model["nodedef"] for model in models}), 6)
        self.assertEqual(len({model["nodegraph"] for model in models}), 6)
        for model in models:
            with self.subTest(model=model["name"]):
                self.assertTrue(model["nodedef"])
                self.assertTrue(model["nodegraph"])
                self.assertTrue(model["inputs"], "Model must expose evaluated per-hit inputs")
                artifact = Path(model["artifact"])
                self.assertFalse(artifact.is_absolute())
                self.assertNotIn("..", artifact.parts)
                digest = hashlib.sha256((self.output / artifact).read_bytes()).hexdigest()
                self.assertEqual(model["artifact_sha256"], digest)

    def test_source_provenance_matches_actual_official_files(self):
        files = self.manifest["source_files"]
        self.assertTrue(files, "Generated code needs upstream source provenance")
        self.assertEqual(len({item["path"] for item in files}), len(files))
        for item in files:
            with self.subTest(source=item["path"]):
                relative = Path(item["path"])
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                self.assertEqual(hashlib.sha256((self.source / relative).read_bytes()).hexdigest(), item["sha256"])
        self.assertTrue(self.manifest["adaptations"], "Path-tracer adaptations must be explicit")

    def test_boundary_inputs_preserve_normals_emission_alpha_and_units(self):
        models = {model["name"]: model for model in self.manifest["models"]}
        expected = {
            "standard_surface": {"normal": "normal", "coat_normal": "coatNormal",
                "emission": "1.0", "emission_color": "m.emission.xyz",
                "thin_film_thickness": "m.thinFilm.y"},
            "open_pbr_surface": {"geometry_normal": "normal", "geometry_coat_normal": "coatNormal",
                "emission_luminance": "1.0", "emission_color": "m.emission.xyz",
                "thin_film_thickness": "m.thinFilm.y * 0.001"},
            "disney_principled": {"baseColor": "m.baseColor.xyz", "roughness": "max(m.surface.y, 0.02)",
                # Disney uses inverse roughness; its gloss cap must match the
                # finite .02 roughness used by the coat sampling proposal.
                "clearcoatGloss": "min(m.coat.y, 0.98)"},
            "gltf_pbr": {"normal": "normal", "clearcoat_normal": "coatNormal",
                "emissive": "m.emission.xyz", "emissive_strength": "1.0", "alpha": "m.factors.z"},
            "usd_preview_surface": {"diffuseColor": "m.baseColor.xyz", "emissiveColor": "m.emission.xyz",
                "opacity": "m.factors.z"},
            "simple_hair": {"caustra_normal": "normal", "caustra_tangent": "tangent",
                "tint_R": "m.specularColor.xyz", "tint_TT": "m.transmissionColor.xyz",
                "tint_TRT": "m.sheenColor.xyz"},
        }
        for name, bindings in expected.items():
            metadata = models[name]["inputs"]
            for input_name, expression in bindings.items():
                with self.subTest(model=name, input=input_name):
                    occurrences = [item["expression"] for item in metadata if item["name"] == input_name]
                    self.assertTrue(occurrences, "Missing native input " + input_name)
                    self.assertEqual(set(occurrences), {expression})

    def test_outputs_are_reproducible_and_match_checked_in_artifacts(self):
        second = Path(self.temp.name) / "second"
        self.run_generator(self.source, second)
        self.assertEqual(self.artifact_bytes(self.output), self.artifact_bytes(second))
        self.assertEqual(self.artifact_bytes(self.output), self.artifact_bytes(CHECKED_IN))
        self.run_generator(self.source, self.output, "--check")

    def test_generated_functions_have_no_raster_entry_or_hidden_resources(self):
        sources = list(self.output.glob("*.slang"))
        self.assertGreaterEqual(len(sources), 7)
        forbidden = (r"\b(?:uniform|cbuffer|fragmentMain|vertexMain|SV_Target|SamplerTexture2D|"
                     r"Texture2D|SamplerState|RWStructuredBuffer|StructuredBuffer)\b|\[shader\(")
        for path in sources:
            with self.subTest(artifact=path.name):
                self.assertNotRegex(without_comments(path.read_text(encoding="utf-8")), forbidden)
        aggregator = without_comments((self.output / "materialx_surfaces.slang").read_text(encoding="utf-8"))
        self.assertRegex(aggregator, r"\bmxGeneratedReflection\s*\(")
        self.assertRegex(aggregator, r"\bmxGeneratedEmission\s*\(")

    def test_check_rejects_stale_output_without_rewriting_it(self):
        directory = Path(self.temp.name) / "stale"
        shutil.copytree(self.output, directory)
        target = directory / self.manifest["models"][0]["artifact"]
        target.write_bytes(target.read_bytes() + b"\n// stale artifact\n")
        before = self.artifact_bytes(directory)
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--source-root", str(self.source), "--output", str(directory), "--check"],
            cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0, "--check accepted an edited generated shader")
        self.assertEqual(before, self.artifact_bytes(directory), "--check must be read-only")

    def test_official_graph_semantic_edit_changes_only_its_generated_model(self):
        # A copied source tree intentionally changes Disney's specular strength.
        # This detects stale/handwritten stand-ins which merely claim ShaderGen
        # provenance; the material formula itself must depend on upstream XML.
        source = Path(self.temp.name) / "modified-source"
        shutil.copytree(self.source / "libraries", source / "libraries")
        hair = Path("resources/Materials/Examples/SimpleHair")
        shutil.copytree(self.source / hair, source / hair)
        for name in ("LICENSE", "README.md"):
            if (self.source / name).exists():
                shutil.copy2(self.source / name, source / name)
        path = source / "libraries/bxdf/disney_principled.mtlx"
        tree = ET.parse(path)
        strength = tree.find("./nodegraph[@name='NG_disney_principled']/multiply[@name='dielectric_intensity']/input[@name='in2']")
        self.assertIsNotNone(strength)
        self.assertEqual(float(strength.get("value")), .08)
        strength.set("value", "0.123456")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        output = Path(self.temp.name) / "modified-generated"
        self.run_generator(source, output)
        for model in self.manifest["models"]:
            before = (self.output / model["artifact"]).read_text(encoding="utf-8")
            after = (output / model["artifact"]).read_text(encoding="utf-8")
            with self.subTest(model=model["name"]):
                if model["name"] == "disney_principled":
                    self.assertNotEqual(without_comments(before), without_comments(after))
                    self.assertRegex(after, r"\b0\.123456\b")
                else:
                    self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
