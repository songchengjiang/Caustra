#!/usr/bin/env python3
"""Generate parameterized path-tracing closure evaluators with MaterialX ShaderGen.

The official node graphs and Slang primitive implementations supply all reflection
and emission mathematics.  Only the graph's surface terminal is replaced by a
BSDF/EDF output.  Raster entry points, resources and environment approximations
are removed; Caustra owns path sampling, textures and volume transport.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import MaterialX as mx
import MaterialX.PyMaterialXGenShader as gs
import MaterialX.PyMaterialXGenSlang as slang

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caustra_assets.materialx_compiler import (SURFACE_INPUTS, OPENPBR_INPUTS,
    DISNEY_INPUTS, GLTF_INPUTS, USD_INPUTS, HAIR_INPUTS)

VERSION = "1.39.5"
MODELS = [
    (0, "standard_surface", "ND_standard_surface_surfaceshader", SURFACE_INPUTS),
    (1, "open_pbr_surface", "ND_open_pbr_surface_surfaceshader", OPENPBR_INPUTS),
    (2, "disney_principled", "ND_disney_principled", DISNEY_INPUTS),
    (3, "gltf_pbr", "ND_gltf_pbr_surfaceshader", GLTF_INPUTS),
    (4, "usd_preview_surface", "ND_UsdPreviewSurface_surfaceshader", USD_INPUTS),
    (5, "simple_hair", "ND_simple_hair_shader", HAIR_INPUTS),
]
ROWS = ["baseColor", "emission", "surface", "specular", "specularColor", "coat",
    "coatColor", "sheen", "sheenColor", "factors", "transmission",
    "transmissionColor", "transmissionScatter", "subsurface", "subsurfaceColor",
    "subsurfaceRadius", "thinFilm", "fuzz", "fuzzColor", "volume",
    "volumeAbsorption", "volumeScattering", "volumeEmission"]


def digest(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def closing_brace(text: str, opening: int) -> int:
    """Locate a source scope's end, ignoring comments and quoted strings."""
    token = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|[{}]', re.S)
    depth = 0
    for match in token.finditer(text, opening):
        if match.group() == "{": depth += 1
        elif match.group() == "}":
            depth -= 1
            if not depth: return match.end()
    raise ValueError("Unclosed generated Slang scope")


def remove_scope(source: str, pattern: str, required=False) -> str:
    match = re.search(pattern, source, re.M)
    if match is None:
        if required: raise ValueError("Missing generated scope: " + pattern)
        return source
    opening = source.index("{", match.start())
    end = closing_brace(source, opening)
    if source[end:end+1] == ";": end += 1
    return source[:match.start()] + source[end:]


def extract_compute_library(source: str, namespace: str) -> tuple[str, list[str]]:
    # These scopes implement only the unused raster ABI or texture environment
    # helpers.  Material graph textures are evaluated by the existing atlas IR.
    for pattern in [r'^struct SamplerTexture2D\b', r'^cbuffer pixelCB\b',
        r'^struct VertexData\b', r'^float4 fragmentMain\b',
        r'^float4 textureLod\b', r'^float4 texture\b', r'^float4 textureGrad\b',
        r'^int2 textureSize\b', r'^float3 mx_latlong_map_lookup\b']:
        source = remove_scope(source, pattern)
    source = re.sub(r'^static VertexData vd;\s*', '', source, flags=re.M)
    source = source.replace('[shader("fragment")]', '')
    match = re.search(r'^float3 mx_subsurface_scattering_approx\b', source, re.M)
    if match:
        opening = source.index("{", match.start()); end = closing_brace(source, opening)
        source = source[:opening] + ("{\n    // Caustra adapter: screen-space curvature (fwidth) has no meaning at\n"
            "    // a ray hit. The integrator routes SSS through its random walk;\n"
            "    // this generated surface evaluator contributes no local BSSRDF.\n"
            "    return float3(0.0);\n}") + source[end:]
    # Reflection and emission never execute the raster transmission branch.
    # Keep its helper well formed without retaining a renderer uniform.
    source = re.sub(r'\bu_refractionTwoSided\b', 'false', source)
    macros = sorted(set(re.findall(r'^#define\s+(\w+)', source, re.M)))
    # Namespaces do not isolate preprocessor macros.  Prefix every generated
    # macro so the checked-in libraries coexist with the old sampling helpers.
    renamed = {name: namespace.upper() + "_" + name for name in macros}
    source = re.sub(r'\b(' + '|'.join(map(re.escape, macros)) + r')\b',
        lambda m: renamed[m.group()], source)
    if re.search(r'\b(?:cbuffer|fragmentMain|SamplerTexture2D|VertexData)\b|\bvd\.', source):
        raise ValueError("Unexpected hardware interface remains in generated closure")
    # Only optional table lookup branches may refer to uniforms, and the
    # generator's analytic directional-albedo method removes them at compile.
    return source.strip() + "\n", list(renamed.values())


def make_closure(doc, original, nd, name, kind):
    custom = doc.addNodeDef("ND_caustra_" + name + "_" + kind, kind,
        "caustra_" + name + "_" + kind.lower())
    for inp in nd.getActiveInputs():
        custom.addInput(inp.getName(), inp.getType()).copyContentFrom(inp)
    graph = doc.addNodeGraph("NG_caustra_" + name + "_" + kind)
    graph.copyContentFrom(original)
    graph.setSourceUri("")
    graph.setNodeDef(custom)
    surface = next(n for n in graph.getNodes() if n.getCategory() == "surface")
    terminal = surface.getInput("bsdf" if kind == "BSDF" else "edf")
    if terminal is None or not terminal.getNodeName():
        doc.removeNodeGraph(graph.getName()); doc.removeNodeDef(custom.getName())
        return None
    for output in list(graph.getOutputs()): graph.removeOutput(output.getName())
    graph.addOutput("out", kind).setNodeName(terminal.getNodeName())
    graph.removeNode(surface.getName())
    if name == "standard_surface":
        # Caustra's public IR permits an independent coat tangent.  Standard
        # Surface otherwise uses its single tangent for both layers.  Promote
        # only the coat rotation's source; keep the official rotation and
        # normalization nodes, with the renderer supplying the shared default.
        custom.addInput("caustra_coat_tangent", "vector3")
        graph.getNode("coat_tangent_rotate").getInput("in").setInterfaceName("caustra_coat_tangent")
    if name == "usd_preview_surface":
        # The asset compiler has already evaluated UsdPreviewSurface's signed
        # tangent-space normal input into a world-space MaterialEvaluation.
        # Pass that result through rather than applying its normalmap twice.
        for node in graph.getNodes():
            for port in node.getInputs():
                if port.getNodeName() == "surface_normal":
                    port.removeAttribute("nodename"); port.setInterfaceName("normal")
        graph.removeNode("surface_normal")
    # Disney/USD/Hair graphs leave some primitive geometry inputs implicit.
    # Promote default world geometry properties to explicit per-hit arguments.
    for node in graph.getNodes():
        primitive = node.getNodeDef()
        if primitive is None: continue
        for port in primitive.getActiveInputs():
            geomprop = port.getAttribute("defaultgeomprop")
            if geomprop not in ("Nworld", "Tworld", "Bworld"): continue
            if node.getInput(port.getName()) is not None: continue
            interface = {"Nworld": "caustra_normal", "Tworld": "caustra_tangent", "Bworld": "caustra_bitangent"}[geomprop]
            if custom.getInput(interface) is None: custom.addInput(interface, "vector3")
            node.addInput(port.getName(), "vector3").setInterfaceName(interface)
    return custom, graph


def input_expression(name, port, mapping, model):
    special = {"normal": "normal", "coat_normal": "coatNormal",
        "clearcoat_normal": "coatNormal", "tangent": "tangent",
        "caustra_normal": "normal", "caustra_tangent": "tangent", "caustra_bitangent": "cross(normal, tangent)",
        "caustra_coat_tangent": "coatTangent",
        "geometry_normal": "normal", "geometry_coat_normal": "coatNormal",
        "geometry_tangent": "tangent", "geometry_coat_tangent": "coatTangent",
        "emission": "1.0", "emission_luminance": "1.0", "emissive_strength": "1.0",
        "emission_color": "m.emission.xyz", "emissive": "m.emission.xyz",
        "geometry_opacity": "m.factors.z", "opacity": "float3(m.factors.z)",
        # These are bulk dispersion inputs.  They do not participate in either
        # generated reflection or emission graph and remain core-owned.
        "transmission_dispersion": "0.0", "transmission_dispersion_scale": "0.0",
        "transmission_dispersion_abbe_number": "20.0"}
    if name in mapping:
        _, row, lane = mapping[name]
        expr = "m." + ROWS[row] + (".xyz" if lane is None else "." + "xyzw"[lane])
        if port.getType() == "boolean": expr = "(" + expr + " > 0.5)"
        elif port.getType() == "integer": expr = "int(round(" + expr + "))"
        # OpenPBR's input thickness is micrometers; the portable material ABI
        # stores nanometers to share its physical thin-film implementation.
        if model == "open_pbr_surface" and name == "thin_film_thickness": expr += " * 0.001"
        if name in ("specular_roughness", "coat_roughness", "roughness", "clearcoat_roughness", "clearcoatRoughness"):
            expr = "max(" + expr + ", 0.02)"
        if model == "disney_principled" and name == "clearcoatGloss":
            # The official graph maps gloss to roughness as 1 - gloss.
            # Apply the same finite roughness floor before that conversion so
            # the generated evaluator and finite GGX proposal remain aligned.
            expr = "min(" + expr + ", 0.98)"
        return expr
    if name in special: return special[name]
    # Inputs not exposed by our asset compiler must remain the official
    # NodeDef default.  Its compiler rejects nondefault active values.
    value = port.getValue()
    if value is None:
        if port.getType() == "float": return "0.0"
        raise ValueError("Unmapped input has no default: " + model + "." + name)
    if port.getType() == "boolean": return "true" if value else "false"
    if port.getType() == "integer": return str(int(value))
    if port.getType() == "float": return format(float(value), ".9g") + (".0" if float(value).is_integer() else "")
    if port.getType() in ("color3", "vector3"):
        return "float3(" + ", ".join(format(float(v), ".9g") for v in value) + ")"
    raise ValueError("Unsupported interface default: " + model + "." + name)


def generate(source_root: Path):
    if mx.getVersionString() != VERSION:
        raise RuntimeError("Expected MaterialX " + VERSION + "; got " + mx.getVersionString())
    library = mx.createDocument()
    search = mx.FileSearchPath(str(source_root))
    mx.loadLibraries(["libraries"], search, library)
    hair = source_root / "resources/Materials/Examples/SimpleHair/simple_hair_default.mtlx"
    hairdoc = mx.createDocument(); mx.readFromXmlFile(hairdoc, str(hair))
    for element in hairdoc.getChildren():
        if element.getCategory() in ("nodedef", "nodegraph"):
            library.addChildOfCategory(element.getCategory(), element.getName()).copyContentFrom(element)
    artifacts = {}
    models = []
    for model_id, name, nodedef_name, mapping in MODELS:
        nd = library.getNodeDef(nodedef_name)
        if nd is None: raise ValueError("Missing official NodeDef: " + nodedef_name)
        original = nd.getImplementation()
        doc = mx.createDocument(); doc.setDataLibrary(library)
        closures = [item for kind in ("BSDF", "EDF")
            if (item := make_closure(doc, original, nd, name, kind)) is not None]
        gen = slang.SlangShaderGenerator.create(); ctx = gs.GenContext(gen)
        ctx.registerSourceCodeSearchPath(search)
        options = ctx.getOptions()
        options.hwSpecularEnvironmentMethod = gs.HwSpecularEnvironmentMethod.SPECULAR_ENVIRONMENT_NONE
        options.hwMaxActiveLightSources = 0
        # One generated material connects both wrappers so ShaderGen emits their
        # primitive dependency union only once, with no hand merged functions.
        bsdfnode = doc.addNodeInstance(closures[0][0], "reflection")
        surf = doc.addNode("surface", "closure_surface", "surfaceshader")
        surf.addInput("bsdf", "BSDF").setNodeName(bsdfnode.getName())
        if len(closures) > 1:
            edfnode = doc.addNodeInstance(closures[1][0], "emission")
            surf.addInput("edf", "EDF").setNodeName(edfnode.getName())
        else:
            # Preserve the public renderer API's emission for every material.
            # This is a Caustra adapter, not part of Disney/Hair's source graph.
            edfnode = doc.addNode("uniform_edf", "caustra_emission_adapter", "EDF")
            surf.addInput("edf", "EDF").setNodeName(edfnode.getName())
        shader = gen.generate("caustra_" + name, surf, ctx)
        namespace = "caustra_mx_" + name
        code, macros = extract_compute_library(shader.getSourceCode(gs.PIXEL_STAGE), namespace)
        # The synthetic HwSurface function still executes a raster light loop.
        # Its body is at graph scope in fragmentMain, removed above.  It must not
        # have introduced light uniforms/functions with max lights == 0.
        metadata = []
        wrappers = []
        for kind in ("BSDF", "EDF"):
            item = next((x for x in closures if x[0].getType() == kind), None)
            output_type = "BSDF" if kind == "BSDF" else "float3"
            function_name = "reflection" if kind == "BSDF" else "emission"
            signature = ("float3 " + function_name + "(GpuMaterial m, float3 normal, float3 coatNormal, "
                "float3 tangent, float3 coatTangent, float3 view, float3 light)")
            if item is None:
                closure_type = namespace.upper() + "_CLOSURE_TYPE_EMISSION"
                wrappers.append(signature + "\n{\n    // Caustra general emission adapter; the source model has no EDF.\n"
                    "    ClosureData cd = makeClosureData(" + closure_type +
                    ", light, view, normal, float3(0.0), 1.0);\n    float3 result = float3(0.0);\n"
                    "    mx_uniform_edf(cd, m.emission.xyz, result);\n    return result;\n}\n")
                metadata.append({"name": "caustra_emission", "type": "color3",
                    "expression": "m.emission.xyz", "closure": "EDF", "adapter": "caustra_uniform_edf"})
                continue
            custom, graph = item
            inputs = list(custom.getActiveInputs())
            expressions = [input_expression(p.getName(), p, mapping, name) for p in inputs]
            metadata.extend({"name": p.getName(), "type": p.getType(), "expression": e, "closure": kind}
                for p, e in zip(inputs, expressions))
            closure_type = namespace.upper() + "_CLOSURE_TYPE_" + ("REFLECTION" if kind == "BSDF" else "EMISSION")
            init = "BSDF(float3(0.0), float3(1.0))" if kind == "BSDF" else "float3(0.0)"
            wrappers.append(signature + "\n{\n    ClosureData cd = makeClosureData(" + closure_type +
                ", light, view, normal, float3(0.0), 1.0);\n    " + output_type + " result = " + init +
                ";\n    " + graph.getName() + "(cd, " + ", ".join(expressions) + ", result);\n    return " +
                ("result.response" if kind == "BSDF" else "result") + ";\n}\n")
        filename = "materialx_" + name + ".slang"
        header = ("// Copyright Contributors to the MaterialX Project\n// SPDX-License-Identifier: Apache-2.0\n"
            "// Generated by tools/generate_materialx_surfaces.py; do not edit.\n"
            "// MaterialX 1.39.5 official Slang ShaderGen; renderer adapter described in manifest.\n")
        artifacts[filename] = header + "namespace " + namespace + "\n{\n" + code + "\n" + "\n".join(wrappers) + "}\n" + "\n".join("#undef " + m for m in macros) + "\n"
        models.append({"id": model_id, "name": name, "nodedef": nd.getName(), "nodegraph": original.getName(),
            "artifact": filename, "artifact_sha256": digest(artifacts[filename]), "inputs": metadata,
            "has_emission": len(closures) > 1,
            "emission_adapter": None if len(closures) > 1 else "caustra_uniform_edf"})
    aggregator = ["// SPDX-License-Identifier: Apache-2.0", "// Generated by tools/generate_materialx_surfaces.py; do not edit.",
        "#ifndef CAUSTRA_GENERATED_MATERIALX_SURFACES", "#define CAUSTRA_GENERATED_MATERIALX_SURFACES"]
    aggregator.extend('#include "' + m["artifact"] + '"' for m in models)
    for function in ("Reflection", "Emission"):
        aggregator.append("float3 mxGenerated" + function + "(GpuMaterial m, float3 normal, float3 coatNormal, float3 tangent, float3 coatTangent, float3 view, float3 light)\n{")
        for model in models:
            aggregator.append("    if (m.program.z == " + str(model["id"]) + ") return caustra_mx_" + model["name"] + "::" + function.lower() + "(m, normal, coatNormal, tangent, coatTangent, view, light);")
        aggregator.append("    return float3(0.0);\n}")
    aggregator.append("#endif\n")
    artifacts["materialx_surfaces.slang"] = "\n".join(aggregator)
    source_files = sorted((source_root / "libraries").rglob("*")) + [hair]
    manifest = {"materialx_version": VERSION, "generator": "MaterialXGenSlang.SlangShaderGenerator",
        "models": models, "source_files": [{"path": p.relative_to(source_root).as_posix(),
            "sha256": digest(p.read_bytes())} for p in source_files if p.is_file()],
        "adaptations": ["Clone official surface node graph with BSDF and EDF terminals; all NodeDef inputs retained.",
            "Remove raster ABI, texture environment helpers and fragment entry; use analytic directional albedo and no environment shader.",
            "Prefix macros and namespace each model; supply per-hit normal/tangent parameters.",
            "Standard Surface coat tangent source is promoted to the existing Caustra independent CoatTangent API; official rotation and normalization remain intact.",
            "UsdPreviewSurface normalmap is already evaluated by the asset IR; connect its resolved world normal directly.",
            "Finite GGX proposal adapter floors specular and coat roughness to 0.02; Disney inverse clearcoatGloss is capped at 0.98.",
            "Remove screen-space SSS curvature helper (zero local response); bulk subsurface is integrated by the core random walk.",
            "Existing asset IR supplies parameters and linear textures; folded emission uses unit strength and stored RGB.",
            "Disney and Hair have no source EDF; Caustra general emission API uses the official uniform_edf primitive.",
            "Core owns sampling/PDF, geometric transmission and random-walk subsurface/volume integration."]}
    artifacts["materialx_surfaces.json"] = json.dumps(manifest, indent=2) + "\n"
    return artifacts


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
        default=root / "build/deps/materialx-official/MaterialX-1.39.5")
    parser.add_argument("--output", type=Path, default=root / "shaders/generated")
    parser.add_argument("--check", action="store_true", help="Verify committed artifacts without writing")
    args = parser.parse_args()
    artifacts = generate(args.source_root.resolve())
    if not args.check: args.output.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        target = args.output / name
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                raise RuntimeError("Generated artifact differs: " + str(target))
        else: target.write_text(content, encoding="utf-8", newline="\n")
    print(("Verified" if args.check else "Generated") + " six MaterialX " + VERSION + " closure models")


if __name__ == "__main__": main()
