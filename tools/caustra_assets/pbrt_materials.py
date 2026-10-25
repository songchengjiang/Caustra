"""PBRT v4 San Miguel materials -> editable MaterialX and USD mtlx networks.

This is an offline adapter, deliberately separate from the rendering library.
Only explicitly handled PBRT parameters are accepted.  Model differences and
the height-to-normal bake are recorded per material in ``MaterialLibrary.report``.
Reference semantics: https://github.com/mmp/pbrt-v4/tree/master/src/pbrt
"""
from pathlib import Path
import hashlib
import math
import shutil

import MaterialX as mx
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, UsdShade

from .materialx_compiler import standard_library, srgb_to_linear
from .model import AssetError


TYPE_NAMES = {
    "float": Sdf.ValueTypeNames.Float, "integer": Sdf.ValueTypeNames.Int,
    "boolean": Sdf.ValueTypeNames.Bool, "string": Sdf.ValueTypeNames.String,
    "filename": Sdf.ValueTypeNames.Asset, "vector2": Sdf.ValueTypeNames.Float2,
    "vector3": Sdf.ValueTypeNames.Vector3f, "color3": Sdf.ValueTypeNames.Color3f,
    "vector4": Sdf.ValueTypeNames.Float4, "color4": Sdf.ValueTypeNames.Color4f,
}


def export_material(stage, path, surface):
    """Embed the connected MaterialX NodeDef graph as standard UsdShade shaders.

    Filename inputs are relative to the USD/MaterialX output directory.  Callers
    keep both files together; no runtime MaterialX source-asset import is needed.
    """
    material = UsdShade.Material.Define(stage, path)
    translated = {}

    def convert(node):
        key = node.getNamePath()
        if key in translated:
            return translated[key]
        definition = node.getNodeDef()
        if definition is None:
            raise AssetError("No MaterialX NodeDef for " + key)
        shader = UsdShade.Shader.Define(stage, str(path)+"/"+mx.createValidName(key.replace("/", "_")))
        shader.CreateIdAttr(definition.getName())
        translated[key] = shader
        for output in definition.getActiveOutputs():
            shader.CreateOutput(output.getName(), TYPE_NAMES.get(output.getType(), Sdf.ValueTypeNames.Token))
        if not shader.GetOutputs():
            shader.CreateOutput("out", TYPE_NAMES.get(node.getType(), Sdf.ValueTypeNames.Token))
        for port in node.getInputs():
            target = shader.CreateInput(port.getName(), TYPE_NAMES.get(port.getType(), Sdf.ValueTypeNames.Token))
            if port.getConnectedNode() is not None:
                target.ConnectToSource(convert(port.getConnectedNode()).ConnectableAPI(), port.getOutputString() or "out")
            elif port.hasValueString():
                value = port.getValue()
                kind = port.getType()
                if kind == "filename":
                    value = Sdf.AssetPath(port.getValueString())
                elif kind == "boolean":
                    value = port.getValueString().lower() == "true"
                elif kind in ("vector2", "vector3", "color3", "vector4", "color4"):
                    value = {"vector2":Gf.Vec2f, "vector3":Gf.Vec3f, "color3":Gf.Vec3f,
                             "vector4":Gf.Vec4f, "color4":Gf.Vec4f}[kind](*value)
                target.Set(value)
            else:
                raise AssetError("Unresolved generated MaterialX port " + port.getNamePath())
            if port.getColorSpace():
                target.GetAttr().SetMetadata("colorSpace", port.getColorSpace())
        return shader

    material.CreateSurfaceOutput("mtlx").ConnectToSource(convert(surface).ConnectableAPI(), "out")
    return material


def _single(definition, name, default=None):
    values = definition.get("params", {}).get(name)
    if values is None:
        return default
    if len(values) != 1:
        raise AssetError("PBRT parameter %s requires one value" % name)
    return values[0]


def _finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise AssetError("Nonfinite PBRT " + label)
    return result


def _boolean(value, label):
    if value in (True, "true"):
        return True
    if value in (False, "false"):
        return False
    raise AssetError("Expected PBRT boolean for " + label)


def _known(definition, allowed, label):
    unknown = set(definition.get("params", {}))-set(allowed)
    if unknown:
        raise AssetError("Unsupported PBRT %s parameters: %s" % (label, ", ".join(sorted(unknown))))


class MaterialLibrary:
    """Compile parser-neutral material/texture dictionaries from pbrt_scene.

    A definition has ``type``, ``params`` (name -> tuple), and ``param_types``
    (name -> PBRT type). Texture definitions also have ``value_type``.
    ``bump_metrics`` contains area-weighted representative dPdu/dPdv lengths in
    scene units for each material; the converter computes these from geometry.
    """
    def __init__(self, materials, textures, source_dir, output_dir, bump_metrics=None, thin_walled_overrides=None):
        self.materials = materials
        self.textures = textures
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bump_metrics = bump_metrics or {}
        self.thin_walled_overrides = thin_walled_overrides or {}
        self.libraries = standard_library()
        self.document = mx.createDocument()
        self.document.setDataLibrary(self.libraries)
        self.document.setColorSpace("lin_rec709")
        self.surfaces = {}
        self.texture_nodes = {}
        self.active_textures = set()
        self.copy_cache = {}
        self.image_info = {}
        self.baked_normals = {}
        self.report = {"materials": [], "textures": [], "approximations": [],
                       "semantics_reference": "https://github.com/mmp/pbrt-v4/tree/master/src/pbrt",
                       "scalar_image_rule": "PBRT bilinear/trilinear: non-opaque RGBA alpha, otherwise RGB average or grayscale; point/EWA: first RGB channel. Caustra mip filtering differs from PBRT's special coarsest-level first-channel lookup.",
                       "uv_rule": "u=uscale*u+udelta, v=vscale*v+vdelta; image origin upper-left, UV origin lower-left"}

    def node(self, category, kind, name, **inputs):
        node = self.document.addNode(category, self.document.createValidChildName(mx.createValidName(name)), kind)
        # These categories overload input vector widths independently of their
        # output type. Select the intended standard definition explicitly.
        if category == "dotproduct":
            node.setNodeDefString("ND_dotproduct_vector3")
        elif category == "normalmap":
            node.setNodeDefString("ND_normalmap_float")
        for key, value in inputs.items():
            self.set(node, key, value)
        return node

    def set(self, node, name, value):
        definition = node.getNodeDef()
        expected = definition.getActiveInput(name) if definition else None
        if expected is None:
            raise AssetError("Unknown generated MaterialX input %s.%s" % (node.getCategory(), name))
        port = node.getInput(name) or node.addInput(name, expected.getType())
        if isinstance(value, mx.Node):
            port.setConnectedNode(value)
        else:
            if isinstance(value, (tuple, list, np.ndarray)):
                value = ", ".join(str(float(v)) for v in value)
            elif isinstance(value, bool):
                value = "true" if value else "false"
            else:
                value = str(value)
            port.setValueString(value)
        return port

    def binary(self, category, kind, name, a, b):
        return self.node(category, kind, name, in1=a, in2=b)

    def convert(self, value, kind, name):
        if isinstance(value, mx.Node):
            if value.getType() == kind:
                return value
            node = self.document.addNode("convert", self.document.createValidChildName(mx.createValidName(name)), kind)
            node.addInput("in", value.getType()).setConnectedNode(value)
            return node
        if kind in ("color3", "vector3") and not isinstance(value, (tuple, list)):
            return (value, value, value)
        return value

    def _resource(self, filename):
        source = (self.source_dir/filename).resolve()
        try:
            relative = source.relative_to(self.source_dir)
        except ValueError as error:
            raise AssetError("PBRT texture escapes source directory: " + str(filename)) from error
        if not source.is_file():
            raise AssetError("PBRT texture does not exist: " + str(source))
        destination = self.output_dir/relative
        if source not in self.copy_cache:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source != destination:
                shutil.copy2(source, destination)
            self.copy_cache[source] = relative.as_posix()
        return source, self.copy_cache[source]

    def _image(self, name, definition):
        _known(definition, {"filename", "mapping", "uscale", "vscale", "udelta", "vdelta", "wrap",
                            "maxanisotropy", "encoding", "gamma", "filter", "scale", "invert"}, "texture " + name)
        if _single(definition, "mapping", "uv") != "uv":
            raise AssetError("Only PBRT UV mappings are supported: " + name)
        filename = str(_single(definition, "filename", ""))
        source, relative = self._resource(filename)
        if source.suffix.lower() != ".png":
            raise AssetError("PBRT material image conversion currently supports PNG only; environment HDR is converted separately: " + filename)
        header = source.read_bytes()[:25]
        if len(header) != 25 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[24] == 16:
            raise AssetError("Unsupported PBRT PNG encoding (16-bit input needs a lossless float decoder): " + filename)
        encoding = str(_single(definition, "encoding", "sRGB" if source.suffix.lower() == ".png" else "linear"))
        gamma = None
        if encoding.lower() in ("srgb", "srgb_texture"):
            space = "srgb_texture"
        elif encoding.lower() in ("linear", "gamma 1", "gamma 1.0"):
            space = "lin_rec709"
        elif encoding.lower().startswith("gamma "):
            gamma = _finite(encoding.split()[1], "texture gamma")
            if gamma <= 0:
                raise AssetError("PBRT texture gamma must be positive")
            space = "lin_rec709"
        else:
            raise AssetError("Unsupported PBRT image encoding: " + encoding)
        with Image.open(source) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise AssetError("Animated PBRT material images are unsupported: " + filename)
            rgba = np.asarray(image.convert("RGBA"))
            # PBRT ReadPNG keeps A only for PNG color type RGBA. Its gray-alpha
            # and palette/tRNS paths intentionally decode grayscale/RGB only.
            has_alpha = image.mode == "RGBA" if source.suffix.lower() == ".png" else "A" in image.getbands()
            alpha_active = has_alpha and bool(np.any(rgba[..., 3] != 255))
            dimensions = list(image.size)
        uvscale = tuple(_finite(_single(definition, key, 1), key) for key in ("uscale", "vscale"))
        uvdelta = tuple(_finite(_single(definition, key, 0), key) for key in ("udelta", "vdelta"))
        wrap = str(_single(definition, "wrap", "repeat"))
        if wrap not in ("repeat", "clamp", "black"):
            raise AssetError("Unsupported PBRT texture wrap: " + wrap)
        filtering = str(_single(definition, "filter", "bilinear")).lower()
        if filtering not in ("bilinear", "trilinear", "point", "ewa"):
            raise AssetError("Unsupported PBRT texture filter: " + filtering)
        info = {"name": name, "source": filename, "file": relative, "encoding": encoding, "color_space": space,
                "gamma": gamma, "size": dimensions, "scalar_uses_alpha": alpha_active,
                "uv_scale": uvscale, "uv_offset": uvdelta, "wrap": wrap, "filter": filtering,
                "scale": _finite(_single(definition, "scale", 1), "image scale"),
                "invert": _boolean(_single(definition, "invert", False), name+" invert"),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        info["scalar_channel"] = "red" if filtering in ("point", "ewa") else "alpha" if alpha_active else "rgb_average"
        info["scalar_uses_alpha"] = info["scalar_channel"] == "alpha"
        if "gamma" in definition.get("params", {}):
            info["legacy_unused_parameter"] = {"gamma": list(definition["params"]["gamma"])}
            info["legacy_note"] = "PBRT v4 reads encoding; the legacy float gamma parameter is unused upstream."
        if filtering in ("ewa", "bilinear", "trilinear"):
            info["filter_approximation"] = "Caustra uses ray-cone isotropic trilinear mip filtering; PBRT derivative footprint/filter differs."
        self.image_info[name] = info
        self.report["textures"].append(info)
        return info

    def _decode_alpha(self, value, info, prefix):
        """Match PBRT's encoding of every U256 PNG channel, including A.

        MaterialX/our image loader correctly treats ordinary alpha as linear;
        the extra scalar graph is specific to PBRT compatibility.
        """
        if info["color_space"] == "srgb_texture":
            low = self.binary("divide", "float", prefix+"_linear_segment", value, 12.92)
            high = self.binary("power", "float", prefix+"_gamma_segment",
                               self.binary("divide", "float", prefix+"_srgb_scale",
                                           self.binary("add", "float", prefix+"_srgb_offset", value, .055), 1.055), 2.4)
            selector = self._clamp(self.node("ceil", "float", prefix+"_srgb_above_threshold",
                                            **{"in":self.binary("subtract", "float", prefix+"_srgb_threshold", value, .04045)}),
                                   "float", prefix+"_srgb_selector")
            return self.node("mix", "float", prefix+"_srgb_decode", bg=low, fg=high, mix=selector)
        if info["gamma"] is not None:
            return self.binary("power", "float", prefix+"_gamma_decode", value, info["gamma"])
        return value

    def _uv(self, info, prefix):
        uv = self.node("texcoord", "vector2", prefix+"_uv", index=0)
        if tuple(info["uv_scale"]) != (1, 1):
            uv = self.binary("multiply", "vector2", prefix+"_scale", uv, info["uv_scale"])
        if tuple(info["uv_offset"]) != (0, 0):
            uv = self.binary("add", "vector2", prefix+"_offset", uv, info["uv_offset"])
        return uv

    def _image_node(self, name, info, kind, filename=None):
        uv = self._uv(info, name)
        image = self.node("image", kind, name+"_image", file=filename or info["file"], texcoord=uv,
                          uaddressmode={"repeat":"periodic", "clamp":"clamp", "black":"constant"}[info["wrap"]],
                          vaddressmode={"repeat":"periodic", "clamp":"clamp", "black":"constant"}[info["wrap"]],
                          filtertype="closest" if info["filter"] == "point" else "linear")
        image.getInput("file").setColorSpace("lin_rec709" if filename else info["color_space"])
        return image

    def texture(self, name):
        if name in self.texture_nodes:
            node = self.texture_nodes[name]
            if node is None:
                raise AssetError("Cycle in PBRT texture graph: " + name)
            return node
        if name not in self.textures:
            raise AssetError("Unknown PBRT texture: " + name)
        self.texture_nodes[name] = None
        self.active_textures.add(name)
        definition = self.textures[name]
        if definition.get("value_type") not in ("float", "spectrum"):
            raise AssetError("Unsupported PBRT texture value type: " + str(definition.get("value_type")))
        kind = "float" if definition.get("value_type") == "float" else "color3"
        category = definition["type"]
        if category == "imagemap":
            info = self.image_info.get(name) or self._image(name, definition)
            image_kind = "color4" if kind == "float" and info["scalar_uses_alpha"] else "color3"
            node = self._image_node(name, info, image_kind)
            if info["gamma"] is not None:
                # Decode color before scalar averaging. A is decoded separately
                # below to match PBRT's unusual PNG alpha encoding convention.
                if image_kind != "color4":
                    node = self.binary("power", image_kind, name+"_decode", node, (info["gamma"],)*3)
            if kind == "float":
                if info["scalar_channel"] in ("alpha", "red"):
                    extract = self.document.addNode("extract", self.document.createValidChildName(mx.createValidName(name+"_channel")), "float")
                    extract.addInput("in", node.getType()).setConnectedNode(node)
                    extract.addInput("index", "integer").setValueString("3" if info["scalar_uses_alpha"] else "0")
                    node = extract
                    if info["scalar_uses_alpha"]:
                        node = self._decode_alpha(node, info, name+"_alpha")
                else:
                    node = self.node("dotproduct", "float", name+"_average", in1=self.convert(node, "vector3", name+"_rgb"), in2=(1/3,)*3)
            if info["scale"] != 1:
                node = self.binary("multiply", kind, name+"_intensity", node, self.convert(info["scale"], kind, name+"_scale"))
            if info["invert"]:
                node = self.node("max", kind, name+"_inversion_clamp",
                                 in1=self.binary("subtract", kind, name+"_inversion", self.convert(1, kind, name+"_one"), node),
                                 in2=self.convert(0, kind, name+"_zero"))
        elif category == "scale":
            _known(definition, {"tex", "scale"}, "scaled texture " + name)
            a = self.parameter(definition, "tex", kind, 1, name+"_tex")
            b = self.parameter(definition, "scale", "float", 1, name+"_factor")
            node = self.binary("multiply", kind, name+"_multiply", a, self.convert(b, kind, name+"_broadcast"))
        elif category == "constant":
            _known(definition, {"value"}, "constant texture " + name)
            node = self.node("constant", kind, name, value=self.parameter(definition, "value", kind, 1, name+"_value"))
        else:
            raise AssetError("Unsupported PBRT texture type: " + category)
        self.texture_nodes[name] = node
        return node

    def parameter(self, definition, name, kind, default, prefix):
        values = definition.get("params", {}).get(name)
        if values is None:
            return self.convert(default, kind, prefix)
        source_type = definition.get("param_types", {}).get(name, "float")
        if source_type == "texture":
            if len(values) != 1:
                raise AssetError("Expected one PBRT texture reference: " + prefix)
            result = self.texture(str(values[0]))
            if result.getType() != kind and not (result.getType() == "float" and kind == "color3"):
                raise AssetError("PBRT texture type mismatch: " + prefix)
            return self.convert(result, kind, prefix+"_convert")
        numeric = tuple(_finite(v, prefix) for v in values)
        if kind == "float":
            if len(numeric) != 1:
                raise AssetError("Expected scalar PBRT parameter " + prefix)
            return numeric[0]
        if len(numeric) == 1:
            return numeric*3
        if len(numeric) != 3 or source_type not in ("rgb", "color"):
            raise AssetError("Only RGB PBRT color values are supported: " + prefix)
        return numeric

    def _clamp(self, value, kind, prefix):
        return self.node("clamp", kind, prefix, **{"in":value, "low":self.convert(0, kind, prefix), "high":self.convert(1, kind, prefix)})

    def _height(self, name, factor=1, seen=None):
        seen = set() if seen is None else set(seen)
        if name in seen:
            raise AssetError("Cycle in PBRT height graph: " + name)
        seen.add(name)
        definition = self.textures.get(name)
        if definition is None:
            raise AssetError("Unknown PBRT bump texture: " + name)
        if definition["type"] == "scale":
            if definition.get("param_types", {}).get("scale") == "texture":
                raise AssetError("Spatially varying PBRT bump multipliers require an explicit bake: " + name)
            return self._height(str(_single(definition, "tex")), factor*_finite(_single(definition, "scale", 1), name), seen)
        if definition["type"] != "imagemap":
            raise AssetError("PBRT bump conversion requires imagemap/constant-scale: " + name)
        info = self.image_info.get(name) or self._image(name, definition)
        with Image.open(self.source_dir/info["source"]) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)/255
        if info["scalar_uses_alpha"]:
            height = rgba[..., 3]
            if info["color_space"] == "srgb_texture":
                height = srgb_to_linear(height)
            elif info["gamma"] is not None:
                height = height**info["gamma"]
        else:
            rgb = rgba[..., :3]
            if info["color_space"] == "srgb_texture":
                rgb = srgb_to_linear(rgb)
            elif info["gamma"] is not None:
                rgb = rgb**info["gamma"]
            height = rgb[..., 0] if info["scalar_channel"] == "red" else rgb.mean(axis=-1)
        height = height*info["scale"]
        if info["invert"]:
            height = np.maximum(0, 1-height)
        return info, height*factor

    def _bump(self, material_name, definition, record):
        if definition.get("param_types", {}).get("displacement") != "texture":
            # A constant height has zero derivatives; preserve it in the report.
            if len(definition["params"]["displacement"]) != 1:
                raise AssetError("Invalid constant PBRT bump displacement")
            record["constant_bump"] = _finite(_single(definition, "displacement"), "bump")
            return None
        height_name = str(_single(definition, "displacement"))
        # Retain the original editable image/scale graph in the .mtlx document.
        self.texture(height_name)
        info, height = self._height(height_name)
        metric = tuple(float(v) for v in self.bump_metrics.get(material_name, (1, 1)))
        if len(metric) != 2 or not all(math.isfinite(v) and v > 0 for v in metric):
            raise AssetError("Invalid bump dPdu/dPdv metric for " + material_name)
        key = (height_name, metric)
        if key not in self.baked_normals:
            rows, columns = height.shape
            def derivative(axis, size):
                if size == 1:
                    return np.zeros_like(height)
                if info["wrap"] == "repeat":
                    return (np.roll(height, -1, axis)-np.roll(height, 1, axis))*size*.5
                mode = "edge" if info["wrap"] == "clamp" else "constant"
                pads = [(0, 0), (0, 0)]; pads[axis] = (1, 1)
                extended = np.pad(height, pads, mode=mode)
                return (np.take(extended, range(2, size+2), axis)-np.take(extended, range(size), axis))*size*.5
            ds = derivative(1, columns)
            dt = -derivative(0, rows)  # file rows down; texture v increases up
            normal = np.stack((-ds*info["uv_scale"][0]/metric[0], -dt*info["uv_scale"][1]/metric[1], np.ones_like(height)), axis=-1)
            normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
            data = np.rint(np.clip(normal*.5+.5, 0, 1)*255).astype(np.uint8)
            digest = hashlib.sha256((repr(key)+info["sha256"]).encode("utf-8")).hexdigest()[:20]
            relative = "derived_normals/"+digest+".png"
            output = self.output_dir/relative
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(data).save(output)
            self.baked_normals[key] = relative
        relative = self.baked_normals[key]
        image = self._image_node(material_name+"_bump", info, "vector3", relative)
        normalmap = self.node("normalmap", "vector3", material_name+"_normalmap", **{"in":image, "scale":1})
        record["bump"] = {"source_texture":height_name, "normal_file":relative, "metric_dPdu_dPdv":list(metric),
                          "approximation":"Central-difference tangent normal bake with representative geometric UV metric; excludes per-hit curvature, skew/nonorthogonal UV basis, varying metric and ray differential filtering. Original PBRT height graph and image are retained in the MaterialX source.",
                          "metric_source":"area-weighted geometry" if material_name in self.bump_metrics else "unit-UV fallback"}
        return normalmap

    def build(self):
        if self.surfaces:
            return
        for index, (name, definition) in enumerate(self.materials.items()):
            category = definition["type"]
            allowed = {"type", "displacement", "alpha"}
            if category == "diffuse":
                allowed |= {"reflectance", "sigma"}
            elif category == "coateddiffuse":
                allowed |= {"reflectance", "roughness", "eta", "remaproughness", "sigma"}
            elif category == "diffusetransmission":
                allowed |= {"reflectance", "transmittance", "scale"}
            elif category in ("dielectric", "thindielectric"):
                allowed |= {"roughness", "eta", "remaproughness"}
            else:
                raise AssetError("Unsupported PBRT material type: " + category)
            _known(definition, allowed, "material " + name)
            prefix = "m%04d_" % index+mx.createValidName(name)
            surface = self.node("standard_surface", "surfaceshader", prefix, base=1, specular=0, metalness=0,
                                specular_roughness=0, diffuse_roughness=0, coat=0, transmission=0)
            record = {"name":name, "surface_node":surface.getName(), "pbrt_type":category,
                      "source":definition.get("source", ""), "source_parameters":definition.get("params", {}),
                      "source_parameter_types":definition.get("param_types", {}), "model":"standard_surface"}
            if "sigma" in definition.get("params", {}):
                record["unused_legacy_sigma"] = "PBRT v4 diffuse/coateddiffuse does not read legacy sigma; retained in source_parameters and original texture nodes."
                if definition.get("param_types", {}).get("sigma") == "texture":
                    self.texture(str(_single(definition, "sigma")))
            reflectance = self.parameter(definition, "reflectance", "color3", .25 if category == "diffusetransmission" else .5, prefix+"_R")
            self.set(surface, "base_color", self._clamp(reflectance, "color3", prefix+"_R_clamp"))
            if category == "coateddiffuse":
                self.set(surface, "specular", 1)
                self.set(surface, "specular_IOR", self.parameter(definition, "eta", "float", 1.5, prefix+"_eta"))
                record["approximation"] = "MaterialX dielectric layer over diffuse replaces PBRT's stochastic finite-thickness layered BSDF; internal layer multiple scattering is different."
            if category in ("coateddiffuse", "dielectric", "thindielectric"):
                roughness = self.parameter(definition, "roughness", "float", 0, prefix+"_roughness")
                remap = _boolean(_single(definition, "remaproughness", True), name+" remaproughness")
                self.set(surface, "specular_roughness", self.binary("power", "float", prefix+"_roughness_to_materialx",
                         self._clamp(roughness, "float", prefix+"_roughness_clamp"), .25 if remap else .5))
                record["roughness_mapping"] = "MaterialX roughness = PBRT roughness^%s; alpha matches before renderer's existing roughness floor." % ("0.25" if remap else "0.5")
            if category == "diffusetransmission":
                scale = self.parameter(definition, "scale", "float", 1, prefix+"_scale")
                transmittance = self.parameter(definition, "transmittance", "color3", .25, prefix+"_T")
                reflectance = self._clamp(self.binary("multiply", "color3", prefix+"_scaled_R", reflectance, self.convert(scale, "color3", prefix)), "color3", prefix+"_R_saturate")
                transmittance = self._clamp(self.binary("multiply", "color3", prefix+"_scaled_T", transmittance, self.convert(scale, "color3", prefix)), "color3", prefix+"_T_saturate")
                # Standard Surface's thin-wall SSS selects its actual Lambert
                # translucent BSDF. Half weights and doubled colors recover R,T.
                self.set(surface, "thin_walled", True)
                self.set(surface, "subsurface", .5)
                self.set(surface, "base_color", self.binary("multiply", "color3", prefix+"_twice_R", reflectance, (2, 2, 2)))
                self.set(surface, "subsurface_color", self.binary("multiply", "color3", prefix+"_twice_T", transmittance, (2, 2, 2)))
                record["mapping"] = "Thin-walled Standard Surface Lambert reflection/translucency: subsurface=.5, base_color=2R, subsurface_color=2T, specular=0."
            if category in ("dielectric", "thindielectric"):
                self.set(surface, "base", 0)
                self.set(surface, "specular", 1)
                self.set(surface, "transmission", 1)
                self.set(surface, "transmission_color", (1, 1, 1))
                self.set(surface, "specular_IOR", self.parameter(definition, "eta", "float", 1.5, prefix+"_eta"))
                if category == "thindielectric":
                    self.set(surface, "thin_walled", True)
                record["approximation"] = "RGB MaterialX dielectric with Caustra transport adapter; PBRT spectral integration and exact smooth delta events differ from the existing 0.02 roughness floor."
                if name in self.thin_walled_overrides:
                    self.set(surface, "thin_walled", True)
                    record["thin_walled_override"] = {
                        "reason":str(self.thin_walled_overrides[name]),
                        "approximation":"Open/nonmanifold PBRT dielectric geometry has no well-defined bounded interior. Thin-wall transport avoids inventing a medium, while preserving IOR/Fresnel; it omits solid thickness and bulk refraction."
                    }
            if "alpha" in definition.get("params", {}):
                alpha = self._clamp(self.parameter(definition, "alpha", "float", 1, prefix+"_alpha"), "float", prefix+"_alpha_clamp")
                self.set(surface, "opacity", self.convert(alpha, "color3", prefix+"_coverage"))
                record["alpha"] = "PBRT shape alpha retained as scalar stochastic surface coverage."
            if "displacement" in definition.get("params", {}):
                normal = self._bump(name, definition, record)
                if normal is not None:
                    self.set(surface, "normal", normal)
            material = self.document.addNode("surfacematerial", self.document.createValidChildName(prefix+"_material"), "material")
            material.addInput("surfaceshader", "surfaceshader").setConnectedNode(surface)
            self.surfaces[name] = surface
            self.report["materials"].append(record)
            for feature in ("approximation", "bump", "unused_legacy_sigma", "thin_walled_override"):
                if feature in record:
                    self.report["approximations"].append({"material":name, "feature":feature,
                                                           "details":record[feature]})
        valid, message = self.document.validate()
        if not valid:
            raise AssetError("Converted PBRT MaterialX is invalid: " + message)

    def write(self, stage, material_root="/World/Materials"):
        self.build()
        result = {}
        for index, (name, surface) in enumerate(self.surfaces.items()):
            path = material_root+"/m%04d_" % index+mx.createValidName(name)
            result[name] = export_material(stage, path, surface)
            result[name].GetPrim().SetCustomDataByKey("pbrtMaterialName", name)
        return result

    def write_mtlx(self, path):
        self.build()
        options = mx.XmlWriteOptions()
        options.writeXIncludeEnable = False
        mx.writeToXmlFile(self.document, str(path), options)
