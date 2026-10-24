#!/usr/bin/env python3
"""Build one generated-model gallery containing every official MaterialX example.

The input is the unmodified MaterialX resources/Materials/Examples directory.
No source graph is replaced with a hand-written material and compilation is
all-or-nothing. Outputs are a portable .caustra package, an editable .usda scene
with the original MaterialX node graphs, embedded-resource copies, and a JSON
material/index/provenance manifest. USD and MaterialX stay outside the renderer.
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
from io import BytesIO
from pathlib import Path

import MaterialX as mx
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from caustra_assets.materialx_compiler import (
    MaterialCompiler, TextureStore, color_space, material_terminals,
    read_resource, srgb_to_linear,
)
from caustra_assets.model import Asset, AssetError, Material, Mesh, Texture
from caustra_assets.package import write_asset


VERSION = "1.39.5"
FAMILY_ORDER = ("StandardSurface", "OpenPbr", "UsdPreviewSurface", "GltfPbr", "DisneyPrincipled", "SimpleHair")
IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
# A small, dependency-free pixel font becomes ordinary scene geometry.
FONT = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/.:() ", [
    "01110/10001/10001/11111/10001/10001/10001", "11110/10001/10001/11110/10001/10001/11110",
    "01111/10000/10000/10000/10000/10000/01111", "11110/10001/10001/10001/10001/10001/11110",
    "11111/10000/10000/11110/10000/10000/11111", "11111/10000/10000/11110/10000/10000/10000",
    "01111/10000/10000/10111/10001/10001/01111", "10001/10001/10001/11111/10001/10001/10001",
    "11111/00100/00100/00100/00100/00100/11111", "00111/00010/00010/00010/10010/10010/01100",
    "10001/10010/10100/11000/10100/10010/10001", "10000/10000/10000/10000/10000/10000/11111",
    "10001/11011/10101/10101/10001/10001/10001", "10001/11001/10101/10011/10001/10001/10001",
    "01110/10001/10001/10001/10001/10001/01110", "11110/10001/10001/11110/10000/10000/10000",
    "01110/10001/10001/10001/10101/10010/01101", "11110/10001/10001/11110/10100/10010/10001",
    "01111/10000/10000/01110/00001/00001/11110", "11111/00100/00100/00100/00100/00100/00100",
    "10001/10001/10001/10001/10001/10001/01110", "10001/10001/10001/10001/10001/01010/00100",
    "10001/10001/10001/10101/10101/10101/01010", "10001/10001/01010/00100/01010/10001/10001",
    "10001/10001/01010/00100/00100/00100/00100", "11111/00001/00010/00100/01000/10000/11111",
    "01110/10001/10011/10101/11001/10001/01110", "00100/01100/00100/00100/00100/00100/01110",
    "01110/10001/00001/00010/00100/01000/11111", "11110/00001/00001/01110/00001/00001/11110",
    "00010/00110/01010/10010/11111/00010/00010", "11111/10000/10000/11110/00001/00001/11110",
    "01110/10000/10000/11110/10001/10001/01110", "11111/00001/00010/00100/01000/01000/01000",
    "01110/10001/10001/01110/10001/10001/01110", "01110/10001/10001/01111/00001/00001/01110",
    "00000/00000/00000/11111/00000/00000/00000", "00000/00000/00000/00000/00000/00000/11111",
    "00001/00010/00010/00100/01000/01000/10000", "00000/00000/00000/00000/00000/01100/01100",
    "00000/01100/01100/00000/01100/01100/00000", "00010/00100/01000/01000/01000/00100/00010",
    "01000/00100/00010/00010/00010/00100/01000", "00000/00000/00000/00000/00000/00000/00000",
]))


def examples_root(path):
    path = Path(path).resolve()
    for candidate in (path, path / "resources/Materials/Examples", path / "Materials/Examples"):
        if candidate.is_dir() and any(candidate.glob("*/*.mtlx")):
            return candidate
    raise AssetError("Cannot find official resources/Materials/Examples below " + str(path))


def inventory(root):
    files = list(root.rglob("*.mtlx"))
    if not files:
        raise AssetError("The official examples directory contains no .mtlx files")
    rank = {name.lower(): index for index, name in enumerate(FAMILY_ORDER)}
    return sorted(files, key=lambda path: (rank.get(path.relative_to(root).parts[0].lower(), 100), path.relative_to(root).as_posix().lower()))


def vertex(p, n=(0, 1, 0), uv=(0, 0), tangent=(0, 0, 0)):
    return list(p) + [0] + list(n) + [0] + list(uv) + [0, 0] + list(tangent) + [1 if any(tangent) else 0]


def add_triangle(mesh, a, b, c, outward=None):
    if outward is not None:
        p, q, r = (np.asarray(mesh.vertices[i][:3]) for i in (a, b, c))
        if np.dot(np.cross(q-p, r-p), outward) < 0:
            b, c = c, b
    mesh.indices.extend((a, b, c))


def sphere(material, segments, rings, scale=(.84, .84, .84), center=(0, 0, 0), fiber=False):
    mesh = Mesh(material)
    for i in range(rings+1):
        theta = math.pi*i/rings
        sin_theta = 0.0 if i in (0, rings) else math.sin(theta)
        for j in range(segments+1):
            # Force the duplicated seam to exactly identical positions/normals.
            phi = 2*math.pi*(j % segments)/segments
            p = np.asarray((sin_theta*math.cos(phi), math.cos(theta), sin_theta*math.sin(phi)))
            n = p / np.asarray(scale); n /= np.linalg.norm(n)
            if fiber:
                tangent = np.array((0., 1., 0.)); tangent -= n*np.dot(tangent, n)
                if np.linalg.norm(tangent) < 1e-8:
                    tangent = np.array((math.cos(phi), 0, math.sin(phi)))
                tangent /= np.linalg.norm(tangent)
                uv = (1-i/rings, j/segments)
            else:
                tangent = (-math.sin(phi), 0, math.cos(phi)); uv = (j/segments, 1-i/rings)
            mesh.vertices.append(vertex(p*np.asarray(scale)+center, n, uv, tangent))
    stride = segments+1
    for i in range(rings):
        for j in range(segments):
            a = i*stride+j; b = a+1; c = a+stride; d = c+1
            if i:
                add_triangle(mesh, a, b, c, np.asarray(mesh.vertices[a][4:7]))
            if i < rings-1:
                add_triangle(mesh, b, d, c, np.asarray(mesh.vertices[c][4:7]))
    return mesh


def append_mesh(target, source):
    first = len(target.vertices)
    target.vertices.extend(source.vertices)
    target.indices.extend(first+i for i in source.indices)


def hair_bundle(material):
    mesh = Mesh(material)
    # Chiang's analytic TT/TRT lobes already integrate a fiber cross-section.
    # Represent each strand by a single open curved ribbon, so that transmitted
    # scattering is not evaluated again on a second, geometric back interface.
    # Different ribbon angles remain visible while orbiting the gallery.
    for row in range(5):
        for col in range(9):
            x = (col-4)*.135; z = (row-2)*.075
            height = 1.7 + .12*math.sin(col*1.7+row*2.3)
            angle = (col-4)*.20+(row-2)*.35
            first=len(mesh.vertices); longitudinal=16; across=3
            for i in range(longitudinal+1):
                t=i/longitudinal; phase=row*.8
                center=np.array((x+.06*math.sin(math.pi*t)+(col-4)*.015*t,
                                 height*(t-.5),z+.05*math.sin(2*math.pi*t+phase)))
                tangent=np.array((.06*math.pi*math.cos(math.pi*t)+(col-4)*.015,
                                  height,.10*math.pi*math.cos(2*math.pi*t+phase)))
                tangent/=np.linalg.norm(tangent)
                for j in range(across+1):
                    phi=angle+(j/across-.5)*1.6
                    radial=np.array((math.sin(phi),0,math.cos(phi)))
                    normal=radial-tangent*np.dot(radial,tangent); normal/=np.linalg.norm(normal)
                    mesh.vertices.append(vertex(center+.037*radial,normal,(t,j/across),tangent))
            stride=across+1
            for i in range(longitudinal):
                for j in range(across):
                    a=first+i*stride+j; b=a+1; c=a+stride; d=c+1
                    add_triangle(mesh,a,b,c,mesh.vertices[a][4:7])
                    add_triangle(mesh,b,d,c,mesh.vertices[d][4:7])
    return mesh


def quad(mesh, points, normal=(0, 1, 0)):
    first = len(mesh.vertices)
    mesh.vertices.extend(vertex(p, normal, uv) for p, uv in zip(points, ((0, 0), (1, 0), (1, 1), (0, 1))))
    add_triangle(mesh, first, first+1, first+2, normal)
    add_triangle(mesh, first, first+2, first+3, normal)


def box(material, minimum, maximum):
    mesh = Mesh(material)
    x0, y0, z0 = minimum; x1, y1, z1 = maximum
    for points, normal in (
        (((x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)), (0,-1,0)),
        (((x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)), (0,1,0)),
        (((x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)), (0,0,-1)),
        (((x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)), (0,0,1)),
        (((x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)), (-1,0,0)),
        (((x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0)), (1,0,0)),
    ):
        quad(mesh, points, normal)
    return mesh


def text(mesh, value, x, y, z, width=2.6, height=.18):
    value = value.upper()
    pixel = min(height/7, width/max(1, len(value)*6-1))
    left = x - (len(value)*6-1)*pixel/2
    for index, char in enumerate(value):
        pattern = FONT.get(char, FONT[" "]).split("/")
        for row, line in enumerate(pattern):
            for col, bit in enumerate(line):
                if bit == "1":
                    x0 = left+(index*6+col)*pixel; y0 = y+(6-row)*pixel
                    quad(mesh, ((x0,y0,z),(x0+pixel*.9,y0,z),(x0+pixel*.9,y0+pixel*.9,z),(x0,y0+pixel*.9,z)), (0,0,1))


def sample_scale(entry):
    # Preserve the official physical radiance, using a physically small bulb.
    return .15 if entry["source"].replace("\\","/").lower()=="openpbr/open_pbr_lightbulb.mtlx" else 1.


def sample_display_name(entry):
    name=entry["name"]
    prefix={"standardsurface":"standard_surface_","openpbr":"open_pbr_","usdpreviewsurface":"usd_preview_surface_",
            "gltfpbr":"gltf_pbr_","disneyprincipled":"disney_principled_","simplehair":"simple_hair_"}.get(entry["family"].lower(),entry["family"].lower()+"_")
    if name.lower().startswith(prefix): name=name[len(prefix):]
    if name.startswith("chess_set/"): name="Chess "+name.split("/",1)[1]
    if name.startswith("look_"):
        name="Look Calibration" if "Calibration" in name else "Look "+("Brass" if "brass" in name else "Wood")
    if name=="greysphere_calibration": name="Grey Calibration"
    return name.replace("_"," ")


def label_plate(pedestals,labels,entry,x,base,z):
    """A raised two-line sign facing the fitted camera instead of the horizon."""
    family=entry["family"]
    tag={"standardsurface":"SS","openpbr":"OPBR","usdpreviewsurface":"USD","gltfpbr":"GLTF",
         "disneyprincipled":"DISNEY","simplehair":"HAIR"}.get(family.lower(),family)
    plate=Mesh(pedestals.material)
    quad(plate,((-1.30,-.28,0),(1.30,-.28,0),(1.30,.28,0),(-1.30,.28,0)),(0,0,1))
    lettering=Mesh(labels.material)
    text(lettering,"%02d / %s" % (entry["id"],tag),0,.025,.003,2.42,.20)
    name=sample_display_name(entry)
    text(lettering,name,0,-.225,.003,2.42,.20)
    sine=math.sin(math.radians(60)); cosine=math.cos(math.radians(60))
    matrix=np.array(((1,0,0),(0,cosine,sine),(0,-sine,cosine)))
    origin=np.array((x,base+.47,z+1.35))
    for mesh in (plate,lettering):
        for v in mesh.vertices:
            v[:3]=(matrix@np.asarray(v[:3])+origin).tolist()
            v[4:7]=(matrix@np.asarray(v[4:7])).tolist()
    append_mesh(pedestals,plate); append_mesh(labels,lettering)
    return {"type":"raised_inclined_two_line_plate","display_name":name,"family_label":tag,
            "height":.56,"line_height":.20,"normal":[0,sine,cosine],"front_z_offset":1.35+.28*sine}


def plain_material(color, roughness=.6, emission=None):
    material = Material()
    material.values[0][:3] = color
    material.values[2][1] = roughness
    if emission is not None:
        material.values[1][:3] = emission
    return material


class GalleryTextureStore(TextureStore):
    """Explicit gallery resolution budget; shader graphs and values stay intact."""
    def __init__(self, asset, maximum):
        super().__init__(asset); self.maximum=maximum; self.resolutions=[]

    def add(self, path, space):
        if not self.maximum:
            return super().add(path,space)
        space=color_space(space); key=(str(path),space)
        if key in self.cache:
            return self.cache[key]
        try:
            with Image.open(BytesIO(read_resource(path))) as source:
                if getattr(source,"n_frames",1)!=1:
                    raise AssetError("Animated textures are unsupported")
                if source.width*source.height>64*1024*1024:
                    raise AssetError("Texture exceeds the 64 megapixel limit")
                original=source.size
                if source.mode=="F":
                    grey=np.asarray(source,dtype=np.float32)
                    pixels=np.stack([grey,grey,grey,np.ones_like(grey)],axis=-1)
                else:
                    pixels=np.asarray(source.convert("RGBA"),dtype=np.float32)/255
                if space=="srgb":
                    pixels[...,:3]=srgb_to_linear(pixels[...,:3])
                ratio=min(1,self.maximum/max(original))
                size=(max(1,round(original[0]*ratio)),max(1,round(original[1]*ratio)))
                if size!=original:
                    # Resize linear channels, rather than averaging sRGB bytes.
                    minimum=pixels.min(axis=(0,1)); maximum=pixels.max(axis=(0,1))
                    pixels=np.stack([np.asarray(Image.fromarray(pixels[...,lane],mode="F").resize(size,Image.Resampling.LANCZOS)) for lane in range(4)],axis=-1)
                    # Lanczos ringing can exceed physical texture ranges.
                    pixels=np.clip(pixels,minimum,maximum)
        except (OSError,ValueError,KeyError) as error:
            raise AssetError("Cannot load texture %s: %s" % (path,error)) from error
        if not np.isfinite(pixels).all():
            raise AssetError("Texture contains nonfinite values: "+str(path))
        height,width=pixels.shape[:2]; index=len(self.asset.textures)
        self.asset.textures.append(Texture(width,height,pixels.reshape(-1,4).tolist()))
        self.resolutions.append({"texture_index":index,"source_name":Path(path).name,"color_space":space,
                                 "source_size":list(original),"package_size":[width,height]})
        self.cache[key]=index
        return index


class UsdMaterialWriter:
    """Author actual NodeDef networks, retaining graph connections and values."""
    TYPE_NAMES = {
        "float": Sdf.ValueTypeNames.Float, "integer": Sdf.ValueTypeNames.Int,
        "boolean": Sdf.ValueTypeNames.Bool, "string": Sdf.ValueTypeNames.String,
        "filename": Sdf.ValueTypeNames.Asset, "vector2": Sdf.ValueTypeNames.Float2,
        "vector3": Sdf.ValueTypeNames.Vector3f, "color3": Sdf.ValueTypeNames.Color3f,
        "vector4": Sdf.ValueTypeNames.Float4, "color4": Sdf.ValueTypeNames.Color4f,
        "matrix33": Sdf.ValueTypeNames.Matrix3d, "matrix44": Sdf.ValueTypeNames.Matrix4d,
    }

    def __init__(self, stage, resource_dir):
        self.stage = stage; self.resource_dir = resource_dir
        self.resources = {}; self.source_files=set()

    def resource(self, port, base_directory):
        resolved = port.getResolvedValueString() or port.getValueString()
        path = Path(resolved)
        if not path.is_absolute():
            path = base_directory/path
        self.source_files.add(path.resolve())
        content = read_resource(path)
        digest = hashlib.sha256(content).hexdigest()
        suffix = path.suffix.lower()
        name = digest[:20]+suffix
        destination = self.resource_dir/name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        self.resources[digest] = {"file": destination.name, "sha256": digest, "source_name": path.name}
        return Sdf.AssetPath(self.resource_dir.name+"/"+name)

    def material(self, path, document, surface, base_directory):
        # Preserve local NodeDefs as standard USD sourceAsset definitions.
        # In particular the official Hair convenience graph is a real local
        # implementation, rather than a node in the standard data library.
        libraries = document.getDataLibrary()
        local = [n for n in document.getNodes() if n.getNodeDef() is not None and libraries.getNodeDef(n.getNodeDef().getName()) is None]
        definition_asset=None
        if local:
            definitions=mx.createDocument()
            for child in document.getChildren():
                if child.getCategory() in ("nodedef","nodegraph","implementation"):
                    definitions.addChildOfCategory(child.getCategory(),child.getName()).copyContentFrom(child)
            xml_options=mx.XmlWriteOptions(); xml_options.writeXIncludeEnable=False
            content=mx.writeToXmlString(definitions,xml_options).encode("utf-8")
            digest=hashlib.sha256(content).hexdigest(); destination=self.resource_dir/(digest[:20]+".mtlx")
            destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(content)
            self.resources[digest]={"file":destination.name,"sha256":digest,"kind":"local_materialx_definitions"}
            definition_asset=Sdf.AssetPath(self.resource_dir.name+"/"+destination.name)
        result = UsdShade.Material.Define(self.stage, path)
        translated = {}; visiting = set()

        def resolve(port, seen=None):
            seen = set() if seen is None else set(seen)
            key = port.getNamePath()
            if key in seen:
                raise AssetError("Cycle while exporting MaterialX port " + key)
            seen.add(key)
            if port.getNodeName():
                node = port.getConnectedNode()
                if node is None:
                    raise AssetError("Broken MaterialX connection " + key)
                return node, port.getOutputString() or "out", None
            if port.getNodeGraphString():
                graph = document.getNodeGraph(port.getNodeGraphString())
                output = graph.getOutput(port.getOutputString() or "out") if graph else None
                if output is None:
                    raise AssetError("Broken MaterialX graph output " + key)
                return resolve(output, seen)
            if port.getInterfaceName():
                interface = port.getParent().getParent().getInput(port.getInterfaceName())
                if interface is None:
                    raise AssetError("Broken MaterialX interface " + key)
                return resolve(interface, seen)
            return None, None, port

        def convert(node):
            key = node.getNamePath()
            if key in translated:
                return translated[key]
            if key in visiting:
                raise AssetError("Cycle while exporting MaterialX node " + key)
            visiting.add(key)
            definition = node.getNodeDef()
            if definition is None:
                raise AssetError("MaterialX node has no NodeDef: " + key)
            name = mx.createValidName(key.replace("/", "_"))
            shader = UsdShade.Shader.Define(self.stage, str(path)+"/"+name)
            shader.CreateIdAttr(definition.getName())
            if libraries.getNodeDef(definition.getName()) is None:
                shader.SetSourceAsset(definition_asset,"mtlx")
                shader.SetSourceAssetSubIdentifier(definition.getName(),"mtlx")
            translated[key] = shader
            for output in definition.getActiveOutputs():
                shader.CreateOutput(output.getName(), self.TYPE_NAMES.get(output.getType(), Sdf.ValueTypeNames.Token))
            if not shader.GetOutputs():
                shader.CreateOutput("out", self.TYPE_NAMES.get(node.getType(), Sdf.ValueTypeNames.Token))
            for port in node.getInputs():
                upstream, output_name, literal = resolve(port)
                target = shader.CreateInput(port.getName(), self.TYPE_NAMES.get(port.getType(), Sdf.ValueTypeNames.Token))
                if upstream is not None:
                    target.ConnectToSource(convert(upstream).ConnectableAPI(), output_name)
                elif literal.hasValueString():
                    value = literal.getValue()
                    kind = literal.getType()
                    if kind == "filename":
                        value = self.resource(literal, base_directory)
                    elif kind == "boolean":
                        value = literal.getValueString().lower() == "true"
                    elif kind in ("vector2", "vector3", "color3", "vector4", "color4"):
                        constructor = {"vector2":Gf.Vec2f,"vector3":Gf.Vec3f,"color3":Gf.Vec3f,"vector4":Gf.Vec4f,"color4":Gf.Vec4f}[kind]
                        value = constructor(*value)
                    elif kind in ("matrix33", "matrix44"):
                        constructor = Gf.Matrix3d if kind == "matrix33" else Gf.Matrix4d
                        value = constructor(*list(value))
                    target.Set(value)
                if port.getColorSpace():
                    target.GetAttr().SetMetadata("colorSpace", port.getColorSpace())
            visiting.remove(key)
            return shader

        if surface is not None:
            result.CreateSurfaceOutput("mtlx").ConnectToSource(convert(surface).ConnectableAPI(), "out")
        for kind, node in material_terminals(document, surface).items():
            accessor = result.CreateVolumeOutput if kind == "volumeshader" else result.CreateDisplacementOutput
            accessor("mtlx").ConnectToSource(convert(node).ConnectableAPI(), "out")
        return result


def usd_mesh(stage, path, mesh, transform, material, uv_names=()):
    result = UsdGeom.Mesh.Define(stage, path)
    result.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    result.CreatePointsAttr([Gf.Vec3f(*v[:3]) for v in mesh.vertices])
    result.CreateFaceVertexCountsAttr([3]*(len(mesh.indices)//3))
    result.CreateFaceVertexIndicesAttr(mesh.indices)
    result.CreateNormalsAttr([Gf.Vec3f(*v[4:7]) for v in mesh.vertices])
    result.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    primvars = UsdGeom.PrimvarsAPI(result)
    for name in set(uv_names)|{"st"}:
        primvars.CreatePrimvar(name, Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set([Gf.Vec2f(*v[8:10]) for v in mesh.vertices])
    matrix = Gf.Matrix4d(1)
    for row in range(3):
        for col in range(4):
            matrix[col,row] = transform[row*4+col]
    result.AddTransformOp().Set(matrix)
    UsdShade.MaterialBindingAPI.Apply(result.GetPrim()).Bind(material)
    return result


def usd_plain_material(stage, path, material):
    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, str(path)+"/surface")
    shader.CreateIdAttr("ND_standard_surface_surfaceshader")
    shader.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*material.values[0][:3]))
    shader.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(material.values[2][1])
    if any(material.values[1][:3]):
        shader.CreateInput("emission", Sdf.ValueTypeNames.Float).Set(1)
        shader.CreateInput("emission_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*material.values[1][:3]))
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    result.CreateSurfaceOutput("mtlx").ConnectToSource(shader.ConnectableAPI(), "out")
    return result


def copy_provenance(root, files, writer):
    """Keep original source paths/XIncludes and notices with the editable scene."""
    resource_dir=writer.resource_dir
    upstream=root.parents[2] if root.name=="Examples" and root.parent.name=="Materials" and root.parent.parent.name=="resources" else root
    source_dir=resource_dir/"sources"; license_dir=resource_dir/"LICENSES"
    copied=[]; notices=[]
    for source in set(files)|writer.source_files:
        source=source.resolve()
        try: relative=source.relative_to(upstream)
        except ValueError:
            raise AssetError("Official resource escaped the pinned source tree: "+str(source))
        destination=source_dir/relative
        destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,destination)
        copied.append(relative.as_posix())
    candidates=set(upstream.glob("*"))|set(root.rglob("*"))
    images=upstream/"resources/Images"
    if images.is_dir(): candidates.update(images.rglob("*"))
    for source in sorted(candidates):
        name=source.name.lower()
        if not source.is_file() or not any(name.startswith(prefix) for prefix in ("license","third-party","third_party","readme","notice","copying")):
            continue
        relative=source.relative_to(upstream); destination=license_dir/relative
        destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,destination)
        notices.append(relative.as_posix())
        # Keep a resource's local notice next to its reviewable source copy too.
        if relative.parts[0]=="resources":
            destination=source_dir/relative
            destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,destination)
    source_text=("MaterialX official example gallery\n\nVersion: v%s\nUpstream: https://github.com/AcademySoftwareFoundation/MaterialX/tree/v%s\n\n"
                 "sources/ retains the original .mtlx files and every referenced original image, with their upstream relative paths. "
                 "Original source documents are not rewritten. Standard data libraries come from the pinned MaterialX Python package.\n"
                 "LICENSES/ retains the upstream license, third-party notices, and available resource README/license files.\n"
                 "The generated USD uses content-addressed original images and sourceAsset metadata for local MaterialX definitions. "
                 "The portable .caustra gallery separately applies its explicitly recorded texture resolution budget.\n") % (VERSION,VERSION)
    resource_dir.mkdir(parents=True,exist_ok=True)
    (resource_dir/"SOURCE.txt").write_text(source_text,encoding="utf-8")
    return {"source_directory":"sources","source_files":sorted(copied),"license_directory":"LICENSES","notices":notices,"description":"SOURCE.txt"}


def build(options):
    root = examples_root(options.source_root); files = inventory(root)
    output = options.output.with_suffix("") if options.output.suffix in (".caustra", ".usda") else options.output
    output = output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_relative_to(root):
        raise AssetError("Gallery output must be outside the original examples directory")
    asset = Asset(); compiler = MaterialCompiler(asset)
    compiler.textures=GalleryTextureStore(asset,options.texture_max_size)
    stage = None if options.validate_only else Usd.Stage.CreateNew(str(output.with_suffix(".usda")))
    if stage is not None:
        world = UsdGeom.Xform.Define(stage, "/Gallery")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y); UsdGeom.SetStageMetersPerUnit(stage, 1)
    writer = UsdMaterialWriter(stage, output.parent/(output.name+"_resources")) if stage is not None else None
    usd_materials = []; entries = []; errors = []
    for file_index, path in enumerate(files):
        relative = path.relative_to(root).as_posix()
        try:
            document=mx.createDocument()
            mx.readFromXmlFile(document,str(path),mx.FileSearchPath(str(path.parent)))
            document.setDataLibrary(compiler.libraries)
            surfaces=[node for node in document.getNodes() if node.getType()=="surfaceshader"]
            if not surfaces:
                if "volumeshader" not in material_terminals(document):
                    raise AssetError("Example has no surface or volume material")
                surfaces=[None]
            for surface in surfaces:
                identifier=len(entries)+1
                material = compiler.compile(document, surface, path.parent)
                usd_material = writer.material("/Gallery/Materials/M%03d" % identifier, document, surface, path.parent) if writer is not None else None
                if usd_material:
                    usd_material.GetPrim().SetCustomDataByKey("officialMaterialXSource", relative)
                usd_materials.append(usd_material)
                name=path.stem+("/"+surface.getName() if len(surfaces)>1 else "")
                origin=Path(surface.getSourceUri()) if surface and surface.getSourceUri() else path
                if not origin.is_absolute(): origin=path.parent/origin
                origin=origin.resolve()
                identity=origin.relative_to(root).as_posix()+"#"+(surface.getName() if surface else "volume")
                entries.append({"id":identifier, "source":relative, "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                "family":path.relative_to(root).parts[0], "name":name, "material_index":material,
                                "source_surface":surface.getName() if surface else None,
                                "source_include":surface.getSourceUri() if surface else None,
                                "source_identity":identity,
                                "surface_category":surface.getCategory() if surface else "volume",
                                "instructions":asset.materials[material].program[1]})
                print("[%d/%d] %s%s: %d instructions" % (file_index+1,len(files),relative,"/"+surface.getName() if len(surfaces)>1 else "",asset.materials[material].program[1]), flush=True)
        except Exception as error:
            errors.append(relative+": "+str(error))
    if errors:
        # A failure cannot leave a partial portable gallery that looks complete.
        if stage is not None:
            stage.GetRootLayer().Clear(); stage.GetRootLayer().Save()
        raise AssetError("Official gallery requires every material; %d/%d source files failed:\n%s" % (len(errors),len(files),"\n".join(errors)))
    if options.validate_only:
        print("All %d official MaterialX files / %d material entries compiled successfully" % (len(files),len(entries))); return
    first_support = len(asset.materials)
    for name, color, roughness, emission in (
        ("Pedestal",(.10,.13,.17),.38,None), ("Ground",(.22,.25,.29),.65,None),
        ("Lettering",(.80,.88,.95),.6,(.6,.65,.7)),
        ("KeyLight",(0,0,0),.5,(19.6,18.2,16.24)), ("FillLight",(0,0,0),.5,(12.6,14.56,19.6)),
    ):
        material = plain_material(color,roughness,emission); asset.materials.append(material)
        usd_materials.append(usd_plain_material(stage,"/Gallery/Materials/"+name,material))
    pedestals = Mesh(first_support); labels = Mesh(first_support+2)
    max_row=0; row_spacing=4.2
    for index, entry in enumerate(entries):
        family = entry["family"]
        row=index//options.columns; column=index%options.columns
        x = (column-(options.columns-1)/2)*3.1; z = -row*row_spacing; base = row*.20
        transform = IDENTITY.copy(); transform[3]=x; transform[7]=base+1.21; transform[11]=z
        scale=sample_scale(entry)
        for lane in (0,5,10): transform[lane]=scale
        mesh = hair_bundle(entry["material_index"]) if family.lower()=="simplehair" else sphere(entry["material_index"],options.segments,options.rings)
        spherical_interior=bool(asset.materials[entry["material_index"]].program[3]&1) and family.lower()!="simplehair"
        if spherical_interior:
            # SSS/volume graphs are evaluated throughout this generated sphere.
            # Extend UVs radially into the solid, rather than freezing entry UVs.
            asset.materials[entry["material_index"]].program[3] |= 2
            entry["interior_uv"]={"type":"spherical_radial_extension", "object_center":[0,0,0],
                                  "u":"wrap(atan2(z,x)/(2*pi))", "v":"1-acos(y/length(position))/pi"}
        displacement = compiler.material_displacements.get(entry["material_index"])
        if displacement:
            from caustra_assets.displacement import displace_mesh
            mesh = displace_mesh(mesh,displacement,asset.textures,transform,options.displacement_subdivisions)
        mesh_index=len(asset.meshes); asset.meshes.append(mesh); asset.instances.append((mesh_index,transform))
        source_mesh=usd_mesh(stage,"/Gallery/Samples/M%03d" % entry["id"],mesh,transform,usd_materials[index],compiler.material_uv_names[entry["material_index"]])
        if spherical_interior:
            source_mesh.GetPrim().SetCustomDataByKey("caustra:sphericalInteriorUV",True)
        append_mesh(pedestals,box(first_support,(x-1.32,base,z-1.23),(x+1.32,base+.29,z+1.23)))
        if scale!=1:
            append_mesh(pedestals,box(first_support,(x-.035,base+.29,z-.035),(x+.035,base+1.21-.84*scale,z+.035)))
        label=label_plate(pedestals,labels,entry,x,base,z)
        entry.update({"mesh_index":mesh_index,"position":[x,base+1.21,z],"geometry":"curved_fiber_ribbon_bundle" if family.lower()=="simplehair" else "uv_sphere"})
        entry.update({"instance_scale":[scale,scale,scale],"label":label})
        if scale!=1:
            entry.update({"geometry":"small_lightbulb_sphere_with_non_emissive_stem","world_radius":.84*scale,
                          "stem":{"material":"Pedestal","emissive":False,"half_width":.035,"bottom":base+.29,"top":base+1.21-.84*scale}})
        max_row=max(max_row,row)
    width=options.columns*3.1; depth=max_row*row_spacing+5
    # Fit generated samples, excluding off-camera light geometry and the ground.
    # A fixed pitch makes the row spacing readable while keeping a 3:2 overview
    # filled with samples instead of empty floor/large visible softboxes.
    minimum=np.array((-width/2+.15,0.,-max_row*row_spacing-1.24))
    maximum=np.array((width/2-.15,max_row*.20+2.1,1.61))
    center=(minimum+maximum)*.5
    pitch=math.radians(60); offset=np.array((0.,math.sin(pitch),math.cos(pitch)))
    up=np.array((0.,math.cos(pitch),-math.sin(pitch)))
    tangent=math.tan(math.radians(20)); distance=0.
    for x in (minimum[0],maximum[0]):
        for y in (minimum[1],maximum[1]):
            for z in (minimum[2],maximum[2]):
                point=np.array((x,y,z))-center
                along=np.dot(point,offset)
                distance=max(distance,abs(point[0])/(tangent*1.5)+along,abs(np.dot(point,up))/tangent+along)
    camera_position=center+offset*distance*1.08
    asset.camera=camera_position.tolist()+[0]+center.tolist()+[0]+[0,1,0,0,40]
    asset.has_camera=True
    ground=box(first_support+1,(-width*.65,-.10,-depth+2),(width*.65,-.02,3.7))
    key=Mesh(first_support+3); fill=Mesh(first_support+4)
    # Side-facing vertical softboxes stay outside every column. Their apparent
    # footprint cannot cover the central spheres in the fitted gallery view.
    left=-width/2-18; right=width/2+18
    quad(key,((left,2,3),(left,8,3),(left,8,-depth),(left,2,-depth)),(1,0,0))
    quad(fill,((right,2,3),(right,8,3),(right,8,-depth),(right,2,-depth)),(-1,0,0))
    for name,mesh in (("Pedestals",pedestals),("Labels",labels),("Ground",ground),("KeyLight",key),("FillLight",fill)):
        asset.instances.append((len(asset.meshes),IDENTITY.copy())); asset.meshes.append(mesh)
        usd_mesh(stage,"/Gallery/Environment/"+name,mesh,IDENTITY,usd_materials[mesh.material])
    positions=[]
    for mesh_index,transform in asset.instances:
        matrix=np.asarray(transform).reshape(3,4)
        positions.extend(np.asarray([v[:3] for v in asset.meshes[mesh_index].vertices])@matrix[:,:3].T+matrix[:,3])
    positions=np.asarray(positions)
    asset.bounds_min=np.min(positions,axis=0).tolist()+[0]; asset.bounds_max=np.max(positions,axis=0).tolist()+[0]
    camera=UsdGeom.Camera.Define(stage,"/Gallery/Camera")
    view=Gf.Matrix4d(1); view.SetLookAt(Gf.Vec3d(*camera_position),Gf.Vec3d(*center),Gf.Vec3d(0,1,0))
    camera.AddTransformOp().Set(view.GetInverse())
    camera.CreateVerticalApertureAttr(24); camera.CreateHorizontalApertureAttr(36)
    camera.CreateFocalLengthAttr(12/math.tan(math.radians(40/2)))
    camera.CreateFocusDistanceAttr(float(np.linalg.norm(camera_position-center)))
    stage.GetRootLayer().customLayerData={"description":"Every official MaterialX %s material example; generated geometry, no graph substitutions" % VERSION}
    stage.GetRootLayer().Save()
    write_asset(asset,output.with_suffix(".caustra"))
    provenance=copy_provenance(root,files,writer)
    manifest={"materialx_version":VERSION,"upstream":"https://github.com/AcademySoftwareFoundation/MaterialX/tree/v"+VERSION+"/resources/Materials/Examples",
              "source_file_count":len(files),"example_count":len(entries),
              "unique_material_count":len({entry["source_identity"] for entry in entries}),
              "look_reuse_count":len(entries)-len({entry["source_identity"] for entry in entries}),
              "all_examples_required":True,"materials":entries,"resources":list(writer.resources.values()),
              "texture_max_size":options.texture_max_size,"texture_resolutions":compiler.textures.resolutions,
              "heterogeneous_interior":{"grid_resolution":[32,32,32],"representation":"piecewise_constant_object_space_cells","transport":"cell_boundary_DDA","spherical_uv":"radial_extension"},
              "provenance":provenance,
              "scene":{"meshes":len(asset.meshes),"instances":len(asset.instances),"materials":len(asset.materials),"textures":len(asset.textures),"instructions":len(asset.instructions)},
              "camera":asset.camera,"source_scene":output.with_suffix(".usda").name,"package":output.with_suffix(".caustra").name}
    output.with_suffix(".json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Wrote %s: all %d examples, %d meshes, %d textures, %d instructions" % (output.with_suffix(".caustra"),len(entries),len(asset.meshes),len(asset.textures),len(asset.instructions)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True,help="MaterialX source root or resources/Materials/Examples directory")
    parser.add_argument("--output",type=Path,default=Path("renders/materialx_official"),help="Output base path (.caustra, .usda, .json)")
    parser.add_argument("--columns",type=int,default=11)
    parser.add_argument("--segments",type=int,default=48)
    parser.add_argument("--rings",type=int,default=24)
    parser.add_argument("--displacement-subdivisions",type=int,default=1)
    parser.add_argument("--texture-max-size",type=int,default=512,help="Explicit package texture resolution limit; 0 preserves source resolution. Editable USD resources always preserve originals.")
    parser.add_argument("--validate-only",action="store_true",help="Compile the complete inventory without writing render geometry")
    options=parser.parse_args(argv)
    if not 1<=options.columns<=16 or not 8<=options.segments<=128 or not 4<=options.rings<=64 or not 0<=options.displacement_subdivisions<=4 or not 0<=options.texture_max_size<=16384:
        parser.error("columns must be 1..16, segments 8..128, rings 4..64, displacement subdivisions 0..4")
    try:
        build(options)
    except (AssetError,OSError,RuntimeError) as error:
        print("Caustra official MaterialX gallery: "+str(error),file=sys.stderr); return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
