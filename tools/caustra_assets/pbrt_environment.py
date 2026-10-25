# SPDX-License-Identifier: Apache-2.0
# Equal-area mapping adapted from pbrt, Copyright(c) 1998-2020 Matt Pharr,
# Wenzel Jakob, and Greg Humphreys. Modified: Python offline conversion,
# adaptive patches, finite geometry and flux-preserving palette generation.
"""PBRT's equal-area HDR sky as a finite, inward-emitting MaterialX backdrop.

The sphere mapping follows Clarberg's equal-area construction, as documented
by PBRT v4 (https://pbr-book.org/4ed/Geometry_and_Transformations/Spherical_Geometry).
The angular piecewise-constant approximation and finite distance are recorded
in the conversion report; the original EXR remains beside the converted asset.
"""
from collections import defaultdict
from pathlib import Path
import math
import shutil
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
from .model import AssetError


def equal_area_direction(u, v):
    x, y = 2*u-1, 2*v-1
    a, b = abs(x), abs(y)
    distance = 1-a-b
    r = 1-abs(distance)
    phi = (1 if r == 0 else (b-a)/r+1)*math.pi/4
    z = math.copysign(1-r*r, distance)
    radial = r*math.sqrt(max(0, 2-r*r))
    return np.array((math.copysign(math.cos(phi), x)*radial,
                     math.copysign(math.sin(phi), y)*radial, z))


def rgb_to_xyz_matrix(chromaticities):
    primaries = np.array(chromaticities[:6], dtype=float).reshape(3, 2)
    white = np.array(chromaticities[6:], dtype=float)
    columns = np.array([[x/y, 1, (1-x-y)/y] for x, y in primaries]).T
    white_xyz = np.array((white[0]/white[1], 1, (1-white.sum())/white[1]))
    return columns @ np.diag(np.linalg.solve(columns, white_xyz)), white_xyz


def convert_to_rec709(pixels, chromaticities):
    """Linear primary conversion with Bradford adaptation to the D65 white."""
    source, white = rgb_to_xyz_matrix(chromaticities)
    target, target_white = rgb_to_xyz_matrix((.64,.33,.30,.60,.15,.06,.3127,.3290))
    bradford = np.array(((.8951,.2664,-.1614),(-.7502,1.7135,.0367),(.0389,-.0685,1.0296)))
    adaptation = np.linalg.inv(bradford) @ np.diag((bradford@target_white)/(bradford@white)) @ bradford
    matrix = np.linalg.inv(target) @ adaptation @ source
    converted = np.asarray(pixels, dtype=np.float64) @ matrix.T
    negative = np.minimum(converted, 0)
    report = {"source_chromaticities": list(chromaticities), "linear_to_rec709": matrix.tolist(),
              "negative_rgb_clipped_absolute_mean": (-negative).mean(axis=(0,1)).tolist()}
    return np.maximum(converted, 0), report


def sky_patches(pixels, base_resolution=64):
    """Preserve each patch's RGB integral and refine concentrated bright sources.

    Child vertices interpolate the base polyhedron's triangles, so adaptive
    subdivision creates no gaps between fine and coarse neighboring patches.
    """
    h, w, _ = pixels.shape
    if h != w or w % base_resolution:
        raise AssetError("PBRT equal-area sky must be square and divisible by the base resolution")
    stride = w//base_resolution
    result = []
    for by in range(base_resolution):
        for bx in range(base_resolution):
            x0, y0 = bx*stride, by*stride
            corners = [equal_area_direction((x0+dx)/w, (y0+dy)/h)
                       for dx,dy in ((0,0),(stride,0),(0,stride),(stride,stride))]

            def position(x, y):
                s, t = (x-x0)/stride, (y-y0)/stride
                a,b,c,d = corners
                return a+s*(b-a)+t*(c-a) if s+t <= 1 else d+(1-t)*(b-d)+(1-s)*(c-d)

            def emit(x, y, size):
                region = pixels[y:y+size, x:x+size]
                mean = region.mean(axis=(0,1))
                luminance = region@np.array((.2126,.7152,.0722))
                if size > 1 and luminance.max() > max(2., 2*luminance.mean()):
                    half = size//2
                    for dx,dy in ((0,0),(half,0),(0,half),(half,half)):
                        emit(x+dx, y+dy, half)
                    return
                points = np.array([position(x+dx,y+dy) for dx,dy in ((0,0),(size,0),(0,size),(size,size))])
                result.append({"points": points, "color": mean, "weight": size*size,
                               "pixel_rect": (x,y,size)})
            emit(x0,y0,stride)
    return result


def palette_patches(patches, maximum=256):
    """Bound material/BLAS counts; preserve the equal-area RGB integral exactly."""
    for precision in (32,16,8,4,2,1):
        groups = defaultdict(list)
        for patch in patches:
            color = patch["color"]
            luminance = float(color@np.array((.2126,.7152,.0722)))
            if luminance <= 1e-12:
                key = ("black",)
            else:
                chroma = color/max(float(color.sum()),1e-30)
                key = (round(math.log2(luminance)*max(1,precision//8)),
                       round(float(chroma[0])*precision), round(float(chroma[1])*precision))
            groups[key].append(patch)
        if len(groups) <= maximum:
            break
    if len(groups) > maximum:
        raise AssetError("Sky palette cannot fit the material budget")
    result = []
    for key in sorted(groups, key=str):
        members = groups[key]
        weight = sum(p["weight"] for p in members)
        color = sum((p["color"]*p["weight"] for p in members), np.zeros(3))/weight
        result.append((color, members))
    return result


def _emitter(stage, library, name, color):
    surface = library.document.addNodeInstance(library.libraries.getNodeDef("ND_standard_surface_surfaceshader"), name)
    for input_name,value in (("base",0.),("specular",0.),("emission",1.)):
        surface.addInput(input_name,"float").setValue(value)
    surface.addInput("emission_color","color3").setValueString(", ".join(format(float(c),".9g") for c in color))
    material_node = library.document.addNode("surfacematerial",name+"_material","material")
    material_node.addInput("surfaceshader","surfaceshader").setConnectedNode(surface)
    material = UsdShade.Material.Define(stage,"/World/SkyMaterials/"+name)
    shader = UsdShade.Shader.Define(stage,material.GetPath().AppendChild("Surface"))
    shader.CreateIdAttr("ND_standard_surface_surfaceshader")
    for input_name,value in (("base",0.),("specular",0.),("emission",1.)):
        shader.CreateInput(input_name,Sdf.ValueTypeNames.Float).Set(value)
    shader.CreateInput("emission_color",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float,color)))
    shader.CreateOutput("out",Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mtlx").ConnectToSource(shader.ConnectableAPI(),"out")
    return material


def add_environment(stage, scene, materials, output_dir, bounds):
    """Conversion callback: one PBRT image infinite light, in PBRT Z-up space."""
    if not scene.lights:
        return {"type":"none"}
    if len(scene.lights) != 1:
        raise AssetError("San Miguel conversion expects exactly one HDR infinite light")
    light = scene.lights[0]
    command = light.command
    if command.args != ("infinite",) or set(command.params)-{"filename","scale"}:
        raise AssetError("Unsupported PBRT environment light type/parameters")
    # Command API is supplied by the PBRT parser; keep value handling explicit.
    params = command.params
    def value(name, default=None):
        entry = params.get(name, default)
        if hasattr(entry,"values"):
            entry = entry.values
        if isinstance(entry,(tuple,list)) and len(entry)==1:
            return entry[0]
        return entry
    filename = value("filename")
    if not filename:
        raise AssetError("Expected a PBRT HDR image filename")
    source_root = Path(scene.asset_root).resolve()
    source = (source_root/str(filename)).resolve()
    if not source.is_relative_to(source_root):
        raise AssetError("HDR environment escapes source directory")
    try:
        import OpenEXR
    except ImportError as error:
        raise AssetError("San Miguel HDR conversion needs requirements-sanmiguel.txt") from error
    image = OpenEXR.File(str(source))
    channels = image.channels()
    if "RGB" not in channels:
        raise AssetError("Expected RGB channels in the PBRT sky EXR")
    pixels = np.asarray(channels["RGB"].pixels,dtype=float)
    if not np.isfinite(pixels).all() or (pixels<0).any():
        raise AssetError("Invalid PBRT sky radiance")
    chroma = image.header().get("chromaticities",(.64,.33,.30,.60,.15,.06,.3127,.3290))
    pixels, color_report = convert_to_rec709(pixels,chroma)
    scale = float(value("scale",1.))
    if not math.isfinite(scale) or scale < 0:
        raise AssetError("Invalid infinite light scale")
    pixels *= scale
    patches = sky_patches(pixels)
    palette = palette_patches(patches)
    minimum, maximum = np.asarray(bounds[0],dtype=float)[:3],np.asarray(bounds[1],dtype=float)[:3]
    center = (minimum+maximum)/2
    radius = max(1000.,100*float(np.linalg.norm(maximum-minimum)))
    rotation = np.asarray(light.transform,dtype=float)[:3,:3]
    # PBRT lights accept only rigid directional transforms here.
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6):
        raise AssetError("Non-rigid HDR light transform is unsupported")
    triangle_count = 0
    UsdGeom.Scope.Define(stage,"/World/Sky")
    for index,(color,members) in enumerate(palette):
        mesh = UsdGeom.Mesh.Define(stage,"/World/Sky/PatchGroup_%03d"%index)
        vertices, indices = [], []
        for patch in members:
            points = (patch["points"]@rotation.T)*radius+center
            for triangle in ((0,1,2),(1,3,2)):
                a,b,c = (points[i] for i in triangle)
                if np.linalg.norm(np.cross(b-a,c-a)) < radius*radius*1e-14:
                    continue
                normal = np.cross(b-a,c-a)
                if np.dot(normal,(a+b+c)/3-center)>0:
                    b,c = c,b
                first = len(vertices)
                vertices.extend((a.tolist(),b.tolist(),c.tolist()))
                indices.extend((first,first+1,first+2))
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices,dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]*(len(indices)//3)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.GetPrim().SetCustomDataByKey("caustra:environmentBackdrop",True)
        material = _emitter(stage,materials,"sky_%03d"%index,color)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        triangle_count += len(indices)//3
    target = Path(output_dir)/"textures"/source.name
    target.parent.mkdir(parents=True,exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source,target)
    source_integral = pixels.mean(axis=(0,1))*4*math.pi
    palette_integral = sum((color*sum(p["weight"] for p in members) for color,members in palette),np.zeros(3))/(pixels.shape[0]*pixels.shape[1])*4*math.pi
    return {"type":"finite_piecewise_constant_emissive_backdrop", "source":"textures/"+source.name,
            "mapping":"PBRT v4 equal-area square", "source_resolution":list(pixels.shape[:2]),
            "source_scale":scale,"rotation":rotation.tolist(),"radius_m":radius,
            "base_grid":64,"adaptive_patches":len(patches),"materials":len(palette),"triangles":triangle_count,
            "source_rgb_spherical_integral":source_integral.tolist(),"palette_rgb_spherical_integral":palette_integral.tolist(),
            "color_conversion":color_report,
            "approximation":"Finite polyhedral backdrop; constant angular patches with flux-preserving color quantization. Bright sources refined to source pixels; not a native infinite-light implementation."}
