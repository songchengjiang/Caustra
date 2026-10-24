"""Validate MaterialX graphs and lower supported patterns to the renderer IR.

This deliberately does not guess how an unknown node should shade. MaterialX
NodeDefs supply input types/defaults; supported graphs keep all their spatial
and texture-dependent computation. Unsupported closures and active inputs fail.
"""
from pathlib import Path
from io import BytesIO
import math
import zipfile
import numpy as np
from PIL import Image
import MaterialX as mx
from pxr import Ar
from .model import AssetError, Material, MaterialModel, Texture, Op, Parameter, float4


def standard_library():
    doc=mx.createDocument()
    mx.loadLibraries(mx.getDefaultDataLibraryFolders(),mx.getDefaultDataSearchPath(),doc)
    return doc


def connected_node(port,document,seen=None):
    """Resolve closure connections through the same graph/interface boundaries."""
    if port is None: return None
    seen=set() if seen is None else set(seen)
    key=port.getNamePath()
    if key in seen: raise AssetError("Cycle in MaterialX closure connection")
    seen.add(key)
    if port.getNodeName():
        node=port.getConnectedNode()
        if node is None: raise AssetError("Unresolved MaterialX closure node: "+key)
        return node
    if port.getNodeGraphString():
        graph=document.getNodeGraph(port.getNodeGraphString())
        output=graph.getOutput(port.getOutputString() or "out") if graph else None
        if output is None: raise AssetError("Unresolved MaterialX closure graph: "+key)
        return connected_node(output,document,seen)
    if port.getInterfaceName():
        parent=port.getParent().getParent();interface=parent.getInput(port.getInterfaceName())
        if interface is None: raise AssetError("Unresolved MaterialX closure interface: "+key)
        return connected_node(interface,document,seen)
    return None


def material_terminals(document,surface=None):
    """Return explicitly connected auxiliary terminals for one material document."""
    result={}
    for material in document.getNodes():
        if material.getType()!="material": continue
        for port in material.getInputs():
            if port.getType() not in ("displacementshader","volumeshader"): continue
            node=connected_node(port,document)
            if node is None:
                if MaterialCompiler.connected(port): raise AssetError("Unresolved material terminal "+port.getNamePath())
                continue
            key=port.getType()
            if key in result and result[key]!=node: raise AssetError("Multiple "+key+" terminals in one material")
            result[key]=node
    return result


def color_space(name):
    name=str(name or "linear").lower()
    if name in ("srgb", "srgb_texture", "srgb_rec709"): return "srgb"
    if name in ("raw", "linear", "lin_rec709", "none"): return "linear"
    raise AssetError("Unsupported color space: "+name)


def srgb_to_linear(values):
    x=np.asarray(values,dtype=np.float32)
    return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)


def read_resource(path):
    # Ar asset paths may identify an image inside a USDZ archive.
    text=str(path)
    if Ar.IsPackageRelativePath(text):
        resolver=Ar.GetResolver()
        if hasattr(resolver,"OpenAsset"):
            resolved=resolver.Resolve(text)
            asset=resolver.OpenAsset(resolved or Ar.ResolvedPath(text))
            if not asset: raise AssetError("USD resolver cannot open texture: "+text)
            if asset.GetSize()>512*1024*1024: raise AssetError("Texture archive member is too large")
            return asset.GetBuffer()
        # Compatibility with bindings without Ar.OpenAsset: read bytes directly,
        # never extract archive members to filesystem paths.
        archive,member=Ar.SplitPackageRelativePathOuter(text)
        with zipfile.ZipFile(archive) as package:
            entry=package.getinfo(member)
            if entry.file_size>512*1024*1024: raise AssetError("Texture archive member is too large")
            return package.read(entry)
    return Path(path).read_bytes()


class TextureStore:
    def __init__(self,asset):
        self.asset=asset
        self.cache={}

    def add(self,path,space):
        space=color_space(space)
        key=(str(path),space)
        if key in self.cache: return self.cache[key]
        try:
            with Image.open(BytesIO(read_resource(path))) as source:
                if getattr(source,"n_frames",1)!=1: raise AssetError("Animated textures are unsupported")
                if source.width*source.height>64*1024*1024: raise AssetError("Texture exceeds the 64 megapixel limit")
                if source.mode=="F":
                    grey=np.asarray(source,dtype=np.float32)
                    pixels=np.stack([grey,grey,grey,np.ones_like(grey)],axis=-1)
                else: pixels=np.asarray(source.convert("RGBA"),dtype=np.float32)/255
        except (OSError,ValueError,KeyError) as error:
            raise AssetError("Cannot load texture %s: %s" % (path,error)) from error
        if space=="srgb": pixels[...,:3]=srgb_to_linear(pixels[...,:3])
        if not np.isfinite(pixels).all(): raise AssetError("Texture contains nonfinite values: "+str(path))
        height,width=pixels.shape[:2]
        index=len(self.asset.textures)
        self.asset.textures.append(Texture(width,height,pixels.reshape(-1,4).tolist()))
        self.cache[key]=index
        return index


COMPONENTS={"float":1,"integer":1,"boolean":1,"vector2":2,"vector3":3,"color3":3,"vector4":4,"color4":4}
# parameter, POD float4 row, lane (None means RGB)
SURFACE_INPUTS={
    "base_color":(Parameter.BaseColor,0,None), "base":(Parameter.BaseWeight,9,0),
    "metalness":(Parameter.Metalness,2,0), "specular_roughness":(Parameter.Roughness,2,1),
    "diffuse_roughness":(Parameter.DiffuseRoughness,9,1),
    "specular":(Parameter.SpecularWeight,3,0), "specular_IOR":(Parameter.SpecularIOR,3,1),
    "specular_anisotropy":(Parameter.Anisotropy,3,2), "specular_rotation":(Parameter.Rotation,3,3),
    "specular_color":(Parameter.SpecularColor,4,None), "coat":(Parameter.CoatWeight,5,0),
    "coat_roughness":(Parameter.CoatRoughness,5,1), "coat_IOR":(Parameter.CoatIOR,5,2),
    "coat_color":(Parameter.CoatColor,6,None), "sheen":(Parameter.SheenWeight,7,0),
    "sheen_roughness":(Parameter.SheenRoughness,7,1), "sheen_color":(Parameter.SheenColor,8,None),
}
OPENPBR_INPUTS={
    "base_color":(Parameter.BaseColor,0,None), "base_weight":(Parameter.BaseWeight,9,0),
    "base_metalness":(Parameter.Metalness,2,0), "specular_roughness":(Parameter.Roughness,2,1),
    "base_diffuse_roughness":(Parameter.DiffuseRoughness,9,1),
    "specular_weight":(Parameter.SpecularWeight,3,0), "specular_ior":(Parameter.SpecularIOR,3,1),
    "specular_roughness_anisotropy":(Parameter.Anisotropy,3,2),
    "specular_color":(Parameter.SpecularColor,4,None), "coat_weight":(Parameter.CoatWeight,5,0),
    "coat_roughness":(Parameter.CoatRoughness,5,1), "coat_ior":(Parameter.CoatIOR,5,2),
    "coat_roughness_anisotropy":(Parameter.CoatAnisotropy,5,3),
    "coat_color":(Parameter.CoatColor,6,None), "coat_darkening":(Parameter.CoatDarkening,9,3),
}

SURFACE_INPUTS.update({
    "transmission":(Parameter.TransmissionWeight,10,0),
    "transmission_color":(Parameter.TransmissionColor,11,None),
    "transmission_depth":(Parameter.TransmissionDepth,10,1),
    "transmission_scatter":(Parameter.TransmissionScatter,12,None),
    "transmission_scatter_anisotropy":(Parameter.TransmissionAnisotropy,10,2),
    "transmission_extra_roughness":(Parameter.TransmissionExtraRoughness,10,3),
    "subsurface":(Parameter.SubsurfaceWeight,13,0),
    "subsurface_color":(Parameter.SubsurfaceColor,14,None),
    "subsurface_radius":(Parameter.SubsurfaceRadius,15,None),
    "subsurface_scale":(Parameter.SubsurfaceScale,13,1),
    "subsurface_anisotropy":(Parameter.SubsurfaceAnisotropy,13,2),
    "thin_walled":(Parameter.ThinWalled,13,3),
    "thin_film_thickness":(Parameter.ThinFilmThickness,16,1),
    "thin_film_IOR":(Parameter.ThinFilmIOR,16,2),
    "coat_anisotropy":(Parameter.CoatAnisotropy,5,3),
    "coat_rotation":(Parameter.CoatRotation,17,2),
    "coat_affect_color":(Parameter.CoatAffectColor,17,3),
    "coat_affect_roughness":(Parameter.CoatAffectRoughness,19,3),
})
OPENPBR_INPUTS.update({
    "transmission_weight":(Parameter.TransmissionWeight,10,0),
    "transmission_color":(Parameter.TransmissionColor,11,None),
    "transmission_depth":(Parameter.TransmissionDepth,10,1),
    "transmission_scatter":(Parameter.TransmissionScatter,12,None),
    "transmission_scatter_anisotropy":(Parameter.TransmissionAnisotropy,10,2),
    "subsurface_weight":(Parameter.SubsurfaceWeight,13,0),
    "subsurface_color":(Parameter.SubsurfaceColor,14,None),
    "subsurface_radius":(Parameter.SubsurfaceScale,13,1),
    "subsurface_radius_scale":(Parameter.SubsurfaceRadius,15,None),
    "subsurface_scatter_anisotropy":(Parameter.SubsurfaceAnisotropy,13,2),
    "geometry_thin_walled":(Parameter.ThinWalled,13,3),
    "thin_film_weight":(Parameter.ThinFilmWeight,16,0),
    "thin_film_thickness":(Parameter.ThinFilmThickness,16,1),
    "thin_film_ior":(Parameter.ThinFilmIOR,16,2),
    "fuzz_weight":(Parameter.FuzzWeight,17,0),
    "fuzz_color":(Parameter.FuzzColor,18,None),
    "fuzz_roughness":(Parameter.FuzzRoughness,17,1),
})

# These are native closure payloads, not conversions to Standard Surface.
# Slots are shared by the fixed renderer ABI; the model tag defines their
# interpretation. Keep these mappings synchronized with official_models.slang.
DISNEY_INPUTS={
    "baseColor":(Parameter.BaseColor,0,None), "metallic":(Parameter.Metalness,2,0),
    "roughness":(Parameter.Roughness,2,1), "anisotropic":(Parameter.Anisotropy,3,2),
    "specular":(Parameter.SpecularWeight,3,0), "specularTint":(Parameter.CoatDarkening,9,3),
    "sheen":(Parameter.SheenWeight,7,0), "sheenTint":(Parameter.CoatAffectColor,17,3),
    "clearcoat":(Parameter.CoatWeight,5,0), "clearcoatGloss":(Parameter.CoatRoughness,5,1),
    "specTrans":(Parameter.TransmissionWeight,10,0), "ior":(Parameter.SpecularIOR,3,1),
    "subsurface":(Parameter.SubsurfaceWeight,13,0), "subsurfaceDistance":(Parameter.SubsurfaceRadius,15,None),
}
GLTF_INPUTS={
    "base_color":(Parameter.BaseColor,0,None), "metallic":(Parameter.Metalness,2,0),
    "roughness":(Parameter.Roughness,2,1), "occlusion":(Parameter.BaseWeight,9,0),
    "transmission":(Parameter.TransmissionWeight,10,0), "specular":(Parameter.SpecularWeight,3,0),
    "specular_color":(Parameter.SpecularColor,4,None), "ior":(Parameter.SpecularIOR,3,1),
    "alpha":(Parameter.Opacity,9,2), "alpha_mode":(Parameter.CoatRotation,17,2),
    "alpha_cutoff":(Parameter.CoatAffectRoughness,19,3),
    "iridescence":(Parameter.ThinFilmWeight,16,0), "iridescence_ior":(Parameter.ThinFilmIOR,16,2),
    "iridescence_thickness":(Parameter.ThinFilmThickness,16,1),
    "sheen_color":(Parameter.SheenColor,8,None), "sheen_roughness":(Parameter.SheenRoughness,7,1),
    "clearcoat":(Parameter.CoatWeight,5,0), "clearcoat_roughness":(Parameter.CoatRoughness,5,1),
    "thickness":(Parameter.SubsurfaceScale,13,1),
    "attenuation_distance":(Parameter.TransmissionDepth,10,1),
    "attenuation_color":(Parameter.TransmissionColor,11,None),
    "anisotropy_strength":(Parameter.Anisotropy,3,2), "anisotropy_rotation":(Parameter.Rotation,3,3),
    "dispersion":(Parameter.Dispersion,16,3),
}
USD_INPUTS={
    "diffuseColor":(Parameter.BaseColor,0,None), "emissiveColor":(Parameter.Emission,1,None),
    "useSpecularWorkflow":(Parameter.CoatDarkening,9,3), "specularColor":(Parameter.SpecularColor,4,None),
    "metallic":(Parameter.Metalness,2,0), "roughness":(Parameter.Roughness,2,1),
    "clearcoat":(Parameter.CoatWeight,5,0), "clearcoatRoughness":(Parameter.CoatRoughness,5,1),
    "opacity":(Parameter.Opacity,9,2), "opacityMode":(Parameter.CoatAffectColor,17,3),
    "opacityThreshold":(Parameter.Rotation,3,3), "ior":(Parameter.SpecularIOR,3,1),
    "occlusion":(Parameter.BaseWeight,9,0),
}
HAIR_INPUTS={
    "baseColor":(Parameter.BaseColor,0,None), "melaninConcentration":(Parameter.BaseWeight,9,0),
    "melaninRedness":(Parameter.DiffuseRoughness,9,1), "melaninMix":(Parameter.CoatDarkening,9,3),
    "longitudinalRoughness":(Parameter.Roughness,2,1), "azimuthalRoughness":(Parameter.Anisotropy,3,2),
    "cuticleAngle":(Parameter.CoatRoughness,5,1), "tint_R":(Parameter.SpecularColor,4,None),
    "tint_TT":(Parameter.TransmissionColor,11,None), "tint_TRT":(Parameter.SheenColor,8,None),
    "ior":(Parameter.SpecularIOR,3,1),
}
NATIVE_INPUTS={"disney_principled":(MaterialModel.DisneyPrincipled,DISNEY_INPUTS),
               "gltf_pbr":(MaterialModel.GltfPbr,GLTF_INPUTS),
               "UsdPreviewSurface":(MaterialModel.UsdPreviewSurface,USD_INPUTS),
               "simple_hair":(MaterialModel.SimpleHair,HAIR_INPUTS)}


class MaterialCompiler:
    def __init__(self,asset,libraries=None,unit_scale=1.0):
        self.asset=asset
        self.libraries=libraries or standard_library()
        self.textures=TextureStore(asset)
        self.material_uv_names={}
        self.material_displacements={}
        self.unit_scale=unit_scale

    def compile(self,document,surface,base_directory=Path(".")):
        document.setDataLibrary(self.libraries)
        # Expand these official graph implementations, rather than hand-copying
        # their UV transform, factor, and alpha semantics. Surface closures stay
        # intact and retain their own native model tag.
        expand={"tiledimage","place2d","gltf_image","gltf_colorimage","gltf_normalmap"}
        for graph in [document]+list(document.getNodeGraphs()):
            graph.flattenSubgraphs("",lambda node:node.getCategory() in expand)
        valid,message=document.validate()
        if not valid: raise AssetError("Invalid MaterialX graph: "+message)
        terminals=material_terminals(document,surface)
        category=surface.getCategory() if surface is not None else "volume_only";openpbr=category=="open_pbr_surface"
        if category not in ("standard_surface","open_pbr_surface","volume_only",*NATIVE_INPUTS):
            raise AssetError("Unsupported MaterialX surface "+category)
        nd=surface.getNodeDef() if surface is not None else None
        if surface is not None and nd is None: raise AssetError("Surface has no registered MaterialX NodeDef")
        self.document=document;self.base_directory=Path(base_directory)
        self.program=[];self.registers={};self.constants={};self.visiting=set();self.uv_names=set();self.heterogeneous=False
        material=Material()
        if category in NATIVE_INPUTS:
            return self.compile_native(surface,material,terminals)
        if surface is None:
            if "volumeshader" not in terminals: raise AssetError("Material needs a surface or volume terminal")
            material.values[0][:3]=[0,0,0];material.values[3][0]=0
            material.values[19][2]=1
            self.compile_volume(terminals["volumeshader"],material)
            return self.finish_material(material,False,terminals)
        mapping=OPENPBR_INPUTS if openpbr else SURFACE_INPUTS
        emission_name="emission_luminance" if openpbr else "emission"
        opacity_name="geometry_opacity" if openpbr else "opacity"
        normals=(("geometry_normal",Parameter.Normal),("geometry_coat_normal",Parameter.CoatNormal),
                 ("geometry_tangent",Parameter.Tangent),("geometry_coat_tangent",Parameter.CoatTangent)) if openpbr else (
                 ("normal",Parameter.Normal),("coat_normal",Parameter.CoatNormal),("tangent",Parameter.Tangent))
        dispersion_names={"transmission_dispersion_scale","transmission_dispersion_abbe_number"} if openpbr else {"transmission_dispersion"}
        supported=set(mapping)|{emission_name,"emission_color",opacity_name}|{name for name,_ in normals}|dispersion_names
        inactive=set()
        thin_name="geometry_thin_walled" if openpbr else "thin_walled"
        thin=float(self.constant_port(self.input(surface,thin_name))[0])
        if thin not in (0,1): raise AssetError("Thin-walled flag must be constant boolean")
        interior_weights=("transmission_weight","subsurface_weight") if openpbr else ("transmission","subsurface")
        solid="volumeshader" in terminals
        constant_weights={}
        for name in interior_weights:
            port=self.input(surface,name)
            if not thin:
                values=self.interior_constant(port)
                if values is None: solid=True
                else:
                    value=float(values[0]);constant_weights[name]=value
                    solid|=value>0
        homogeneous={Parameter.TransmissionWeight,Parameter.TransmissionColor,Parameter.TransmissionDepth,
                     Parameter.TransmissionScatter,Parameter.TransmissionAnisotropy,
                     Parameter.SubsurfaceWeight,Parameter.SubsurfaceColor,Parameter.SubsurfaceRadius,
                     Parameter.SubsurfaceScale,Parameter.SubsurfaceAnisotropy,Parameter.SpecularIOR}
        # Standard Surface clear-coat changes the observed SSS color before
        # converting it to homogeneous medium coefficients, so these inputs
        # cannot vary by the boundary entry point either.
        if solid and not openpbr and constant_weights.get("subsurface",1)>0:
            homogeneous.update((Parameter.CoatWeight,Parameter.CoatAffectColor))
        if solid and openpbr:
            # Native OpenPBR specular weight modulates the actual dielectric
            # IOR, including the homogeneous interior's refractive index.
            homogeneous.add(Parameter.SpecularWeight)
        # Ignore graphs only when their entire closure is provably disabled.
        weights=(("transmission","transmission_weight"),("subsurface","subsurface_weight"),("thin_film","thin_film_weight"),("fuzz","fuzz_weight")) if openpbr else (("transmission","transmission"),("subsurface","subsurface"))
        for prefix,weight_name in weights:
            weight=self.input(surface,weight_name)
            if constant_weights.get(weight_name,1)==0 or (not self.connected(weight) and self.numeric(weight)==0):
                inactive.update(i.getName() for i in nd.getActiveInputs() if i.getName().startswith(prefix+"_") and i.getName()!=weight_name)
        for definition in nd.getActiveInputs():
            name=definition.getName()
            if name in supported or name in inactive: continue
            port=surface.getInput(name)
            if port is None: continue
            if self.connected(port) or (port.getValueString() and port.getValueString()!=definition.getValueString()):
                # Numeric equality avoids rejecting spelling-only differences, e.g. 0 vs 0.0.
                if not self.connected(port) and self.equivalent(port.getValue(),definition.getValue()): continue
                raise AssetError("Unsupported %s input: %s" % (category,name))
        for name,(parameter,row,lane) in mapping.items():
            if name in inactive: continue
            port=self.input(surface,name)
            scale=self.unit_scale if parameter in (Parameter.TransmissionDepth,Parameter.SubsurfaceScale) else 1000. if openpbr and parameter==Parameter.ThinFilmThickness else 1.
            fold=parameter==Parameter.ThinWalled or (solid and parameter in homogeneous) or (
                not thin and parameter in (Parameter.TransmissionWeight,Parameter.SubsurfaceWeight))
            uniform_ior=parameter==Parameter.SpecularIOR or (openpbr and parameter==Parameter.SpecularWeight)
            folded_values=self.interior_constant(port,allow_spatial=not uniform_ior) if self.connected(port) and fold and parameter!=Parameter.ThinWalled else None
            if self.connected(port) and (not fold or (parameter!=Parameter.ThinWalled and folded_values is None)):
                reg=self.compile_port(port)
                if scale!=1: reg=self.emit(Op.Multiply,reg,self.constant(scale))
                self.emit(Op.Output,reg,int(parameter))
            else:
                if self.connected(port):
                    values=folded_values if folded_values is not None else self.constant_port(port)
                    value=values[:COMPONENTS[port.getType()]].tolist() if COMPONENTS[port.getType()]>1 else float(values[0])
                else: value=self.numeric(port)
                if scale!=1: value=[v*scale for v in value] if isinstance(value,list) else value*scale
                if lane is None: material.values[row][:3]=float4(value)[:3]
                else: material.values[row][lane]=float(value)
        emission=self.input(surface,emission_name);tint=self.input(surface,"emission_color")
        if self.connected(emission) or self.connected(tint):
            reg=self.emit(Op.Multiply,self.compile_port(emission),self.compile_port(tint))
            self.emit(Op.Output,reg,int(Parameter.Emission))
        else: material.values[1][:3]=[float(self.numeric(emission))*v for v in float4(self.numeric(tint))[:3]]
        for name,parameter in normals:
            port=self.input(surface,name)
            if self.connected(port) or port.getValueString(): self.emit(Op.Output,self.compile_port(port),int(parameter))
        opacity=self.input(surface,opacity_name)
        if solid:
            values=self.constant_port(opacity)[:3]
            if max(values)-min(values)>1e-7 or abs(float(values[0])-1)>1e-7:
                raise AssetError("Solid interior materials require constant opacity 1; alpha holes are unsupported")
            material.values[9][2]=1
        elif self.connected(opacity): self.emit(Op.Output,self.opacity_register(opacity),int(Parameter.Opacity))
        else:
            values=float4(self.numeric(opacity))[:3]
            if max(values)-min(values)>1e-7: raise AssetError("Colored opacity/transmission is unsupported")
            material.values[9][2]=values[0]
        # Standard Surface enables thin film through nonzero thickness.
        if not openpbr: material.values[16][0]=1
        self.compile_dispersion(surface,material,openpbr,inactive,solid)
        if "volumeshader" in terminals: self.compile_volume(terminals["volumeshader"],material)
        return self.finish_material(material,openpbr,terminals)

    def compile_native(self,surface,material,terminals):
        """Keep official closure families and every connected numeric graph."""
        category=surface.getCategory();model,mapping=NATIVE_INPUTS[category]
        special={"gltf_pbr":{"normal","tangent","clearcoat_normal","emissive","emissive_strength"},
                 "UsdPreviewSurface":{"normal","displacement"}}.get(category,set())
        for port in surface.getInputs():
            if port.getName() not in set(mapping)|special:
                raise AssetError("Unsupported %s input: %s" % (category,port.getName()))
        # Fold uniform interior graphs into POD values; preserve spatial
        # coefficients for bulk grids. Optical IOR still requires uniformity.
        folded=set()
        if category=="disney_principled":
            folded={"specTrans","subsurface","subsurfaceDistance","ior"}
        elif category=="gltf_pbr":
            folded={"transmission","thickness","attenuation_distance","attenuation_color","ior","dispersion"}
        elif category=="UsdPreviewSurface": folded={"ior"}
        for name,(parameter,row,lane) in mapping.items():
            port=self.input(surface,name)
            # MaterialX specifies an unbounded attenuation distance by omitting
            # its literal. The native backend uses zero for no attenuation.
            if category=="gltf_pbr" and name=="attenuation_distance" and not self.connected(port) and port.getValue() is None:
                value=0.
            elif self.connected(port):
                vector=self.interior_constant(port,allow_spatial=name not in ("ior","dispersion")) if name in folded else None
                if vector is None:
                    reg=self.compile_port(port)
                    if name in ("thickness","attenuation_distance","subsurfaceDistance") and self.unit_scale!=1:
                        reg=self.emit(Op.Multiply,reg,self.constant(self.unit_scale))
                    self.emit(Op.Output,reg,int(parameter));continue
                value=vector[:COMPONENTS[port.getType()]].tolist() if COMPONENTS[port.getType()]>1 else float(vector[0])
            else: value=self.numeric(port)
            if name in ("thickness","attenuation_distance","subsurfaceDistance"):
                value=[v*self.unit_scale for v in value] if isinstance(value,list) else value*self.unit_scale
            if lane is None: material.values[row][:3]=float4(value)[:3]
            else: material.values[row][lane]=float(value)
        if category=="disney_principled":
            port=self.input(surface,"baseColor")
            if self.connected(port): self.emit(Op.Output,self.compile_port(port),int(Parameter.SubsurfaceColor))
            else: material.values[14][:3]=float4(self.numeric(port))[:3]
            if material.values[13][0]>0 or self.heterogeneous:
                if self.connected(port): self.interior_constant(port)
        elif category=="gltf_pbr":
            # The official graph has a surface BTDF; thickness and dispersion
            # are declared but unconnected in MaterialX 1.39.5's implementation.
            material.values[13][3]=1.
            tint=self.input(surface,"emissive");strength=self.input(surface,"emissive_strength")
            if self.connected(tint) or self.connected(strength):
                self.emit(Op.Output,self.emit(Op.Multiply,self.compile_port(tint),self.compile_port(strength)),int(Parameter.Emission))
            else: material.values[1][:3]=[float(self.numeric(strength))*v for v in float4(self.numeric(tint))[:3]]
            for name,parameter in (("normal",Parameter.Normal),("tangent",Parameter.Tangent),("clearcoat_normal",Parameter.CoatNormal)):
                port=self.input(surface,name)
                if self.connected(port) or port.getValueString(): self.emit(Op.Output,self.compile_port(port),int(parameter))
        elif category=="UsdPreviewSurface":
            mode=self.input(surface,"opacityMode");threshold=self.input(surface,"opacityThreshold");opacity=self.input(surface,"opacity")
            if self.connected(mode) or self.connected(threshold) or self.connected(opacity):
                for port in (mode,threshold,opacity):
                    if self.connected(port): self.interior_constant(port)
                active=lambda port:self.emit(Op.OneMinus,self.emit(Op.Clamp,self.emit(Op.Ceil,self.emit(Op.Abs,self.compile_port(port))),self.constant(0),self.constant(1)))
                transmission=self.emit(Op.Multiply,self.emit(Op.OneMinus,self.compile_port(opacity)),self.emit(Op.Multiply,active(mode),active(threshold)))
                self.emit(Op.Output,transmission,int(Parameter.TransmissionWeight))
            elif self.numeric(mode)==0 and self.numeric(threshold)<=0:
                material.values[10][0]=1-float(self.numeric(opacity))
            port=self.input(surface,"normal")
            if self.connected(port) or not self.equivalent(self.numeric(port),[0,0,1]):
                color=self.emit(Op.Add,self.emit(Op.Multiply,self.compile_port(port),self.constant(.5)),self.constant(.5))
                self.emit(Op.Output,self.emit(Op.Normalmap,color,self.constant(1)),int(Parameter.Normal))
            displacement=self.input(surface,"displacement")
            if self.connected(displacement) or float(self.numeric(displacement))!=0:
                # Preview Surface exposes displacement as a numeric input.
                # Use the same external geometric displacement path as an
                # explicit MaterialX displacement terminal.
                if "displacementshader" in terminals: raise AssetError("Both USD surface and terminal displacement are active")
                node=self.document.addNode("displacement",self.document.createValidChildName("caustra_usd_displacement"),"displacementshader")
                node.addInput("displacement","float").copyContentFrom(displacement)
                terminals=dict(terminals,displacementshader=node)
        if "volumeshader" in terminals: self.compile_volume(terminals["volumeshader"],material)
        return self.finish_material(material,model,terminals)

    def finish_material(self,material,model,terminals):
        outputs={code[2] for code,_ in self.program if code[0]==Op.Output}
        explicit_volume=(material.values[19][0]>0 or material.values[19][2]>0 or
                         bool(outputs & {Parameter.VolumeDensity,Parameter.VolumeBoundaryOnly}))
        solid_lobe=(model!=MaterialModel.SimpleHair and material.values[13][3]==0 and
                    (material.values[10][0]>0 or material.values[13][0]>0 or
                     bool(outputs & {Parameter.TransmissionWeight,Parameter.SubsurfaceWeight})))
        # Spatial surface inputs remain live graphs on thin sheets and fibers.
        # They cannot create a bulk coefficient field unless a solid interior
        # or an independent explicit volume is actually/potentially active.
        heterogeneous=self.heterogeneous and (explicit_volume or solid_lobe)
        material.program=[len(self.asset.instructions),len(self.program),int(model),int(heterogeneous)]
        self.asset.instructions.extend(self.program)
        index=len(self.asset.materials);self.asset.materials.append(material)
        self.material_uv_names[index]=set(self.uv_names)
        displacement=terminals.get("displacementshader")
        if displacement is not None:
            if displacement.getCategory()!="displacement": raise AssetError("Unsupported displacement terminal "+displacement.getCategory())
            self.program=[];self.registers={};self.constants={};self.visiting=set()
            port=self.input(displacement,"displacement")
            if port.getType() not in ("float","vector3"): raise AssetError("Displacement must be scalar or vector3")
            reg=self.emit(Op.Multiply,self.compile_port(port),self.compile_port(self.input(displacement,"scale")))
            self.material_displacements[index]=(self.program,reg,port.getType())
            self.material_uv_names[index].update(self.uv_names)
        return index

    def compile_dispersion(self,surface,material,openpbr,inactive,solid=False):
        name="transmission_dispersion_scale" if openpbr else "transmission_dispersion"
        if name in inactive: return
        port=self.input(surface,name)
        abbe=self.input(surface,"transmission_dispersion_abbe_number") if openpbr else port
        if solid or (not self.connected(port) and (not openpbr or not self.connected(abbe))):
            value=float(self.constant_port(port)[0]);denom=float(self.constant_port(abbe)[0])
            if value<0 or (openpbr and value>0 and denom<=0): raise AssetError("Invalid transmission dispersion/Abbe number")
            material.values[16][3]=(value/denom if openpbr else 1/value) if value else 0
        else:
            denominator=self.emit(Op.Max,self.compile_port(abbe),self.constant(1e-6))
            if openpbr: numerator=self.compile_port(port)
            else: numerator=self.emit(Op.Min,self.emit(Op.Ceil,self.emit(Op.Max,self.compile_port(port),self.constant(0))),self.constant(1))
            self.emit(Op.Output,self.emit(Op.Divide,numerator,denominator),int(Parameter.Dispersion))

    def constant_port(self,port):
        from .pattern_cpu import evaluate_program
        saved=(self.program,self.registers,self.constants,self.visiting)
        self.program=[];self.registers={};self.constants={};self.visiting=set()
        try:
            reg=self.compile_port(port)
            return evaluate_program(self.program,self.asset.textures,allow_spatial=False)[reg]
        finally: self.program,self.registers,self.constants,self.visiting=saved

    def interior_constant(self,port,allow_spatial=True):
        try: return self.constant_port(port)
        except AssetError as error:
            if not allow_spatial or "Homogeneous interior coefficients must be spatially constant" not in str(error): raise
            self.heterogeneous=True
            return None

    def compile_volume(self,node,material):
        if node.getCategory()!="volume": raise AssetError("Unsupported volume terminal: "+node.getCategory())
        material.values[19][0]=1
        vdf=connected_node(self.input(node,"vdf"),self.document)
        edf=connected_node(self.input(node,"edf"),self.document)
        if vdf is not None:
            if vdf.getCategory() not in ("anisotropic_vdf","absorption_vdf"): raise AssetError("Unsupported volume VDF: "+vdf.getCategory())
            material.values[20][:3]=self.constant_port(self.input(vdf,"absorption"))[:3].tolist()
            if vdf.getCategory()=="anisotropic_vdf":
                material.values[21][:3]=self.constant_port(self.input(vdf,"scattering"))[:3].tolist()
                material.values[19][1]=float(self.constant_port(self.input(vdf,"anisotropy"))[0])
        if edf is not None:
            if edf.getCategory()!="uniform_edf": raise AssetError("Unsupported volume EDF: "+edf.getCategory())
            material.values[22][:3]=self.constant_port(self.input(edf,"color"))[:3].tolist()
        if vdf is None and edf is None: raise AssetError("Volume has no VDF or EDF")
        if any(x<0 for row in (20,21,22) for x in material.values[row][:3]) or abs(material.values[19][1])>=1:
            raise AssetError("Invalid homogeneous volume coefficients/anisotropy")

    @staticmethod
    def equivalent(a,b):
        try:
            x=float4(a);y=float4(b)
            return all(abs(c-d)<1e-7 for c,d in zip(x,y))
        except (TypeError,ValueError): return a==b

    @staticmethod
    def connected(port):
        return bool(port and (port.getNodeName() or port.getNodeGraphString() or port.getInterfaceName()))

    def input(self,node,name):
        port=node.getInput(name)
        if port is None:
            nd=node.getNodeDef();port=nd.getActiveInput(name) if nd else None
        if port is None: raise AssetError("Missing MaterialX input %s.%s" % (node.getNamePath(),name))
        return port

    def numeric(self,port):
        value=port.getValue()
        if value is None: raise AssetError("Missing numeric value on "+port.getNamePath())
        if port.getType() not in COMPONENTS: raise AssetError("Unsupported numeric type: "+port.getType())
        components=float4(value)
        if not all(math.isfinite(v) for v in components): raise AssetError("Nonfinite MaterialX value")
        if port.getType().startswith("color") and port.getColorSpace():
            if color_space(port.getColorSpace())=="srgb": components[:3]=srgb_to_linear(components[:3]).tolist()
        return components[0] if COMPONENTS[port.getType()]==1 else components[:COMPONENTS[port.getType()]]

    def text(self,node,name):
        port=self.input(node,name)
        if self.connected(port): raise AssetError("Dynamic string input is unsupported: "+port.getNamePath())
        return port.getValueString()

    def emit(self,op,a=0,b=0,c=0,value=(0,0,0,0)):
        result=len(self.program)
        if result>=128: raise AssetError("Material graph exceeds 128 renderer instructions")
        self.program.append(([int(op),int(a),int(b),int(c)],float4(value)))
        return result

    def constant(self,value):
        vector=tuple(float4(value))
        if vector not in self.constants: self.constants[vector]=self.emit(Op.Constant,value=vector)
        return self.constants[vector]

    def opacity_register(self,port):
        node=port.getConnectedNode()
        if node is not None:
            if node.getType()=="float": return self.compile_node(node,port.getOutputString())
            if node.getCategory()=="convert" and self.input(node,"in").getType()=="float":
                return self.compile_port(self.input(node,"in"))
            if node.getCategory()=="constant":
                value=float4(self.numeric(self.input(node,"value")))[:3]
                if max(value)-min(value)<=1e-7: return self.constant(value[0])
        raise AssetError("Opacity must be an equal-channel constant or a scalar-to-color3 graph; colored transmission is unsupported")

    def compile_port(self,port):
        if port.getInterfaceName():
            parent=port.getParent().getParent()
            interface=parent.getInput(port.getInterfaceName())
            if interface is None: raise AssetError("Unresolved MaterialX interface input")
            return self.compile_port(interface)
        if port.getNodeGraphString():
            graph=self.document.getNodeGraph(port.getNodeGraphString())
            output=graph.getOutput(port.getOutputString() or "out") if graph else None
            if output is None: raise AssetError("Unresolved MaterialX nodegraph output")
            return self.compile_port(output)
        if port.getNodeName():
            node=port.getConnectedNode()
            if node is None: raise AssetError("Unresolved MaterialX node connection: "+port.getNodeName())
            return self.compile_node(node,port.getOutputString())
        if port.getValue() is not None: return self.constant(self.numeric(port))
        geom=port.getAttribute("defaultgeomprop")
        # Active defaults from a NodeDef are kept, including geometric inputs.
        if not geom:
            parent=port.getParent();nd=parent.getNodeDef() if isinstance(parent,mx.Node) else None
            default=nd.getActiveInput(port.getName()) if nd else None
            geom=default.getAttribute("defaultgeomprop") if default else ""
        if geom in ("UV0","st"): return self.emit(Op.Texcoord)
        if geom=="Pobject": return self.emit(Op.ObjectPosition)
        if geom=="Pworld": return self.emit(Op.Position)
        if geom=="Nworld": return self.emit(Op.Normal)
        if geom=="Tworld": return self.emit(Op.Tangent)
        raise AssetError("Unresolved geometric/default input: "+port.getNamePath())

    def compile_node(self,node,output=""):
        key=(node.getNamePath(),output)
        if key in self.registers: return self.registers[key]
        if key in self.visiting: raise AssetError("Cycle in MaterialX graph")
        self.visiting.add(key)
        nd=node.getNodeDef()
        if nd is None: raise AssetError("Unknown MaterialX NodeDef on "+node.getNamePath())
        category=node.getCategory();kind=node.getType()
        p=lambda name:self.compile_port(self.input(node,name))
        unary={"absval":Op.Abs,"invert":Op.OneMinus,"sin":Op.Sin,"cos":Op.Cos,"floor":Op.Floor,"ceil":Op.Ceil,"fract":Op.Fract}
        binary={"add":Op.Add,"subtract":Op.Subtract,"multiply":Op.Multiply,"divide":Op.Divide,"power":Op.Power,"min":Op.Min,"max":Op.Max,"crossproduct":Op.Cross}
        if category=="constant": reg=p("value")
        elif category in binary: reg=self.emit(binary[category],p("in1"),p("in2"))
        elif category in unary:
            if category=="invert":
                # MaterialX invert is amount - in, rather than always 1 - in.
                reg=self.emit(Op.Subtract,p("amount"),p("in"))
            else: reg=self.emit(unary[category],p("in"))
        elif category=="mix": reg=self.emit(Op.Mix,p("bg"),p("fg"),p("mix"))
        elif category=="clamp": reg=self.emit(Op.Clamp,p("in"),p("low"),p("high"))
        elif category in ("normalize","magnitude","dotproduct"):
            source=self.input(node,"in1" if category=="dotproduct" else "in")
            count=COMPONENTS.get(source.getType(),0)
            if count<2: raise AssetError("Invalid vector operation type")
            op={"normalize":Op.Normalize,"magnitude":Op.Length,"dotproduct":Op.Dot}[category]
            reg=self.emit(op,self.compile_port(source),p("in2") if category=="dotproduct" else 0,value=[count,0,0,0])
        elif category=="texcoord":
            if self.text(node,"index") not in ("0","0.0"): raise AssetError("Only texture coordinate set 0 is supported")
            reg=self.emit(Op.Texcoord)
        elif category in ("position","normal","tangent"):
            space=self.text(node,"space")
            if category=="position" and space in ("object","world"): reg=self.emit(Op.ObjectPosition if space=="object" else Op.Position)
            elif category in ("normal","tangent") and space=="world":
                if category=="tangent" and self.text(node,"index")!="0": raise AssetError("Only tangent set 0 is supported")
                reg=self.emit(Op.Normal if category=="normal" else Op.Tangent)
            else: raise AssetError("Unsupported %s space: %s" % (category,space))
        elif category=="geompropvalue" and kind=="vector2":
            name=self.text(node,"geomprop")
            if not name: raise AssetError("Empty texture coordinate geomprop")
            if not self.equivalent(self.numeric(self.input(node,"default")),[0,0]): raise AssetError("Nonzero UV geomprop defaults are unsupported")
            self.uv_names.add(name);reg=self.emit(Op.Texcoord)
        elif category.startswith("combine") and category[-1:] in ("2","3","4"):
            count=int(category[-1]);zero=self.constant(0)
            reg=self.emit(Op.Combine,p("in1"),p("in2"),p("in3") if count>=3 else zero)
            if count==4: reg=self.emit(Op.Add,reg,self.emit(Op.Multiply,p("in4"),self.constant([0,0,0,1])))
        elif category=="extract":
            index=int(self.text(node,"index"));count=COMPONENTS.get(self.input(node,"in").getType(),0)
            if not 0<=index<count: raise AssetError("Extract channel index is out of range")
            reg=self.emit(Op.Swizzle,p("in"),value=[index]*4)
        elif category in ("separate2","separate3","separate4"):
            outputs=[out.getName() for out in nd.getActiveOutputs()]
            if output not in outputs: raise AssetError("Missing MaterialX output selector on "+node.getNamePath())
            reg=self.emit(Op.Swizzle,p("in"),value=[outputs.index(output)]*4)
        elif category=="convert":
            source=self.input(node,"in");a=COMPONENTS.get(source.getType(),0);b=COMPONENTS.get(kind,0)
            if not a or not b or (b==1 and a!=1) or (b>a and a not in (1,3)):
                raise AssetError("Unsupported MaterialX conversion %s to %s" % (source.getType(),kind))
            reg=self.compile_port(source)
            if b>1:
                reg=self.emit(Op.Multiply,reg,self.constant([1]*min(a,b)+[0]*(4-min(a,b))) if a!=1 else self.constant([1]*b+[0]*(4-b)))
            if b>a and a!=1:
                reg=self.emit(Op.Add,reg,self.constant([0]*a+[1]*(b-a)+[0]*(4-b)))
        elif category=="dot": reg=p("in")
        elif category=="switch":
            # MaterialX switch uses a zero-based, clamped integer selector.
            # A constant selector also prunes the unused UV-transform branch.
            selector=int(float(self.constant_port(self.input(node,"which"))[0]))
            count=sum(i.getName().startswith("in") for i in nd.getActiveInputs())
            reg=p("in"+str(min(max(selector,0),count-1)+1))
        elif category=="rotate2d":
            vector=p("in");angle=self.emit(Op.Multiply,p("amount"),self.constant(math.pi/180))
            sine=self.emit(Op.Sin,angle);cosine=self.emit(Op.Cos,angle)
            x=self.emit(Op.Swizzle,vector,value=[0]*4);y=self.emit(Op.Swizzle,vector,value=[1]*4)
            # The reference uses row-vector multiplication, as rotate3d does.
            rx=self.emit(Op.Add,self.emit(Op.Multiply,x,cosine),self.emit(Op.Multiply,y,sine))
            ry=self.emit(Op.Subtract,self.emit(Op.Multiply,y,cosine),self.emit(Op.Multiply,x,sine))
            reg=self.emit(Op.Combine,rx,ry,self.constant(0))
        elif category=="rotate3d":
            vector=p("in");axis=self.emit(Op.Normalize,p("axis"),value=[3,0,0,0])
            angle=self.emit(Op.Multiply,p("amount"),self.constant(math.pi/180))
            sine=self.emit(Op.Sin,angle);cosine=self.emit(Op.Cos,angle)
            # MaterialX uses row-vector rotation; its reference implementation
            # takes cross(vector, axis), the reverse of column-vector Rodrigues.
            first=self.emit(Op.Multiply,vector,cosine)
            second=self.emit(Op.Multiply,self.emit(Op.Cross,vector,axis),sine)
            third=self.emit(Op.Multiply,self.emit(Op.Multiply,axis,self.emit(Op.Dot,axis,vector,value=[3,0,0,0])),self.emit(Op.OneMinus,cosine))
            reg=self.emit(Op.Add,self.emit(Op.Add,first,second),third)
        elif category=="image": reg=self.image(node,p)
        elif category=="normalmap":
            if self.input(node,"scale").getType()!="float": raise AssetError("Vector normalmap scales are unsupported")
            for name in ("normal","tangent","bitangent"):
                actual=node.getInput(name)
                if actual is not None and (self.connected(actual) or actual.getValueString()): raise AssetError("Custom normalmap basis is unsupported: "+name)
            reg=self.emit(Op.Normalmap,p("in"),p("scale"))
        elif category=="checkerboard":
            uv=self.emit(Op.Add,self.emit(Op.Multiply,p("texcoord"),p("uvtiling")),p("uvoffset"))
            reg=self.emit(Op.Checker,uv,p("color1"),p("color2"),[1,1,0,0])
        elif category=="noise3d" and kind in ("float","color3","vector3"):
            reg=self.emit(Op.Noise3D,p("position"),p("amplitude"),p("pivot"),[3 if kind!="float" else 0,0,0,0])
        elif category=="fractal3d" and kind in ("float","color3","vector3"):
            octaves=int(float(self.constant_port(self.input(node,"octaves"))[0]))
            if not 0<=octaves<=16: raise AssetError("Fractal noise supports zero to sixteen constant octaves")
            position=p("position");amplitude=p("amplitude");lacunarity=p("lacunarity");diminish=p("diminish")
            reg=self.constant(0)
            for octave in range(octaves):
                reg=self.emit(Op.Add,reg,self.emit(Op.Noise3D,position,amplitude,self.constant(0),[3 if kind!="float" else 0,0,0,0]))
                if octave+1<octaves:
                    position=self.emit(Op.Multiply,position,lacunarity);amplitude=self.emit(Op.Multiply,amplitude,diminish)
        elif category in ("rgbtohsv","hsvtorgb"):
            reg=self.emit(Op.RgbToHsv if category=="rgbtohsv" else Op.HsvToRgb,p("in"))
        elif category=="hextiledimage" and kind in ("color3","color4"):
            texture=self.texture_index(node)
            if texture is None: reg=p("default")
            else:
                uv=self.emit(Op.Multiply,p("texcoord"),p("tiling"))
                combine4=lambda x,y,z,w:self.emit(Op.Add,self.emit(Op.Combine,x,y,z),self.emit(Op.Multiply,w,self.constant([0,0,0,1])))
                first=combine4(p("rotation"),p("scale"),p("offset"),p("falloff"))
                rotation=p("rotationrange");scale=p("scalerange");offset=p("offsetrange")
                lane=lambda vector,index:self.emit(Op.Swizzle,vector,value=[index]*4)
                second=combine4(lane(rotation,0),lane(rotation,1),lane(scale,0),lane(scale,1))
                third=self.emit(Op.Combine,lane(offset,0),lane(offset,1),p("falloffcontrast"))
                fourth=p("lumacoeffs");one=self.constant(1)
                base=len(self.program)
                for source in (first,second,third,fourth): self.emit(Op.Multiply,source,one)
                reg=self.emit(Op.HexImage,uv,texture,base)
        elif category in ("remap","smoothstep"):
            a=self.emit(Op.Subtract,p("in"),p("inlow" if category=="remap" else "low"))
            b=self.emit(Op.Subtract,p("inhigh" if category=="remap" else "high"),p("inlow" if category=="remap" else "low"))
            t=self.emit(Op.Divide,a,b)
            if category=="remap": reg=self.emit(Op.Add,p("outlow"),self.emit(Op.Multiply,t,self.emit(Op.Subtract,p("outhigh"),p("outlow"))))
            else:
                t=self.emit(Op.Clamp,t,self.constant(0),self.constant(1))
                reg=self.emit(Op.Multiply,self.emit(Op.Multiply,t,t),self.emit(Op.Subtract,self.constant(3),self.emit(Op.Multiply,self.constant(2),t)))
        else: raise AssetError("Unsupported MaterialX pattern node: %s (%s)" % (node.getNamePath(),nd.getName()))
        self.visiting.remove(key);self.registers[key]=reg
        return reg

    def texture_index(self,node):
        file=self.input(node,"file");filename=file.getResolvedValueString()
        if self.connected(file): raise AssetError("Dynamic texture filenames are unsupported")
        if not filename: return None
        if "<UDIM>" in filename or "%" in filename: raise AssetError("UDIM and frame-sequence textures are unsupported")
        base=self.base_directory
        source=file.getActiveSourceUri()
        if source and Path(source).suffix.lower()==".mtlx":
            source_path=Path(source)
            if not source_path.is_absolute():
                source_path=source_path.resolve() if source_path.exists() else base/source_path
            base=source_path.parent
        path=filename if Path(filename).is_absolute() else base/filename
        space=file.getColorSpace() or node.getColorSpace() or self.document.getColorSpace() or "linear"
        return self.textures.add(path,space)

    def image(self,node,p):
        for name in ("layer","framerange"):
            if self.text(node,name): raise AssetError("Unsupported image input: "+name)
        if self.text(node,"frameoffset") not in ("0","0.0"): raise AssetError("Animated image offsets are unsupported")
        texture=self.texture_index(node)
        if texture is None: return p("default")
        modes={"periodic":0,"clamp":1,"mirror":2,"constant":3}
        try: u=modes[self.text(node,"uaddressmode")];v=modes[self.text(node,"vaddressmode")]
        except KeyError as error: raise AssetError("Unsupported texture address mode") from error
        if 3 in (u,v) and not self.equivalent(self.numeric(self.input(node,"default")),0):
            raise AssetError("Constant texture borders currently require a black default")
        filtering=self.text(node,"filtertype")
        if filtering not in ("linear","closest"): raise AssetError("Unsupported texture filter: "+filtering)
        return self.emit(Op.Image,p("texcoord"),texture,u|(v<<2)|(int(filtering=="closest")<<4))


def load_materialx(path,libraries=None):
    # Parse before attaching the data library: its typedef named "material"
    # otherwise makes the XML reader skip a legal local node of that name.
    path=Path(path);doc=mx.createDocument()
    try: mx.readFromXmlFile(doc,str(path),mx.FileSearchPath(str(path.parent)))
    except Exception as error: raise AssetError("Cannot read MaterialX: "+str(error)) from error
    doc.setDataLibrary(libraries or standard_library())
    surfaces=[node for node in doc.getNodes() if node.getType()=="surfaceshader"]
    if len(surfaces)>1 or (not surfaces and "volumeshader" not in material_terminals(doc)): raise AssetError("Standalone MaterialX must contain one surface or volume material")
    return doc,surfaces[0] if surfaces else None
