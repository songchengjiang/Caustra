"""Strict, offline PBRT scene reader used by the San Miguel converter.

Transforms use column vectors. PBRT's textual matrices are column-major; USD's
Gf matrices use row vectors and are transposed at the authoring boundary.
This is deliberately outside the rendering library and is not a general PBRT
renderer: unsupported active commands, shapes and animated transforms fail.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
import math
import re
import numpy as np

from .model import AssetError


@dataclass(frozen=True, slots=True)
class Parameter:
    type: str
    name: str
    values: tuple

    @property
    def value(self):
        return self.values[0] if len(self.values) == 1 else self.values


@dataclass(frozen=True, slots=True)
class Command:
    name: str
    args: tuple
    params: dict
    file: Path
    line: int

    def fail(self, message):
        raise AssetError(f"{self.file}:{self.line}: {message}")


_ARITY = {
    "Include": 1, "Import": 1, "WorldBegin": 0, "WorldEnd": 0,
    "AttributeBegin": 0, "AttributeEnd": 0, "TransformBegin": 0,
    "TransformEnd": 0, "ObjectBegin": 1, "ObjectEnd": 0,
    "ObjectInstance": 1, "Identity": 0, "Translate": 3, "Scale": 3,
    "Rotate": 4, "LookAt": 9, "Transform": 1, "ConcatTransform": 1,
    "CoordinateSystem": 1, "CoordSysTransform": 1, "ActiveTransform": 1,
    "TransformTimes": 2, "ReverseOrientation": 0, "Camera": 1,
    "Film": 1, "Sampler": 1, "Integrator": 1, "PixelFilter": 1,
    "Accelerator": 1, "LightSource": 1, "AreaLightSource": 1,
    "Texture": 3, "Shape": 1, "Material": 1, "MakeNamedMaterial": 1,
    "NamedMaterial": 1, "MakeNamedMedium": 1, "MediumInterface": 2,
    "Option": 0,
}
_TOKEN = re.compile(r'\s+|\#[^\n]*|"(?:\\.|[^"\\])*"|\[|\]|[^\s\[\]"#]+')
_NUMBER = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')


def _tokens(text, path):
    line = 1
    for match in _TOKEN.finditer(text):
        token = match.group()
        token_line = line
        line += token.count("\n")
        if token.isspace() or token.startswith("#"):
            continue
        if token.startswith('"'):
            # PBRT string escapes, including Windows paths, are not Python or
            # JSON escapes. Only escaped quote/backslash/control codes matter.
            value = re.sub(r'\\(["\\nrt])', lambda m: {"n":"\n","r":"\r","t":"\t"}.get(m[1],m[1]), token[1:-1])
            yield value, True, token_line
        elif _NUMBER.fullmatch(token):
            yield float(token), False, token_line
        else:
            yield token, False, token_line


def read_commands(path):
    """Read one file without expanding Includes; values are immutable tuples."""
    path = Path(path).resolve()
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise AssetError(f"Cannot read PBRT file {path}: {error}") from error
    iterator = iter(_tokens(text, path))
    pending = next(iterator, None)

    def take():
        nonlocal pending
        result = pending
        if result is None:
            raise AssetError(f"Unexpected end of PBRT file {path}")
        pending = next(iterator, None)
        return result

    def value():
        token, quoted, line = take()
        if token == "[" and not quoted:
            result = []
            while pending and pending[:2] != ("]", False):
                v, q, unused = take()
                if not q and isinstance(v, str):
                    raise AssetError(f"{path}:{line}: Unexpected token {v} in array")
                result.append(v)
            if not pending:
                raise AssetError(f"{path}:{line}: Unterminated array")
            take()
            return tuple(result)
        return token

    while pending:
        name, quoted, line = take()
        if quoted or name not in _ARITY:
            raise AssetError(f"{path}:{line}: Unsupported PBRT command {name!r}")
        args = tuple(value() for _ in range(_ARITY[name]))
        params = {}
        while pending and pending[1]:
            declaration, unused, parameter_line = take()
            parts = declaration.split(maxsplit=1)
            if len(parts) != 2:
                raise AssetError(f"{path}:{parameter_line}: Invalid parameter declaration {declaration!r}")
            type_name, parameter_name = parts
            data = value()
            if parameter_name in params:
                raise AssetError(f"{path}:{parameter_line}: Duplicate parameter {parameter_name}")
            params[parameter_name] = Parameter(type_name, parameter_name, data if isinstance(data,tuple) else (data,))
        yield Command(name, args, params, path, line)


def iter_commands(path, expand_includes=True, asset_root=None, _stack=()):
    path = Path(path).resolve()
    root = Path(asset_root).resolve() if asset_root else path.parent
    if path in _stack:
        raise AssetError("Cyclic PBRT Include: " + " -> ".join(map(str, _stack + (path,))))
    for command in read_commands(path):
        if command.name == "Include" and expand_includes:
            included = (root / str(command.args[0])).resolve()
            if not included.is_relative_to(root):
                command.fail("Include escapes the source asset directory")
            yield from iter_commands(included, True, root, _stack + (path,))
        else:
            yield command


@dataclass(slots=True)
class Shape:
    command: Command
    transform: np.ndarray
    material: str
    reverse: bool
    area_light: Command | None


@dataclass(slots=True)
class Instance:
    name: str
    transform: np.ndarray


@dataclass(slots=True)
class Light:
    command: Command
    transform: np.ndarray


@dataclass(slots=True)
class Camera:
    command: Command
    world_from_camera: np.ndarray


@dataclass
class PbrtScene:
    asset_root: Path
    source: Path
    materials: dict = field(default_factory=dict)
    textures: dict = field(default_factory=dict)
    shapes: list = field(default_factory=list)
    objects: dict = field(default_factory=dict)
    instances: list = field(default_factory=list)
    lights: list = field(default_factory=list)
    camera: Camera | None = None
    film: Command | None = None
    options: list = field(default_factory=list)
    source_files: set = field(default_factory=set)
    command_counts: dict = field(default_factory=dict)
    renamed_definitions: list = field(default_factory=list)

    @staticmethod
    def _definition(command, texture=False):
        result = {"params": {n:p.values for n,p in command.params.items()},
                  "param_types": {n:p.type for n,p in command.params.items()},
                  "source": str(command.file)}
        if texture:
            result.update(type=command.args[2], value_type=command.args[1])
        else:
            if "type" not in command.params:
                command.fail("Named material has no type")
            result["type"] = str(command.params["type"].value)
            result["params"].pop("type")
            result["param_types"].pop("type")
        return result

    def material_definitions(self):
        return {name:self._definition(c) for name,c in self.materials.items()}

    def texture_definitions(self):
        return {name:self._definition(c,True) for name,c in self.textures.items()}


def _normalized(value, command):
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm <= 1e-14:
        command.fail("Degenerate transform direction")
    return value / norm


def command_transform(command):
    name, args = command.name, command.args
    result = np.eye(4)
    if name in ("Transform", "ConcatTransform"):
        if not isinstance(args[0], tuple) or len(args[0]) != 16:
            command.fail("Transform requires 16 column-major numbers")
        result = np.asarray(args[0], dtype=np.float64).reshape(4,4).T.copy()
    elif name == "Translate":
        result[:3,3] = args
    elif name == "Scale":
        result[:3,:3] = np.diag(args)
    elif name == "Rotate":
        x,y,z = _normalized(args[1:], command)
        angle = math.radians(args[0]);c=math.cos(angle);s=math.sin(angle);t=1-c
        result[:3,:3] = ((t*x*x+c,t*x*y-s*z,t*x*z+s*y),
                          (t*x*y+s*z,t*y*y+c,t*y*z-s*x),
                          (t*x*z-s*y,t*y*z+s*x,t*z*z+c))
    elif name == "LookAt":
        eye=np.asarray(args[:3]);direction=_normalized(np.asarray(args[3:6])-eye, command)
        right=_normalized(np.cross(_normalized(args[6:9],command),direction),command)
        camera=np.eye(4);camera[:3,:3]=np.column_stack((right,np.cross(direction,right),direction));camera[:3,3]=eye
        result=np.linalg.inv(camera)
    if not np.all(np.isfinite(result)) or not np.allclose(result[3], (0,0,0,1), atol=1e-10):
        command.fail("Only finite affine transforms are supported")
    if abs(np.linalg.det(result[:3,:3])) < 1e-14:
        command.fail("Singular transform")
    return result


def load_scene(path):
    """Interpret the source scene, preserving object definitions and instances."""
    path=Path(path).resolve();scene=PbrtScene(path.parent,path)
    transform=np.eye(4);material=None;reverse=False;area_light=None
    stack=[];transform_stack=[];object_name=None;coordinate_systems={};world=False
    texture_aliases={};material_aliases={}

    def canonical_definition(c, table, aliases):
        original=str(c.args[0]);params={}
        for name,p in c.params.items():
            if p.type=="texture":
                if len(p.values)!=1 or p.value not in texture_aliases:
                    c.fail("Undefined texture reference "+str(p.value))
                p=replace(p,values=(texture_aliases[p.value],))
            params[name]=p
        c=replace(c,params=params)
        if original in table and table[original].args[1:]==c.args[1:] and table[original].params==params:
            aliases[original]=original
            return
        key=original
        if key in table:
            number=2
            while f"{original}__definition_{number}" in table: number+=1
            key=f"{original}__definition_{number}"
            scene.renamed_definitions.append({"kind":c.name,"original":original,"resolved":key,"source":str(c.file.relative_to(scene.asset_root)),"line":c.line})
        table[key]=replace(c,args=(key,)+c.args[1:]);aliases[original]=key
    for c in iter_commands(path):
        scene.source_files.add(c.file)
        scene.command_counts[c.name]=scene.command_counts.get(c.name,0)+1
        if c.name == "WorldBegin":
            if world or stack: c.fail("Unbalanced or repeated WorldBegin")
            transform=np.eye(4);world=True;coordinate_systems["world"]=transform.copy()
        elif c.name == "WorldEnd":
            if stack or object_name: c.fail("Unbalanced WorldEnd")
        elif c.name == "AttributeBegin": stack.append((transform,material,reverse,area_light));transform=transform.copy()
        elif c.name == "AttributeEnd":
            if not stack: c.fail("Unmatched AttributeEnd")
            transform,material,reverse,area_light=stack.pop()
        elif c.name == "TransformBegin": transform_stack.append(transform);transform=transform.copy()
        elif c.name == "TransformEnd":
            if not transform_stack: c.fail("Unmatched TransformEnd")
            transform=transform_stack.pop()
        elif c.name == "Identity": transform=np.eye(4)
        elif c.name in ("Transform","ConcatTransform","Translate","Scale","Rotate","LookAt"):
            matrix=command_transform(c)
            transform=matrix if c.name=="Transform" else transform @ matrix
        elif c.name == "CoordinateSystem": coordinate_systems[str(c.args[0])]=transform.copy()
        elif c.name == "CoordSysTransform":
            if c.args[0] not in coordinate_systems: c.fail("Unknown coordinate system")
            transform=coordinate_systems[c.args[0]].copy()
        elif c.name == "ActiveTransform":
            if c.args != ("All",): c.fail("Animated transforms are unsupported")
        elif c.name == "ReverseOrientation": reverse=not reverse
        elif c.name == "Camera":
            if world or scene.camera: c.fail("Camera must be unique and before WorldBegin")
            scene.camera=Camera(c,np.linalg.inv(transform));coordinate_systems["camera"]=np.linalg.inv(transform)
        elif c.name == "Film":
            if scene.film: c.fail("Multiple Film declarations")
            scene.film=c
        elif c.name in ("Sampler","Integrator","PixelFilter","Accelerator"):
            scene.options.append(c)
        elif c.name == "Texture":
            canonical_definition(c,scene.textures,texture_aliases)
        elif c.name == "MakeNamedMaterial":
            canonical_definition(c,scene.materials,material_aliases)
        elif c.name == "NamedMaterial":
            if c.args[0] not in material_aliases: c.fail("Undefined named material "+str(c.args[0]))
            material=material_aliases[c.args[0]]
        elif c.name == "ObjectBegin":
            if object_name is not None: c.fail("Nested ObjectBegin is unsupported")
            object_name=str(c.args[0])
            if object_name in scene.objects: c.fail("Duplicate ObjectBegin "+object_name)
            scene.objects[object_name]=[]
        elif c.name == "ObjectEnd":
            if object_name is None: c.fail("Unmatched ObjectEnd")
            object_name=None
        elif c.name == "ObjectInstance":
            if object_name is not None: c.fail("Nested ObjectInstance is unsupported")
            scene.instances.append(Instance(str(c.args[0]),transform.copy()))
        elif c.name == "LightSource": scene.lights.append(Light(c,transform.copy()))
        elif c.name == "AreaLightSource": area_light=c if c.args[0] else None
        elif c.name == "Shape":
            if c.args != ("plymesh",): c.fail("Unsupported shape "+str(c.args))
            unknown=set(c.params)-{"filename","alpha"}
            if unknown: c.fail("Unsupported shape parameters: "+", ".join(sorted(unknown)))
            if "filename" not in c.params: c.fail("plymesh has no filename")
            if material is None: c.fail("Shape has no explicit named material")
            if "alpha" in c.params and c.params["alpha"].type=="texture":
                alpha=c.params["alpha"]
                if alpha.value not in texture_aliases: c.fail("Undefined alpha texture "+str(alpha.value))
                params=dict(c.params);params["alpha"]=replace(alpha,values=(texture_aliases[alpha.value],));c=replace(c,params=params)
            shape=Shape(c,transform.copy(),material,reverse,area_light)
            (scene.shapes if object_name is None else scene.objects[object_name]).append(shape)
        else:
            c.fail("Unsupported active command "+c.name)
    if not world or stack or transform_stack or object_name is not None:
        raise AssetError("Unbalanced PBRT world/attribute/object scope")
    for shape in scene.shapes+[s for group in scene.objects.values() for s in group]:
        if shape.material not in scene.materials: shape.command.fail("Undefined named material "+shape.material)
        if shape.area_light: shape.command.fail("Active emissive surface material conversion is not implemented")
        if "alpha" in shape.command.params:
            alpha=shape.command.params["alpha"]
            derived=shape.material+"__alpha_"+str(alpha.values[0])
            if derived not in scene.materials:
                original=scene.materials[shape.material];params=dict(original.params);params["alpha"]=alpha
                scene.materials[derived]=replace(original,args=(derived,),params=params)
            shape.material=derived
    for instance in scene.instances:
        if instance.name not in scene.objects: raise AssetError("Undefined ObjectInstance "+instance.name)
    return scene


_PLY_TYPES={"float":"f4","float32":"f4","double":"f8","float64":"f8",
            "int":"i4","int32":"i4","uint":"u4","uint32":"u4",
            "uchar":"u1","uint8":"u1","char":"i1","int8":"i1",
            "short":"i2","int16":"i2","ushort":"u2","uint16":"u2"}


def read_ply(path):
    """Read binary little-endian triangle PLY without expanding indexed vertices."""
    path=Path(path)
    with path.open("rb") as stream:
        if stream.readline().strip()!=b"ply": raise AssetError("Not a PLY file: "+str(path))
        vertices=None;faces=None;vertex_properties=[];face_property=None;element=None;format_name=None
        while True:
            raw=stream.readline()
            if not raw: raise AssetError("Truncated PLY header: "+str(path))
            words=raw.decode("ascii").split()
            if not words: continue
            if words[0]=="end_header": break
            if words[0] in ("comment","obj_info"): continue
            if words[0]=="format": format_name=words[1];continue
            if words[0]=="element":
                element=words[1]
                if element=="vertex": vertices=int(words[2])
                elif element=="face": faces=int(words[2])
                else: raise AssetError(f"Unsupported PLY element {element}: {path}")
            elif words[0]=="property":
                if element=="vertex" and len(words)==3 and words[1] in _PLY_TYPES:
                    vertex_properties.append((words[2],"<"+_PLY_TYPES[words[1]]))
                elif element=="face" and len(words)==5 and words[1]=="list" and words[4] in ("vertex_indices","vertex_index"):
                    if face_property: raise AssetError("Multiple PLY face properties: "+str(path))
                    face_property=("<"+_PLY_TYPES[words[2]],"<"+_PLY_TYPES[words[3]])
                else: raise AssetError("Unsupported PLY property: "+raw.decode().strip())
            else: raise AssetError("Unsupported PLY header declaration: "+raw.decode().strip())
        if format_name!="binary_little_endian" or vertices is None or faces is None or face_property is None:
            raise AssetError("Expected binary_little_endian triangle PLY with vertices/faces: "+str(path))
        names={n for n,t in vertex_properties}
        if not {"x","y","z"}<=names or names-{"x","y","z","nx","ny","nz","u","v","s","t"}:
            raise AssetError("Unsupported PLY vertex layout: "+str(path))
        if bool(names&{"nx","ny","nz"}) and not {"nx","ny","nz"}<=names:
            raise AssetError("Incomplete PLY normals: "+str(path))
        uv_names=("u","v") if {"u","v"}<=names else (("s","t") if {"s","t"}<=names else None)
        if names&{"u","v","s","t"} and uv_names is None: raise AssetError("Incomplete PLY UVs: "+str(path))
        data=np.fromfile(stream,dtype=np.dtype(vertex_properties),count=vertices)
        if len(data)!=vertices: raise AssetError("Truncated PLY vertices: "+str(path))
        face_data=np.fromfile(stream,dtype=np.dtype([("count",face_property[0]),("indices",face_property[1],(3,))]),count=faces)
        if len(face_data)!=faces or np.any(face_data["count"]!=3):
            raise AssetError("PLY must contain triangles only: "+str(path))
        if stream.read(1): raise AssetError("Unexpected trailing PLY data: "+str(path))
        indices=np.ascontiguousarray(face_data["indices"],dtype=np.int32)
        if indices.size and (indices.min()<0 or indices.max()>=vertices): raise AssetError("PLY index out of bounds: "+str(path))
        def columns(keys): return np.ascontiguousarray(np.column_stack([data[k] for k in keys]),dtype=np.float32)
        points=columns(("x","y","z"));normals=columns(("nx","ny","nz")) if {"nx","ny","nz"}<=names else None
        uv=columns(uv_names) if uv_names else None
        if not all(np.all(np.isfinite(a)) for a in (points,normals,uv) if a is not None): raise AssetError("Nonfinite PLY attributes: "+str(path))
        return points,indices,normals,uv
