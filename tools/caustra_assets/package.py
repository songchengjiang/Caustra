"""Portable .caustra package: little-endian PODs and embedded linear RGBA textures.

Header: magic[8], version u32, flags u32 (bit 0: camera), five u32 counts
(materials, meshes, instances, instructions, textures), camera[13] f32,
boundsMin[4] f32, boundsMax[4] f32. Version 1 header size is 120 bytes.
Payload: v2 materials (92 f32 + 4 u32; v1 used40 f32), instructions (4 u32 + 4 f32),
textures (width/height u32 + RGBA f32 pixels), meshes (material/vertex/index
counts u32 + 16 f32 per vertex + u32 indices), instances (mesh u32 + 12 f32).
There are no absolute paths, external resources, native pointers or padding.
"""
from pathlib import Path
import math
import struct
import numpy as np
from .model import Asset, AssetError, Material, Mesh, Texture

MAGIC = b"CAUSTRA\0"
VERSION = 2
MAX_BYTES = 2*1024*1024*1024
HEADER = struct.Struct("<8s7I21f")


def _floats(values):
    if not all(math.isfinite(float(v)) for v in values):
        raise AssetError("Package contains nonfinite numeric values")
    return struct.pack("<%df" % len(values), *values)


def _write_rows(out,values,columns):
    """Batch large geometry/images without per-vertex Python pack calls."""
    data=np.asarray(values,dtype="<f4")
    if data.ndim!=2 or data.shape[1]!=columns or not np.isfinite(data).all():
        raise AssetError("Invalid/nonfinite numeric rows in package")
    out.write(np.ascontiguousarray(data).tobytes())


def write_asset(asset, path):
    path=Path(path)
    if len(asset.camera)!=13 or len(asset.bounds_min)!=4 or len(asset.bounds_max)!=4:
        raise AssetError("Invalid camera or bounds layout")
    _floats(asset.camera+asset.bounds_min+asset.bounds_max)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(HEADER.pack(MAGIC,VERSION,int(asset.has_camera),len(asset.materials),len(asset.meshes),
                              len(asset.instances),len(asset.instructions),len(asset.textures),
                              *asset.camera,*asset.bounds_min,*asset.bounds_max))
        for material in asset.materials:
            if len(material.values)!=23 or any(len(row)!=4 for row in material.values): raise AssetError("Invalid material layout")
            out.write(_floats([v for row in material.values for v in row]))
            out.write(struct.pack("<4I",*material.program))
        for code,value in asset.instructions:
            if len(code)!=4 or len(value)!=4: raise AssetError("Invalid material instruction layout")
            out.write(struct.pack("<4I",*code));out.write(_floats(value))
        for texture in asset.textures:
            if not texture.width or not texture.height or texture.width*texture.height!=len(texture.pixels) or any(len(p)!=4 for p in texture.pixels):
                raise AssetError("Texture dimensions do not match its pixels")
            out.write(struct.pack("<2I",texture.width,texture.height))
            _write_rows(out,texture.pixels,4)
        for mesh in asset.meshes:
            out.write(struct.pack("<3I",mesh.material,len(mesh.vertices),len(mesh.indices)))
            _write_rows(out,mesh.vertices,16)
            out.write(struct.pack("<%dI" % len(mesh.indices),*mesh.indices))
        for mesh,transform in asset.instances:
            if len(transform)!=12: raise AssetError("Instance transform must contain 12 affine values")
            out.write(struct.pack("<I",mesh));out.write(_floats(transform))


def read_asset(path):
    data=Path(path).read_bytes()
    if len(data)>MAX_BYTES: raise AssetError("Asset exceeds the 2 GiB package limit")
    offset=0
    def take(fmt):
        nonlocal offset
        s=struct.Struct("<"+fmt)
        if s.size>len(data)-offset: raise AssetError("Truncated Caustra asset")
        values=s.unpack_from(data,offset);offset+=s.size
        if any(isinstance(v,float) and not math.isfinite(v) for v in values): raise AssetError("Nonfinite value in Caustra asset")
        return values
    head=take("8s7I21f")
    if head[0]!=MAGIC or head[1] not in (1,VERSION) or head[2]&~1:
        raise AssetError("Unsupported Caustra package header")
    nm,nmesh,ni,nops,nt=head[3:8]
    material_floats=40 if head[1]==1 else 92
    if nm*(material_floats*4+16)+nmesh*12+ni*52+nops*32+nt*8>len(data)-offset:
        raise AssetError("Impossible Caustra package counts")
    asset=Asset(camera=list(head[8:21]),bounds_min=list(head[21:25]),bounds_max=list(head[25:29]),has_camera=bool(head[2]&1))
    for _ in range(nm):
        values=take("%df" % material_floats);material=Material()
        material.values[:material_floats//4]=[list(values[i:i+4]) for i in range(0,material_floats,4)]
        material.program=list(take("4I"));asset.materials.append(material)
        first,count,model,_=asset.materials[-1].program
        if first+count>nops or count>128 or model>5: raise AssetError("Invalid material program range/model")
    for _ in range(nops): asset.instructions.append((list(take("4I")),list(take("4f"))))
    for _ in range(nt):
        width,height=take("2I")
        if not width or not height or width*height*16>len(data)-offset: raise AssetError("Invalid texture dimensions")
        values=take("%df" % (width*height*4))
        asset.textures.append(Texture(width,height,[list(values[i:i+4]) for i in range(0,len(values),4)]))
    for _ in range(nmesh):
        material,nv,nx=take("3I")
        if nv*64+nx*4>len(data)-offset: raise AssetError("Invalid mesh counts")
        mesh=Mesh(material,[list(take("16f")) for _ in range(nv)],list(take("%dI" % nx)))
        if material>=nm or nx%3 or any(i>=nv for i in mesh.indices): raise AssetError("Invalid mesh indices")
        asset.meshes.append(mesh)
    for _ in range(ni):
        mesh,=take("I")
        if mesh>=nmesh: raise AssetError("Invalid instance mesh")
        asset.instances.append((mesh,list(take("12f"))))
    if offset!=len(data): raise AssetError("Trailing bytes in Caustra package")
    return asset
