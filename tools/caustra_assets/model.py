"""Renderer-neutral data and the version 2 material instruction ABI."""
from dataclasses import dataclass, field
from enum import IntEnum


class AssetError(ValueError):
    pass


class MaterialModel(IntEnum):
    StandardSurface=0; OpenPbr=1; DisneyPrincipled=2; GltfPbr=3
    UsdPreviewSurface=4; SimpleHair=5


class Op(IntEnum):
    Constant=0; Texcoord=1; Position=2; ObjectPosition=3; Normal=4; Tangent=5
    Add=10; Subtract=11; Multiply=12; Divide=13; Mix=14; Clamp=15; Power=16
    Min=17; Max=18; Abs=19; OneMinus=20; Sin=21; Cos=22; Floor=23; Ceil=24
    Fract=25; Dot=26; Normalize=27; Length=28; Cross=29; Combine=30
    Swizzle=31; Image=32; Checker=33; Noise3D=34; Normalmap=35
    RgbToHsv=36; HsvToRgb=37; HexImage=38; Output=63


class Parameter(IntEnum):
    BaseColor=0; Metalness=1; Roughness=2; Emission=3; Normal=4
    BaseWeight=5; DiffuseRoughness=6; SpecularWeight=7; SpecularColor=8
    SpecularIOR=9; Anisotropy=10; Rotation=11; CoatWeight=12; CoatColor=13
    CoatRoughness=14; CoatIOR=15; CoatNormal=16; SheenWeight=17
    SheenColor=18; SheenRoughness=19; Opacity=20; Tangent=21; CoatTangent=22
    CoatAnisotropy=23; CoatDarkening=24
    TransmissionWeight=25; TransmissionColor=26; TransmissionDepth=27
    TransmissionScatter=28; TransmissionAnisotropy=29; TransmissionExtraRoughness=30
    SubsurfaceWeight=31; SubsurfaceColor=32; SubsurfaceRadius=33; SubsurfaceScale=34
    SubsurfaceAnisotropy=35; ThinWalled=36; ThinFilmWeight=37; ThinFilmThickness=38
    ThinFilmIOR=39; FuzzWeight=40; FuzzColor=41; FuzzRoughness=42; Dispersion=43
    VolumeDensity=44; VolumeAbsorption=45; VolumeScattering=46; VolumeAnisotropy=47
    VolumeEmission=48; VolumeBoundaryOnly=49; CoatRotation=50; CoatAffectColor=51
    CoatAffectRoughness=52


@dataclass
class Material:
    # Twenty-three float4s and a program uint4 match caustra::Material (384B).
    values: list = field(default_factory=lambda: [
        [0.8,0.8,0.8,1], [0,0,0,0], [0,0.5,0,0], [1,1.5,0,0],
        [1,1,1,0], [0,0.1,1.5,0], [1,1,1,0], [0,0.3,0,0],
        [1,1,1,0], [1,0,1,1],
        [0,0,0,0], [1,1,1,0], [0,0,0,0], [0,1,0,0], [1,1,1,0],
        [1,1,1,0], [1,0,1.5,0], [0,.5,0,0], [1,1,1,0],
        [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]])
    program: list = field(default_factory=lambda: [0,0,0,0])


@dataclass
class Texture:
    width: int
    height: int
    pixels: list


@dataclass
class Mesh:
    material: int
    vertices: list = field(default_factory=list)  # position, normal, UV, tangent: 16 floats
    indices: list = field(default_factory=list)


@dataclass
class Asset:
    materials: list = field(default_factory=list)
    meshes: list = field(default_factory=list)
    instances: list = field(default_factory=list)  # mesh index, 12-float affine transform
    instructions: list = field(default_factory=list)  # uint4, float4
    textures: list = field(default_factory=list)
    camera: list = field(default_factory=lambda: [0,1.3,4,0, 0,1,0,0, 0,1,0,0, 45])
    bounds_min: list = field(default_factory=lambda: [0,0,0,0])
    bounds_max: list = field(default_factory=lambda: [0,0,0,0])
    has_camera: bool = False


def float4(value, scalar=False):
    if isinstance(value, (float, int)):
        return [float(value)]*4
    result = [float(v) for v in value]
    if not 1 <= len(result) <= 4:
        raise AssetError("Expected one to four numeric components")
    return result + [0.]*(4-len(result))
