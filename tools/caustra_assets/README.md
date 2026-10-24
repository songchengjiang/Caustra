# Optional USD and MaterialX asset compiler

The renderer does not import USD, XML or image files. This Python tool resolves
scene data and validates MaterialX outside the rendering library, then writes a
self-contained `.caustra` package. `caustra::loadAsset` reads that package using
C++ and the renderer's CPU validation; it does not create a Vulkan device.

```powershell
python -m venv build/asset-tools-env
./build/asset-tools-env/Scripts/python.exe -m pip install -r requirements-assets.txt
./build/asset-tools-env/Scripts/python.exe tools/compile_asset.py scene.usdc scene.caustra
./build/asset-tools-env/Scripts/python.exe tests/asset_tests.py -v
./build/asset-tools-env/Scripts/python.exe tests/gallery_tests.py -v
```

Gallery regressions use the checked-in
[`materialx_official_inventory.json`](../../tests/assets/materialx_official_inventory.json)
as an independent coverage reference: 50 original MaterialX files with SHA256,
66 displayed material entries, and 62 unique source identities. When the pinned
source tree exists in `build/deps/materialx-official/MaterialX-1.39.5`, tests
verify every source hash and the generator's complete enumeration. Set
`CAUSTRA_MATERIALX_SOURCE` to use another source location. Without a source tree,
only those two optional checks skip; generated geometry, USD round trips, fixed
inventory consistency and offline fetch safety tests still run. No test downloads
assets or requires copying the upstream textures into the repository.

`--material surface.mtlx` explicitly applies one standalone MaterialX material to
all imported meshes. `--time N` selects a USD time-code snapshot; without it,
authored default values are used. Camera, instance transforms, Y/Z up axes and
meters-per-unit are converted to the renderer's Y-up meter coordinate system.
Face-varying and indexed UVs, normal interpolation and face material subsets are
preserved. Concave planar polygons are triangulated. Subdivision surfaces must
be tessellated before export. A camera's positive focus distance sets its orbit
target; otherwise the bounding-box center is projected onto its viewing ray.

Materials are selected from the USD `mtlx` render context, including NodeGraph
outputs and interface connections. The tool rebuilds and validates a native
MaterialX document and uses actual standard-library NodeDefs for defaults. It
does not substitute USD Preview Surface when a graph cannot be compiled.

Standard Surface, OpenPBR, Disney Principled, glTF PBR, USD Preview Surface and
the official SimpleHair example use separate native renderer material model
tags. The latter four follow their MaterialX 1.39.5 nodegraph definitions;
they are not converted to Standard Surface. A valid MaterialX
`ND_UsdPreviewSurface_surfaceshader` is supported explicitly; an unrelated USD
shader identifier is not used as a fallback for unsupported graphs.
Inputs cover base/diffuse roughness, metallic/specular color and IOR, anisotropy,
coat, normals/tangent graphs, transmission/tint/scattering, subsurface radii,
thin-walled transmission, thin film and OpenPBR fuzz (Standard Surface sheen).
Standard Surface coat rotation and coat-affect controls are retained. Thin-film
thickness is stored in nanometers (OpenPBR micrometers are converted); dispersion
is stored as inverse effective Abbe number. Transmission depth and subsurface
distance scales use stage units and are converted to world meters. Solid
interiors require full coverage. Constant arithmetic graphs are evaluated into
the material POD. Spatial transmission and subsurface weights, colors, radii
and scattering parameters retain their GPU programs and enable the
heterogeneous interior flag. The renderer samples those programs throughout
an object-space coefficient grid and transports paths through the resulting
piecewise constant field; it does not freeze the coefficients at the entry
surface. The current grid is 32 cubed cells per heterogeneous instance, so
features smaller than a cell require a finer future grid. Volume UVs use
object-space XY by default; the generated sphere gallery explicitly selects a
radial projection with the same UV convention as its surface geometry.
IOR remains spatially constant within each solid.
Native OpenPBR `specular_weight` modulates the actual dielectric IOR and must
therefore also be spatially constant when a solid interior is active.
Thin-walled transmission can retain spatial color/weight graphs. Unsupported
active inputs or nodes produce an error instead of a silent fallback.

SSS color represents observed diffuse reflectance; the renderer inverts the
OpenPBR van-de-Hulst calibration to obtain single-scattering albedo. The product
of radius and distance scale is the RGB mean free path (`1 / sigma_t`), and
transport uses interior random walks with the actual dielectric boundary.
The albedo calibration assumes an index-matched boundary, so exact finite-object
appearance can differ. Standard Surface applies its coat color-affect exponent
before that inversion; spatial coat controls therefore also enable the
heterogeneous grid when solid SSS is active.

MaterialX `volume` terminals support homogeneous `anisotropic_vdf` or
`absorption_vdf`, with optional `uniform_edf` emission. Absorption and scattering
are nonnegative inverse world meters; phase anisotropy is strictly between -1
and 1. Connected constant graphs are evaluated, while position/UV-dependent
coefficients are rejected. A volume without a surface is a boundary-only medium.
Spatially varying explicit volume-terminal fields and arbitrary closure
composition remain unsupported.

Scalar and vector3 MaterialX displacement terminals are evaluated on offline
microtriangle geometry before building the renderer scene. Scalars move along
the interpolated object-space normal; vectors use MaterialX's `(dPdu,dPdv,N)`
tangent/normal basis derived from UVs, then become object-space offsets. Vector
displacement requires nondegenerate UV derivatives. The instance transform then
converts those coordinates to world meters. Authored
UV/normal seams are preserved and normals/bounds are rebuilt after displacement.
Fields that disagree at a seam within the same material mesh fail with an
explicit crack diagnostic; separate material subsets are not automatically stitched. Set
`--displacement-subdivisions N` (0..7, default 3); each level quadruples triangle
count, bounded to one million triangles per displaced mesh. Noise and supported
image/arithmetic patterns are evaluated on the CPU using the same IR. Normalmap
nodes are shading perturbations and are rejected as geometry displacement inputs.

The current pattern subset includes constants, arithmetic, mix/clamp, common
scalar functions, vector math, extract/separate/combine, selected numeric
conversions, position/UV/world normal/world tangent, rotate2d/rotate3d, image,
normalmap, checkerboard, scalar/color/vector 3D noise, constant-octave fractal
noise, RGB/HSV conversion, and MaterialX hex-tiled images. Official tiledimage,
place2d and glTF image subgraphs are expanded with MaterialX's own graph
implementation; connected numeric graphs are retained. File prefixes and
included-material texture origins are resolved before loading. Programs are limited to 128 instructions per
material. Texture coordinates use MaterialX conventions; the renderer converts
V for top-row-first texture storage exactly once. Image wrap U/V is independent
(repeat, clamp, mirror or black), and closest/linear filtering is supported.
Nonblack constant image borders, cubic filtering, UDIM and animated textures
are explicitly unsupported.

Textures are decoded by Pillow to floating-point RGBA and embedded. Tagged
`srgb_texture`, `srgb` and `srgb_rec709` RGB are converted to linear; alpha remains
linear. `lin_rec709`, `linear` and `raw` are unchanged. Untagged image values are
treated as linear working-space data. Unknown color spaces fail. OpenEXR and
other image formats unsupported by the installed Pillow build fail at decoding.

USD RectLight and DiskLight become one-sided emissive meshes facing local -Z.
Radiance is `intensity * 2^exposure * color`, including optional USD blackbody
temperature. Normalization divides by the transformed emitting triangle area,
including stage units and nonuniform scale. A disk uses 96 triangles. Light
values map to this renderer's relative linear radiance, without an ACES exposure
pipeline. Other light types, textured area lights and shaping explicitly fail.

The version 2 little-endian binary layout is documented in `package.py`; readers
also expand version 1 materials with zero-weight defaults for the new lobes. It
contains POD scene data, material programs and texture pixels, with no external
paths. A compiled asset can be moved independently of its source USD, MaterialX
and image files. Packages larger than 2 GiB are rejected by the reader.
