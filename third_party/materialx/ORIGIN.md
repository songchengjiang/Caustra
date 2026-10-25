# MaterialX noise provenance

## Generated surface backend

`shaders/generated/` is produced by `tools/generate_materialx_surfaces.py` using
the official v1.39.5 `SlangShaderGenerator`. The generator clones the upstream
six model graphs with BSDF/EDF terminals and records source and artifact SHA-256
in `materialx_surfaces.json`. This replaces the ordinary surface reflection and
emission evaluators; the core still owns parameter IR, sampling/PDF and physical
dielectric/SSS/volume transport. The generated source is covered by the upstream
Apache-2.0 license reproduced in this directory.

Compute adapters remove raster entry points/resources, namespace symbols and
prefix macros; use analytic directional albedo; disable the screen-curvature
SSS helper (actual SSS uses the core random walk); floor GGX roughness at 0.02;
map folded emission and OpenPBR film units; and bypass a second USD normalmap
conversion because the parameter IR already supplies world normals. Original
MaterialX source files are not modified. The full adaptation list is in the
generated manifest and `docs/materialx_backend.md`.

Standard Surface's coat rotation input is promoted to a per-hit coat tangent
to preserve the existing renderer-neutral `CoatTangent` IR output. Without that
explicit output the parameter interpreter follows the authored base tangent,
as the official graph does. The rotation and normalization math is unchanged.
Disney/Hair, whose model graphs have no EDF, use the official generated
`uniform_edf` primitive for the existing generic `Material.emission` field.

Generated Simple Hair retains the upstream TRRT+ `1 / 2 * pi` expression.
The corrected normalization described for the older handwritten fiber evaluator
below does not apply to this generated backend.

Upstream: [AcademySoftwareFoundation/MaterialX v1.39.5](https://github.com/AcademySoftwareFoundation/MaterialX/tree/v1.39.5), commit `7b64921ef1d42f2d57871e9d2c43dc11f041f26b`.

- `mx_noise.glsl` is an unmodified copy of `libraries/stdlib/genglsl/lib/mx_noise.glsl` (upstream Git blob `ac3de6b87aa0c27e7cf7f62ff7881613f86187b5`).
- `mx_perlin.slang` is a modified subset of that file: scalar Perlin noise and its dependencies; GLSL vector type spellings converted to Slang. Hashing, gradients, interpolation, and normalization coefficients are unchanged. The original 2D scalar overload and shared helper overloads are retained to keep the extracted dependency block auditable.
- `LICENSE` reproduces the upstream MaterialX Apache-2.0 license. The noise file's original Sony Pictures Imageworks / Open Shading Language BSD notice is retained in both source copies and must also accompany binary redistribution.

Caustra's MaterialOp::Noise3D evaluates the MaterialX noise3d operation: `perlin(position) * amplitude + pivot`. Instruction value.x = 3 selects native vector/color 3D Perlin; zero selects the scalar operation. The vector overload was extracted from the same unmodified source and only its type spellings were converted to Slang.


## Official example native surface models

`shaders/official_models.slang` implements these upstream v1.39.5 graphs directly
as separate model tags, without translating them to Standard Surface:

- `libraries/bxdf/disney_principled.mtlx`: its Burley diffuse, subsurface/sheen,
  smooth dielectric transmission, weighted generalized-Schlick dielectric,
  metallic mixture and weighted clearcoat in the original node order.
- `libraries/bxdf/gltf_pbr.mtlx`: metallic/roughness and anisotropy, IOR/specular
  extensions, generalized-Schlick thin film, sheen, dielectric coat and alpha modes.
- `libraries/bxdf/usd_preview_surface.mtlx`: distinct specular/metalness workflows,
  artistic-IOR conductor mixture, coat, transparent opacity and presence opacity.
- `resources/Materials/Examples/SimpleHair/simple_hair_default.mtlx`: its native
  melanin/color absorption mix, roughness mapping and Chiang R/TT/TRT/TRRT+ fiber model.

Shared GGX directional albedo, compensation and optical primitives reuse Caustra's
already attributed MaterialX ports in `third_party/aurora`. Burley diffuse, current
Imageworks sheen fit and generalized-Schlick layering follow the corresponding
`libraries/pbrlib/genglsl/mx_burley_diffuse_bsdf.glsl`,
`mx_sheen_bsdf.glsl`, `mx_generalized_schlick_bsdf.glsl`,
`mx_dielectric_bsdf.glsl` and `lib/mx_microfacet_diffuse.glsl` implementations.
The path tracer supplies its own normalized sampling distributions and physical
geometry/refraction integration; a finite roughness floor is retained for numerical
stability. Homogeneous SSS uses the renderer's random walk rather than a raster
screen-space approximation. DLSS guides compress the multilobe closures.

`mx_hair_math.slang` ports the helper functions from
`libraries/pbrlib/genglsl/mx_chiang_hair_bsdf.glsl`: GLSL vector names, helper
prefixes and the pi symbol are changed; mathematical formulas are retained.
These helpers now support the full-sphere mixture of normalized truncated
logistic sampling proposals in `official_models.slang`; their exact PDF is used
in MIS. Before ShaderGen integration, the handwritten fiber evaluator also used
them for scattering and corrected the upstream final closure's `1 / 2 * pi`
expression to the helper's `1 / (2 * pi)` density. That evaluator has been removed.
The new scattering response is generated from the upstream Chiang function,
including its original TRRT+ expression, and applies no additional surface cosine
or raster indirect/environment contribution.

All upstream sources in this section are covered by the MaterialX Apache-2.0
license reproduced in `LICENSE`.

## Official example patterns

- `mx_hsv.glsl` and `mx_hextile.glsl` are unmodified copies of `libraries/stdlib/genglsl/lib` from the same v1.39.5 source. `mx_hsv.slang` changes vector type names only and retains the Open Shading Language notice.
- `shaders/material_patterns.slang` adapts `mx_hextile.glsl` and `mx_hextiledimage.glsl` to Slang: scalar vector formulas replace GLSL matrix conventions, explicit conservative ray-cone footprints replace screen derivatives, and all-black luminance weights fall back to geometric weights to avoid 0/0. Tiling, randomized rotation/scale/offset, luminance blending, falloff and independent alpha blending are retained. The CPU pattern evaluator follows the same operation at LOD zero.
- Official gallery assets are fetched unchanged from the v1.39.5 tag, archive SHA256 `830b33e96069a69440df175d8ab0951b292893030d825235fbb3b3f846604dda`. `tools/build_materialx_gallery.py` retains their source documents, asset licenses, and original-resolution resources in the generated editable gallery. Geometry and gallery composition are generated by Caustra.
