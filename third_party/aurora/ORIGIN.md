# Autodesk Aurora MaterialX shading math

`standard_surface_math.slang` is adapted from the local source file
`C:/Dev/Formark/Aurora/Libraries/Aurora/Source/Shaders/StandardSurfaceBSDF.slang`,
copyright 2026 Autodesk, Inc., distributed under Apache-2.0 (see `LICENSE`).
Upstream project: <https://github.com/Autodesk/Aurora>.

Source SHA-256:

- `StandardSurfaceBSDF.slang`:
  `D1328C57C130D8D3D9B61B55466EBE42302DD7563D61B49D14E5BEE2796C0729`
- `BSDFCommon.slang`:
  `D2C652CDAE98D77E2C23A091CC5A65DB0C48189618999CA92CFAE55E004E6127`

Aurora identifies these functions as an HLSL port of the MaterialX GLSL PBR
implementations. This import retains the independent Oren-Nayar, GGX,
directional-albedo/energy-compensation, artistic conductor IOR and Imageworks
sheen evaluators, including its 16x16 sheen directional-albedo lookup table.
The original Autodesk copyright and license notice are preserved.

Changes in the imported file:

- Removed the dependency on Aurora's `BSDFCommon.slang`, globals, application
  material types, shading types, and renderer-specific sampling policy.
- Extracted only the independent mathematical functions before the subsurface
  approximation. Replaced the `BSDF` alias with `float3` and `BLACK` with a
  zero vector; renamed the include guard, supplied `M_FLOAT_EPS`, and used
  separately named constants for pi to avoid depending on renderer globals.

`shaders/surface_bsdf.slang` is Caustra's adapter. Its visible-normal GGX
proposal is adapted from the Heitz 2018 implementation in Aurora's
`BSDFCommon.slang` (copyright 2025 Autodesk, Inc., Apache-2.0). It evaluates
and samples a deterministic **full mixture** instead of Aurora's stochastic
selected-lobe interface. It adds finite roughness floors, independent coat
normal frames, MaterialX coat-color filtering with white coat reflection,
normal-map hemisphere safeguards, and renderer-neutral denoiser features.

The Standard Surface branch supports diffuse weight/color
and roughness, metalness, dielectric specular color/IOR/roughness/anisotropy and
rotation, sheen, and colored clear coat with independent normal/IOR/roughness.
The closure now also supports rough dielectric transmission, thin-walled
transmission, spectral thin-film reflection and coat rotation/affect-color/
affect-roughness. Subsurface and volume transport are implemented separately in
the path integrator, not by treating them as a local reflection BRDF. This is
not an arbitrary MaterialX graph evaluator; unsupported graph nodes and
displacement must still be rejected by the external compiler.

The GGX masking and energy compensation remain Aurora's MaterialX-port
approximations. A single GGX lobe is used to propose base specular/metallic
reflection, a second for coat, and cosine sampling for diffuse plus sheen;
returned PDFs are the complete proposal mixture, and weights use the full
evaluated BSDF divided by that PDF. Some GGX directions below the geometric
hemisphere are absorption events; they are not resampled.

## Native OpenPBR primitives

`open_pbr_math.slang` is a modified extraction/port of the Apache-2.0 MaterialX
v1.39.5 reference PBR library, commit
`7b64921ef1d42f2d57871e9d2c43dc11f041f26b`. Upstream:
<https://github.com/AcademySoftwareFoundation/MaterialX/tree/v1.39.5>.
The local sources are under
`build/asset-tools-env/Lib/site-packages/MaterialX/libraries/`:

| Source | SHA-256 |
| --- | --- |
| `bxdf/open_pbr_surface.mtlx` | `2904F680A939215409FEA90E3586DB1E2087C5E29000611C1BF4BE518FC65F83` |
| `pbrlib/genglsl/lib/mx_microfacet_diffuse.glsl` | `B6105919EE8EE8B61D00CAFA78447D3D5CFED4CE8001E6E0D08CB1570D46A684` |
| `pbrlib/genglsl/lib/mx_microfacet_specular.glsl` | `EE390C557E1A0028169664ADB0A73BF7F33C4F599DDE0EB47A36B1FE3487240C` |

Modified by converting GLSL spelling to Slang, extracting only independent
opaque-surface functions, prefixing names, and adding finite denominator and
alpha safeguards. The alpha floor intentionally replaces ideal delta surfaces
with finite-width reflection. Dielectric IOR=1 is handled explicitly as no
interface to avoid the reference Fresnel expression's 0/0 at normal incidence.
GGX directional albedo uses the reference analytic rational fit rather than
an external lookup texture or dynamic Monte Carlo integration.

The `program.z == 1` branch in `surface_bsdf.slang` implements the native
OpenPBR 1.1.1 reference graph's supported opaque composition: Fujii
energy-compensated diffuse, OpenPBR anisotropy, F0 modulation by specular weight,
coat-relative substrate IOR with the graph's ratio-inversion fix, roughness
broadening from coat, generalized Hoffman-Schlick metallic reflection with
color82, coat darkening and colored coat attenuation. Independent base/coat
normal and tangent frames are supported. Coated emission uses the graph's
angle-dependent generalized Schlick EDF. It does not translate OpenPBR to
Standard Surface.

Its sampler shares the full-mixture proposal/PDF machinery with the Standard
Surface branch while each model has its own parameter semantics and evaluators.

## Extended optics and transport (2026-10-24)

`surface_optics_math.slang` ports the same MaterialX v1.39.5
`mx_microfacet_specular.glsl` spectral Airy model and its polarized conductor
interface/phase helpers. It retains the three-term Belcour spectral sensitivity
fit and explicit CIE RGB conversion. The file also ports the Zeltner LTC sheen
directional-albedo and LTC coefficient fits, density and importance sampler
from `pbrlib/genglsl/lib/mx_microfacet_sheen.glsl` (SHA-256
`C105CA7788FAAD788B605945830D6FA998A41AC085BD920DB246C69D80C3A54C`).
These MaterialX sources are Apache-2.0, with the license retained in `LICENSE`.
The original Zeltner algorithm is credited in the source to
<https://github.com/tizian/ltc-sheen>.

Port changes: GLSL vector types and names were converted to Slang; orthonormal
bases and RGB conversion use explicit vector arithmetic to remove matrix-layout
ambiguity; denominators/normals receive finite-domain guards. Film IORs are
relative to the incident medium rather than assuming vacuum, admitting an
exiting film with lower IOR and handling film-interface TIR explicitly. Film
coverage blends ordinary Fresnel with Airy reflectance. Film thickness is in nm.
The Zeltner Gaussian albedo fit is clamped to a physical [0,1] range; roughness
uses the reference [0.01,1] domain. Native OpenPBR fuzz is layered over the coat,
including its directional attenuation and exact LTC proposal density.
Thin-wall subsurface follows the canonical OpenPBR 1.1.1 specification's
single-color hemispherical albedos S*(1-g)/2 and S*(1+g)/2:
<https://academysoftwarefoundation.github.io/OpenPBR/>. This intentionally
corrects the v1.39.5 reference nodegraph's duplicated subsurface-color multiply;
the color is applied once, not squared.

The new dielectric transport in `surface_bsdf.slang` is Caustra's path-tracing
implementation of the Walter rough dielectric model, not MaterialX's
environment-map transmission approximation. Its visible-GGX proposal handles
Fresnel reflection/TIR and rough refraction; evaluation uses the refractive
half-vector Jacobian and radiance transport factor etaI^2/etaT^2. Mathematical
reference: <https://www.pbr-book.org/4ed/Reflection_Models/Rough_Dielectric_BSDF>.
No PBRT implementation code was imported. Thin sheets use the incoherent
parallel-interface effective Fresnel 2F/(1+F) and a mirrored GGX transmission
lobe. Exactly equal IORs form a null transmission delta, marked explicitly for
the integrator's emitter MIS; other surfaces retain finite roughness floors.

All finite proposals (diffuse, base GGX, coat GGX, boundary transmission,
thin-wall diffuse transmission and native Zeltner fuzz) contribute to the full
directional mixture density. Returned weights evaluate all contributing finite
lobes divided by that density. Negative geometric hemispheres are legal only
for transmission events. The transmission and SSS interior form one physical
coefficient mixture, avoiding stochastic selection of incompatible path domains.
Event 4 labels SSS entry; exit is event 3. The integrator owns homogeneous
free-flight, phase scattering, absorption, volume emission and medium stacks.

Dispersion uses a Cauchy fit to the Fraunhofer C/F separation around the d-line
IOR. Native OpenPBR first modulates dielectric F0 by specular weight and
converts it back to IOR, as required for both reflection and refraction by the
canonical specification; Cauchy dispersion uses this modulated d-line IOR.
Native OpenPBR specular color tints exterior reflection only; solid internal
reflection and boundary transmission are untinted. Thin sheets expose an
exterior reflection from either side. Denoiser guides follow the same rule.
The opaque reference branch keeps its existing relative coating IOR treatment.
For actual volume propagation, the coating is a surface layer rather than a
second finite-thickness medium: its relative-IOR preview approximation is not
baked into the physical bulk IOR. The material ABI stores inverse effective
Abbe number; the path integrator
selects and reuses one RGB hero wavelength (610/550/460 nm) through nested entry
and exit boundaries. This is RGB hero-channel spectral transport, not a full
continuous-wavelength renderer. Thin-film reflection itself uses the reference
spectral sensitivity fit, not independent RGB sine waves.
