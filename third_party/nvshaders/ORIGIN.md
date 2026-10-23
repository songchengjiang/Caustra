# NVIDIA shader helpers

These nine shader headers are copied unchanged from the local
`C:/Dev/nvpro_core2/nvshaders/` checkout supplied alongside
`vk_gltf_renderer`. Upstream project:
<https://github.com/nvpro-samples/nvpro_core2/tree/main/nvshaders>.
They are the minimal transitive include set for `bsdfEvaluateSimple`,
`bsdfSampleSimple`, the random number generator and `offsetRay`.

The original NVIDIA copyright and Apache-2.0 headers are preserved. The
upstream Apache-2.0 license is included in `LICENSE`. No glTF scene I/O,
material evaluation, model loaders or texture descriptors are included.
The BSDF helper header still defines the upstream full BSDF functions; only
its simple opaque metallic/roughness model is used and compiled into Caustra.

Source checkout commit: `7cfe3ca9dc53ab23aeb8b55787fa676f1ff6c3d0`.
These SHA-256 hashes additionally pin the exact imported source files:

| File | SHA-256 |
| --- | --- |
| bsdf_functions.h.slang | 7da82efe3b6ff635093840305b94555c78fcb1f1d4731c8ae7b43ee7f58bc44b |
| bsdf_types.h.slang | 427d92bf6082dc8abf25f94eb94b369cf490441120d5d2f17d9d887570c586e9 |
| pbr_ggx_microfacet.h.slang | bb6bc74767419874a5546928147404ae7dabb0f1520d23e13ee82f77dc980205 |
| pbr_material_types.h.slang | 1f7afa694b323ab0cf370c43d9f2f1f4971384abd4123a2fcd33a88f186554ed |
| functions.h.slang | 18def828c41117d044948915951996bf2a754f2d34fbc86a7dc9de9302adf1b5 |
| constants.h.slang | bdcf7e3a0853c8fc253559494724c9d5078790353feed37940527084c30614d8 |
| random.h.slang | 295ed8cbfa66e9bf665972616f0f0b9a99ea41278a72a26b962b2f6b98efc8b2 |
| slang_types.h | 4892ba1852d20ee7d0722dd5287be5c4ad5c966ee5684dbb63e89441e43b3414 |
| ray_utils.h.slang | 90b3141b98f9a929f6ffaf36d9cb109312fc0ac71628d778246de4988b28813b |

`shaders/pathtrace.slang` is Caustra's new scene-independent integrator. It
retains the source renderer's iterative path tracing design, BSDF sampling,
next-event estimation, balance-heuristic MIS and Russian roulette. Area lights
use world-area sampling rather than the original renderer's power weighting.
The environment is constant and sampled by BSDF paths only. Phase 1 does not
expose the upstream full BSDF, transmission, textures or glTF extensions.
