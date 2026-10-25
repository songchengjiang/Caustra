# Caustra renderer

This is a standalone C++20/Vulkan Ray Query path-tracing library with headless
and Windows real-time viewer examples. The user explicitly requested removing model import and other application
capabilities. The user subsequently requested DLSS Ray Reconstruction as an
optional rendering-core feature. Keep model import, editor and rasterizer
dependencies outside the renderer library. Future asset loaders belong outside it.
The user subsequently requested Blender USD + MaterialX as the primary material
workflow. USD/MaterialX parsing lives in tools/caustra_assets; the optional
caustra_assets C++ library only reads compiled portable packages. Keep these
dependencies outside caustra. Unsupported active material features must fail
explicitly; never silently replace MaterialX graphs with constant PBR values.

Read README.md and docs/architecture.md for scope and extension boundaries.
Keep include/caustra/renderer.hpp and src/shader_types.hpp synchronized with the
buffer and push-constant layouts in shaders/pathtrace.slang. Preserve original
NVIDIA notices in third_party/nvshaders; record any modifications there.
Material IR opcodes/layouts must also match tools/caustra_assets/model.py and
shaders/material_program.slang. Keep native OpenPBR and Standard Surface model
semantics distinct. Preserve MaterialX/Aurora licenses and provenance records.
Ordinary surface reflection and EDF use official ShaderGen output in
shaders/generated; regenerate with tools/generate_materialx_surfaces.py instead
of editing generated math. Preserve raw per-hit inputs separately from sampling
frames. Real dielectric/SSS/volume transport remains an explicit core adapter.

Build: cmake --preset windows; cmake --build --preset windows-release.
Test: ctest --preset windows-release (GPU test requires Vulkan validation layer
and ray-query hardware). See README for Linux presets and dependency overrides.
Core methods are synchronous and single-thread owned. Reset accumulated samples
whenever the camera, scene, material or integration settings change. `setCamera`
is a full camera cut; `updateCamera` preserves DLSS history using camera motion
vectors for static geometry. Keep window/input/presentation code in examples/.
The viewer presents GPU HDR through examples/vulkan_presenter.* and a compute
display transform. Do not restore per-frame HDR readback or CPU image display.
Optional Vulkan interop lives in include/caustra/vulkan.hpp; borrowed output must
finish GPU consumption before renderer mutations. Keep headless defaults free of
surface/swapchain requirements. Raw and DLSS readbacks are explicit and cached.
