# Caustra renderer

This is a standalone C++20/Vulkan Ray Query path-tracing library with a headless
example. The user explicitly requested removing model import and other application
capabilities. The user subsequently requested DLSS Ray Reconstruction as an
optional rendering-core feature. Keep model import, editor and rasterizer
dependencies outside the renderer library. Future asset loaders belong outside it.

Read README.md and docs/architecture.md for scope and extension boundaries.
Keep include/caustra/renderer.hpp and src/shader_types.hpp synchronized with the
buffer and push-constant layouts in shaders/pathtrace.slang. Preserve original
NVIDIA notices in third_party/nvshaders; record any modifications there.

Build: cmake --preset windows; cmake --build --preset windows-release.
Test: ctest --preset windows-release (GPU test requires Vulkan validation layer
and ray-query hardware). See README for Linux presets and dependency overrides.
Core methods are synchronous and single-thread owned. Reset accumulated samples
whenever the camera, scene, material or integration settings change.
