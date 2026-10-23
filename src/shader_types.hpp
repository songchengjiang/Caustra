#pragma once
#include <caustra/renderer.hpp>

namespace caustra {
struct GpuInstance { Uint4 offsets; }; // vertex offset, index offset, material index, reserved
struct GpuLight {
  Float4 v0; // xyz world-space vertex; w = triangle area
  Float4 edge1; // xyz edge; w = cumulative area
  Float4 edge2;
  Float4 emission;
};
struct PushConstants {
  Float4 origin, lowerLeft, horizontal, vertical;
  Float4 environment;
  Uint4 dimensions; // width, height, first sample, max bounces
  Uint4 counts; // light count, DLSS enabled, unused, unused
  Float4 params; // total light area, ray epsilon, jitter X, jitter Y (input pixels)
};
static_assert(sizeof(Vertex) == 32 && sizeof(Material) == 48);
static_assert(sizeof(GpuInstance) == 16 && sizeof(GpuLight) == 64);
static_assert(sizeof(PushConstants) == 128);
} // namespace caustra
