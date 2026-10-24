#pragma once
#include <caustra/renderer.hpp>
#include <cmath>

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
  Uint4 counts; // light count, DLSS enabled, random frame sequence, history valid
  Float4 params; // total light area, ray epsilon, jitter X, jitter Y (input pixels)
};
using Matrix4 = std::array<float, 16>;
struct alignas(16) GpuCameraHistory {
  // Row-major, column-vector transforms. Projection includes Vulkan's Y flip,
  // so NDC (-1,-1) maps to the top-left input pixel; no temporal jitter here.
  Matrix4 currentWorldToClip;
  Matrix4 previousWorldToClip;
};
struct CameraTransforms { Matrix4 view, projection, worldToClip; };

// Shared with the renderer's analytic projection tests. Camera inputs have
// already passed validateCamera; calculations retain precision at large origins.
inline CameraTransforms makeCameraTransforms(const Camera& camera, uint32_t width, uint32_t height) {
  auto normalized=[](double x,double y,double z) -> Float4 {
    const auto magnitude=std::hypot(x,y,z);
    return {float(x/magnitude),float(y/magnitude),float(z/magnitude),0};
  };
  const auto f=normalized(double(camera.target.x)-camera.position.x,
                          double(camera.target.y)-camera.position.y,
                          double(camera.target.z)-camera.position.z);
  const auto up=normalized(camera.up.x,camera.up.y,camera.up.z);
  const auto r=normalized(double(f.y)*up.z-double(f.z)*up.y,
                          double(f.z)*up.x-double(f.x)*up.z,
                          double(f.x)*up.y-double(f.y)*up.x);
  const Float4 u{r.y*f.z-r.z*f.y,r.z*f.x-r.x*f.z,r.x*f.y-r.y*f.x,0};
  auto dotPosition=[&](Float4 v) { return v.x*camera.position.x+v.y*camera.position.y+v.z*camera.position.z; };
  CameraTransforms result{};
  result.view={r.x,r.y,r.z,-dotPosition(r), u.x,u.y,u.z,-dotPosition(u),
               -f.x,-f.y,-f.z,dotPosition(f), 0,0,0,1};
  const float y=1/std::tan(camera.verticalFov*3.14159265358979323846f/360);
  const float x=y*height/width;
  constexpr float nearPlane=0.01f,farPlane=10000;
  result.projection={x,0,0,0, 0,-y,0,0,
                     0,0,farPlane/(nearPlane-farPlane),(farPlane*nearPlane)/(nearPlane-farPlane), 0,0,-1,0};
  for(size_t row=0;row<4;++row)
    for(size_t column=0;column<4;++column)
      for(size_t k=0;k<4;++k)
        result.worldToClip[4*row+column]+=result.projection[4*row+k]*result.view[4*k+column];
  return result;
}
static_assert(sizeof(Vertex) == 32 && sizeof(Material) == 48);
static_assert(sizeof(GpuInstance) == 16 && sizeof(GpuLight) == 64);
static_assert(sizeof(PushConstants) == 128);
static_assert(sizeof(GpuCameraHistory) == 128);
} // namespace caustra
