#pragma once
#include <caustra/renderer.hpp>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <stdexcept>

namespace caustra {
struct GpuInstance {
  Uint4 offsets; // vertex offset, index offset, material index, heterogeneous grid base row (0 if absent)
  std::array<Float4,3> worldToObject, objectToWorld;
  Float4 objectMin, objectMax; // xyz bounds; objectMin.w bit-casts the first emitter triangle index (UINT32_MAX if absent)
};
struct GpuLight {
  Float4 v0; // xyz world-space vertex; w = triangle area
  Float4 edge1; // xyz edge; w = exact cumulative 24-bit selection tickets
  Float4 edge2; // xyz edge; w = instance orientation sign for outward emission
  Uint4 source; // global vertex indices and material index
  Float4 normal0, normal1, normal2; // xyz inverse-transposed normals; normal0.w bit-casts source instance index
  Float4 tangent0, tangent1, tangent2; // world tangents, mirrored handedness in w
};
inline constexpr uint32_t kLightSelectionTickets=1u<<24;

// Integer-valued floats remain exact through 2^24. Reserve one uniform RNG
// ticket for every triangle, then distribute the remainder by double-precision
// area*radiance weights. Small positive lights cannot disappear behind a large
// cumulative float value. Both MIS paths use actual adjacent CDF differences,
// rather than the unquantized proposal weights, so this floor does not bias L.
inline void buildLightCdf(std::vector<GpuLight>& lights,const std::vector<double>& weights) {
  if(lights.size()!=weights.size() || lights.size()>=kLightSelectionTickets)
    throw std::invalid_argument("Emitter distribution exceeds the 24-bit selection ticket capacity");
  double total=0;
  for(double weight:weights) {
    if(!(weight>0) || !std::isfinite(weight)) throw std::invalid_argument("Invalid emitter sampling weight");
    total+=weight;
  }
  if(!std::isfinite(total)) throw std::invalid_argument("Emitter sampling weight sum is not finite");
  double cumulative=0;const auto remaining=kLightSelectionTickets-static_cast<uint32_t>(lights.size());
  for(size_t i=0;i<lights.size();++i) {
    cumulative+=weights[i];
    const uint32_t proportional=static_cast<uint32_t>(std::floor(std::min(1.0,cumulative/total)*remaining));
    lights[i].edge1.w=float(i+1+proportional);
  }
  if(!lights.empty()) lights.back().edge1.w=float(kLightSelectionTickets);
}
struct GpuTextureInfo { Uint4 size; }; // width, height, base texel, mip count
// Bindings 15/16 use Uint4(first instruction,count,flags,reserved) per material
// and the same 32-byte MaterialInstruction as the public IR. These derived
// coverage programs do not change Material or the portable asset layout.
inline constexpr uint32_t kMediumStackCapacity = 16;
inline constexpr uint32_t kCameraHistoryValid = 1u;
inline constexpr uint32_t kSkipInvalidMediumPath = 1u << 1;
inline constexpr uint32_t kDisableMaterialCoverageOptimization = 1u << 2;
inline constexpr size_t kRenderDiagnosticWords = 20;
// Only the optional CAUSTRA_PROFILE_RAYS pipeline accesses the words after the
// ordinary transport diagnostics. One shard per pixel modulo 1024 limits
// atomic contention without requiring shaderInt64/64-bit atomics. Each shard's
// final word records any local or atomic uint overflow; affected counts are not
// valid measurements. MaxScatteringDepth reduces by max; all other counters sum.
enum class RayProfileCounter : uint32_t {
  PathSamples = 0, PrimaryRayQueries, IndirectRayQueries, ShadowRayQueries,
  AuxiliaryRayQueries, BoundaryContinuationQueries, QueryHits, QueryMisses,
  CandidateVisits, CoverageEvaluations, FullMaterialEvaluations,
  AlphaRejectedHits, ShadowTransparentHits, AcceptedSurfaceHits,
  SurfaceScatters, MediumScatters, RussianRouletteTerminations,
  MaxScatteringDepth, Count
};
inline constexpr size_t kRayProfileBaseWord = kRenderDiagnosticWords;
inline constexpr size_t kRayProfileCounterCount = static_cast<size_t>(RayProfileCounter::Count);
inline constexpr size_t kRayProfileShardCount = 1024;
inline constexpr size_t kRayProfileShardWords = kRayProfileCounterCount + 1;
inline constexpr size_t kRayProfileOverflowOffset = kRayProfileCounterCount;
inline constexpr size_t kRayProfileDiagnosticWords = kRayProfileBaseWord + kRayProfileShardCount*kRayProfileShardWords;
inline constexpr uint32_t kInitCameraMedium = 1u << 31;
inline constexpr uint32_t kInitMaterialMedium = 1u << 30;
inline constexpr uint32_t kInitHeterogeneousMedium = 1u << 29;
inline constexpr uint32_t kMediumGridResolution = 32;
inline constexpr uint32_t kMediumGridCells = kMediumGridResolution*kMediumGridResolution*kMediumGridResolution;
// Binding 14 begins with eight uint4 rows per material (uniform optical IOR
// and homogeneous coefficients). Optional 32^3 per-instance coefficient grids
// follow that table. Surface parameters continue to be evaluated at each hit.
struct alignas(16) GpuMaterialMediumData {
  Uint4 metadata; // has interior, invisible boundary, initialized, reserved
  Float4 absorption, scatteringVolume, scatteringTransmission, scatteringSubsurface;
  Float4 emission, anisotropy, ior;
};
// Binding 13 holds containment only. Per-path stacks use these instance IDs to
// look up the shared coefficients without copying them into thread-local arrays.
struct alignas(16) GpuCameraMediumCache {
  Uint4 header; // count, contains dispersion, state (0 unset/1 valid/2 failed probe), error flags
  std::array<Uint4, kMediumStackCapacity / 4> instances;
};
struct PushConstants {
  Float4 origin, lowerLeft, horizontal, vertical;
  Float4 environment; // RGB constant radiance; w enables medium boundary handling
  Uint4 dimensions; // width, height, first sample, max bounces
  Uint4 counts; // light count, DLSS enabled, random frame sequence, flags (history 0, medium recovery 1, coverage reference 2, init 29..31)
  Float4 params; // total light selection tickets, guide/depth floor, jitter X, jitter Y (input pixels)
};
using Matrix4 = std::array<float, 16>;
struct alignas(16) GpuCameraHistory {
  // Row-major, column-vector transforms. Projection includes Vulkan's Y flip,
  // so NDC (-1,-1) maps to the top-left input pixel; no temporal jitter here.
  Matrix4 currentWorldToClip;
  Matrix4 previousWorldToClip;
  Float4 transport; // x: mismatched medium boundary tolerance in meters; yzw reserved
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
static_assert(sizeof(Vertex) == 64 && sizeof(Material) == 384);
static_assert(offsetof(Material,transmission)==160 && offsetof(Material,program)==368);
static_assert(sizeof(MaterialInstruction) == 32 && sizeof(GpuTextureInfo) == 16);
static_assert(sizeof(Uint4) == 16);
static_assert(sizeof(GpuMaterialMediumData) == 128 && offsetof(GpuMaterialMediumData,ior) == 112);
static_assert(sizeof(GpuCameraMediumCache) == 80 && offsetof(GpuCameraMediumCache,instances) == 16);
static_assert(sizeof(GpuInstance) == 144 && sizeof(GpuLight) == 160);
static_assert(sizeof(PushConstants) == 128);
static_assert(sizeof(GpuCameraHistory) == 144 && offsetof(GpuCameraHistory,transport) == 128);
} // namespace caustra
