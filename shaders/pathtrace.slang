// SPDX-License-Identifier: Apache-2.0
// Rendering algorithm adapted from vk_gltf_renderer,
// Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
// Modified for a standalone scene API; see NOTICE for provenance.
// Caustra's standalone path tracing integrator reuses the NVIDIA RNG and ray
// offset helpers. Material programs, surface closures and participating media
// have separate implementations below. No glTF scene structures or loader remain.
#include "nvshaders/random.h.slang"
#include "nvshaders/ray_utils.h.slang"

// These storage-buffer and push-constant layouts match renderer.hpp and
// shader_types.hpp. Keep float4/uint4 boundaries when extending this interface.
struct Vertex { float4 position; float4 normal; float4 texcoord; float4 tangent; };
// objectMin.w bit-casts the first light triangle index, UINT32_MAX if absent.
struct GpuInstance { uint4 offsets; float4 worldToObject[3],objectToWorld[3]; float4 objectMin,objectMax; };
// v0.w is area; edge1.w is a ticket CDF; normal0.w bit-casts the source instance.
struct GpuLight { float4 v0; float4 edge1; float4 edge2; uint4 source; float4 normal0,normal1,normal2,tangent0,tangent1,tangent2; };
struct PushConstants
{
  float4 origin, lowerLeft, horizontal, vertical;
  float4 environment;
  uint4 dimensions;
  uint4 counts;
  float4 params;
};
struct GpuCameraHistory
{
  float4x4 currentWorldToClip;
  float4x4 previousWorldToClip;
  float4 transport; // x: mismatched medium boundary tolerance in meters
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] StructuredBuffer<Vertex> vertices;
[[vk::binding(2, 0)]] StructuredBuffer<uint> indices;
[[vk::binding(3, 0)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(5, 0)]] StructuredBuffer<GpuLight> lights;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> accumulation;
// Optional DLSS RR input planes; byte offsets match renderer.cpp. Raw
// progressive radiance stays in accumulation, independently of DLSS history.
[[vk::binding(7, 0)]] RWByteAddressBuffer guideBuffer;
[[vk::binding(8, 0)]] ConstantBuffer<GpuCameraHistory> cameraHistory;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const float FAR_DISTANCE = 1.0e30f;
static const float GUIDE_FAR_DISTANCE = 1.0e4f;
// counts.w shares the history flag with material/camera initialization modes.
// The host uses the same pipeline and inserts compute-write/read barriers
// between initialization dispatches and the ordinary path dispatch.
static const uint CAMERA_HISTORY_VALID = 1u;
static const uint SKIP_INVALID_MEDIUM_PATH = 1u << 1;
static const uint DISABLE_MATERIAL_COVERAGE_OPTIMIZATION = 1u << 2;
static const uint INIT_CAMERA_MEDIUM = 1u << 31;
static const uint INIT_MATERIAL_MEDIUM = 1u << 30;
static const uint INIT_HETEROGENEOUS_MEDIUM = 1u << 29;

bool finite3(float3 v) { return !any(isnan(v)) && !any(isinf(v)); }
bool positivePdf(float p) { return p > 0.0f && !isnan(p) && !isinf(p); }

// Balance heuristic without a potentially overflowing sum of large PDFs.
float balanceWeight(float selectedPdf, float otherPdf)
{
  if(!positivePdf(selectedPdf)) return 0.0f;
  if(!positivePdf(otherPdf)) return 1.0f;
  float scale = max(selectedPdf, otherPdf);
  return (selectedPdf / scale) / (selectedPdf / scale + otherPdf / scale);
}

float3 safeNormal(float3 v, float3 fallback)
{
  float l2 = dot(v, v);
  return l2 > 1.0e-20f && !isinf(l2) ? v * rsqrt(l2) : fallback;
}

struct SurfaceHit
{
  float3 position;
  float3 geometricNormal;
  float3 shadingNormal;
  float3 objectPosition, tangent, bitangent;
  float2 texcoord, uvDensity;
  float distance, positionError;
  bool frontFacing;
  uint materialIndex, instanceIndex, primitiveIndex;
};

#include "material_program.slang"
#include "surface_bsdf.slang"

// A separately compiled pipeline owns profiling. The normal shader preprocesses
// every declaration, wrapper and call below away, retaining its register usage
// and ray/material program. No shader clock or vendor-only atomic is required.
#ifndef CAUSTRA_PROFILE_RAYS
#define CAUSTRA_PROFILE_RAYS 0
#endif
#if CAUSTRA_PROFILE_RAYS
// Exact order/stride matches RayProfileCounter in shader_types.hpp.
static const uint RAY_PROFILE_PATH_SAMPLES = 0;
static const uint RAY_PROFILE_PRIMARY_QUERIES = 1;
static const uint RAY_PROFILE_INDIRECT_QUERIES = 2;
static const uint RAY_PROFILE_SHADOW_QUERIES = 3;
static const uint RAY_PROFILE_AUXILIARY_QUERIES = 4;
static const uint RAY_PROFILE_BOUNDARY_CONTINUATION_QUERIES = 5;
static const uint RAY_PROFILE_QUERY_HITS = 6;
static const uint RAY_PROFILE_QUERY_MISSES = 7;
static const uint RAY_PROFILE_CANDIDATE_VISITS = 8;
static const uint RAY_PROFILE_COVERAGE_EVALUATIONS = 9;
static const uint RAY_PROFILE_FULL_MATERIAL_EVALUATIONS = 10;
static const uint RAY_PROFILE_ALPHA_REJECTED_HITS = 11;
static const uint RAY_PROFILE_SHADOW_TRANSPARENT_HITS = 12;
static const uint RAY_PROFILE_ACCEPTED_SURFACE_HITS = 13;
static const uint RAY_PROFILE_SURFACE_SCATTERS = 14;
static const uint RAY_PROFILE_MEDIUM_SCATTERS = 15;
static const uint RAY_PROFILE_RUSSIAN_ROULETTE_TERMINATIONS = 16;
static const uint RAY_PROFILE_MAX_SCATTERING_DEPTH = 17;
static const uint RAY_PROFILE_COUNTER_COUNT = 18;
static const uint RAY_PROFILE_SHARD_COUNT = 1024;
static const uint RAY_PROFILE_BASE_WORD = 20;
static const uint RAY_PROFILE_SHARD_WORDS = RAY_PROFILE_COUNTER_COUNT + 1;
static uint rayProfileCounters[RAY_PROFILE_COUNTER_COUNT];
static uint rayProfileQueryKind = RAY_PROFILE_AUXILIARY_QUERIES;
static bool rayProfileActive = false;
static uint rayProfileOverflow = 0;

void rayProfileAdd(uint counter)
{
  if(!rayProfileActive) return;
  uint previous=rayProfileCounters[counter];
  rayProfileCounters[counter]=previous+1u;
  if(previous==0xffffffffu) rayProfileOverflow=1u;
}
void rayProfileDepth(uint depth)
{
  if(rayProfileActive)
    rayProfileCounters[RAY_PROFILE_MAX_SCATTERING_DEPTH]=max(rayProfileCounters[RAY_PROFILE_MAX_SCATTERING_DEPTH],depth);
}
MaterialEvaluation rayProfileEvaluateMaterialProgram(uint materialIndex, MaterialContext ctx)
{
  rayProfileAdd(RAY_PROFILE_FULL_MATERIAL_EVALUATIONS);
  return evaluateMaterialProgram(materialIndex,ctx);
}
MaterialEvaluation rayProfileEvaluateMaterialProgramCompact(uint materialIndex, MaterialContext ctx)
{
  rayProfileAdd(RAY_PROFILE_FULL_MATERIAL_EVALUATIONS);
  return evaluateMaterialProgramCompact(materialIndex,ctx);
}
float rayProfileEvaluateMaterialCoverage(uint materialIndex, MaterialContext ctx)
{
  rayProfileAdd(RAY_PROFILE_COVERAGE_EVALUATIONS);
  // A large/unsupported coverage slice executes a full program internally.
  // Count that call as well; ordinary coverage instructions are not full shading.
  if((coveragePrograms[materialIndex].z&1u)!=0)
    rayProfileAdd(RAY_PROFILE_FULL_MATERIAL_EVALUATIONS);
  return evaluateMaterialCoverage(materialIndex,ctx);
}
#define evaluateMaterialProgram rayProfileEvaluateMaterialProgram
#define evaluateMaterialProgramCompact rayProfileEvaluateMaterialProgramCompact
#define evaluateMaterialCoverage rayProfileEvaluateMaterialCoverage
#define RAY_PROFILE_ADD(counter) rayProfileAdd(counter)
#define RAY_PROFILE_DEPTH(depth) rayProfileDepth(depth)
#else
#define RAY_PROFILE_ADD(counter)
#define RAY_PROFILE_DEPTH(depth)
#endif

bool optimizedMaterialCoverage()
{
  return (pc.counts.w & DISABLE_MATERIAL_COVERAGE_OPTIMIZATION) == 0;
}
MaterialEvaluation evaluateShadingMaterial(uint materialIndex, MaterialContext ctx)
{
  if(optimizedMaterialCoverage()) return evaluateMaterialProgramCompact(materialIndex, ctx);
  return evaluateMaterialProgram(materialIndex, ctx);
}

MaterialContext materialContext(SurfaceHit hit,float footprint)
{
  MaterialContext ctx;
  ctx.position=hit.position;ctx.objectPosition=hit.objectPosition;
  ctx.geometricNormal=hit.geometricNormal;ctx.normal=hit.shadingNormal;
  ctx.tangent=hit.tangent;ctx.bitangent=hit.bitangent;
  ctx.texcoord=hit.texcoord;ctx.uvFootprint=max(0.0f,footprint)*hit.uvDensity;
  return ctx;
}

void surfaceFrame(inout SurfaceHit hit,Vertex a,Vertex b,Vertex c,float3 e1,float3 e2,float3 tangent,float handedness)
{
  float2 d1=b.texcoord.xy-a.texcoord.xy,d2=c.texcoord.xy-a.texcoord.xy;
  float determinant=d1.x*d2.y-d1.y*d2.x;
  float3 du=float3(0.0f),dv=float3(0.0f);
  hit.uvDensity=float2(0.0f);
  if(abs(determinant)>1e-12f)
  {
    du=(e1*d2.y-e2*d1.y)/determinant;dv=(e2*d1.x-e1*d2.x)/determinant;
    float g11=dot(e1,e1),g12=dot(e1,e2),g22=dot(e2,e2),metric=g11*g22-g12*g12;
    if(metric>1e-20f) hit.uvDensity=float2(length(e1*(d1.x*g22-d2.x*g12)+e2*(d2.x*g11-d1.x*g12)),
                                                       length(e1*(d1.y*g22-d2.y*g12)+e2*(d2.y*g11-d1.y*g12)))/metric;
  }
  float3 fallback=abs(hit.shadingNormal.z)<0.99f?cross(float3(0,0,1),hit.shadingNormal):cross(float3(0,1,0),hit.shadingNormal);
  if(dot(tangent,tangent)<1e-20f) tangent=dot(du,du)>1e-20f?du:fallback;
  hit.tangent=safeNormal(tangent-hit.shadingNormal*dot(tangent,hit.shadingNormal),safeNormal(fallback,float3(1,0,0)));
  float sign=handedness==0.0f?(dot(cross(hit.shadingNormal,hit.tangent),dv)<0.0f?-1.0f:1.0f):(handedness<0.0f?-1.0f:1.0f);
  hit.bitangent=sign*cross(hit.shadingNormal,hit.tangent);
}

struct DenoiserGuides
{
  float3 diffuseAlbedo;
  float3 specularAlbedo;
  float3 normal;
  float roughness;
  float linearDepth;
  float specularHitDistance;
  float2 motion;
};

// NGX expects current-to-previous displacement in input-resolution pixels.
// Both transforms are unjittered: the primary sample's jitter is supplied to
// NGX separately and must not be added to these motion vectors. A direction
// (w=0) reprojects the constant environment without camera translation.
float2 cameraMotion(float4 worldPositionOrDirection)
{
  if((pc.counts.w & CAMERA_HISTORY_VALID) == 0) return float2(0.0f);
  float4 current = mul(cameraHistory.currentWorldToClip, worldPositionOrDirection);
  float4 previous = mul(cameraHistory.previousWorldToClip, worldPositionOrDirection);
  if(current.w <= pc.params.y || previous.w <= pc.params.y
     || !finite3(float3(current.xy, current.w)) || !finite3(float3(previous.xy, previous.w)))
    return float2(0.0f);
  float2 motion = (previous.xy / previous.w - current.xy / current.w) * 0.5f * float2(pc.dimensions.xy);
  return !any(isnan(motion)) && !any(isinf(motion)) ? motion : float2(0.0f);
}

// Adapted from vk_gltf_renderer/shaders/dlss_util.h (NVIDIA, Apache-2.0):
// Ray Tracing Gems, Chapter 32's integrated GGX specular reflectance fit.
// alpha is squared perceptual roughness, as in the simple BSDF above.
float3 specularGuideAlbedo(float3 f0, float alpha, float nDotV)
{
  float4 X = float4(1.0f, nDotV, nDotV * nDotV, nDotV * nDotV * nDotV);
  float4 Y = float4(1.0f, alpha, alpha * alpha, alpha * alpha * alpha);
  float2x2 M1 = float2x2(0.99044f, -1.28514f, 1.29678f, -0.755907f);
  float3x3 M2 = float3x3(1.0f, 2.92338f, 59.4188f, 20.3225f, -27.0302f, 222.592f,
                        121.563f, 626.13f, 316.627f);
  float2x2 M3 = float2x2(0.0365463f, 3.32707f, 9.0632f, -9.04756f);
  float3x3 M4 = float3x3(1.0f, 3.59685f, -1.36772f, 9.04401f, -16.3174f, 9.22949f,
                        5.56589f, 19.7886f, -20.2123f);
  float bias = dot(mul(M1, X.xy), Y.xy) / dot(mul(M2, X.xyw), Y.xyw);
  float scale = dot(mul(M3, X.xy), Y.xy) / dot(mul(M4, X.xzw), Y.xyw);
  bias *= saturate(f0.g * 50.0f);
  float3 albedo = f0 * max(0.0f, scale) + max(0.0f, bias);
  return finite3(albedo) ? saturate(albedo) : saturate(f0);
}

void writeDenoiserGuides(uint index, float3 noisyRadiance, DenoiserGuides guides)
{
  uint pixels = pc.dimensions.x * pc.dimensions.y;
  guideBuffer.Store4(16 * index, asuint(float4(noisyRadiance, 1.0f)));
  guideBuffer.Store4(16 * pixels + 16 * index, asuint(float4(guides.diffuseAlbedo, 1.0f)));
  guideBuffer.Store4(32 * pixels + 16 * index, asuint(float4(guides.specularAlbedo, 1.0f)));
  guideBuffer.Store4(48 * pixels + 16 * index, asuint(float4(guides.normal, guides.roughness)));
  guideBuffer.Store2(64 * pixels + 8 * index, asuint(guides.motion));
  guideBuffer.Store(72 * pixels + 4 * index, asuint(guides.linearDepth));
  guideBuffer.Store(76 * pixels + 4 * index, asuint(guides.specularHitDistance));
}

float3 triangleObjectPosition(float3 a,float3 b,float3 c,float2 uv)
{
  precise float3 edge1=b-a,edge2=c-a;
  precise float3 position=a+fma(float3(uv.x),edge1,uv.y*edge2);
  return position;
}

float3 triangleWorldPosition(float3 objectPosition,float3x4 objectToWorld)
{
  precise float3 position;
  position.x=objectToWorld[0].w+fma(objectToWorld[0].x,objectPosition.x,
    fma(objectToWorld[0].y,objectPosition.y,objectToWorld[0].z*objectPosition.z));
  position.y=objectToWorld[1].w+fma(objectToWorld[1].x,objectPosition.x,
    fma(objectToWorld[1].y,objectPosition.y,objectToWorld[1].z*objectPosition.z));
  position.z=objectToWorld[2].w+fma(objectToWorld[2].x,objectPosition.x,
    fma(objectToWorld[2].y,objectPosition.y,objectToWorld[2].z*objectPosition.z));
  return position;
}

float trianglePositionError(float3 a,float3 b,float3 c,float3 objectPosition,float3 worldPosition,
                            float3 geometricNormal,float3x4 objectToWorld,float3x4 worldToObject)
{
  // NVIDIA's triangle reconstruction, traversal, and affine round-trip bounds.
  // Project onto the actual geometric normal: unrelated large coordinates must
  // not turn a micrometer normal error into a millimeter surface displacement.
  const float reconstructionRoundoff=5.9604644775390625e-8f;
  const float triangleRoundoff=1.7881397695873602e-7f;
  const float transformRoundoff=1.1920931797249068e-7f;
  float3 edge1=b-a,edge2=c-a;
  float3 extent3=abs(edge1)+abs(edge2)+abs(abs(edge1)-abs(edge2));
  float extent=max(extent3.x,max(extent3.y,extent3.z));
  float3 objectError=reconstructionRoundoff*abs(a)+triangleRoundoff*extent;
  objectError+=transformRoundoff*mul(abs(worldToObject),float4(abs(worldPosition),1));
  float3 worldError=triangleRoundoff*mul(abs((float3x3)objectToWorld),abs(objectPosition))+
    transformRoundoff*abs(float3(objectToWorld[0].w,objectToWorld[1].w,objectToWorld[2].w));
  float3 objectNormal=cross(edge1,edge2);
  float3 transformedNormal=objectNormal.x*worldToObject[0].xyz+
    objectNormal.y*worldToObject[1].xyz+objectNormal.z*worldToObject[2].xyz;
  float inverseNormalLength=rsqrt(max(dot(transformedNormal,transformedNormal),1.0e-30f));
  return 2.0f*(dot(objectError,abs(objectNormal))*inverseNormalLength+
    dot(worldError,abs(geometricNormal)));
}

bool traceGeometry(RayDesc ray, out SurfaceHit hit,uint excludedInstance=0xffffffffu,float processedDistance=-1.0f)
{
  hit = (SurfaceHit)0;
  bool continuation=excludedInstance!=0xffffffffu;
#if CAUSTRA_PROFILE_RAYS
  RAY_PROFILE_ADD(rayProfileQueryKind);
  if(continuation) { RAY_PROFILE_ADD(RAY_PROFILE_BOUNDARY_CONTINUATION_QUERIES); }
#endif
  RayQuery<RAY_FLAG_NONE> query;
  query.TraceRayInline(topLevelAS,continuation?RAY_FLAG_FORCE_NON_OPAQUE:RAY_FLAG_FORCE_OPAQUE,0xff,ray);
  while(query.Proceed())
  {
    // Proceed exposes shader-side candidate events, not hardware BVH node or
    // triangle test counts. FORCE_OPAQUE traversal normally exposes none.
    RAY_PROFILE_ADD(RAY_PROFILE_CANDIDATE_VISITS);
    if(continuation && query.CandidateType()==CANDIDATE_NON_OPAQUE_TRIANGLE)
    {
      float distance=query.CandidateTriangleRayT();
      if(distance<processedDistance || (distance==processedDistance && query.CandidateInstanceID()==excludedInstance)) continue;
      query.CommitNonOpaqueTriangleHit();
    }
  }
  if(query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
  {
    RAY_PROFILE_ADD(RAY_PROFILE_QUERY_MISSES);
    return false;
  }
  RAY_PROFILE_ADD(RAY_PROFILE_QUERY_HITS);

  GpuInstance instance = instances[query.CommittedInstanceID()];
  hit.instanceIndex=query.CommittedInstanceID();
  hit.primitiveIndex=query.CommittedPrimitiveIndex();
  uint first = instance.offsets.y + query.CommittedPrimitiveIndex() * 3;
  Vertex a = vertices[instance.offsets.x + indices[first]];
  Vertex b = vertices[instance.offsets.x + indices[first + 1]];
  Vertex c = vertices[instance.offsets.x + indices[first + 2]];
  float2 uv = query.CommittedTriangleBarycentrics();
  float3 bary = float3(1.0f - uv.x - uv.y, uv.x, uv.y);
  float3x4 objectToWorld = query.CommittedObjectToWorld3x4();
  float3x4 worldToObject = query.CommittedWorldToObject3x4();

  // Add the base vertex and affine translation last. Precise operations keep
  // the reconstruction compatible with the error bound below.
  precise float3 objectPosition=triangleObjectPosition(a.position.xyz,b.position.xyz,c.position.xyz,uv);
  precise float3 worldPosition=triangleWorldPosition(objectPosition,objectToWorld);
  hit.position=worldPosition;
  hit.objectPosition=objectPosition;
  hit.texcoord=a.texcoord.xy*bary.x+b.texcoord.xy*bary.y+c.texcoord.xy*bary.z;
  hit.distance=query.CommittedRayT();
  // Deriving the geometric normal from world-space edges also handles mirrored
  // instances. The interpolated normal needs the inverse transpose under scale.
  float3 edge1 = mul(objectToWorld, float4(b.position.xyz - a.position.xyz, 0.0f));
  float3 edge2 = mul(objectToWorld, float4(c.position.xyz - a.position.xyz, 0.0f));
  hit.geometricNormal = safeNormal(cross(edge1, edge2), -ray.Direction);
  // Normal clearance must include object/world round trips, not just one
  // world-space ULP (a small world coordinate may cancel a large translation).
  // Numerical analysis: NVIDIA, "Solving Self-Intersection Artifacts in
  // DirectX Raytracing" (2023); formulas apply to Vulkan triangle traversal.
  hit.positionError=trianglePositionError(a.position.xyz,b.position.xyz,c.position.xyz,objectPosition,
    worldPosition,hit.geometricNormal,objectToWorld,worldToObject);
  // Use traversal's object-space winding decision. Recomputing it from a
  // world-space cross/dot can disagree at grazing facets after rounding and
  // turn an exterior entry into an unmatched exit. TLAS transforms, including
  // mirrored instances, do not change Vulkan's object-space facing flag.
  hit.frontFacing=query.CommittedTriangleFrontFace();
  float3 n = a.normal.xyz * bary.x + b.normal.xyz * bary.y + c.normal.xyz * bary.z;
  float3 worldNormal = n.x * worldToObject[0].xyz + n.y * worldToObject[1].xyz + n.z * worldToObject[2].xyz;
  hit.shadingNormal = safeNormal(worldNormal, hit.geometricNormal);

  // Face the local shading frame toward the incident ray, while preserving
  // physical frontFacing for medium transitions and one-sided emission.
  if(dot(hit.geometricNormal, ray.Direction) > 0.0f) hit.geometricNormal = -hit.geometricNormal;
  if(dot(hit.shadingNormal, hit.geometricNormal) < 0.0f) hit.shadingNormal = -hit.shadingNormal;
  // Prevent the reflection lobe of a strongly interpolated normal from sending
  // the incoming ray through the geometric surface.
  if(dot(reflect(ray.Direction, hit.shadingNormal), hit.geometricNormal) <= 0.0f)
    hit.shadingNormal = hit.geometricNormal;
  float4 tangent=a.tangent*bary.x+b.tangent*bary.y+c.tangent*bary.z;
  float orientation=dot(cross(objectToWorld[0].xyz,objectToWorld[1].xyz),objectToWorld[2].xyz)<0.0f?-1.0f:1.0f;
  surfaceFrame(hit,a,b,c,edge1,edge2,mul(objectToWorld,float4(tangent.xyz,0.0f)),tangent.w*orientation);
  hit.materialIndex = instance.offsets.z;
  return true;
}

#include "medium.slang"

#if CAUSTRA_PROFILE_RAYS
void flushRayProfile(uint pixelIndex)
{
  uint base=RAY_PROFILE_BASE_WORD+(pixelIndex%RAY_PROFILE_SHARD_COUNT)*RAY_PROFILE_SHARD_WORDS;
  for(uint counter=0;counter<RAY_PROFILE_COUNTER_COUNT;++counter)
  {
    uint value=rayProfileCounters[counter];
    if(value==0) continue;
    if(counter==RAY_PROFILE_MAX_SCATTERING_DEPTH)
    {
      InterlockedMax(renderErrors[base+counter],value);
    }
    else
    {
      uint previous;
      InterlockedAdd(renderErrors[base+counter],value,previous);
      if(previous>0xffffffffu-value) rayProfileOverflow=1u;
    }
  }
  if(rayProfileOverflow!=0)
    InterlockedOr(renderErrors[base+RAY_PROFILE_COUNTER_COUNT],1u);
}
#endif

float3 rayOrigin(SurfaceHit hit, float3 normal)
{
  // Use the triangle/instance error bound instead of a world-position-sized
  // epsilon. The latter can jump through an otherwise resolvable glass wall.
  // Round each changed coordinate into the chosen geometric halfspace, then
  // repair any projection lost to float rounding, reusing medium spawn tools.
  precise float3 position=fma(normal,float3(hit.positionError),hit.position);
  position=mediumNextPoint(position,normal);
  for(uint pass=0;pass<2;++pass)
  {
    float clearance=dot(position-hit.position,normal);
    if(clearance>=hit.positionError) break;
    position=mediumNextPoint(fma(normal,float3(hit.positionError-clearance),position),normal);
  }
  return mediumPointInHalfspace(position,hit.position,normal);
}

bool traceSurface(inout RayDesc ray,inout float coneWidth,float coneSpread,inout uint seed,out SurfaceHit hit,out MaterialEvaluation material,
                  uint excludedInstance=0xffffffffu,float processedDistance=-1.0f)
{
  material=(MaterialEvaluation)0;hit=(SurfaceHit)0;
  float initialWidth=coneWidth;
  // Advance the intersection interval, preserving the exact original ray line.
  // Repeated normal offsets would accumulate drift through deep alpha layers.
  for(;;)
  {
    if(ray.TMin>ray.TMax) return false;
    if(!traceGeometry(ray,hit,excludedInstance,processedDistance)) return false;
    coneWidth=initialWidth+hit.distance*coneSpread;
    MaterialContext ctx=materialContext(hit,coneWidth);
    if(optimizedMaterialCoverage())
    {
      float opacity=evaluateMaterialCoverage(hit.materialIndex,ctx);
      if(opacity>=1.0f || (opacity>0.0f && rand(seed)<opacity))
      {
        material=evaluateShadingMaterial(hit.materialIndex,ctx);
        if(material.material.surface.z==1.0f && !hit.frontFacing) material.material.emission.xyz=float3(0.0f);
        RAY_PROFILE_ADD(RAY_PROFILE_ACCEPTED_SURFACE_HITS);
        return true;
      }
      RAY_PROFILE_ADD(RAY_PROFILE_ALPHA_REJECTED_HITS);
      ray.TMin=asfloat(max(asuint(hit.distance)+1u,0x00800000u));
      if(ray.TMin>ray.TMax) return false;
      continue;
    }
    material=evaluateMaterialProgram(hit.materialIndex,ctx);
    if(material.material.surface.z==1.0f && !hit.frontFacing) material.material.emission.xyz=float3(0.0f);
    float opacity=materialOpacity(material.material);
    if(opacity>=1.0f || (opacity>0.0f && rand(seed)<opacity))
    {
      RAY_PROFILE_ADD(RAY_PROFILE_ACCEPTED_SURFACE_HITS);
      return true;
    }
    RAY_PROFILE_ADD(RAY_PROFILE_ALPHA_REJECTED_HITS);
    ray.TMin=asfloat(max(asuint(hit.distance)+1u,0x00800000u));
    if(ray.TMin>ray.TMax) return false;
  }
}

float3 visible(float3 origin, float3 target,float coneWidth,float coneSpread,MediumStack media,uint source)
{
  float3 delta = target - origin;
  float distance = length(delta);
  if(!(distance>0.0f) || isinf(distance)) return float3(0.0f);
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = delta / distance;
  ray.TMin = 0.0f;
  // Both endpoints have reconstruction-sized normal clearance. Keep
  // real short segments instead of subtracting a global shadow epsilon.
  ray.TMax = mediumResumeDistance(distance);
  float3 transmittance=float3(1.0f);
  float initialWidth=coneWidth;
  float previousDistance=0.0f;
  uint nearContacts=0;
  MediumBoundaryProgress previousBoundary=(MediumBoundaryProgress)0;
  for(;;)
  {
    SurfaceHit hit;
    bool found=traceGeometry(ray,hit,previousBoundary.active?previousBoundary.instanceIndex:0xffffffffu,previousBoundary.processedDistance);
    GpuMaterial material=(GpuMaterial)0;
    if(found)
    {
      coneWidth=initialWidth+hit.distance*coneSpread;
      // Only homogeneous materials whose initialized table proves that they
      // have no interior may use the coverage-only shadow path. Heterogeneous
      // media keep per-hit evaluation even when the calibration point is empty.
      bool coverageOnly=optimizedMaterialCoverage() &&
        materialMediumTable[hit.materialIndex*8].x==0 && (materials[hit.materialIndex].program.w&1u)==0;
      if(coverageOnly)
      {
        float opacity=evaluateMaterialCoverage(hit.materialIndex,materialContext(hit,coneWidth));
        transmittance*=mediumRayTransmittance(media,fma(ray.Direction,float3(previousDistance),ray.Origin),
          ray.Direction,max(0.0f,hit.distance-previousDistance));
        transmittance*=1.0f-opacity;
        if(!any(transmittance>0.0f)) return float3(0.0f);
        RAY_PROFILE_ADD(RAY_PROFILE_SHADOW_TRANSPARENT_HITS);
        previousDistance=hit.distance;
        ray.TMin=asfloat(max(asuint(hit.distance)+1u,0x00800000u));
        if(ray.TMin>ray.TMax) return transmittance;
        continue;
      }
      material=evaluateShadingMaterial(hit.materialIndex,materialContext(hit,coneWidth)).material;
      if(material.volume.z==1.0f && repeatedMediumBoundary(previousBoundary,hit))
      {
        previousBoundary.processedDistance=max(previousBoundary.processedDistance,hit.distance);
        ray.TMin=mediumResumeDistance(previousBoundary.processedDistance);
        if(ray.TMin>ray.TMax) return transmittance;
        continue;
      }
    }
    if(found && skipNearMediumBoundary(media,hit,material,max(0.0f,hit.distance-previousDistance)))
    {
      if(++nearContacts > MEDIUM_CONTACT_LIMIT) { mediumError(2); return float3(0.0f); }
      previousBoundary=recordMediumBoundary(ray,hit);
      previousBoundary.planeActive=false;
      previousDistance=hit.distance;
      ray.TMin=mediumResumeDistance(hit.distance);
      if(ray.TMin>ray.TMax) return transmittance;
      continue;
    }
    float endpoint=found?hit.distance:ray.TMax;
    transmittance*=mediumRayTransmittance(media,fma(ray.Direction,float3(previousDistance),ray.Origin),ray.Direction,max(0.0f,endpoint-previousDistance));
    if(!found) return transmittance;
    bool tangent=material.volume.z!=1.0f && materialHasInterior(material) &&
      mediumBoundaryTangent(ray.Origin,ray.Direction,hit);
    if(material.volume.z==1.0f || tangent)
    {
      uint previousCount=media.count;
      if(!tangent && !mediumCross(media,hit,material,source,ray.Origin,ray.Direction)) return float3(0.0f);
      previousBoundary=recordMediumBoundary(ray,hit);
      previousBoundary.planeActive=media.count!=previousCount;
    }
    else
    {
      transmittance*=1.0f-materialOpacity(material);
#if CAUSTRA_PROFILE_RAYS
      if(any(transmittance>0.0f)) { RAY_PROFILE_ADD(RAY_PROFILE_SHADOW_TRANSPARENT_HITS); }
#endif
    }
    if(!any(transmittance>0.0f)) return float3(0.0f);
    previousDistance=hit.distance;
    ray.TMin=material.volume.z==1.0f || tangent?mediumResumeDistance(hit.distance):asfloat(max(asuint(hit.distance)+1u,0x00800000u));
    if(ray.TMin>ray.TMax) return transmittance;
  }
}

// The CPU proposal is area * constant emission luminance, with a positive
// fallback for graph emission and one ticket per triangle. Use the quantized
// CDF's actual probability in both MIS paths, including that small-light floor.
float emitterPdf(uint lightIndex,float distanceSquared, float cosine)
{
  if(lightIndex>=pc.counts.x || pc.params.x<=0.0f || cosine<=0.0f) return 0.0f;
  float previous=lightIndex>0?lights[lightIndex-1].edge1.w:0.0f;
  float probability=(lights[lightIndex].edge1.w-previous)/pc.params.x;
  float pdf=(probability/lights[lightIndex].v0.w)*(distanceSquared/cosine);
  return positivePdf(pdf) ? pdf : 0.0f;
}

uint selectLight(inout uint seed)
{
  // Direct 24-bit integer tickets are uniform, exact in float, and never hit
  // the CDF's upper endpoint. rand() can round its uint conversion to 1.0.
  float ticket=float(pcg(seed)>>8);
  uint lo = 0, hi = pc.counts.x - 1;
  while(lo < hi)
  {
    uint mid = lo + (hi - lo) / 2;
    if(ticket < lights[mid].edge1.w) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

float3 sampleDirect(SurfaceHit hit, SurfaceClosure closure, float3 view, float footprint, float coneSpread,
                    inout uint seed,MediumStack media,bool phaseEvent,MediumCoefficients medium)
{
  if(pc.counts.x == 0 || pc.params.x <= 0.0f) return float3(0.0f);
  uint lightIndex=selectLight(seed);
  GpuLight light = lights[lightIndex];
  float u = sqrt(rand(seed));
  float v = rand(seed);
  Vertex a=vertices[light.source.x],b=vertices[light.source.y],c=vertices[light.source.z];
  GpuInstance lightInstance=instances[asuint(light.normal0.w)];
  float3x4 lightObjectToWorld=float3x4(lightInstance.objectToWorld[0],lightInstance.objectToWorld[1],lightInstance.objectToWorld[2]);
  float3x4 lightWorldToObject=float3x4(lightInstance.worldToObject[0],lightInstance.worldToObject[1],lightInstance.worldToObject[2]);
  precise float3 lightObjectPosition=triangleObjectPosition(a.position.xyz,b.position.xyz,c.position.xyz,float2(u*(1.0f-v),u*v));
  precise float3 lightPosition=triangleWorldPosition(lightObjectPosition,lightObjectToWorld);
  float3 delta = lightPosition - hit.position;
  float distanceSquared = dot(delta, delta);
  if(!(distanceSquared>0.0f) || !finite3(delta)) return float3(0.0f);
  float3 direction = delta * rsqrt(distanceSquared);
  float3 lightNormal = safeNormal(cross(light.edge1.xyz, light.edge2.xyz)*light.edge2.w, -direction);
  bool lightFront=dot(lightNormal,-direction)>0.0f;
  float lightPdf = emitterPdf(lightIndex,distanceSquared, abs(dot(lightNormal, -direction)));
  if(!positivePdf(lightPdf)) return float3(0.0f);

  SurfaceEval eval;
  if(phaseEvent)
  {
    eval.fCos=mediumPhase(medium,-view,direction);
    eval.pdf=mediumPhasePdf(medium,-view,direction);
  }
  else eval=evaluateSurface(closure,view,direction);
  float3 value=eval.fCos;
  if(!finite3(value) || !any(value > 0.0f)) return float3(0.0f);
  if(dot(lightNormal, -direction) < 0.0f) lightNormal = -lightNormal;
  float3 origin=hit.position;
  if(!phaseEvent)
  {
    bool across=dot(direction,hit.geometricNormal)<0.0f;
    origin=rayOrigin(hit,across?-hit.geometricNormal:hit.geometricNormal);
    if(across)
    {
      // Closure material has already received surface-only color adjustments;
      // interior calibration takes the original graph parameters exactly once.
      GpuMaterial boundary=evaluateShadingMaterial(hit.materialIndex,materialContext(hit,footprint)).material;
      if(!mediumCross(media,hit,boundary,2,origin,direction)) return float3(0.0f);
    }
  }
  SurfaceHit lightHit=(SurfaceHit)0;
  lightHit.position=lightPosition;lightHit.geometricNormal=lightNormal;
  lightHit.positionError=trianglePositionError(a.position.xyz,b.position.xyz,c.position.xyz,lightObjectPosition,
    lightPosition,lightNormal,lightObjectToWorld,lightWorldToObject);
#if CAUSTRA_PROFILE_RAYS
  uint previousQueryKind=rayProfileQueryKind;
  rayProfileQueryKind=RAY_PROFILE_SHADOW_QUERIES;
#endif
  float3 visibility=visible(origin,rayOrigin(lightHit,lightNormal),footprint,coneSpread,media,phaseEvent?1:5);
#if CAUSTRA_PROFILE_RAYS
  rayProfileQueryKind=previousQueryKind;
#endif
  if(!any(visibility>0.0f)) return float3(0.0f);
  float3 bary=float3(1.0f-u,u*(1.0f-v),u*v);
  lightHit.shadingNormal=safeNormal(light.normal0.xyz*bary.x+light.normal1.xyz*bary.y+light.normal2.xyz*bary.z,lightNormal);
  if(dot(lightHit.shadingNormal,lightNormal)<0.0f) lightHit.shadingNormal=-lightHit.shadingNormal;
  if(dot(reflect(direction,lightHit.shadingNormal),lightNormal)<=0.0f) lightHit.shadingNormal=lightNormal;
  lightHit.objectPosition=lightObjectPosition;
  lightHit.texcoord=a.texcoord.xy*bary.x+b.texcoord.xy*bary.y+c.texcoord.xy*bary.z;
  float4 tangent=light.tangent0*bary.x+light.tangent1*bary.y+light.tangent2*bary.z;
  surfaceFrame(lightHit,a,b,c,light.edge1.xyz,light.edge2.xyz,tangent.xyz,tangent.w);
  float lightFootprint=footprint+sqrt(distanceSquared)*coneSpread;
  MaterialEvaluation emitter=evaluateShadingMaterial(light.source.w,materialContext(lightHit,lightFootprint));
  if(emitter.material.surface.z==1.0f && !lightFront) return float3(0.0f);
  SurfaceClosure emitterClosure=makeSurfaceClosure(emitter,materialContext(lightHit,lightFootprint),-direction);
  float3 emission=surfaceEmission(emitterClosure,-direction)*emitterClosure.opacity;
  return visibility*value*emission*(balanceWeight(lightPdf,eval.pdf)/lightPdf);
}

float3 pathTrace(RayDesc ray, inout uint seed, out DenoiserGuides guides)
{
  guides = (DenoiserGuides)0;
  guides.roughness = 1.0f;
  guides.linearDepth = GUIDE_FAR_DISTANCE;
  bool dlssEnabled = pc.counts.y == 1;
  if(dlssEnabled) guides.motion = cameraMotion(float4(ray.Direction, 0.0f));
  bool captureSpecularHit = false, previousDelta = true;
  float3 radiance = float3(0.0f), throughput = float3(1.0f);
  float3 previousPosition = ray.Origin;
  float previousPdf = 0.0f;
  float coneWidth = 0.0f, coneSpread = length(pc.vertical.xyz) / float(pc.dimensions.y);
  float segmentBegin = 0.0f;
  float3 segmentStartPosition = ray.Origin;
  uint bounce = 0, hero = MEDIUM_NO_HERO;
  uint nearContacts = 0;
  MediumStack media = (MediumStack)0;
  MediumBoundaryProgress previousBoundary = (MediumBoundaryProgress)0;
  if(pc.environment.w > 0.0f && !loadCameraMediumStack(seed, hero, media)) return float3(0.0f);
  if(hero < 3) { throughput = float3(0.0f); throughput[hero] = 3.0f; }

  // Only scattering events consume the integration depth. Invisible medium
  // interfaces and coverage-zero layers preserve the exact original ray line.
  for(;;)
  {
#if CAUSTRA_PROFILE_RAYS
    rayProfileQueryKind=bounce==0?RAY_PROFILE_PRIMARY_QUERIES:RAY_PROFILE_INDIRECT_QUERIES;
#endif
    RAY_PROFILE_DEPTH(bounce);
    SurfaceHit hit;
    MaterialEvaluation material;
    float hitConeWidth = coneWidth;
    bool found = traceSurface(ray, hitConeWidth, coneSpread, seed, hit, material,
      previousBoundary.active?previousBoundary.instanceIndex:0xffffffffu,previousBoundary.processedDistance);
    if(found && material.material.volume.z == 1.0f && repeatedMediumBoundary(previousBoundary, hit))
    {
      // Filter the processed contact explicitly, leaving even its immediately
      // following representable ray parameter available to a real thin exit.
      previousBoundary.processedDistance=max(previousBoundary.processedDistance,hit.distance);
      ray.TMin=mediumResumeDistance(previousBoundary.processedDistance);
      continue;
    }
    if(found && skipNearMediumBoundary(media,hit,material.material,max(0.0f,hit.distance-segmentBegin)))
    {
      if(++nearContacts > MEDIUM_CONTACT_LIMIT) { mediumError(2); break; }
      previousBoundary=recordMediumBoundary(ray,hit);
      previousBoundary.planeActive=false;
      segmentBegin=hit.distance;
      segmentStartPosition=hit.position;
      ray.TMin=mediumResumeDistance(hit.distance);
      continue;
    }
    float endpoint = found ? hit.distance : FAR_DISTANCE;
    float segmentLength = max(endpoint - segmentBegin, 0.0f);
    MediumCoefficients medium;
    float3 mediumSource;
    MediumFlight flight = sampleCurrentMedium(media,segmentStartPosition,ray.Direction,segmentLength,throughput,seed,medium,mediumSource);
    radiance += throughput * mediumSource;
    throughput *= flight.weight;
    if(!finite3(throughput) || !finite3(radiance)) { mediumError(4); break; }
    if(!any(throughput > 0.0f)) break;

    if(flight.scattered)
    {
      if(bounce == pc.dimensions.w) break;
      float distance = segmentBegin + flight.distance;
      float3 position = mediumFlightPoint(ray, segmentStartPosition, previousBoundary,
        hit, found, flight.distance, segmentLength);
      float scatterCone = coneWidth + distance * coneSpread;
      float scatterSpread = min(1.0f, coneSpread + 0.25f);
      SurfaceHit phaseHit = (SurfaceHit)0;
      phaseHit.position = position;
      radiance += throughput * sampleDirect(phaseHit, (SurfaceClosure)0, -ray.Direction,
        scatterCone, scatterSpread, seed, media, true, medium);
      float phasePdf;
      float3 phaseWeight;
      float3 direction = sampleMediumPhase(medium, ray.Direction, seed, phasePdf, phaseWeight);
      throughput *= phaseWeight;
      previousPosition = position; previousPdf = phasePdf; previousDelta = false;
      if(dlssEnabled && bounce == 0)
      {
        float3 forward = normalize(pc.lowerLeft.xyz + 0.5f * pc.horizontal.xyz + 0.5f * pc.vertical.xyz);
        guides.linearDepth = min(dot(position - pc.origin.xyz, forward), GUIDE_FAR_DISTANCE);
        guides.motion = cameraMotion(float4(position, 1.0f));
      }
      captureSpecularHit = false;
      ++bounce;
      RAY_PROFILE_ADD(RAY_PROFILE_MEDIUM_SCATTERS);
      RAY_PROFILE_DEPTH(bounce);
      if(bounce >= 3)
      {
        float survival = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05f, 0.95f);
        if(rand(seed) >= survival)
        {
          RAY_PROFILE_ADD(RAY_PROFILE_RUSSIAN_ROULETTE_TERMINATIONS);
          break;
        }
        throughput /= survival;
      }
      ray.Origin = position; ray.Direction = direction; ray.TMin = 0.0f; ray.TMax = FAR_DISTANCE;
      segmentStartPosition = position;
      previousBoundary.active = false;
      segmentBegin = 0.0f; coneWidth = scatterCone; coneSpread = scatterSpread;
      continue;
    }

    if(!found)
    {
      if(captureSpecularHit) guides.specularHitDistance = GUIDE_FAR_DISTANCE;
      radiance += throughput * pc.environment.xyz;
      break;
    }
    MaterialContext ctx = materialContext(hit, hitConeWidth);
    // Both null volumes and solid glass/SSS meshes can have zero-length edge
    // contacts. Reject a proven contact before selecting an optical interface
    // or evaluating a BSDF, rather than treating its lone exit as refraction.
    bool tangent=material.material.volume.z!=1.0f && materialHasInterior(material.material) &&
      mediumBoundaryTangent(ray.Origin,ray.Direction,hit);
    if(material.material.volume.z == 1.0f || tangent)
    {
      uint previousCount=media.count;
      if(!tangent && !mediumCross(media, hit, material.material,3,ray.Origin,ray.Direction)) break;
      previousBoundary = recordMediumBoundary(ray, hit);
      previousBoundary.planeActive=media.count!=previousCount;
      segmentBegin = hit.distance;
      segmentStartPosition = hit.position;
      ray.TMin=mediumResumeDistance(hit.distance);
      continue;
    }
    if(hero == MEDIUM_NO_HERO && material.material.thinFilm.w > 0.0f && material.material.program.z!=3)
    {
      hero = min(uint(rand(seed) * 3.0f), 2);
      float selected = throughput[hero] * 3.0f;
      throughput = float3(0.0f); throughput[hero] = selected;
    }
    SurfaceClosure closure = makeSurfaceClosure(material, ctx, -ray.Direction);
    float etaI = currentMediumIor(media, hero);
    float etaT = materialHasInterior(material.material) && material.material.subsurface.w == 0.0f ?
      destinationMediumIor(media, hit, hero) : materialInteriorIor(material.material, hero);
    configureSurfaceInterface(closure, etaI, etaT);
    closure.entering = hit.frontFacing;
    SurfaceGuides features = surfaceGuides(closure, -ray.Direction);
    if(captureSpecularHit)
    {
      guides.specularHitDistance = min(hit.distance, GUIDE_FAR_DISTANCE);
      captureSpecularHit = false;
    }
    if(dlssEnabled && bounce == 0)
    {
      guides.diffuseAlbedo = features.diffuseAlbedo; guides.specularAlbedo = features.specularAlbedo;
      guides.roughness = features.roughness; guides.normal = features.normal;
      float3 forward = normalize(pc.lowerLeft.xyz + 0.5f * pc.horizontal.xyz + 0.5f * pc.vertical.xyz);
      guides.linearDepth = clamp(dot(hit.position - pc.origin.xyz, forward), pc.params.y, GUIDE_FAR_DISTANCE);
      guides.motion = cameraMotion(float4(hit.position, 1.0f));
    }
    if(any(closure.emission > 0.0f) && (material.material.surface.z != 1.0f || hit.frontFacing))
    {
      float weight = 1.0f;
      if(bounce > 0 && !previousDelta)
      {
        float3 delta = hit.position - previousPosition;
        uint firstLight=asuint(instances[hit.instanceIndex].objectMin.w);
        uint lightIndex=firstLight==0xffffffffu?firstLight:firstLight+hit.primitiveIndex;
        float lightPdf = emitterPdf(lightIndex,dot(delta, delta), abs(dot(hit.geometricNormal, -ray.Direction)));
        weight = balanceWeight(previousPdf, lightPdf);
      }
      radiance += throughput * surfaceEmission(closure, -ray.Direction) * weight;
    }
    if(bounce == pc.dimensions.w) break;

    float scatteredSpread = min(1.0f, coneSpread + features.roughness * features.roughness * 0.25f);
    radiance += throughput * sampleDirect(hit, closure, -ray.Direction, hitConeWidth,
      scatteredSpread, seed, media, false, medium);
    SurfaceSample sample = sampleSurface(closure, -ray.Direction, float3(rand(seed), rand(seed), rand(seed)));
    if(sample.event == 0 || !positivePdf(sample.pdf) || !finite3(sample.weight)) break;
    bool across = dot(sample.direction, hit.geometricNormal) < 0.0f;
    if(across && !mediumCross(media, hit, material.material,4,ray.Origin,ray.Direction)) break;
    throughput *= sample.weight;
    if(!finite3(throughput)) { mediumError(4); break; }
    float maximum = max(throughput.x, max(throughput.y, throughput.z));
    if(maximum <= 0.0f) break;
    ++bounce;
    RAY_PROFILE_ADD(RAY_PROFILE_SURFACE_SCATTERS);
    RAY_PROFILE_DEPTH(bounce);
    if(bounce >= 3)
    {
      float survival = clamp(maximum, 0.05f, 0.95f);
      if(rand(seed) >= survival)
      {
        RAY_PROFILE_ADD(RAY_PROFILE_RUSSIAN_ROULETTE_TERMINATIONS);
        break;
      }
      throughput /= survival;
    }
    previousPosition = hit.position; previousPdf = sample.pdf; previousDelta = sample.delta;
    // NGX derives reflection motion from this distance. Transmission and SSS
    // exits are not reflected geometry and must not populate that guide.
    if(dlssEnabled && bounce == 1) captureSpecularHit = sample.event == 2;
    coneWidth = hitConeWidth; coneSpread = scatteredSpread; segmentBegin = 0.0f;
    ray.Origin = rayOrigin(hit, across ? -hit.geometricNormal : hit.geometricNormal);
    segmentStartPosition = ray.Origin;
    ray.Direction = normalize(sample.direction); ray.TMin = 0.0f; ray.TMax = FAR_DISTANCE;
    previousBoundary.active = false;
  }
  return finite3(radiance) ? max(radiance, float3(0.0f)) : float3(0.0f);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 thread : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
  if((pc.counts.w & INIT_HETEROGENEOUS_MEDIUM) != 0)
  {
    initializeHeterogeneousMedium(pc.dimensions.x,groupID.x*64+groupIndex);
    return;
  }
  if((pc.counts.w & INIT_MATERIAL_MEDIUM) != 0)
  {
    // Only this initialization mode uses dimensions.x as the dispatch X group
    // count. A two-dimensional grid supports the device's dispatch limits.
    uint materialIndex = (groupID.x + groupID.y * pc.dimensions.x) * 64 + groupIndex;
    initializeMaterialMedium(materialIndex);
    return;
  }
  if((pc.counts.w & INIT_CAMERA_MEDIUM) != 0)
  {
    if(all(thread == uint3(0))) initializeCameraMediumCache();
    return;
  }
  if(thread.x >= pc.dimensions.x || thread.y >= pc.dimensions.y) return;
  uint index = thread.y * pc.dimensions.x + thread.x;
#if CAUSTRA_PROFILE_RAYS
  rayProfileActive=true;
  rayProfileQueryKind=RAY_PROFILE_AUXILIARY_QUERIES;
  rayProfileOverflow=0;
  for(uint counter=0;counter<RAY_PROFILE_COUNTER_COUNT;++counter) rayProfileCounters[counter]=0;
  rayProfileCounters[RAY_PROFILE_PATH_SAMPLES]=1;
#endif
  uint seed = xxhash32(uint3(thread.xy, pc.counts.z));
  bool dlssEnabled = pc.counts.y == 1;
  // A single frame-wide Halton offset permits DLSS temporal reconstruction.
  // Independent per-pixel stochastic primary jitter remains the raw mode.
  float2 jitter = dlssEnabled ? float2(0.5f) + pc.params.zw : float2(rand(seed), rand(seed));
  float2 uv = (float2(thread.xy) + jitter) / float2(pc.dimensions.xy);
  // The film vertical vector points up; the first output row is the top row.
  // lowerLeft is a direction relative to the origin. Keeping the camera
  // translation out of film arithmetic preserves precision far from zero.
  float3 filmDirection = pc.lowerLeft.xyz + uv.x * pc.horizontal.xyz + (1.0f - uv.y) * pc.vertical.xyz;
  RayDesc ray;
  ray.Origin = pc.origin.xyz;
  ray.Direction = normalize(filmDirection);
  ray.TMin = 0.0f;
  ray.TMax = FAR_DISTANCE;
  DenoiserGuides guides;
  float3 sample = pathTrace(ray, seed, guides);
  if(dlssEnabled) writeDenoiserGuides(index, sample, guides);
  float3 average = sample;
  if(pc.dimensions.z > 0)
  {
    float3 previous = accumulation[index].xyz;
    average = previous + (sample - previous) / (float(pc.dimensions.z) + 1.0f);
  }
  accumulation[index] = float4(average, 1.0f);
#if CAUSTRA_PROFILE_RAYS
  flushRayProfile(index);
#endif
}
