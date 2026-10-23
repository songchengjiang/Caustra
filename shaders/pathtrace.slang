// SPDX-License-Identifier: Apache-2.0
// Rendering algorithm adapted from vk_gltf_renderer,
// Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
// Modified for a standalone scene API; see NOTICE for provenance.
// Caustra's standalone path tracing integrator. The BSDF, RNG and ray offset
// helpers are the unmodified nvpro_core2/nvshaders implementations used by
// vk_gltf_renderer. No glTF scene structures, texture access or loader remain.
#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/ray_utils.h.slang"

// These storage-buffer and push-constant layouts match renderer.hpp and
// shader_types.hpp. Keep float4/uint4 boundaries when extending this interface.
struct Vertex { float4 position; float4 normal; };
struct Material { float4 baseColor; float4 emission; float4 surface; };
struct GpuInstance { uint4 offsets; };
struct GpuLight { float4 v0; float4 edge1; float4 edge2; float4 emission; };
struct PushConstants
{
  float4 origin, lowerLeft, horizontal, vertical;
  float4 environment;
  uint4 dimensions;
  uint4 counts;
  float4 params;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(1, 0)]] StructuredBuffer<Vertex> vertices;
[[vk::binding(2, 0)]] StructuredBuffer<uint> indices;
[[vk::binding(3, 0)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(4, 0)]] StructuredBuffer<Material> materials;
[[vk::binding(5, 0)]] StructuredBuffer<GpuLight> lights;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> accumulation;
// Optional DLSS RR input planes; byte offsets match renderer.cpp. Raw
// progressive radiance stays in accumulation, independently of DLSS history.
[[vk::binding(7, 0)]] RWByteAddressBuffer guideBuffer;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const float FAR_DISTANCE = 1.0e30f;
static const float GUIDE_FAR_DISTANCE = 1.0e4f;

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
  uint materialIndex;
};

struct DenoiserGuides
{
  float3 diffuseAlbedo;
  float3 specularAlbedo;
  float3 normal;
  float roughness;
  float linearDepth;
  float specularHitDistance;
};

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
  // The synchronous API resets history on every camera or scene update.
  // Between resets the unjittered image is static, so pixel motion is zero.
  guideBuffer.Store2(64 * pixels + 8 * index, asuint(float2(0.0f)));
  guideBuffer.Store(72 * pixels + 4 * index, asuint(guides.linearDepth));
  guideBuffer.Store(76 * pixels + 4 * index, asuint(guides.specularHitDistance));
}

bool traceSurface(RayDesc ray, out SurfaceHit hit)
{
  hit = (SurfaceHit)0;
  RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
  query.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xff, ray);
  while(query.Proceed()) {}
  if(query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return false;

  GpuInstance instance = instances[query.CommittedInstanceID()];
  uint first = instance.offsets.y + query.CommittedPrimitiveIndex() * 3;
  Vertex a = vertices[instance.offsets.x + indices[first]];
  Vertex b = vertices[instance.offsets.x + indices[first + 1]];
  Vertex c = vertices[instance.offsets.x + indices[first + 2]];
  float2 uv = query.CommittedTriangleBarycentrics();
  float3 bary = float3(1.0f - uv.x - uv.y, uv.x, uv.y);
  float3x4 objectToWorld = query.CommittedObjectToWorld3x4();
  float3x4 worldToObject = query.CommittedWorldToObject3x4();

  float3 objectPosition = a.position.xyz * bary.x + b.position.xyz * bary.y + c.position.xyz * bary.z;
  hit.position = mul(objectToWorld, float4(objectPosition, 1.0f));
  // Deriving the geometric normal from world-space edges also handles mirrored
  // instances. The interpolated normal needs the inverse transpose under scale.
  float3 edge1 = mul(objectToWorld, float4(b.position.xyz - a.position.xyz, 0.0f));
  float3 edge2 = mul(objectToWorld, float4(c.position.xyz - a.position.xyz, 0.0f));
  hit.geometricNormal = safeNormal(cross(edge1, edge2), -ray.Direction);
  float3 n = a.normal.xyz * bary.x + b.normal.xyz * bary.y + c.normal.xyz * bary.z;
  float3 worldNormal = n.x * worldToObject[0].xyz + n.y * worldToObject[1].xyz + n.z * worldToObject[2].xyz;
  hit.shadingNormal = safeNormal(worldNormal, hit.geometricNormal);

  // Phase 1 materials and area emitters are opaque and two-sided.
  if(dot(hit.geometricNormal, ray.Direction) > 0.0f) hit.geometricNormal = -hit.geometricNormal;
  if(dot(hit.shadingNormal, hit.geometricNormal) < 0.0f) hit.shadingNormal = -hit.shadingNormal;
  // Prevent the reflection lobe of a strongly interpolated normal from sending
  // the incoming ray through the geometric surface.
  if(dot(reflect(ray.Direction, hit.shadingNormal), hit.geometricNormal) <= 0.0f)
    hit.shadingNormal = hit.geometricNormal;
  hit.materialIndex = instance.offsets.z;
  return true;
}

float3 rayOrigin(float3 position, float3 normal)
{
  // NVIDIA's relative offset plus an absolute floor close to the world origin.
  return offsetRay(position, normal) + normal * pc.params.y;
}

bool visible(float3 origin, float3 target)
{
  float3 delta = target - origin;
  float distance = length(delta);
  if(distance <= pc.params.y) return false;
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = delta / distance;
  ray.TMin = 0.0f;
  ray.TMax = max(0.0f, distance - pc.params.y);
  RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
  query.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xff, ray);
  while(query.Proceed()) {}
  return query.CommittedStatus() == COMMITTED_NOTHING;
}

// Each triangle is selected in proportion to world area, then sampled uniformly
// on its surface. The combined area PDF is therefore 1 / totalArea.
float emitterPdf(float distanceSquared, float cosine)
{
  if(pc.counts.x == 0 || pc.params.x <= 0.0f || cosine <= 0.0f) return 0.0f;
  float pdf = distanceSquared / (pc.params.x * cosine);
  return positivePdf(pdf) ? pdf : 0.0f;
}

uint selectLight(float randomValue)
{
  float area = randomValue * pc.params.x;
  uint lo = 0, hi = pc.counts.x - 1;
  while(lo < hi)
  {
    uint mid = lo + (hi - lo) / 2;
    if(area < lights[mid].edge1.w) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

float3 sampleDirect(SurfaceHit hit, PbrMaterial pbr, float3 view, inout uint seed)
{
  if(pc.counts.x == 0 || pc.params.x <= 0.0f) return float3(0.0f);
  GpuLight light = lights[selectLight(rand(seed))];
  float u = sqrt(rand(seed));
  float v = rand(seed);
  float3 lightPosition = light.v0.xyz + light.edge1.xyz * (u * (1.0f - v)) + light.edge2.xyz * (u * v);
  float3 delta = lightPosition - hit.position;
  float distanceSquared = dot(delta, delta);
  if(distanceSquared <= pc.params.y * pc.params.y || !finite3(delta)) return float3(0.0f);
  float3 direction = delta * rsqrt(distanceSquared);
  if(dot(direction, hit.geometricNormal) <= 0.0f) return float3(0.0f);
  float3 lightNormal = safeNormal(cross(light.edge1.xyz, light.edge2.xyz), -direction);
  float lightPdf = emitterPdf(distanceSquared, abs(dot(lightNormal, -direction)));
  if(!positivePdf(lightPdf)) return float3(0.0f);

  BsdfEvaluateData eval = (BsdfEvaluateData)0;
  eval.k1 = view;
  eval.k2 = direction;
  bsdfEvaluateSimple(eval, pbr); // includes the surface cosine
  float3 value = eval.bsdf_diffuse + eval.bsdf_glossy;
  if(!finite3(value) || !any(value > 0.0f)) return float3(0.0f);
  if(dot(lightNormal, -direction) < 0.0f) lightNormal = -lightNormal;
  if(!visible(rayOrigin(hit.position, hit.geometricNormal), rayOrigin(lightPosition, lightNormal))) return float3(0.0f);
  return value * light.emission.xyz * (balanceWeight(lightPdf, eval.pdf) / lightPdf);
}

float3 pathTrace(RayDesc ray, inout uint seed, out DenoiserGuides guides)
{
  guides = (DenoiserGuides)0;
  guides.roughness = 1.0f;
  guides.linearDepth = GUIDE_FAR_DISTANCE;
  bool dlssEnabled = pc.counts.y == 1;
  bool captureSpecularHit = false;
  float3 radiance = float3(0.0f);
  float3 throughput = float3(1.0f);
  float3 previousPosition = ray.Origin;
  float previousPdf = 0.0f;

  // maxBounces counts scattering events. The final trace still collects an
  // emitter or environment endpoint, so NEE and BSDF MIS remain paired at depth.
  for(uint bounce = 0; bounce <= pc.dimensions.w; ++bounce)
  {
    SurfaceHit hit;
    if(!traceSurface(ray, hit))
    {
      if(captureSpecularHit) guides.specularHitDistance = GUIDE_FAR_DISTANCE;
      // The constant environment is sampled only by BSDF paths, so weight = 1.
      radiance += throughput * pc.environment.xyz;
      break;
    }
    Material material = materials[hit.materialIndex];
    if(captureSpecularHit)
    {
      guides.specularHitDistance = min(length(hit.position - ray.Origin), GUIDE_FAR_DISTANCE);
      captureSpecularHit = false;
    }
    if(dlssEnabled && bounce == 0)
    {
      // Capture the same jittered primary surface that produced this frame's
      // noisy color. Metal has no diffuse lobe; its tint belongs in F0.
      float metallic = saturate(material.surface.x);
      guides.diffuseAlbedo = saturate(material.baseColor.xyz) * (1.0f - metallic);
      guides.roughness = clamp(material.surface.y, 0.02f, 1.0f);
      guides.normal = hit.shadingNormal;
      float3 f0 = lerp(float3(0.04f), saturate(material.baseColor.xyz), metallic);
      guides.specularAlbedo = specularGuideAlbedo(f0, guides.roughness * guides.roughness,
                                                 saturate(dot(hit.shadingNormal, -ray.Direction)));
      float3 viewForward = normalize(pc.lowerLeft.xyz + 0.5f * pc.horizontal.xyz + 0.5f * pc.vertical.xyz);
      guides.linearDepth = clamp(dot(hit.position - pc.origin.xyz, viewForward), pc.params.y, GUIDE_FAR_DISTANCE);
    }
    if(any(material.emission.xyz > 0.0f))
    {
      float weight = 1.0f;
      if(bounce > 0)
      {
        float3 delta = hit.position - previousPosition;
        float lightPdf = emitterPdf(dot(delta, delta), abs(dot(hit.geometricNormal, -ray.Direction)));
        weight = balanceWeight(previousPdf, lightPdf);
      }
      radiance += throughput * material.emission.xyz * weight;
    }
    if(bounce == pc.dimensions.w) break;

    // NV's constructor converts perceptual roughness to GGX alpha (squared).
    // A finite floor keeps the simple BSDF's GGX model away from singular PDFs.
    PbrMaterial pbr = defaultPbrMaterial(material.baseColor.xyz, material.surface.x,
                                        max(material.surface.y, 0.02f), hit.shadingNormal, hit.geometricNormal);
    radiance += throughput * sampleDirect(hit, pbr, -ray.Direction, seed);

    BsdfSampleData sample = (BsdfSampleData)0;
    sample.k1 = -ray.Direction;
    sample.xi = float3(rand(seed), rand(seed), rand(seed));
    bsdfSampleSimple(sample, pbr);
    if(sample.event_type == BSDF_EVENT_ABSORB || !positivePdf(sample.pdf)
       || !finite3(sample.bsdf_over_pdf) || dot(sample.k2, hit.geometricNormal) <= 0.0f) break;
    throughput *= sample.bsdf_over_pdf;
    float maximum = max(throughput.x, max(throughput.y, throughput.z));
    if(!finite3(throughput) || maximum <= 0.0f) break;

    // Russian roulette after three scattering events; the surviving throughput
    // compensates for its probability. Directional PDFs stay conditional on RR.
    if(bounce >= 2)
    {
      float survival = clamp(maximum, 0.05f, 0.95f);
      if(rand(seed) >= survival) break;
      throughput /= survival;
    }
    previousPosition = hit.position;
    previousPdf = sample.pdf;
    if(dlssEnabled && bounce == 0)
      captureSpecularHit = sample.event_type == BSDF_EVENT_GLOSSY_REFLECTION;
    ray.Origin = rayOrigin(hit.position, hit.geometricNormal);
    ray.Direction = normalize(sample.k2);
    ray.TMin = 0.0f;
    ray.TMax = FAR_DISTANCE;
  }
  return finite3(radiance) ? max(radiance, float3(0.0f)) : float3(0.0f);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 thread : SV_DispatchThreadID)
{
  if(thread.x >= pc.dimensions.x || thread.y >= pc.dimensions.y) return;
  uint index = thread.y * pc.dimensions.x + thread.x;
  uint seed = xxhash32(uint3(thread.xy, pc.dimensions.z));
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
}
