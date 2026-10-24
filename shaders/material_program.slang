// SPDX-License-Identifier: Apache-2.0
// Renderer-neutral, per-hit material graph evaluation. Layouts and opcodes match
// include/caustra/renderer.hpp; no raster stages, global shading state or ddx/ddy.
#ifndef CAUSTRA_MATERIAL_PROGRAM_SLANG
#define CAUSTRA_MATERIAL_PROGRAM_SLANG

#include "../third_party/materialx/mx_perlin.slang"

struct GpuMaterial
{
  float4 baseColor, emission, surface;
  float4 specular, specularColor;
  float4 coat, coatColor;
  float4 sheen, sheenColor;
  float4 factors;
  float4 transmission, transmissionColor, transmissionScatter;
  float4 subsurface, subsurfaceColor, subsurfaceRadius;
  float4 thinFilm, fuzz, fuzzColor;
  float4 volume, volumeAbsorption, volumeScattering, volumeEmission;
  uint4 program;
};
struct MaterialInstruction { uint4 code; float4 value; };
struct GpuTextureInfo { uint4 size; }; // width, height, first texel, mip count
struct MaterialContext
{
  float3 position, objectPosition, geometricNormal, normal, tangent, bitangent;
  float2 texcoord, uvFootprint;
};
struct MaterialEvaluation { GpuMaterial material; float3 normal, coatNormal, tangent, coatTangent; };

// Coverage is distinct from the transparent BSDF in the USD surface model.
// Apply alpha modes before traversal acceptance, shadowing and emission alike.
float materialOpacity(GpuMaterial m)
{
  float opacity=saturate(m.factors.z);
  if(m.program.z==3)
  {
    uint mode=uint(round(max(0.0f,m.fuzz.z)));
    return mode==0?1.0f:(mode==1?(opacity>=m.volume.w?1.0f:0.0f):opacity);
  }
  if(m.program.z==4)
    return m.fuzz.w>.5f && m.specular.w<=0.0f?opacity:(opacity>=m.specular.w?1.0f:0.0f);
  return opacity;
}

[[vk::binding(4, 0)]] StructuredBuffer<GpuMaterial> materials;
[[vk::binding(9, 0)]] StructuredBuffer<MaterialInstruction> materialInstructions;
[[vk::binding(10, 0)]] StructuredBuffer<GpuTextureInfo> textureInfos;
[[vk::binding(11, 0)]] StructuredBuffer<float4> texturePixels;

static const uint MATERIAL_REGISTER_COUNT = 128;

float3 materialSafeNormal(float3 v, float3 fallback)
{
  float d = dot(v, v);
  return d > 1.0e-20f && !isinf(d) && !isnan(d) ? v * rsqrt(d) : fallback;
}

// The host creates floor-halved mip dimensions, each at least 1, and stores all
// mip texels consecutively. Address each filter tap, including edge taps, so
// repeat/mirror/black behave correctly at seams with bilinear filtering.
int materialTextureAddress(int coordinate, int extent, uint mode)
{
  if(mode == 1) return clamp(coordinate, 0, extent - 1);
  if(mode == 3) return coordinate < 0 || coordinate >= extent ? -1 : coordinate;
  int period = mode == 2 ? 2 * extent : extent;
  int wrapped = coordinate % period;
  if(wrapped < 0) wrapped += period;
  return mode == 2 && wrapped >= extent ? period - 1 - wrapped : wrapped;
}

float materialWrapUv(float uv, uint mode)
{
  if(mode == 0) return frac(uv);
  if(mode == 1) return clamp(uv, 0.0f, 1.0f);
  if(mode == 2)
  {
    float phase = frac(uv * 0.5f) * 2.0f;
    return phase <= 1.0f ? phase : 2.0f - phase;
  }
  return uv;
}

float4 materialTextureTap(uint first, uint2 extent, int2 pixel, uint wrapU, uint wrapV)
{
  int x = materialTextureAddress(pixel.x, int(extent.x), wrapU);
  int y = materialTextureAddress(pixel.y, int(extent.y), wrapV);
  if(x < 0 || y < 0) return float4(0.0f);
  return texturePixels[first + uint(y) * extent.x + uint(x)];
}

float4 materialTextureLevel(GpuTextureInfo info, uint level, float2 uv, uint wrapU, uint wrapV, bool nearest)
{
  uint first = info.size.z;
  uint2 extent = info.size.xy;
  for(uint i = 0; i < level; ++i)
  {
    first += extent.x * extent.y;
    extent = max(extent / 2, uint2(1));
  }
  if(nearest)
    return materialTextureTap(first, extent, int2(floor(uv * float2(extent))), wrapU, wrapV);
  float2 position = uv * float2(extent) - 0.5f;
  int2 lo = int2(floor(position));
  float2 weight = frac(position);
  float4 a = materialTextureTap(first, extent, lo, wrapU, wrapV);
  float4 b = materialTextureTap(first, extent, lo + int2(1, 0), wrapU, wrapV);
  float4 c = materialTextureTap(first, extent, lo + int2(0, 1), wrapU, wrapV);
  float4 d = materialTextureTap(first, extent, lo + int2(1, 1), wrapU, wrapV);
  return lerp(lerp(a, b, weight.x), lerp(c, d, weight.x), weight.y);
}

float4 materialTextureSample(uint textureIndex, float2 uv, float2 footprint, uint flags, float lodBias)
{
  if(any(isnan(uv)) || any(isinf(uv))) return float4(0.0f);
  GpuTextureInfo info = textureInfos[textureIndex];
  uint wrapU = flags & 3;
  uint wrapV = (flags >> 2) & 3;
  bool nearest = (flags & 16) != 0;
  // Graph UVs have their origin at the bottom; uploaded texels are top-first.
  uv.y = 1.0f - uv.y;
  uv = float2(materialWrapUv(uv.x, wrapU), materialWrapUv(uv.y, wrapV));
  if((wrapU == 3 && (uv.x < -1.0f || uv.x > 2.0f))
     || (wrapV == 3 && (uv.y < -1.0f || uv.y > 2.0f))) return float4(0.0f);
  float rho = max(abs(footprint.x) * info.size.x, abs(footprint.y) * info.size.y);
  float lod = clamp(log2(max(rho, 1.0e-8f)) + lodBias, 0.0f, float(info.size.w - 1));
  if(isnan(lod)) lod = 0.0f;
  if(nearest)
    return materialTextureLevel(info, uint(floor(lod + 0.5f)), uv, wrapU, wrapV, true);
  uint lower = uint(floor(lod));
  uint upper = min(lower + 1, info.size.w - 1);
  return lerp(materialTextureLevel(info, lower, uv, wrapU, wrapV, false),
              materialTextureLevel(info, upper, uv, wrapU, wrapV, false), frac(lod));
}

#include "material_patterns.slang"

float materialVectorDot(float4 a, float4 b, uint lanes)
{
  if(lanes == 2) return dot(a.xy, b.xy);
  if(lanes == 4) return dot(a, b);
  if(lanes == 1) return a.x * b.x;
  return dot(a.xyz, b.xyz);
}

float4 materialVectorMask(float4 value, uint lanes)
{
  if(lanes == 1) return float4(value.x, 0, 0, 0);
  if(lanes == 2) return float4(value.xy, 0, 0);
  if(lanes == 4) return value;
  return float4(value.xyz, 0);
}

float3 materialNormalmap(float3 color, float strength, MaterialContext ctx)
{
  float3 n = materialSafeNormal(ctx.normal, ctx.geometricNormal);
  float3 fallback = abs(n.z) < 0.999f ? cross(float3(0, 0, 1), n) : cross(float3(0, 1, 0), n);
  float3 t = materialSafeNormal(ctx.tangent - n * dot(ctx.tangent, n), materialSafeNormal(fallback, float3(1, 0, 0)));
  float3 b = cross(n, t);
  if(dot(b, ctx.bitangent) < 0.0f) b = -b;
  float3 local = color * 2.0f - 1.0f;
  local.xy *= strength;
  return materialSafeNormal(t * local.x + b * local.y + n * local.z, n);
}

void materialWriteOutput(inout MaterialEvaluation evaluation, uint parameter, float4 value)
{
  switch(parameter)
  {
    case 0: evaluation.material.baseColor.xyz = value.xyz; break;
    case 1: evaluation.material.surface.x = value.x; break;
    case 2: evaluation.material.surface.y = value.x; break;
    case 3: evaluation.material.emission.xyz = value.xyz; break;
    case 4: evaluation.normal = materialSafeNormal(value.xyz, evaluation.normal); break;
    case 5: evaluation.material.factors.x = value.x; break;
    case 6: evaluation.material.factors.y = value.x; break;
    case 7: evaluation.material.specular.x = value.x; break;
    case 8: evaluation.material.specularColor.xyz = value.xyz; break;
    case 9: evaluation.material.specular.y = value.x; break;
    case 10: evaluation.material.specular.z = value.x; break;
    case 11: evaluation.material.specular.w = value.x; break;
    case 12: evaluation.material.coat.x = value.x; break;
    case 13: evaluation.material.coatColor.xyz = value.xyz; break;
    case 14: evaluation.material.coat.y = value.x; break;
    case 15: evaluation.material.coat.z = value.x; break;
    case 16: evaluation.coatNormal = materialSafeNormal(value.xyz, evaluation.coatNormal); break;
    case 17: evaluation.material.sheen.x = value.x; break;
    case 18: evaluation.material.sheenColor.xyz = value.xyz; break;
    case 19: evaluation.material.sheen.y = value.x; break;
    case 20: evaluation.material.factors.z = value.x; break;
    case 21: evaluation.tangent = materialSafeNormal(value.xyz, evaluation.tangent); break;
    case 22: evaluation.coatTangent = materialSafeNormal(value.xyz, evaluation.coatTangent); break;
    case 23: evaluation.material.coat.w = value.x; break;
    case 24: evaluation.material.factors.w = value.x; break;
    case 25: evaluation.material.transmission.x = value.x; break;
    case 26: evaluation.material.transmissionColor.xyz = value.xyz; break;
    case 27: evaluation.material.transmission.y = value.x; break;
    case 28: evaluation.material.transmissionScatter.xyz = value.xyz; break;
    case 29: evaluation.material.transmission.z = value.x; break;
    case 30: evaluation.material.transmission.w = value.x; break;
    case 31: evaluation.material.subsurface.x = value.x; break;
    case 32: evaluation.material.subsurfaceColor.xyz = value.xyz; break;
    case 33: evaluation.material.subsurfaceRadius.xyz = value.xyz; break;
    case 34: evaluation.material.subsurface.y = value.x; break;
    case 35: evaluation.material.subsurface.z = value.x; break;
    case 36: evaluation.material.subsurface.w = value.x; break;
    case 37: evaluation.material.thinFilm.x = value.x; break;
    case 38: evaluation.material.thinFilm.y = value.x; break;
    case 39: evaluation.material.thinFilm.z = value.x; break;
    case 40: evaluation.material.fuzz.x = value.x; break;
    case 41: evaluation.material.fuzzColor.xyz = value.xyz; break;
    case 42: evaluation.material.fuzz.y = value.x; break;
    case 43: evaluation.material.thinFilm.w = value.x; break;
    case 44: evaluation.material.volume.x = value.x; break;
    case 45: evaluation.material.volumeAbsorption.xyz = value.xyz; break;
    case 46: evaluation.material.volumeScattering.xyz = value.xyz; break;
    case 47: evaluation.material.volume.y = value.x; break;
    case 48: evaluation.material.volumeEmission.xyz = value.xyz; break;
    case 49: evaluation.material.volume.z = value.x; break;
    case 50: evaluation.material.fuzz.z = value.x; break;
    case 51: evaluation.material.fuzz.w = value.x; break;
    case 52: evaluation.material.volume.w = value.x; break;
  }
}

MaterialEvaluation evaluateMaterialProgram(uint materialIndex, MaterialContext ctx)
{
  MaterialEvaluation evaluation;
  evaluation.material = materials[materialIndex];
  evaluation.normal = materialSafeNormal(ctx.normal, ctx.geometricNormal);
  evaluation.coatNormal = evaluation.normal;
  evaluation.tangent = ctx.tangent;
  evaluation.coatTangent = ctx.tangent;
  uint count = min(evaluation.material.program.y, MATERIAL_REGISTER_COUNT);
  if(count == 0) return evaluation;
  if(count == 3)
  {
    // Blender's default geometry tangent often survives as this exact program.
    // Recognize existing packages too, before allocating/clearing the generic
    // register and footprint arrays. Every other graph keeps the interpreter.
    uint first = evaluation.material.program.x;
    MaterialInstruction source = materialInstructions[first];
    MaterialInstruction normalized = materialInstructions[first + 1];
    MaterialInstruction output = materialInstructions[first + 2];
    if(all(source.code == uint4(5, 0, 0, 0))
       && all(normalized.code == uint4(27, 0, 0, 0)) && normalized.value.x == 3.0f
       && all(output.code == uint4(63, 1, 21, 0)))
    {
      // Preserve both distinct normalizations from Normalize and Output. In
      // particular their small-vector thresholds differ, and the output's
      // fallback is the original context tangent, not a fixed unit vector.
      float3 tangent = float3(0.0f);
      float magnitude = sqrt(dot(ctx.tangent, ctx.tangent));
      if(magnitude > 1.0e-20f) tangent = ctx.tangent / magnitude;
      evaluation.tangent = materialSafeNormal(tangent, evaluation.tangent);
      return evaluation;
    }
  }
  float4 registers[MATERIAL_REGISTER_COUNT] = {};
  // Conservative per-lane filter widths propagate UV scaling and combinations.
  // These are ray-footprint bounds, not screen-space quad derivatives. Spatial
  // position inputs currently have no footprint in MaterialContext.
  float4 widths[MATERIAL_REGISTER_COUNT] = {};
  for(uint index = 0; index < count; ++index)
  {
    MaterialInstruction instruction = materialInstructions[evaluation.material.program.x + index];
    uint4 code = instruction.code;
    float4 a = float4(0), b = float4(0), c = float4(0);
    float4 da = float4(0), db = float4(0), dc = float4(0);
    // Scene validation rejects forward references. Explicit guards also prevent
    // unused fields (e.g. an Image texture index) from reading an invalid slot.
    if(code.y < index) {a = registers[code.y]; da = widths[code.y];}
    if(code.z < index) {b = registers[code.z]; db = widths[code.z];}
    if(code.w < index) {c = registers[code.w]; dc = widths[code.w];}
    uint lanes = instruction.value.x >= 1 && instruction.value.x <= 4 ? uint(instruction.value.x) : 3;
    float4 value = float4(0), width = float4(0);
    switch(code.x)
    {
      case 0: value = instruction.value; break;
      case 1: value = float4(ctx.texcoord, 0, 0); width = float4(abs(ctx.uvFootprint), 0, 0); break;
      case 2: value = float4(ctx.position, 0); break;
      case 3: value = float4(ctx.objectPosition, 0); break;
      case 4: value = float4(ctx.normal, 0); break;
      case 5: value = float4(ctx.tangent, 0); break;
      case 10: value = a + b; width = da + db; break;
      case 11: value = a - b; width = da + db; break;
      case 12: value = a * b; width = da * abs(b) + db * abs(a); break;
      case 13:
      {
        // Division by zero is outside the valid arithmetic domain; a finite
        // zero result prevents a single invalid lane contaminating radiance.
        float4 divisor = select(abs(b) > 1.0e-20f, b, float4(1));
        value = select(abs(b) > 1.0e-20f, a / divisor, float4(0));
        width = (da + abs(value) * db) / abs(divisor);
        break;
      }
      case 14: value = lerp(a, b, c); width = abs(1.0f - c) * da + abs(c) * db + abs(b - a) * dc; break;
      case 15: value = clamp(a, b, c); width = max(da, max(db, dc)); break;
      case 16:
      {
        value = pow(a, b);
        float4 positive = max(abs(a), float4(1.0e-20f));
        width = abs(b * pow(positive, b - 1.0f)) * da + abs(value * log(positive)) * db;
        break;
      }
      case 17: value = min(a, b); width = max(da, db); break;
      case 18: value = max(a, b); width = max(da, db); break;
      case 19: value = abs(a); width = da; break;
      case 20: value = 1.0f - a; width = da; break;
      case 21: value = sin(a); width = da; break;
      case 22: value = cos(a); width = da; break;
      case 23: value = floor(a); width = da; break;
      case 24: value = ceil(a); width = da; break;
      case 25: value = frac(a); width = da; break;
      case 26:
        value = float4(materialVectorDot(a, b, lanes));
        width = float4(materialVectorDot(da, abs(b), lanes) + materialVectorDot(db, abs(a), lanes));
        break;
      case 27:
      {
        float magnitude = sqrt(materialVectorDot(a, a, lanes));
        if(magnitude > 1.0e-20f)
        {
          value = materialVectorMask(a / magnitude, lanes);
          width = float4(2.0f * sqrt(materialVectorDot(da, da, lanes)) / magnitude);
        }
        break;
      }
      case 28:
        value = float4(sqrt(materialVectorDot(a, a, lanes)));
        width = float4(sqrt(materialVectorDot(da, da, lanes)));
        break;
      case 29:
        value = float4(cross(a.xyz, b.xyz), 0);
        width = float4(abs(a.yzx) * db.zxy + abs(a.zxy) * db.yzx + da.yzx * abs(b.zxy) + da.zxy * abs(b.yzx), 0);
        break;
      case 30: value = float4(a.x, b.x, c.x, instruction.value.w); width = float4(da.x, db.x, dc.x, 0); break;
      case 31:
      {
        uint4 channels = uint4(clamp(instruction.value, float4(0), float4(3)));
        value = float4(a[channels.x], a[channels.y], a[channels.z], a[channels.w]);
        width = float4(da[channels.x], da[channels.y], da[channels.z], da[channels.w]);
        break;
      }
      case 32: value = materialTextureSample(code.z, a.xy, da.xy, code.w, instruction.value.x); break;
      case 33:
      {
        float2 cell = floor(a.xy * instruction.value.xy);
        float parity = frac((cell.x + cell.y) * 0.5f) * 2.0f;
        value = parity < 1.0f ? b : c;
        width = max(db, dc);
        break;
      }
      case 34:
        value = instruction.value.x==3.0f ? float4(mx_perlin_noise_vec3(a.xyz)*b.xyz+c.xyz,0) : float4(mx_perlin_noise_float(a.xyz)*b.x+c.x);
        break;
      case 35: value = float4(materialNormalmap(a.xyz, b.x, ctx), 0); break;
      case 36: value=float4(mx_rgbtohsv(a.xyz),a.w);width=da;break;
      case 37: value=float4(mx_hsvtorgb(a.xyz),a.w);width=da;break;
      case 38: value=materialHexImage(code.z,a.xy,da.xy,registers[code.w],registers[code.w+1],registers[code.w+2],registers[code.w+3]);break;
      case 63: value = a; width = da; materialWriteOutput(evaluation, code.z, a); break;
    }
    registers[index] = value;
    widths[index] = width;
  }
  return evaluation;
}

#endif
