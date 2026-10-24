// SPDX-License-Identifier: Apache-2.0
// Adapted from AcademySoftwareFoundation/MaterialX v1.39.5 PBR library:
// mx_microfacet_diffuse.glsl, mx_microfacet_specular.glsl and open_pbr_surface.mtlx.
// Source provenance and modifications are recorded in ORIGIN.md. These are
// native OpenPBR graph primitives; no OpenPBR-to-Standard-Surface translation.
#ifndef CAUSTRA_OPEN_PBR_MATH
#define CAUSTRA_OPEN_PBR_MATH

float2 openPbrAlpha(float roughness, float anisotropy)
{
  float a = 1.0f - saturate(anisotropy);
  float x = roughness * roughness * sqrt(2.0f / (1.0f + a * a));
  return clamp(float2(x, a * x), float2(0.0004f), float2(1.0f));
}

float openPbrFresnelDielectric(float cosine, float ior)
{
  if(abs(ior - 1.0f) < 0.000001f) return 0.0f;
  float c = clamp(cosine, 0.000001f, 1.0f);
  float g2 = ior * ior + c * c - 1.0f;
  if(g2 < 0.0f) return 1.0f;
  float g = sqrt(g2);
  float a = (g - c) / (g + c);
  float b = ((g + c) * c - 1.0f) / ((g - c) * c + 1.0f);
  return 0.5f * a * a * (1.0f + b * b);
}

float3 openPbrFresnelMetal(float cosine, float3 color0, float3 color82)
{
  const float cMax = 1.0f / 7.0f;
  const float factor = 1.0f / (cMax * pow(1.0f - cMax, 6.0f));
  float c = saturate(cosine);
  float3 correction = lerp(color0, float3(1.0f), pow(1.0f - cMax, 5.0f)) * (float3(1.0f) - color82) * factor;
  return max(float3(0.0f), lerp(color0, float3(1.0f), pow(1.0f - c, 5.0f)) - correction * c * pow(1.0f - c, 6.0f));
}

// MaterialX's current rational-quadratic GGX directional-albedo fit.
float3 openPbrGgxAlbedo(float nDotV, float alpha, float3 f0, float3 f90)
{
  float x = nDotV, y = alpha, x2 = x * x, y2 = y * y;
  float4 r = float4(0.1003f, 0.9345f, 1.0f, 1.0f)
    + float4(-0.6303f, -2.323f, -1.765f, 0.2281f) * x
    + float4(9.748f, 2.229f, 8.263f, 15.94f) * y
    + float4(-2.038f, -3.748f, 11.53f, -55.83f) * x * y
    + float4(29.34f, 1.424f, 28.96f, 13.08f) * x2
    + float4(-8.245f, -0.7684f, -7.507f, 41.26f) * y2
    + float4(-26.44f, 1.436f, -36.11f, 54.9f) * x2 * y
    + float4(19.99f, 0.2913f, 15.86f, 300.2f) * x * y2
    + float4(-5.448f, 0.6286f, 33.37f, -285.1f) * x2 * y2;
  float2 ab = saturate(r.xy / r.zw);
  return f0 * ab.x + f90 * ab.y;
}

float3 openPbrGgxCompensation(float nDotV, float alpha, float3 fresnel)
{
  float ess = max(openPbrGgxAlbedo(nDotV, alpha, float3(1), float3(1)).x, 0.000001f);
  return float3(1.0f) + fresnel * ((1.0f - ess) / ess);
}

float openPbrFujiiDirectionalAlbedo(float cosine, float roughness)
{
  const float k1 = 0.5f - 2.0f / (3.0f * AURORA_PI);
  float c = clamp(cosine, 0.000001f, 1.0f);
  float a = 1.0f / (1.0f + k1 * roughness), b = roughness * a;
  float s = sqrt(max(0.0f, 1.0f - c * c));
  float g = s * (acos(c) - s * c) + 2.0f * ((s / c) * (1.0f - s * s * s) - s) / 3.0f;
  return a + b * g * AURORA_INV_PI;
}

float openPbrFujiiAverageAlbedo(float roughness)
{
  const float k1 = 0.5f - 2.0f / (3.0f * AURORA_PI);
  const float k2 = 2.0f / 3.0f - 28.0f / (15.0f * AURORA_PI);
  return (1.0f + k2 * roughness) / (1.0f + k1 * roughness);
}

float3 openPbrDiffuse(float nDotV, float nDotL, float lDotV, float roughness, float3 color)
{
  const float k1 = 0.5f - 2.0f / (3.0f * AURORA_PI);
  float s = lDotV - nDotL * nDotV;
  float st = s > 0.0f ? s / max(nDotL, nDotV) : s;
  float a = 1.0f / (1.0f + k1 * roughness);
  float3 single = color * a * (1.0f + roughness * st);
  float ev = openPbrFujiiDirectionalAlbedo(nDotV, roughness);
  float el = openPbrFujiiDirectionalAlbedo(nDotL, roughness);
  float average = openPbrFujiiAverageAlbedo(roughness);
  float3 multipleColor = color * color * average / max(float3(0.000001f), float3(1.0f) - color * max(0.0f, 1.0f - average));
  float3 multiple = multipleColor * max(0.000001f, 1.0f - ev) * max(0.000001f, 1.0f - el) / max(0.000001f, 1.0f - average);
  return single + multiple;
}

float3 openPbrDiffuseAlbedo(float nDotV, float roughness, float3 color)
{
  float e = openPbrFujiiDirectionalAlbedo(nDotV, roughness);
  float average = openPbrFujiiAverageAlbedo(roughness);
  float3 multipleColor = color * color * average / max(float3(0.000001f), float3(1.0f) - color * max(0.0f, 1.0f - average));
  return lerp(multipleColor, color, e);
}

#endif // CAUSTRA_OPEN_PBR_MATH
