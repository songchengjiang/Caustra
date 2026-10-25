// SPDX-License-Identifier: Apache-2.0
// Directed integration checks against the independently invoked ShaderGen
// provider. This bypasses scene traversal, Monte Carlo image noise and DLSS.
#include "../shaders/material_program.slang"
#include "../shaders/surface_bsdf.slang"

// Keep this explicit include: the reference below must use the checked-in
// official ShaderGen output rather than a second copy of Caustra's evaluator.
#include "../shaders/generated/materialx_surfaces.slang"

struct ProbeRecord
{
  GpuMaterial material;
  float4 normal, coatNormal, tangent, coatTangent, geometricNormal, view, light, random;
  float4 runtimeValue, referenceValue, runtimeEmission, referenceEmission;
  float4 sampleValue, sampleReference, sampleState;
};
[[vk::binding(0, 0)]] RWStructuredBuffer<ProbeRecord> probes;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID)
{
  uint count, stride; probes.GetDimensions(count, stride);
  uint index = thread.x; if(index >= count) return;
  ProbeRecord p = probes[index];
  if(p.random.w == 1.0f)
  {
    // Independent deterministic quadrature: bypass production closures, PDF
    // evaluation and sampling completely. Directions are uniform in Ng's
    // upper hemisphere; the host integrates this f*cos with measure 2*pi/N.
    probes[index].runtimeValue = float4(mxGeneratedReflection(p.material,
      p.normal.xyz, p.coatNormal.xyz, p.tangent.xyz, p.coatTangent.xyz,
      p.view.xyz, p.light.xyz), 0.0f);
    return;
  }
  MaterialEvaluation e = (MaterialEvaluation)0;
  e.material = p.material;
  e.normal = p.normal.xyz; e.coatNormal = p.coatNormal.xyz;
  e.tangent = p.tangent.xyz; e.coatTangent = p.coatTangent.xyz;
  MaterialContext ctx = (MaterialContext)0;
  ctx.normal = p.normal.xyz; ctx.geometricNormal = p.geometricNormal.xyz;
  ctx.tangent = p.tangent.xyz; ctx.bitangent = cross(ctx.normal, ctx.tangent);
  SurfaceClosure c = makeSurfaceClosure(e, ctx, p.view.xyz);
  configureSurfaceInterface(c, 1.0f, surfaceNominalIor(p.material));
  c.entering = true;
  SurfaceEval value = evaluateSurface(c, p.view.xyz, p.light.xyz);
  p.runtimeValue = float4(value.fCos, value.pdf);
  p.referenceValue = float4(mxGeneratedReflection(p.material, p.normal.xyz,
    p.coatNormal.xyz, p.tangent.xyz, p.coatTangent.xyz, p.view.xyz, p.light.xyz), 0.0f);
  p.runtimeEmission = float4(surfaceEmission(c, p.view.xyz), 0.0f);
  p.referenceEmission = float4(mxGeneratedEmission(p.material, p.normal.xyz,
    p.coatNormal.xyz, p.tangent.xyz, p.coatTangent.xyz, p.view.xyz, p.light.xyz), 0.0f);
  SurfaceSample sample = sampleSurface(c, p.view.xyz, p.random.xyz);
  p.sampleValue = float4(sample.weight * sample.pdf, sample.pdf);
  p.sampleReference = float4(mxGeneratedReflection(p.material, p.normal.xyz,
    p.coatNormal.xyz, p.tangent.xyz, p.coatTangent.xyz, p.view.xyz, sample.direction), float(sample.event));
  p.sampleState = float4(sample.direction, sample.delta ? 1.0f : 0.0f);
  probes[index] = p;
}
