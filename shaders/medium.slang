// SPDX-License-Identifier: Apache-2.0
// Homogeneous and piecewise-constant heterogeneous RGB transport, with bounded,
// instance-keyed optical interfaces.
// The distance estimator selects one RGB extinction channel in proportion to
// current path throughput, then divides by the exact weighted mixture PDF
// (including the discrete surface escape). This remains unbiased while avoiding
// large spectral weight products from repeatedly sampling an irrelevant channel.
#ifndef CAUSTRA_MEDIUM_SLANG
#define CAUSTRA_MEDIUM_SLANG

[[vk::binding(12, 0)]] RWStructuredBuffer<uint> renderErrors;
// Matches GpuCameraMediumCache: one header and four packed instance-ID rows.
// Only the initialization invocation writes this cache; ordinary paths read it.
[[vk::binding(13, 0)]] RWStructuredBuffer<uint4> cameraMediumCache;
// Eight uint4 rows per material match GpuMaterialMediumData, followed by optional
// per-instance 32^3 coefficient grids. Initialized once per scene, before paths.
[[vk::binding(14, 0)]] RWStructuredBuffer<uint4> materialMediumTable;
static const uint MEDIUM_STACK_CAPACITY = 16;
static const uint MEDIUM_PROBE_LIMIT = 4096;
static const uint MEDIUM_NO_HERO = 3;
struct MediumCoefficients
{
  float3 absorption;
  float3 scatteringVolume, scatteringTransmission, scatteringSubsurface;
  float3 emission;
  float3 anisotropy; // volume, transmission, subsurface HG components
};
// IDs keep every path/shadow stack to 68 logical bytes rather than carrying
// 16 copies of RGB optical coefficients. Instances retain their own identity
// even when several closed boundaries share one material.
struct MediumStack { uint entries[MEDIUM_STACK_CAPACITY]; uint count; };
struct MediumFlight { float distance; float3 weight; bool scattered; };
struct MediumBoundaryProgress
{
  bool active, planeActive, frontFacing;
  uint instanceIndex;
  float distance, processedDistance, tolerance, positionError;
  float3 position, normal;
};

// Two adjacent triangles on one closed boundary can report slightly different
// t values along their shared edge. Keep the ray line fixed and recognize only
// same-oriented crossings of that instance at the same numerical surface.
// Do not merge an opposite-facing exit, even for a sub-epsilon thin solid.
bool repeatedMediumBoundary(MediumBoundaryProgress previous, SurfaceHit hit)
{
  return previous.active && previous.instanceIndex == hit.instanceIndex &&
    previous.frontFacing == hit.frontFacing &&
    abs(hit.distance - previous.distance) <= previous.tolerance &&
    all(abs(hit.position - previous.position) <= previous.tolerance);
}
MediumBoundaryProgress recordMediumBoundary(RayDesc ray, SurfaceHit hit)
{
  MediumBoundaryProgress result;
  result.active = true;result.planeActive=true; result.frontFacing = hit.frontFacing;
  result.instanceIndex = hit.instanceIndex; result.distance = hit.distance;result.processedDistance=hit.distance;
  result.position = hit.position; result.normal = hit.geometricNormal;
  result.positionError=hit.positionError;
  float3 magnitude = max(abs(ray.Origin), abs(hit.position));
  float scale = max(1.0f, max(abs(hit.distance), max(magnitude.x, max(magnitude.y, magnitude.z))));
  result.tolerance = 32.0f * 1.1920928955078125e-7f * scale;
  return result;
}
float mediumResumeDistance(float distance)
{
  return distance>0.0f?asfloat(asuint(distance)-1u):0.0f;
}

float mediumNextFloat(float value, float direction)
{
  if(direction == 0.0f) return value;
  // A few minimum normal units survive flush-to-zero during the subsequent
  // dot product, without imposing any world-space epsilon near the origin.
  if(value == 0.0f) return asfloat(direction > 0.0f ? 0x01800000u : 0x81800000u);
  uint bits = asuint(value);
  bool increaseBits = (value > 0.0f) == (direction > 0.0f);
  return asfloat(increaseBits ? bits + 1u : bits - 1u);
}
float3 mediumNextPoint(float3 point, float3 direction)
{
  return float3(mediumNextFloat(point.x, direction.x), mediumNextFloat(point.y, direction.y),
                mediumNextFloat(point.z, direction.z));
}
float3 mediumPointInHalfspace(float3 point, float3 boundary, float3 inward)
{
  float3 next = mediumNextPoint(point, inward);
  float spacing = dot(abs(next - point), abs(inward));
  float distance = dot(point - boundary, inward);
  if(distance > spacing) return point;
  // Projection only repairs the numerical point. Collision/escape selection,
  // sampled distance, transmittance and PDF retain their original values.
  point = fma(inward, float3(max(0.0f, spacing - distance)), point);
  return mediumNextPoint(point, inward);
}
float3 mediumFlightPoint(RayDesc ray, float3 segmentStart, MediumBoundaryProgress entry,
                         SurfaceHit endpoint, bool foundEndpoint, float distance, float segmentLength)
{
  // Keep the sampled distance/PDF intact; only the representable spawn point
  // is moved away from uncertain boundary planes. Stay on the traversed line
  // and cap each clearance at a quarter segment, so thin regions cannot be
  // crossed by a large absolute offset.
  float minimum=0.0f,maximum=segmentLength;
  if(entry.active && entry.planeActive)
  {
    float clearance=dot(segmentStart-entry.position,-entry.normal);
    float speed=max(dot(ray.Direction,-entry.normal),1.0e-30f);
    minimum=min(max(0.0f,(entry.positionError-clearance)/speed),segmentLength*.25f);
  }
  if(foundEndpoint)
  {
    float speed=max(-dot(ray.Direction,endpoint.geometricNormal),1.0e-30f);
    maximum=segmentLength-min(endpoint.positionError/speed,segmentLength*.25f);
  }
  float spawnDistance=clamp(distance,minimum,maximum);
  float remaining = max(segmentLength - spawnDistance, 0.0f);
  // Avoid adding a short flight to a large entry t, which can round straight
  // back to entry (or ahead to exit) before ray(t) is even evaluated.
  float3 point = foundEndpoint && remaining < spawnDistance ?
    fma(-ray.Direction, float3(remaining), endpoint.position) :
    fma(ray.Direction, float3(spawnDistance), segmentStart);
  for(uint pass = 0; pass < 3; ++pass)
  {
    if(entry.active && entry.planeActive) point = mediumPointInHalfspace(point, entry.position, -entry.normal);
    if(foundEndpoint) point = mediumPointInHalfspace(point, endpoint.position, endpoint.geometricNormal);
  }
  if(!finite3(point) || (entry.active && entry.planeActive && dot(point - entry.position, -entry.normal) <= 0.0f) ||
     (foundEndpoint && dot(point - endpoint.position, endpoint.geometricNormal) <= 0.0f))
    mediumError(4); // No representable interior point: fail explicitly.
  return point;
}

void mediumError(uint bits) { InterlockedOr(renderErrors[0], bits); }
void mediumBoundaryError(MediumStack stack,SurfaceHit hit,uint site,float3 origin,float3 direction,uint source)
{
  uint previous;
  InterlockedCompareExchange(renderErrors[1],0,site,previous);
  if(previous==0) {
    renderErrors[2]=hit.instanceIndex;
    uint top=stack.count>0?stack.entries[stack.count-1]:0x00ffffffu;
    renderErrors[3]=(hit.frontFacing?0x80000000u:0u)|(stack.count<<24)|top;
    renderErrors[4]=asuint(hit.distance);
    renderErrors[5]=asuint(hit.position.x);renderErrors[6]=asuint(hit.position.y);renderErrors[7]=asuint(hit.position.z);
    renderErrors[8]=asuint(origin.x);renderErrors[9]=asuint(origin.y);renderErrors[10]=asuint(origin.z);
    renderErrors[11]=asuint(direction.x);renderErrors[12]=asuint(direction.y);renderErrors[13]=asuint(direction.z);
    renderErrors[14]=pc.dimensions.z;renderErrors[15]=source;
  }
  mediumError(2);
}
float mediumMean(float3 value) { return (value.x + value.y + value.z) / 3.0f; }
float mediumLuminance(float3 value) { return dot(value, float3(0.2126f, 0.7152f, 0.0722f)); }
float3 mediumScattering(MediumCoefficients m) { return m.scatteringVolume + m.scatteringTransmission + m.scatteringSubsurface; }
float3 mediumExtinction(MediumCoefficients m) { return m.absorption + mediumScattering(m); }
bool materialHasInterior(GpuMaterial m)
{
  if(m.program.z==5)return m.volume.x>0.0f || m.volume.z==1.0f; // Fiber lobes have no bulk interior; an explicit VDF remains independent.
  // The volume terminal is independent of a thin-walled surface terminal.
  return m.volume.x > 0.0f || m.volume.z == 1.0f ||
    (m.subsurface.w == 0.0f && ((m.program.w&1)!=0 || m.transmission.x > 0.0f || m.subsurface.x > 0.0f));
}

MediumCoefficients materialMedium(GpuMaterial m)
{
  MediumCoefficients result = (MediumCoefficients)0;
  float density = max(m.volume.x, 0.0f);
  result.absorption = max(m.volumeAbsorption.xyz, float3(0.0f)) * density;
  result.scatteringVolume = max(m.volumeScattering.xyz, float3(0.0f)) * density;
  result.emission = max(m.volumeEmission.xyz, float3(0.0f)) * density;
  result.anisotropy = clamp(float3(m.volume.y, m.transmission.z, m.subsurface.z), float3(-0.9999f), float3(0.9999f));
  float transmission = saturate(m.transmission.x);
  float subsurface = (1.0f - transmission) * saturate(m.subsurface.x);
  float total = transmission + subsurface;
  if(total > 0.0f && m.subsurface.w == 0.0f)
  {
    float transmissionShare = transmission / total, subsurfaceShare = subsurface / total;
    if(m.program.z == 1)
    {
      // OpenPBR defines color as extinction over attenuation depth, and scatter
      // as scattering over that same depth. Shift negative absorption by one
      // achromatic offset as specified, preserving its channel differences.
      // At zero depth only the BSDF boundary tint remains; no interior VDF.
      if(m.transmission.y > 0.0f)
      {
        float3 sigmaT = -log(clamp(m.transmissionColor.xyz, float3(1.0e-20f), float3(1.0f))) / m.transmission.y;
        float3 sigmaS = max(m.transmissionScatter.xyz, float3(0.0f)) / m.transmission.y;
        float3 sigmaA = sigmaT - sigmaS;
        sigmaA -= min(0.0f, min(sigmaA.x, min(sigmaA.y, sigmaA.z)));
        result.absorption += sigmaA * transmissionShare;
        result.scatteringTransmission = sigmaS * transmissionShare;
      }
    }
    else
    {
      if(m.transmission.y > 0.0f)
        result.absorption += -log(clamp(m.transmissionColor.xyz, float3(1.0e-20f), float3(1.0f))) / m.transmission.y * transmissionShare;
      // Standard Surface transmission_scatter is a direct inverse-length
      // scattering coefficient, independent of the attenuation depth.
      result.scatteringTransmission = max(m.transmissionScatter.xyz, float3(0.0f)) * transmissionShare;
    }
    if(subsurfaceShare > 0.0f)
    {
      // Both surface models use observed reflection color C and an RGB mean
      // free path radius*scale. Invert the OpenPBR van-de-Hulst calibration in
      // a cancellation-resistant form. Its index-matched-boundary assumption
      // is an albedo calibration approximation; transport itself is a volume
      // random walk with the actual dielectric boundary, not a diffuse fake.
      float3 color = saturate(m.subsurfaceColor.xyz);
      if(m.program.z == 0)
        color = pow(color, float3(1.0f + m.coat.x * saturate(m.fuzz.w)));
      float3 b = 1.139f + 1.17f * color;
      float3 s = 2.0f * (1.0f - color) / (b + sqrt(max(b * b - 0.556f * (1.0f - color), float3(0.0f))));
      float3 rho = saturate((1.0f - s * s) / (1.0f - result.anisotropy.z * s * s));
      float3 radius = max(m.subsurfaceRadius.xyz * m.subsurface.y, float3(1.0e-6f));
      float3 sigmaT = 1.0f / radius;
      result.scatteringSubsurface = rho * sigmaT * subsurfaceShare;
      result.absorption += (1.0f - rho) * sigmaT * subsurfaceShare;
    }
  }
  if(!finite3(result.absorption) || !finite3(mediumScattering(result)) || !finite3(result.emission))
    mediumError(4);
  return result;
}

void initializeMaterialMedium(uint materialIndex)
{
  uint materialCount, materialStride;
  materials.GetDimensions(materialCount, materialStride);
  if(materialIndex >= materialCount) return;
  uint row = materialIndex * 8;
  materialMediumTable[row] = uint4(0);
  // Host validation requires every active interior parameter, including its
  // graph dependencies and SS coat-color controls, to be spatially uniform.
  // Other surface outputs are deliberately discarded here and remain per hit.
  // Evaluate every graph: its constant outputs may activate a raw-zero volume.
  MaterialContext ctx = (MaterialContext)0;
  ctx.normal = ctx.geometricNormal = float3(0, 0, 1);
  ctx.tangent = float3(1, 0, 0); ctx.bitangent = float3(0, 1, 0);
  GpuMaterial material = evaluateMaterialProgram(materialIndex, ctx).material;
  bool interior = materialHasInterior(material);
  MediumCoefficients coefficients = (MediumCoefficients)0;
  float3 ior = float3(0.0f);
  if(interior)
  {
    coefficients = materialMedium(material);
    if(material.volume.z != 1.0f && material.subsurface.w != 1.0f)
      ior = float3(surfaceDispersiveIor(material, 0), surfaceDispersiveIor(material, 1), surfaceDispersiveIor(material, 2));
    if(!finite3(ior)) mediumError(4);
  }
  materialMediumTable[row + 1] = asuint(float4(coefficients.absorption, 0));
  materialMediumTable[row + 2] = asuint(float4(coefficients.scatteringVolume, 0));
  materialMediumTable[row + 3] = asuint(float4(coefficients.scatteringTransmission, 0));
  materialMediumTable[row + 4] = asuint(float4(coefficients.scatteringSubsurface, 0));
  materialMediumTable[row + 5] = asuint(float4(coefficients.emission, 0));
  materialMediumTable[row + 6] = asuint(float4(coefficients.anisotropy, 0));
  materialMediumTable[row + 7] = asuint(float4(ior, 0));
  materialMediumTable[row] = uint4(interior ? 1u : 0u, material.volume.z == 1.0f ? 1u : 0u, 1, 0);
}

static const uint MEDIUM_GRID_RESOLUTION=32;
static const uint MEDIUM_GRID_CELLS=32768;
float3 mediumTransform(float4 rows[3],float3 p,float w)
{
  float4 q=float4(p,w);return float3(dot(rows[0],q),dot(rows[1],q),dot(rows[2],q));
}
// The graph is evaluated once at each object-space cell center, never frozen
// at a surface entry. Transport is exact for this piecewise-constant field.
// UV graphs require an explicit extension inside the object: planar XY by
// default; bit 1 selects radial UVs matching the generated gallery spheres.
void initializeHeterogeneousMedium(uint instanceIndex,uint cellIndex)
{
  if(cellIndex>=MEDIUM_GRID_CELLS)return;
  GpuInstance instance=instances[instanceIndex];
  uint3 cell=uint3(cellIndex%32,(cellIndex/32)%32,cellIndex/1024);
  float3 normalized=(float3(cell)+.5f)/32.0f;
  MaterialContext ctx=(MaterialContext)0;
  ctx.objectPosition=lerp(instance.objectMin.xyz,instance.objectMax.xyz,normalized);
  ctx.position=mediumTransform(instance.objectToWorld,ctx.objectPosition,1);
  ctx.normal=ctx.geometricNormal=float3(0,0,1);ctx.tangent=float3(1,0,0);ctx.bitangent=float3(0,1,0);
  ctx.texcoord=normalized.xy;
  GpuMaterial raw=materials[instance.offsets.z];
  if((raw.program.w&2)!=0)
  {
    float3 radial=safeNormal(ctx.objectPosition-(instance.objectMin.xyz+instance.objectMax.xyz)*.5f,float3(0,1,0));
    float phi=atan2(radial.z,radial.x);
    ctx.texcoord=float2(frac(phi/6.283185307179586f+1.0f),1.0f-acos(clamp(radial.y,-1.0f,1.0f))/3.141592653589793f);
    ctx.normal=ctx.geometricNormal=radial;
    ctx.tangent=float3(-sin(phi),0,cos(phi));ctx.bitangent=cross(radial,ctx.tangent);
  }
  float3 n=ctx.normal;
  ctx.normal=ctx.geometricNormal=safeNormal(float3(
    instance.worldToObject[0].x*n.x+instance.worldToObject[1].x*n.y+instance.worldToObject[2].x*n.z,
    instance.worldToObject[0].y*n.x+instance.worldToObject[1].y*n.y+instance.worldToObject[2].y*n.z,
    instance.worldToObject[0].z*n.x+instance.worldToObject[1].z*n.y+instance.worldToObject[2].z*n.z),float3(0,0,1));
  ctx.tangent=mediumTransform(instance.objectToWorld,ctx.tangent,0);
  ctx.tangent=safeNormal(ctx.tangent-ctx.normal*dot(ctx.tangent,ctx.normal),float3(1,0,0));
  ctx.bitangent=safeNormal(mediumTransform(instance.objectToWorld,ctx.bitangent,0),cross(ctx.normal,ctx.tangent));
  GpuMaterial material=evaluateMaterialProgram(instance.offsets.z,ctx).material;
  MediumCoefficients m=materialMedium(material);
  uint row=instance.offsets.w+cellIndex*8;
  materialMediumTable[row]=uint4(1,0,1,0);
  materialMediumTable[row+1]=asuint(float4(m.absorption,0));
  materialMediumTable[row+2]=asuint(float4(m.scatteringVolume,0));
  materialMediumTable[row+3]=asuint(float4(m.scatteringTransmission,0));
  materialMediumTable[row+4]=asuint(float4(m.scatteringSubsurface,0));
  materialMediumTable[row+5]=asuint(float4(m.emission,0));
  materialMediumTable[row+6]=asuint(float4(m.anisotropy,0));
  materialMediumTable[row+7]=uint4(0);
}

uint mediumInstanceRow(uint instanceIndex)
{
  return instances[instanceIndex].offsets.z * 8;
}

MediumCoefficients mediumTableLookup(uint row)
{
  MediumCoefficients result = (MediumCoefficients)0;
  result.absorption = asfloat(materialMediumTable[row + 1]).xyz;
  result.scatteringVolume = asfloat(materialMediumTable[row + 2]).xyz;
  result.scatteringTransmission = asfloat(materialMediumTable[row + 3]).xyz;
  result.scatteringSubsurface = asfloat(materialMediumTable[row + 4]).xyz;
  result.emission = asfloat(materialMediumTable[row + 5]).xyz;
  result.anisotropy = asfloat(materialMediumTable[row + 6]).xyz;
  return result;
}
MediumCoefficients currentMedium(MediumStack stack)
{
  return stack.count==0?(MediumCoefficients)0:mediumTableLookup(mediumInstanceRow(stack.entries[stack.count-1]));
}

float currentMediumIor(MediumStack stack, uint hero)
{
  // Null volume boundaries change extinction, but not the optical IOR of the
  // containing glass/water region. Zero IOR marks such an invisible boundary.
  for(int index = int(stack.count) - 1; index >= 0; --index)
  {
    float3 ior = asfloat(materialMediumTable[mediumInstanceRow(stack.entries[index]) + 7]).xyz;
    if(ior.x > 0.0f) return ior[hero < 3 ? hero : 1];
  }
  return 1.0f;
}

float materialInteriorIor(GpuMaterial m, uint hero)
{
  return surfaceDispersiveIor(m, hero < 3 ? hero : 1);
}

bool pointNearSharedContact(float3 point, float3 a, float3 b, uint count, float tolerance)
{
  float3 edge=b-a;
  float lengthSquared=dot(edge,edge);
  float along=count>1 && lengthSquared>0.0f?saturate(dot(point-a,edge)/lengthSquared):0.0f;
  return all(abs(point-fma(edge,float3(along),a))<=tolerance);
}

bool mediumBoundaryTangent(float3 origin, float3 direction, SurfaceHit closest)
{
  RayDesc contact;
  contact.Origin=origin;contact.Direction=direction;contact.TMin=0.0f;contact.TMax=FAR_DISTANCE;
  float tolerance=recordMediumBoundary(contact,closest).tolerance;
  GpuInstance instance=instances[closest.instanceIndex];
  uint first=instance.offsets.y+closest.primitiveIndex*3;
  float3 reference[3];
  for(uint i=0;i<3;++i) reference[i]=vertices[instance.offsets.x+indices[first+i]].position.xyz;
  float3 referenceNormal=safeNormal(cross(reference[1]-reference[0],reference[2]-reference[0]),float3(0,0,1));
  contact.Origin=origin;contact.Direction=direction;contact.TMin=max(0.0f,closest.distance-tolerance);
  contact.TMax=closest.distance+tolerance;
  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
  query.TraceRayInline(topLevelAS,RAY_FLAG_NONE,0xff,contact);
  while(query.Proceed())
  {
    if(query.CandidateType()!=CANDIDATE_NON_OPAQUE_TRIANGLE ||
       query.CandidateInstanceID()!=closest.instanceIndex) continue;
    float distance=query.CandidateTriangleRayT();
    if(abs(distance-closest.distance)>tolerance) continue;
    uint first=instance.offsets.y+query.CandidatePrimitiveIndex()*3;
    float3 candidate[3];
    for(uint i=0;i<3;++i) candidate[i]=vertices[instance.offsets.x+indices[first+i]].position.xyz;
    float3 normal=cross(candidate[1]-candidate[0],candidate[2]-candidate[0]);
    // Object-space outward winding is invariant under mirrored instances.
    bool front=query.CandidateTriangleFrontFace();
    if(front==closest.frontFacing) continue;
    float2 uv=query.CandidateTriangleBarycentrics();
    float3 point=mul(query.CandidateObjectToWorld3x4(),float4(candidate[0]*(1.0f-uv.x-uv.y)+candidate[1]*uv.x+candidate[2]*uv.y,1));
    if(distance!=closest.distance)
    {
      // A broad t epsilon would erase real thin volumes. Only collapse a
      // reconstruction-sized contact on shared, nonparallel mesh faces.
      // Face seams can have duplicated vertex indices, so compare positions.
      float3 crease=cross(referenceNormal,safeNormal(normal,referenceNormal));
      if(dot(crease,crease)<=1.0e-12f) continue;
      float3 shared[2];uint count=0;
      for(uint a=0;a<3;++a) for(uint b=0;b<3;++b)
        if(all(reference[a]==candidate[b]) && count<2) shared[count++]=reference[a];
      if(count==0) continue;
      if(count==1) shared[1]=shared[0];
      float3 a=mul(query.CandidateObjectToWorld3x4(),float4(shared[0],1));
      float3 b=mul(query.CandidateObjectToWorld3x4(),float4(shared[1],1));
      if(!pointNearSharedContact(point,a,b,count,tolerance) ||
         !pointNearSharedContact(closest.position,a,b,count,tolerance)) continue;
    }
    return true;
  }
  // Consume only the current report. The query includes earlier faces, so a
  // later report proves the same contact independently. Advancing across the
  // whole tolerance window could skip another instance's real thin boundary.
  // Parallel front/back faces retain every distinct-t optical segment.
  return false;
}

bool mediumCross(inout MediumStack stack, SurfaceHit hit, GpuMaterial material,uint source,float3 origin,float3 direction)
{
  if(!materialHasInterior(material)) return true;
  if(material.volume.z==1.0f && mediumBoundaryTangent(origin,direction,hit)) return true;
  if(hit.frontFacing)
  {
    if(stack.count >= MEDIUM_STACK_CAPACITY) { mediumError(1); return false; }
    for(uint index = 0; index < stack.count; ++index)
      if(stack.entries[index] == hit.instanceIndex) { mediumBoundaryError(stack,hit,source*4+1,origin,direction,source); return false; }
    uint4 metadata = materialMediumTable[hit.materialIndex * 8];
    if(metadata.x != 1 || metadata.z != 1) { mediumError(4); return false; }
    stack.entries[stack.count] = hit.instanceIndex;
    ++stack.count;
    return true;
  }
  if(stack.count == 0 || stack.entries[stack.count - 1] != hit.instanceIndex)
  { mediumBoundaryError(stack,hit,source*4+2,origin,direction,source); return false; }
  --stack.count;
  return true;
}

float destinationMediumIor(MediumStack stack, SurfaceHit hit, uint hero)
{
  // Called only for an actual solid optical interior. Thin surfaces and null
  // boundaries use the per-hit surface path or bypass the BSDF, respectively.
  if(hit.frontFacing)
    return asfloat(materialMediumTable[hit.materialIndex * 8 + 7])[hero < 3 ? hero : 1];
  if(stack.count > 0 && stack.entries[stack.count - 1] == hit.instanceIndex) --stack.count;
  return currentMediumIor(stack, hero);
}

MediumStack initialMediumStack(float3 origin)
{
  MediumStack result = (MediumStack)0;
  RayDesc probe;
  probe.Origin = origin;
  probe.Direction = normalize(float3(0.317f, 0.531f, 0.785f));
  probe.TMin = 0.0f; probe.TMax = FAR_DISTANCE;
  uint outsideDepth = 0;
  MediumBoundaryProgress previousBoundary = (MediumBoundaryProgress)0;
  // Closed, consistently oriented media are nested or disjoint. While probing
  // toward infinity, paired front/back crossings enclose regions outside the
  // camera. Unpaired exits identify its containing media, inner to outer.
  for(uint crossing = 0; crossing < MEDIUM_PROBE_LIMIT; ++crossing)
  {
    SurfaceHit hit;
    if(!traceGeometry(probe, hit,previousBoundary.active?previousBoundary.instanceIndex:0xffffffffu,previousBoundary.processedDistance))
    {
      if(outsideDepth != 0) mediumError(2);
      for(uint a = 0; a < result.count / 2; ++a)
      {
        uint temporary = result.entries[a];
        result.entries[a] = result.entries[result.count - 1 - a];
        result.entries[result.count - 1 - a] = temporary;
      }
      return result;
    }
    uint4 metadata = materialMediumTable[hit.materialIndex * 8];
    if(metadata.z != 1) { mediumError(4); return result; }
    if(metadata.x != 0)
    {
      if(mediumBoundaryTangent(probe.Origin,probe.Direction,hit))
      {
        previousBoundary=recordMediumBoundary(probe,hit);
        previousBoundary.planeActive=false;
        probe.TMin=mediumResumeDistance(hit.distance);
        continue;
      }
      if(repeatedMediumBoundary(previousBoundary, hit))
      {
        previousBoundary.processedDistance=max(previousBoundary.processedDistance,hit.distance);
        probe.TMin=mediumResumeDistance(previousBoundary.processedDistance);
        continue;
      }
      previousBoundary = recordMediumBoundary(probe, hit);
      if(hit.frontFacing) ++outsideDepth;
      else if(outsideDepth > 0) --outsideDepth;
      else
      {
        if(result.count >= MEDIUM_STACK_CAPACITY) { mediumError(1); return result; }
        result.entries[result.count++] = hit.instanceIndex;
      }
    }
    probe.TMin = metadata.x != 0 ? mediumResumeDistance(hit.distance) : asfloat(max(asuint(hit.distance)+1u,0x00800000u));
  }
  mediumError(2); // Explicit bounded-probe safeguard, never silently truncate.
  return result;
}

void initializeCameraMediumCache()
{
  // Publish an empty, invalid header first. A failed probe can therefore never
  // expose partially written entries to the path dispatch in the same submit.
  cameraMediumCache[0] = uint4(0);
  MediumStack stack = (MediumStack)0;
  if(pc.environment.w > 0.0f) stack = initialMediumStack(pc.origin.xyz);
  if(renderErrors[0] != 0) return;
  uint dispersive = 0;
  for(uint row = 0; row < MEDIUM_STACK_CAPACITY / 4; ++row)
  {
    uint4 instanceIDs = uint4(0);
    for(uint lane = 0; lane < 4; ++lane)
    {
      uint index = row * 4 + lane;
      if(index >= stack.count) break;
      uint instanceIndex = stack.entries[index];
      instanceIDs[lane] = instanceIndex;
      float3 ior = asfloat(materialMediumTable[mediumInstanceRow(instanceIndex) + 7]).xyz;
      if(abs(ior.x - ior.z) > 1.0e-6f) dispersive = 1;
    }
    cameraMediumCache[1 + row] = instanceIDs;
  }
  cameraMediumCache[0] = uint4(stack.count, dispersive, 1, 0);
}

MediumStack loadCameraMediumStack(inout uint seed, inout uint hero)
{
  MediumStack stack = (MediumStack)0;
  uint4 header = cameraMediumCache[0];
  if(header.z != 1 || header.x > MEDIUM_STACK_CAPACITY)
  {
    mediumError(4);
    return stack;
  }
  stack.count = header.x;
  for(uint index = 0; index < stack.count; ++index)
    stack.entries[index] = cameraMediumCache[1 + index / 4][index % 4];
  // Containment and material calibration are shared, but this spectral choice
  // remains per path. It consumes exactly the RNG draw used by the old probe.
  if(hero == MEDIUM_NO_HERO && header.y != 0)
    hero = min(uint(rand(seed) * 3.0f), 2);
  return stack;
}

float3 mediumTransmittance(MediumCoefficients m, float distance)
{
  return exp(-mediumExtinction(m) * max(distance, 0.0f));
}
float mediumSourceIntegral(float extinction, float emission, float distance)
{
  if(extinction == 0.0f) return emission * distance;
  float tau = extinction * distance;
  float attenuation = tau < 0.001f ? tau * (1.0f - 0.5f * tau + tau * tau / 6.0f) : 1.0f - exp(-tau);
  return emission * attenuation / extinction;
}
float3 mediumEmission(MediumCoefficients m, float distance)
{
  float3 sigmaT = mediumExtinction(m);
  return float3(mediumSourceIntegral(sigmaT.x, m.emission.x, distance),
                mediumSourceIntegral(sigmaT.y, m.emission.y, distance),
                mediumSourceIntegral(sigmaT.z, m.emission.z, distance));
}

MediumFlight sampleMedium(MediumCoefficients m, float boundaryDistance, float3 throughput, inout uint seed)
{
  MediumFlight result;
  result.distance = boundaryDistance; result.scattered = false;
  float3 sigmaT = mediumExtinction(m), sigmaS = mediumScattering(m);
  if(!any(sigmaS > 0.0f)) { result.weight = mediumTransmittance(m, boundaryDistance); return result; }
  float largest = max(throughput.x, max(throughput.y, throughput.z));
  float3 probability = largest > 0.0f ? max(throughput, float3(0.0f)) / largest : float3(1.0f);
  probability /= probability.x + probability.y + probability.z;
  float choose = rand(seed);
  uint channel = choose < probability.x ? 0 : (choose < probability.x + probability.y ? 1 : 2);
  float distance=FAR_DISTANCE;
  if(sigmaT[channel]>0.0f)
  {
    // Use an open interval for the inverse CDF; rand itself includes 0 and 1.
    float u=clamp(rand(seed),1.1641532182693481e-10f,asfloat(0x3f7fffffu));
    // 1-u rounds to one for very small positive draws. Preserve their short
    // positive flights rather than manufacturing a collision on the boundary.
    float opticalDepth=u<.001f?u*(1.0f+u*(.5f+u/3.0f)):-log(max(1.0f-u,1.0e-30f));
    distance=opticalDepth/sigmaT[channel];
  }
  result.scattered = distance < boundaryDistance;
  result.distance = min(distance, boundaryDistance);
  float3 transmittance = mediumTransmittance(m, result.distance);
  float pdf = dot(probability, result.scattered ? sigmaT * transmittance : transmittance);
  result.weight = pdf > 0.0f ? transmittance * (result.scattered ? sigmaS : float3(1.0f)) / pdf : float3(0.0f);
  return result;
}

struct MediumGridCursor
{
  GpuInstance instance;
  int3 cell,step;
  float3 next,delta;
  float distance;
};
MediumGridCursor mediumGridBegin(uint instanceIndex,float3 origin,float3 direction)
{
  MediumGridCursor c=(MediumGridCursor)0;c.instance=instances[instanceIndex];
  float3 extent=c.instance.objectMax.xyz-c.instance.objectMin.xyz;
  float3 p=(mediumTransform(c.instance.worldToObject,origin,1)-c.instance.objectMin.xyz)*32.0f/extent;
  float3 d=mediumTransform(c.instance.worldToObject,direction,0)*32.0f/extent;
  c.cell=clamp(int3(floor(p)),int3(0),int3(31));
  for(uint axis=0;axis<3;++axis)
  {
    c.step[axis]=d[axis]<0?-1:1;
    c.delta[axis]=d[axis]!=0?1.0f/abs(d[axis]):FAR_DISTANCE;
    float edge=float(c.cell[axis]+(c.step[axis]>0?1:0));
    c.next[axis]=d[axis]!=0?max(0.0f,(edge-p[axis])/d[axis]):FAR_DISTANCE;
  }
  return c;
}
MediumCoefficients mediumGridValue(MediumGridCursor c)
{
  int3 cell=clamp(c.cell,int3(0),int3(31));
  uint index=uint(cell.x+32*cell.y+1024*cell.z);
  return mediumTableLookup(c.instance.offsets.w+index*8);
}
float mediumGridEnd(MediumGridCursor c,float limit)
{
  return min(limit,min(c.next.x,min(c.next.y,c.next.z)));
}
void mediumGridAdvance(inout MediumGridCursor c,float endpoint)
{
  for(uint axis=0;axis<3;++axis)if(c.next[axis]<=endpoint) {
    c.cell[axis]+=c.step[axis];c.next[axis]+=c.delta[axis];
  }
  c.distance=endpoint;
}
bool currentMediumHasGrid(MediumStack stack)
{
  return stack.count>0 && instances[stack.entries[stack.count-1]].offsets.w!=0;
}
float3 mediumRayTransmittance(MediumStack stack,float3 origin,float3 direction,float distance)
{
  if(!currentMediumHasGrid(stack))return mediumTransmittance(currentMedium(stack),distance);
  MediumGridCursor cursor=mediumGridBegin(stack.entries[stack.count-1],origin,direction);
  float3 tau=float3(0);
  for(uint iteration=0;iteration<128;++iteration)
  {
    float end=mediumGridEnd(cursor,distance);
    tau+=mediumExtinction(mediumGridValue(cursor))*max(end-cursor.distance,0.0f);
    if(end>=distance)return exp(-tau);
    mediumGridAdvance(cursor,end);
  }
  mediumError(4);return float3(0);
}
MediumFlight sampleCurrentMedium(MediumStack stack,float3 origin,float3 direction,float distance,
  float3 throughput,inout uint seed,out MediumCoefficients eventMedium,out float3 emission)
{
  eventMedium=currentMedium(stack);
  if(!currentMediumHasGrid(stack)) {
    emission=mediumEmission(eventMedium,distance);
    return sampleMedium(eventMedium,distance,throughput,seed);
  }
  MediumFlight result;result.distance=distance;result.scattered=false;result.weight=float3(1);
  emission=float3(0);
  float largest=max(throughput.x,max(throughput.y,throughput.z));
  float3 probability=largest>0?max(throughput,float3(0))/largest:float3(1);
  probability/=dot(probability,float3(1));
  float choose=rand(seed);
  uint channel=choose<probability.x?0:(choose<probability.x+probability.y?1:2);
  float u=clamp(rand(seed),1.1641532182693481e-10f,asfloat(0x3f7fffffu));
  float target=u<.001f?u*(1.0f+u*(.5f+u/3.0f)):-log(max(1.0f-u,1.0e-30f));
  float3 tau=float3(0);
  bool hasScattering=false;
  MediumGridCursor cursor=mediumGridBegin(stack.entries[stack.count-1],origin,direction);
  for(uint iteration=0;iteration<128;++iteration)
  {
    float end=mediumGridEnd(cursor,distance),length=max(end-cursor.distance,0.0f);
    MediumCoefficients m=mediumGridValue(cursor);float3 sigmaT=mediumExtinction(m);
    hasScattering=hasScattering || any(mediumScattering(m)>0);
    emission+=exp(-tau)*mediumEmission(m,length);
    if(!result.scattered && sigmaT[channel]>0 && target<tau[channel]+sigmaT[channel]*length)
    {
      float local=max(0.0f,(target-tau[channel])/sigmaT[channel]);
      result.distance=cursor.distance+local;result.scattered=true;eventMedium=m;
      float3 transmittance=exp(-tau-sigmaT*local);
      float pdf=dot(probability,sigmaT*transmittance);
      result.weight=pdf>0?transmittance*mediumScattering(m)/pdf:float3(0);
    }
    tau+=sigmaT*length;
    if(end>=distance) {
      if(!hasScattering) {result.scattered=false;result.distance=distance;result.weight=exp(-tau);return result;}
      if(!result.scattered) {
        float3 transmittance=exp(-tau);float pdf=dot(probability,transmittance);
        result.weight=pdf>0?transmittance/pdf:float3(0);
      }
      return result;
    }
    mediumGridAdvance(cursor,end);
  }
  mediumError(4);result.weight=float3(0);return result;
}

float mediumHg(float cosine, float g)
{
  float denominator = max(1.0f + g * g - 2.0f * g * clamp(cosine, -1.0f, 1.0f), 1.0e-12f);
  return (1.0f - g * g) / (12.566370614359172f * denominator * sqrt(denominator));
}
float3 mediumPhaseProbabilities(MediumCoefficients m)
{
  float3 weights = float3(mediumLuminance(m.scatteringVolume), mediumLuminance(m.scatteringTransmission), mediumLuminance(m.scatteringSubsurface));
  float total = weights.x + weights.y + weights.z;
  return total > 0.0f ? weights / total : float3(1, 0, 0);
}
float mediumPhasePdf(MediumCoefficients m, float3 incomingTravel, float3 outgoingTravel)
{
  float cosine = dot(incomingTravel, outgoingTravel);
  float3 phase = float3(mediumHg(cosine, m.anisotropy.x), mediumHg(cosine, m.anisotropy.y), mediumHg(cosine, m.anisotropy.z));
  return dot(mediumPhaseProbabilities(m), phase);
}
float3 mediumPhase(MediumCoefficients m, float3 incomingTravel, float3 outgoingTravel)
{
  float cosine = dot(incomingTravel, outgoingTravel);
  float3 numerator = m.scatteringVolume * mediumHg(cosine, m.anisotropy.x)
                   + m.scatteringTransmission * mediumHg(cosine, m.anisotropy.y)
                   + m.scatteringSubsurface * mediumHg(cosine, m.anisotropy.z);
  float3 sigmaS = mediumScattering(m);
  return float3(sigmaS.x > 0 ? numerator.x / sigmaS.x : 0,
                sigmaS.y > 0 ? numerator.y / sigmaS.y : 0,
                sigmaS.z > 0 ? numerator.z / sigmaS.z : 0);
}
float3 sampleMediumPhase(MediumCoefficients m, float3 incomingTravel, inout uint seed, out float pdf, out float3 weight)
{
  float3 probabilities = mediumPhaseProbabilities(m);
  float choose = rand(seed);
  uint component = choose < probabilities.x ? 0 : (choose < probabilities.x + probabilities.y ? 1 : 2);
  float g = m.anisotropy[component], u = rand(seed);
  float cosine;
  if(abs(g) < 1.0e-4f) cosine = 1.0f - 2.0f * u;
  else
  {
    float value = (1.0f - g * g) / (1.0f - g + 2.0f * g * u);
    cosine = clamp((1.0f + g * g - value * value) / (2.0f * g), -1.0f, 1.0f);
  }
  float phi = 6.283185307179586f * rand(seed), sine = sqrt(max(0.0f, 1.0f - cosine * cosine));
  float3 axis = abs(incomingTravel.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0);
  float3 t = normalize(cross(axis, incomingTravel)), b = cross(incomingTravel, t);
  float3 direction = normalize(t * (cos(phi) * sine) + b * (sin(phi) * sine) + incomingTravel * cosine);
  pdf = mediumPhasePdf(m, incomingTravel, direction);
  weight = mediumPhase(m, incomingTravel, direction) / max(pdf, 1.0e-30f);
  return direction;
}

#endif
