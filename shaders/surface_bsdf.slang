// SPDX-License-Identifier: Apache-2.0
// Caustra's MaterialX Standard Surface / native OpenPBR surface closures.
// Evaluators derive from Autodesk Aurora's MaterialX GLSL-to-HLSL port;
// see third_party/aurora/ORIGIN.md for sources, changes and license.
#ifndef CAUSTRA_SURFACE_BSDF
#define CAUSTRA_SURFACE_BSDF

#include "aurora/standard_surface_math.slang"
#include "aurora/open_pbr_math.slang"
#include "aurora/surface_optics_math.slang"

static const float SURFACE_PI = 3.14159265358979323846f;
static const float SURFACE_INV_PI = 0.31830988618379067154f;
static const float SURFACE_MIN_ROUGHNESS = 0.02f;
static const float SURFACE_MIN_ALPHA = 0.0004f;
static const float SURFACE_COS_EPS = 0.00001f;

struct SurfaceEval { float3 fCos; float pdf; };
// Events: 0 absorb, 1 diffuse reflection, 2 glossy reflection, 3 transmission,
// 4 entry into the same physical transmission/SSS mixture (never an exit).
struct SurfaceSample { float3 direction; float3 weight; float pdf; uint event; bool delta; };
struct SurfaceGuides
{
  float3 diffuseAlbedo, specularAlbedo, normal;
  float roughness;
};
struct SurfaceClosure
{
  GpuMaterial material;
  float3 normal, geometricNormal, tangent, bitangent;
  float3 coatNormal, coatTangent, coatBitangent;
  float2 alpha, coatAlpha;
  float substrateIor;
  float3 substrateDarkening;
  float3 diffuseColor;
  float3 probabilities; // cosine (diffuse + sheen), base GGX, coat GGX
  float transmissionProbability, translucentProbability, fuzzProbability;
  float2 transmissionAlpha;
  float etaI, etaT;
  bool entering, interfaceConfigured;
  float3 emission;
  float opacity;
};

bool surfaceFinite(float3 v) { return !any(isnan(v)) && !any(isinf(v)); }
float surfaceLuminance(float3 v) { return dot(max(v, float3(0.0f)), float3(0.2126f, 0.7152f, 0.0722f)); }
float3 surfaceNormalize(float3 v, float3 fallback)
{
  float lengthSquared = dot(v, v);
  return lengthSquared > 1.0e-20f && !isinf(lengthSquared) ? v * rsqrt(lengthSquared) : fallback;
}

void surfaceFrame(float3 n, float3 inputTangent, out float3 t, out float3 b)
{
  float3 fallbackAxis = abs(n.z) < 0.999f ? float3(0, 0, 1) : float3(0, 1, 0);
  float3 fallbackTangent = normalize(cross(fallbackAxis, n));
  t = surfaceNormalize(inputTangent - n * dot(inputTangent, n), fallbackTangent);
  b = cross(n, t);
}

float3 surfaceOrientedNormal(float3 input, float3 geometricNormal, float3 view)
{
  float3 n = surfaceNormalize(input, geometricNormal);
  if(dot(n, geometricNormal) < 0.0f) n = -n;
  // Very strong normal maps can point away from the viewer. Use the geometric
  // normal there rather than letting a reflected ray cross the physical surface.
  return dot(n, view) > SURFACE_COS_EPS ? n : geometricNormal;
}

float3 surfaceToLocal(float3 v, float3 t, float3 b, float3 n)
{
  return float3(dot(v, t), dot(v, b), dot(v, n));
}

float3 surfaceToWorld(float3 v, float3 t, float3 b, float3 n)
{
  return v.x * t + v.y * b + v.z * n;
}

// The anisotropy conversion is the MaterialX roughness_anisotropy node,
// followed by a finite floor on BOTH axes (the original opaque path's r=0.02).
float2 surfaceAlpha(float roughness, float anisotropy)
{
  return max(float2(SURFACE_MIN_ALPHA), mx_roughness_anisotropy(max(roughness, SURFACE_MIN_ROUGHNESS), anisotropy));
}

// Adapted from Aurora BSDFCommon.slang's Heitz 2018 visible-normal sampler.
// Eval uses the exact distribution of this proposal, independently of the
// MaterialX BRDF's masking approximation and energy compensation.
float3 surfaceSampleGgxNormal(float3 view, float2 alpha, float2 random)
{
  float3 stretched = normalize(float3(alpha.x * view.x, alpha.y * view.y, view.z));
  float planarLength = dot(stretched.xy, stretched.xy);
  float3 t = planarLength > 0.0f ? float3(-stretched.y, stretched.x, 0.0f) * rsqrt(planarLength) : float3(1, 0, 0);
  float3 b = cross(stretched, t);
  float radius = sqrt(random.x), phi = 2.0f * SURFACE_PI * random.y;
  float x = radius * cos(phi), y = radius * sin(phi);
  float s = 0.5f * (1.0f + stretched.z);
  y = (1.0f - s) * sqrt(max(0.0f, 1.0f - x * x)) + s * y;
  float3 h = x * t + y * b + sqrt(max(0.0f, 1.0f - x * x - y * y)) * stretched;
  return normalize(float3(alpha.x * h.x, alpha.y * h.y, max(0.0f, h.z)));
}

float surfaceGgxPdf(float3 view, float3 light, float2 alpha)
{
  if(view.z <= SURFACE_COS_EPS || light.z <= 0.0f) return 0.0f;
  float3 h = normalize(view + light);
  if(h.z <= 0.0f || dot(view, h) <= 0.0f) return 0.0f;
  float3 scaledH = float3(h.xy / alpha, h.z);
  float denominator = dot(scaledH, scaledH);
  float d = 1.0f / (SURFACE_PI * alpha.x * alpha.y * denominator * denominator);
  float projected = dot(alpha * view.xy, alpha * view.xy) / (view.z * view.z);
  float g1 = 2.0f / (1.0f + sqrt(1.0f + projected));
  return d * g1 / (4.0f * view.z);
}

float surfaceNominalIor(GpuMaterial material)
{
  float nd=max(material.specular.y,1.0f);
  if(material.program.z==1)
  {
    // Canonical OpenPBR: refraction uses the same specular-weight-modulated
    // dielectric IOR as reflection (specification 1.1.1, dielectric base).
    float rootF0=sqrt(clamp(max(material.specular.x,0.0f)*mx_ior_to_f0(nd),0.0f,.99999f));
    nd=(1.0f+rootF0)/(1.0f-rootF0);
  }
  return nd;
}

// Cauchy fit through the Fraunhofer C/F separation and modulated d-line IOR.
// thinFilm.w is inverse Abbe number; the integrator selects one RGB hero channel
// once for the path, including all nested medium boundaries.
float surfaceDispersiveIor(GpuMaterial material,uint channel)
{
  // MaterialX 1.39.5 declares glTF dispersion but leaves it disconnected in
  // its official nodegraph. Preserve the input without activating a new VDF.
  if(material.program.z==3)return surfaceNominalIor(material);
  float wavelength=channel==0?0.610f:(channel==1?0.550f:0.460f);
  float nd=surfaceNominalIor(material);
  float b=(nd-1.0f)*max(material.thinFilm.w,0.0f)/(1.0f/(.48613f*.48613f)-1.0f/(.65627f*.65627f));
  return max(1.0f,nd+b*(1.0f/(wavelength*wavelength)-1.0f/(.58756f*.58756f)));
}

void configureSurfaceInterface(inout SurfaceClosure c,float etaI,float etaT)
{
  c.etaI=max(etaI,0.0001f);c.etaT=max(etaT,0.0001f);c.interfaceConfigured=true;
}

float surfaceGgxD(float3 h,float2 alpha)
{
  if(h.z<=0.0f)return 0.0f;
  float3 q=float3(h.xy/alpha,h.z);float d=dot(q,q);
  return 1.0f/(SURFACE_PI*alpha.x*alpha.y*d*d);
}
float surfaceGgxG1(float3 w,float2 alpha)
{
  if(abs(w.z)<=SURFACE_COS_EPS)return 0.0f;
  float q=dot(alpha*w.xy,alpha*w.xy)/(w.z*w.z);
  return 2.0f/(1.0f+sqrt(1.0f+q));
}

bool surfaceHasTransport(SurfaceClosure c)
{
  return c.material.program.z!=5 && (c.material.transmission.x>0.0f || c.material.subsurface.x>0.0f);
}
bool surfaceHasExtendedReflection(SurfaceClosure c)
{
  return surfaceHasTransport(c) || c.material.thinFilm.x*c.material.thinFilm.y>0.0f
    || (c.material.program.z==1 && c.material.fuzz.x>0.0f)
    || c.material.thinFilm.w>0.0f
    || (c.interfaceConfigured && (c.etaI!=1.0f || c.etaT!=surfaceNominalIor(c.material)));
}

// Shared transmission/SSS boundary; the integrator combines their interior
// coefficients into one physical medium, so both terms share one directional PDF.
float surfaceBoundaryWeight(SurfaceClosure c)
{
  float t=c.material.transmission.x;
  float s=c.material.subsurface.w>0.5f?0.0f:(1.0f-t)*c.material.subsurface.x;
  return (1.0f-c.material.surface.x)*(t+s);
}

float3 surfaceFilmFresnel(SurfaceClosure c,float cosine,bool metal)
{
  float eta=surfaceHasTransport(c)?c.etaT/c.etaI:c.substrateIor/c.etaI;
  if(!surfaceHasTransport(c) && c.interfaceConfigured)
    eta*=c.etaT/surfaceNominalIor(c.material);
  float3 ordinary=metal?(c.material.program.z==1?
    openPbrFresnelMetal(cosine,c.material.baseColor.xyz*c.material.factors.x,c.material.specularColor.xyz):
    float3(0.0f)):float3(openPbrFresnelDielectric(cosine,eta));
  FilmFresnelData fd=(FilmFresnelData)0;
  if(metal && c.material.program.z==0)
  {
    mx_artistic_to_complex_ior(c.material.baseColor.xyz*c.material.factors.x,
      c.material.specularColor.xyz*c.material.specular.x,fd.ior,fd.extinction);
    ordinary=film_fresnel_conductor(cosine,fd.ior,fd.extinction);
  }
  else if(metal)
  {
    fd.model=FRESNEL_MODEL_SCHLICK;fd.F0=c.material.baseColor.xyz*c.material.factors.x;
    fd.F82=c.material.specularColor.xyz;fd.F90=float3(1.0f);fd.exponent=5.0f;
  }
  else fd.ior=float3(eta);
  if(!metal)fd.model=FRESNEL_MODEL_DIELECTRIC;
  else if(c.material.program.z==0)fd.model=FRESNEL_MODEL_CONDUCTOR;
  // A real dielectric interface must retain TIR even with a spectral film.
  if(!metal && ordinary.x>=1.0f)return float3(1.0f);
  float filmWeight=c.material.thinFilm.y>0.0f?c.material.thinFilm.x:0.0f;
  if(filmWeight<=0.0f)return ordinary;
  fd.tf_thickness=c.material.thinFilm.y;fd.tf_ior=c.material.thinFilm.z/c.etaI;
  if(abs(fd.tf_ior-1.0f)<1.0e-6f)return ordinary;
  return lerp(ordinary,film_fresnel_airy(clamp(cosine,.000001f,1.0f),fd),filmWeight);
}

float3 surfaceInterfaceFresnel(SurfaceClosure c,float cosine)
{
  float3 f=surfaceFilmFresnel(c,cosine,false);
  // Parallel, zero-thickness sheet: sum all internal specular reflections.
  return c.material.subsurface.w>0.5f?2.0f*f/(float3(1.0f)+f):f;
}

float surfaceInterfaceProbability(SurfaceClosure c,float cosine)
{
  return saturate(surfaceLuminance(surfaceInterfaceFresnel(c,cosine)));
}

float3 surfaceDielectricTint(SurfaceClosure c)
{
  // OpenPBR specular_color colors the first reflection from above, not internal
  // reflection or the BTDF. Either face of a thin sheet is an exterior face.
  return c.material.program.z==1 && surfaceHasTransport(c) && !c.entering
    && c.material.subsurface.w<.5f?float3(1.0f):c.material.specularColor.xyz;
}

SurfaceClosure makeSurfaceClosure(MaterialEvaluation evaluation, MaterialContext context, float3 view)
{
  if(evaluation.material.program.z>=2) return makeOfficialClosure(evaluation,context,view);
  SurfaceClosure c = (SurfaceClosure)0;
  c.material = evaluation.material;
  c.material.baseColor.xyz = max(c.material.baseColor.xyz, float3(0.0f));
  c.material.surface.x = saturate(c.material.surface.x);
  c.material.surface.y = clamp(c.material.surface.y, SURFACE_MIN_ROUGHNESS, 1.0f);
  bool openPbr = c.material.program.z == 1;
  c.material.specular.x = openPbr ? max(0.0f, c.material.specular.x) : saturate(c.material.specular.x);
  c.material.specular.y = max(1.0f, c.material.specular.y);
  c.material.specular.z = saturate(c.material.specular.z);
  c.material.specularColor.xyz = saturate(c.material.specularColor.xyz);
  c.material.coat.x = saturate(c.material.coat.x);
  c.material.coat.y = clamp(c.material.coat.y, SURFACE_MIN_ROUGHNESS, 1.0f);
  c.material.coat.z = max(1.0f, c.material.coat.z);
  c.material.coat.w = saturate(c.material.coat.w);
  c.material.coatColor.xyz = saturate(c.material.coatColor.xyz);
  c.material.sheen.x = saturate(c.material.sheen.x);
  c.material.sheen.y = clamp(c.material.sheen.y, SURFACE_MIN_ROUGHNESS, 1.0f);
  c.material.sheenColor.xyz = saturate(c.material.sheenColor.xyz);
  c.material.factors.x = saturate(c.material.factors.x);
  c.material.factors.y = saturate(c.material.factors.y);
  c.material.factors.w = saturate(c.material.factors.w);
  c.material.transmission.x=saturate(c.material.transmission.x);
  c.material.transmission.w=clamp(c.material.transmission.w,-1.0f,1.0f);
  c.material.transmissionColor.xyz=saturate(c.material.transmissionColor.xyz);
  c.material.subsurface.x=saturate(c.material.subsurface.x);
  c.material.thinFilm.x=saturate(c.material.thinFilm.x);
  c.material.thinFilm.y=max(0.0f,c.material.thinFilm.y);
  c.material.thinFilm.z=max(1.0f,c.material.thinFilm.z);
  c.material.fuzz.x=saturate(c.material.fuzz.x);
  c.material.fuzz.y=clamp(c.material.fuzz.y,0.01f,1.0f);
  c.material.fuzzColor.xyz=saturate(c.material.fuzzColor.xyz);
  c.diffuseColor=c.material.baseColor.xyz;
  if(!openPbr)
  {
    float gamma=1.0f+c.material.coat.x*saturate(c.material.fuzz.w);
    c.diffuseColor=pow(c.material.baseColor.xyz,float3(gamma));
    c.material.subsurfaceColor.xyz=pow(max(c.material.subsurfaceColor.xyz,float3(0)),float3(gamma));
  }
  c.etaI=1.0f;c.etaT=surfaceNominalIor(c.material);c.entering=true;c.interfaceConfigured=false;
  c.emission = max(c.material.emission.xyz, float3(0.0f));
  c.opacity = saturate(c.material.factors.z);

  c.geometricNormal = surfaceNormalize(context.geometricNormal, float3(0, 0, 1));
  if(dot(c.geometricNormal, view) < 0.0f) c.geometricNormal = -c.geometricNormal;
  c.normal = surfaceOrientedNormal(evaluation.normal, c.geometricNormal, view);
  c.coatNormal = surfaceOrientedNormal(evaluation.coatNormal, c.geometricNormal, view);
  surfaceFrame(c.normal, evaluation.tangent, c.tangent, c.bitangent);
  float handedness = dot(c.bitangent, context.bitangent) < 0.0f ? -1.0f : 1.0f;
  float angle = c.material.specular.w * (2.0f * SURFACE_PI);
  c.tangent = cos(angle) * c.tangent + sin(angle) * c.bitangent * handedness;
  c.bitangent = cross(c.normal, c.tangent);
  surfaceFrame(c.coatNormal, evaluation.coatTangent, c.coatTangent, c.coatBitangent);
  float coatAngle=c.material.fuzz.z*(2.0f*SURFACE_PI);
  c.coatTangent=cos(coatAngle)*c.coatTangent+sin(coatAngle)*c.coatBitangent;
  c.coatBitangent=cross(c.coatNormal,c.coatTangent);
  c.substrateIor = c.material.specular.y;
  c.substrateDarkening = float3(1.0f);
  if(openPbr)
  {
    // Native open_pbr_surface graph: coat-induced roughness broadening, then
    // OpenPBR anisotropy. This mapping differs from Standard Surface.
    float r = c.material.surface.y, rc = c.material.coat.y;
    float broadened = pow(min(1.0f, r*r*r*r + 2.0f*rc*rc*rc*rc), 0.25f);
    float effective = lerp(r, broadened, c.material.coat.x);
    c.alpha = openPbrAlpha(effective, c.material.specular.z);
    c.coatAlpha = openPbrAlpha(rc, c.material.coat.w);
    float ratio = c.material.specular.y / c.material.coat.z;
    float eta = lerp(c.material.specular.y, ratio > 1.0f ? ratio : 1.0f / ratio, c.material.coat.x);
    float epsilon = sign(eta - 1.0f) * sqrt(clamp(c.material.specular.x * mx_ior_to_f0(eta), 0.0f, 0.99999f));
    c.substrateIor = (1.0f + epsilon) / (1.0f - epsilon);
    float k = 1.0f - (1.0f - mx_ior_to_f0(c.material.coat.z)) / (c.material.coat.z * c.material.coat.z);
    float3 eBase = lerp(lerp(c.material.baseColor.xyz,c.material.subsurfaceColor.xyz,c.material.subsurface.x), c.material.baseColor.xyz * c.material.specular.x, c.material.surface.x);
    float3 darkening = (1.0f - k) / max(float3(0.000001f), float3(1.0f) - eBase * k);
    c.substrateDarkening = lerp(float3(1.0f), darkening, c.material.coat.x * c.material.factors.w);
  }
  else
  {
    float coatInfluence=c.material.coat.x*c.material.coat.y*saturate(c.material.volume.w);
    c.alpha = surfaceAlpha(lerp(c.material.surface.y,1.0f,coatInfluence), c.material.specular.z);
    c.coatAlpha = surfaceAlpha(c.material.coat.y, c.material.coat.w);
  }
  float rt=clamp(c.material.surface.y+c.material.transmission.w,SURFACE_MIN_ROUGHNESS,1.0f);
  c.transmissionAlpha=openPbr?c.alpha:surfaceAlpha(lerp(rt,1.0f,c.material.coat.x*c.material.coat.y*saturate(c.material.volume.w)),c.material.specular.z);

  float metallic = c.material.surface.x;
  float nDotV = saturate(dot(c.normal, view));
  float coatNDotV = saturate(dot(c.coatNormal, view));
  float f0 = mx_ior_to_f0(c.substrateIor);
  float mainAlbedo = saturate(mx_ggx_directional_albedo(nDotV, mx_average_roughness(c.alpha), f0, 1.0f));
  float coatAlbedo = saturate(mx_ggx_directional_albedo(coatNDotV, mx_average_roughness(c.coatAlpha), mx_ior_to_f0(c.material.coat.z), 1.0f));
  float diffuseScore = (1.0f - metallic) *
    (1.0f-c.material.transmission.x)*
    (c.material.factors.x * surfaceLuminance(c.diffuseColor)*(1.0f-c.material.subsurface.x) + c.material.sheen.x * surfaceLuminance(c.material.sheenColor.xyz));
  if(openPbr && c.material.subsurface.w>.5f)
    diffuseScore+=(1.0f-metallic)*(1.0f-c.material.transmission.x)*c.material.subsurface.x
      *.5f*(1.0f-c.material.subsurface.z)*surfaceLuminance(c.material.subsurfaceColor.xyz);
  float specularScore = metallic * max(surfaceLuminance(c.material.baseColor.xyz * c.material.factors.x), 0.02f) * (openPbr ? c.material.specular.x : 1.0f)
    + (1.0f - metallic) * c.material.specular.x * surfaceLuminance(c.material.specularColor.xyz) * max(mainAlbedo, 0.02f);
  float coatScore = c.material.coat.x * max(coatAlbedo, 0.02f);
  c.transmissionProbability=surfaceBoundaryWeight(c)*max(surfaceLuminance(c.material.transmissionColor.xyz),.05f);
  c.translucentProbability=c.material.subsurface.w>0.5f?
    (1.0f-metallic)*(1.0f-c.material.transmission.x)*c.material.subsurface.x*max(surfaceLuminance(c.material.subsurfaceColor.xyz),.01f)
      *(openPbr?.5f*(1.0f+c.material.subsurface.z):1.0f):0.0f;
  c.fuzzProbability=openPbr?c.material.fuzz.x*max(zeltnerAlbedo(nDotV,c.material.fuzz.y),.02f)*max(surfaceLuminance(c.material.fuzzColor.xyz),.01f):0.0f;
  c.probabilities = float3(diffuseScore, specularScore, coatScore);
  float sum = c.probabilities.x + c.probabilities.y + c.probabilities.z+c.transmissionProbability+c.translucentProbability+c.fuzzProbability;
  c.probabilities = sum > 0.0f ? c.probabilities / sum : float3(1, 0, 0);
  if(sum>0.0f){c.transmissionProbability/=sum;c.translucentProbability/=sum;c.fuzzProbability/=sum;}
  return c;
}

// Current MaterialX dielectric/generalized-Schlick primitives, cosine included.
// The distribution and compensation match the native OpenPBR nodegraph rather
// than translating its parameters to Aurora's older Standard Surface model.
float3 surfaceOpenPbrDielectric(float3 view, float3 light, float weight, float3 tint, float ior,
  float2 alpha, float3 n, float3 t, float3 base)
{
  if(weight <= 0.0f) return base;
  float nv = dot(n, view), nl = dot(n, light);
  if(nv <= SURFACE_COS_EPS || nl <= SURFACE_COS_EPS) return base;
  float3 h = normalize(view + light), b = cross(n, t);
  float f = openPbrFresnelDielectric(saturate(dot(view, h)), ior);
  float averageAlpha = mx_average_roughness(alpha);
  float d = mx_ggx_NDF(t, b, h, dot(n, h), alpha.x, alpha.y);
  float g = mx_ggx_smith_G(nl, nv, averageAlpha);
  float comp = openPbrGgxCompensation(nv, averageAlpha, float3(f)).x;
  float albedo = abs(ior-1.0f) < 0.000001f ? 0.0f : openPbrGgxAlbedo(nv, averageAlpha, float3(mx_ior_to_f0(ior)), float3(1)).x * comp;
  return (d * f * g * comp * weight / (4.0f * nv)) * tint + base * max(0.0f, 1.0f - albedo * weight);
}

float3 surfaceOpenPbrValue(SurfaceClosure c, float3 view, float3 light)
{
  GpuMaterial m = c.material;
  float3 value = float3(0.0f);
  float nv = dot(c.normal, view), nl = dot(c.normal, light);
  if(nv > SURFACE_COS_EPS && nl > SURFACE_COS_EPS)
  {
    float3 diffuse = openPbrDiffuse(nv, nl, clamp(dot(light, view), SURFACE_COS_EPS, 1.0f), m.factors.y, m.baseColor.xyz)
      * (m.factors.x * nl * SURFACE_INV_PI);
    float3 dielectric = surfaceOpenPbrDielectric(view, light, 1.0f, m.specularColor.xyz,
      c.substrateIor, c.alpha, c.normal, c.tangent, diffuse);
    float3 metal = float3(0.0f);
    if(m.surface.x > 0.0f && m.specular.x > 0.0f)
    {
      float3 h = normalize(view + light);
      float3 f = openPbrFresnelMetal(saturate(dot(view,h)), m.baseColor.xyz * m.factors.x, m.specularColor.xyz);
      float averageAlpha = mx_average_roughness(c.alpha);
      float d = mx_ggx_NDF(c.tangent, c.bitangent, h, dot(c.normal,h), c.alpha.x, c.alpha.y);
      float g = mx_ggx_smith_G(nl,nv,averageAlpha);
      metal = d * g * f * openPbrGgxCompensation(nv,averageAlpha,f) * m.specular.x / (4.0f*nv);
    }
    value = lerp(dielectric,metal,m.surface.x) * c.substrateDarkening;
  }
  value *= lerp(float3(1.0f),m.coatColor.xyz,m.coat.x);
  return surfaceOpenPbrDielectric(view,light,m.coat.x,float3(1.0f),m.coat.z,
    c.coatAlpha,c.coatNormal,c.coatTangent,value);
}

float3 surfaceEmission(SurfaceClosure c, float3 view)
{
  if(c.material.program.z != 1) return c.emission;
  // Native OpenPBR coat EDF: generalized Schlick from 1-F0 at normal to zero
  // at grazing, tinting only the covered portion of the emitter.
  float cosine = saturate(abs(dot(c.normal,view)));
  float attenuation = (1.0f-mx_ior_to_f0(c.material.coat.z)) * (1.0f-pow(1.0f-cosine,5.0f));
  return c.emission * lerp(float3(1.0f),c.material.coatColor.xyz*attenuation,c.material.coat.x);
}

float surfaceFuzzAlbedo(SurfaceClosure c,float3 view)
{
  return c.material.program.z==1?c.material.fuzz.x*zeltnerAlbedo(saturate(dot(c.normal,view)),c.material.fuzz.y):0.0f;
}

// Directional layer transmittance uses the same view-dependent closure
// throughput convention as the MaterialX reference layer node.
float3 surfaceOuterAttenuation(SurfaceClosure c,float3 view)
{
  float f=openPbrFresnelDielectric(saturate(dot(c.coatNormal,view)),c.material.coat.z/c.etaI);
  return lerp(float3(1.0f),c.material.coatColor.xyz,c.material.coat.x)
    *(1.0f-c.material.coat.x*f)*(1.0f-surfaceFuzzAlbedo(c,view));
}

float3 surfaceExtendedReflection(SurfaceClosure c,float3 view,float3 light)
{
  GpuMaterial m=c.material;float nv=dot(c.normal,view),nl=dot(c.normal,light);
  float3 value=float3(0.0f);
  if(nv>SURFACE_COS_EPS && nl>SURFACE_COS_EPS)
  {
    float3 v=surfaceToLocal(view,c.tangent,c.bitangent,c.normal);
    float3 l=surfaceToLocal(light,c.tangent,c.bitangent,c.normal);
    float3 h=surfaceNormalize(v+l,float3(0,0,1));
    float3 f=surfaceInterfaceFresnel(c,saturate(dot(v,h)));
    float d=surfaceGgxD(h,c.alpha),g=surfaceGgxG1(v,c.alpha)*surfaceGgxG1(l,c.alpha);
    float reflectionWeight=m.program.z==1?1.0f:m.specular.x;
    if(surfaceHasTransport(c))reflectionWeight=max(reflectionWeight,m.transmission.x+(m.subsurface.w>.5f?0.0f:(1.0f-m.transmission.x)*m.subsurface.x));
    float3 dielectric=d*g*f*surfaceDielectricTint(c)*reflectionWeight/(4.0f*nv);
    float3 diffuse=m.program.z==1?
      openPbrDiffuse(nv,nl,clamp(dot(light,view),-1.0f,1.0f),m.factors.y,m.baseColor.xyz)*(m.factors.x*nl*SURFACE_INV_PI):
      mx_diffuse_brdf_reflection(light,view,m.factors.x,c.diffuseColor,m.factors.y,c.normal);
    diffuse*=(1.0f-m.transmission.x)*(1.0f-m.subsurface.x);
    if(m.program.z==1 && m.subsurface.w>.5f)
      diffuse+=mx_diffuse_brdf_reflection(light,view,1.0f,m.subsurfaceColor.xyz,m.factors.y,c.normal)
        *.5f*(1.0f-m.subsurface.z)*(1.0f-m.transmission.x)*m.subsurface.x;
    if(m.program.z==0)
      diffuse=mx_sheen_brdf_reflection(light,view,m.sheen.x*(1.0f-m.transmission.x),m.sheenColor.xyz,m.sheen.y,c.normal,diffuse);
    float3 fv=surfaceInterfaceFresnel(c,saturate(nv));
    dielectric+=diffuse*max(float3(0.0f),float3(1.0f)-fv*reflectionWeight);
    float3 metalF=surfaceFilmFresnel(c,saturate(dot(v,h)),true);
    float3 metal=d*g*metalF/(4.0f*nv)*(m.program.z==1?m.specular.x:1.0f);
    // Opaque metals retain the graph's multiple-scattering energy fit.
    if(m.surface.x>0.0f)metal*=openPbrGgxCompensation(nv,mx_average_roughness(c.alpha),metalF);
    value=lerp(dielectric,metal,m.surface.x)*c.substrateDarkening;
  }
  value*=surfaceOuterAttenuation(c,view);
  if(m.coat.x>0.0f && dot(c.coatNormal,light)>SURFACE_COS_EPS)
  {
    float3 v=surfaceToLocal(view,c.coatTangent,c.coatBitangent,c.coatNormal);
    float3 l=surfaceToLocal(light,c.coatTangent,c.coatBitangent,c.coatNormal);
    if(v.z>SURFACE_COS_EPS)
    {
      float3 h=surfaceNormalize(v+l,float3(0,0,1));
      float f=openPbrFresnelDielectric(saturate(dot(v,h)),m.coat.z/c.etaI);
      value+=surfaceGgxD(h,c.coatAlpha)*surfaceGgxG1(v,c.coatAlpha)*surfaceGgxG1(l,c.coatAlpha)
        *f*m.coat.x/(4.0f*v.z)*(1.0f-surfaceFuzzAlbedo(c,view));
    }
  }
  if(m.program.z==1 && m.fuzz.x>0.0f)
    value+=m.fuzzColor.xyz*surfaceFuzzAlbedo(c,view)*zeltnerPdf(view,light,c.normal,m.fuzz.y);
  return value;
}

// Full GGX rough dielectric boundary. The half-vector Jacobian and radiance
// etaI^2/etaT^2 factor match Walter's microfacet transmission convention.
float surfaceTransmissionDensity(SurfaceClosure c,float3 view,float3 light,out float value)
{
  value=0.0f;float3 v=surfaceToLocal(view,c.tangent,c.bitangent,c.normal);
  float3 l=surfaceToLocal(light,c.tangent,c.bitangent,c.normal);
  if(v.z<=SURFACE_COS_EPS || l.z>=-SURFACE_COS_EPS)return 0.0f;
  if(c.material.subsurface.w<.5f && abs(c.etaI-c.etaT)<1.0e-5f)return 0.0f; // delta, sampled separately
  bool sheet=c.material.subsurface.w>0.5f;
  float eta=c.etaT/c.etaI;
  float3 h=sheet?surfaceNormalize(v+float3(l.xy,-l.z),float3(0,0,1)):
    surfaceNormalize(v+eta*l,float3(0,0,1));
  if(h.z<0.0f)h=-h;
  float vh=dot(v,h),lh=dot(l,h);
  if(vh<=0.0f || (!sheet && lh>=0.0f))return 0.0f;
  float f=surfaceInterfaceProbability(c,vh);
  float3 fRgb=surfaceInterfaceFresnel(c,vh);
  float d=surfaceGgxD(h,c.transmissionAlpha),g1=surfaceGgxG1(v,c.transmissionAlpha);
  float g=g1*surfaceGgxG1(l,c.transmissionAlpha);
  float factor;
  if(sheet)factor=1.0f/(4.0f*v.z);
  else
  {
    float denominator=vh+eta*lh;
    if(abs(denominator)<1.0e-12f)return 0.0f;
    factor=abs(vh*lh)/(v.z*denominator*denominator);
  }
  // Keep the scalar geometry separate so the caller can use spectral 1-F.
  value=d*g*factor;
  float normalPdf=d*g1*vh/v.z;
  float jacobian=sheet?1.0f/(4.0f*vh):eta*eta*abs(lh)/((vh+eta*lh)*(vh+eta*lh));
  return normalPdf*jacobian*(1.0f-f);
}

float3 surfaceTransmissionValue(SurfaceClosure c,float3 view,float3 light)
{
  float geometric;surfaceTransmissionDensity(c,view,light,geometric);
  if(geometric<=0.0f)return float3(0.0f);
  float3 v=surfaceToLocal(view,c.tangent,c.bitangent,c.normal),l=surfaceToLocal(light,c.tangent,c.bitangent,c.normal);
  bool sheet=c.material.subsurface.w>0.5f || abs(c.etaI-c.etaT)<1.0e-5f;
  float3 h=sheet?surfaceNormalize(v+float3(l.xy,-l.z),float3(0,0,1)):surfaceNormalize(v+(c.etaT/c.etaI)*l,float3(0,0,1));
  if(h.z<0.0f)h=-h;
  float t=c.material.transmission.x,s=c.material.subsurface.w>.5f?0.0f:(1.0f-t)*c.material.subsurface.x;
  // Depth-based attenuation is handled in the volume. Depth zero uses the
  // graph's surface tint; SSS albedo likewise belongs to the interior medium.
  float3 tint=c.material.transmission.y>0.0f?float3(1.0f):c.material.transmissionColor.xyz;
  float3 weight=(t*tint+float3(s))*(1.0f-c.material.surface.x);
  return geometric*(float3(1.0f)-surfaceInterfaceFresnel(c,saturate(dot(v,h))))*weight*surfaceOuterAttenuation(c,view);
}

float surfaceTransmissionReflectionPdf(SurfaceClosure c,float3 view,float3 light)
{
  if(c.material.subsurface.w<.5f && abs(c.etaI-c.etaT)<1.0e-5f)return 0.0f;
  float3 v=surfaceToLocal(view,c.tangent,c.bitangent,c.normal),l=surfaceToLocal(light,c.tangent,c.bitangent,c.normal);
  if(v.z<=SURFACE_COS_EPS||l.z<=0.0f)return 0.0f;
  float3 h=surfaceNormalize(v+l,float3(0,0,1));
  return surfaceGgxPdf(v,l,c.transmissionAlpha)*surfaceInterfaceProbability(c,saturate(dot(v,h)));
}

// Deterministic evaluation of the complete opaque closure, with cosine included.
// It follows the Standard Surface layering order used by Aurora's evaluateMaterial:
// Oren-Nayar -> sheen -> dielectric -> metal mixture -> colored clear coat.
float3 surfaceValue(SurfaceClosure c, float3 view, float3 light)
{
  if(dot(view,c.geometricNormal)<=0.0f)return float3(0.0f);
  if(dot(light,c.geometricNormal)<0.0f)
  {
    float3 value=surfaceTransmissionValue(c,view,light);
    if(c.material.subsurface.w>.5f)
    {
      float3 color=c.material.subsurfaceColor.xyz;
      float factor=1.0f;
      if(c.material.program.z==1)factor=.5f*(1.0f+c.material.subsurface.z);
      float reflectWeight=c.material.program.z==1?1.0f:max(c.material.specular.x,c.material.transmission.x);
      value+=(1.0f-c.material.surface.x)*(1.0f-c.material.transmission.x)*c.material.subsurface.x
        *color*factor*max(0.0f,-dot(c.normal,light))*SURFACE_INV_PI*surfaceOuterAttenuation(c,view)
        *max(float3(0.0f),float3(1.0f)-surfaceInterfaceFresnel(c,saturate(dot(c.normal,view)))*reflectWeight);
    }
    return surfaceFinite(value)?max(value,float3(0.0f)):float3(0.0f);
  }
  if(dot(light,c.geometricNormal)==0.0f)return float3(0.0f);
  if(surfaceHasExtendedReflection(c))
  {
    if(c.material.program.z==1 && !surfaceHasTransport(c)
      && c.material.thinFilm.x*c.material.thinFilm.y<=0.0f
      && c.material.thinFilm.w<=0.0f
      && (!c.interfaceConfigured || (c.etaI==1.0f && c.etaT==surfaceNominalIor(c.material))))
    {
      float fuzz=surfaceFuzzAlbedo(c,view);
      return surfaceOpenPbrValue(c,view,light)*(1.0f-fuzz)
        +c.material.fuzzColor.xyz*fuzz*zeltnerPdf(view,light,c.normal,c.material.fuzz.y);
    }
    float3 value=surfaceExtendedReflection(c,view,light);
    return surfaceFinite(value)?max(value,float3(0.0f)):float3(0.0f);
  }
  if(c.material.program.z == 1)
  {
    float3 openPbr = surfaceOpenPbrValue(c,view,light);
    return surfaceFinite(openPbr) ? max(openPbr,float3(0.0f)) : float3(0.0f);
  }
  GpuMaterial m = c.material;
  float3 value = float3(0.0f);
  if(dot(view, c.normal) > SURFACE_COS_EPS && dot(light, c.normal) > SURFACE_COS_EPS)
  {
    float3 diffuse = mx_diffuse_brdf_reflection(light, view, m.factors.x, c.diffuseColor, m.factors.y, c.normal);
    float3 sheenLayer = mx_sheen_brdf_reflection(light, view, m.sheen.x, m.sheenColor.xyz, m.sheen.y, c.normal, diffuse);
    float3 dielectric = mx_dielectric_brdf_reflection(light, view, m.specular.x, m.specularColor.xyz,
      m.specular.y, c.alpha, c.normal, c.tangent, 0, sheenLayer);
    if(m.surface.x > 0.0f)
    {
      float3 metal = mx_conductor_brdf_reflection(light, view, 1.0f, m.baseColor.xyz * m.factors.x,
        m.specularColor.xyz * m.specular.x, c.alpha, c.normal, c.tangent, 0);
      value = lerp(dielectric, metal, m.surface.x);
    }
    else value = dielectric;
  }
  if(m.coat.x > 0.0f)
  {
    float3 attenuated = value * lerp(float3(1.0f), m.coatColor.xyz, m.coat.x);
    if(dot(view, c.coatNormal) > SURFACE_COS_EPS && dot(light, c.coatNormal) > SURFACE_COS_EPS)
    {
      // MaterialX's coat_color filters the layer below; the coat reflection itself
      // is white. This intentionally corrects Aurora's colored coat-reflection tint.
      value = mx_dielectric_brdf_reflection(light, view, m.coat.x, float3(1.0f), m.coat.z,
        c.coatAlpha, c.coatNormal, c.coatTangent, 0, attenuated);
    }
    else value = attenuated;
  }
  return surfaceFinite(value) ? max(value, float3(0.0f)) : float3(0.0f);
}

float surfacePdf(SurfaceClosure c, float3 view, float3 light)
{
  if(dot(view,c.geometricNormal)<=0.0f)return 0.0f;
  if(dot(light,c.geometricNormal)<0.0f)
  {
    float ignored;
    return c.transmissionProbability*surfaceTransmissionDensity(c,view,light,ignored)
      +c.translucentProbability*max(0.0f,-dot(c.normal,light))*SURFACE_INV_PI;
  }
  if(dot(light,c.geometricNormal)==0.0f)return 0.0f;
  float diffusePdf = max(dot(c.normal, light), 0.0f) * SURFACE_INV_PI;
  float specularPdf = surfaceGgxPdf(surfaceToLocal(view, c.tangent, c.bitangent, c.normal),
    surfaceToLocal(light, c.tangent, c.bitangent, c.normal), c.alpha);
  float coatPdf = surfaceGgxPdf(surfaceToLocal(view, c.coatTangent, c.coatBitangent, c.coatNormal),
    surfaceToLocal(light, c.coatTangent, c.coatBitangent, c.coatNormal), c.coatAlpha);
  return dot(c.probabilities, float3(diffusePdf, specularPdf, coatPdf))
    +c.transmissionProbability*surfaceTransmissionReflectionPdf(c,view,light)
    +c.fuzzProbability*zeltnerPdf(view,light,c.normal,c.material.fuzz.y);
}

SurfaceEval evaluateSurface(SurfaceClosure c, float3 view, float3 light)
{
  if(c.material.program.z>=2) return evaluateOfficialSurface(c,view,light);
  SurfaceEval result;
  result.fCos = surfaceValue(c, view, light);
  result.pdf = surfacePdf(c, view, light);
  return result;
}

SurfaceSample sampleSurface(SurfaceClosure c, float3 view, float3 xi)
{
  if(c.material.program.z>=2) return sampleOfficialSurface(c,view,xi);
  SurfaceSample result = (SurfaceSample)0;
  float2 random = min(saturate(xi.xy), float2(0.99999994f));
  if(xi.z < c.probabilities.x)
  {
    float radius = sqrt(random.x), angle = 2.0f * SURFACE_PI * random.y;
    float3 local = float3(radius * cos(angle), radius * sin(angle), sqrt(max(0.0f, 1.0f - random.x)));
    result.direction = surfaceToWorld(local, c.tangent, c.bitangent, c.normal);
    result.event = 1;
  }
  else if(xi.z<c.probabilities.x+c.probabilities.y+c.probabilities.z)
  {
    bool coat = xi.z >= c.probabilities.x + c.probabilities.y;
    float3 n = coat ? c.coatNormal : c.normal;
    float3 t = coat ? c.coatTangent : c.tangent;
    float3 b = coat ? c.coatBitangent : c.bitangent;
    float3 localView = surfaceToLocal(view, t, b, n);
    if(localView.z <= SURFACE_COS_EPS) return result;
    float3 halfVector = surfaceSampleGgxNormal(localView, coat ? c.coatAlpha : c.alpha, random);
    float3 localLight = reflect(-localView, halfVector);
    if(localLight.z<=0.0f)return (SurfaceSample)0;
    result.direction = surfaceToWorld(localLight, t, b, n);
    result.event = 2;
  }
  else if(xi.z<c.probabilities.x+c.probabilities.y+c.probabilities.z+c.transmissionProbability)
  {
    float choose=(xi.z-c.probabilities.x-c.probabilities.y-c.probabilities.z)/max(c.transmissionProbability,1.0e-20f);
    float3 v=surfaceToLocal(view,c.tangent,c.bitangent,c.normal);
    if(v.z<=SURFACE_COS_EPS)return result;
    if(c.material.subsurface.w<.5f && abs(c.etaI-c.etaT)<1.0e-5f)
    {
      float t=c.material.transmission.x,s=(1.0f-t)*c.material.subsurface.x;
      float3 tint=c.material.transmission.y>0.0f?float3(1.0f):c.material.transmissionColor.xyz;
      result.direction=-view;result.event=c.entering && s>0.0f?4:3;result.delta=true;
      result.pdf=c.transmissionProbability;
      result.weight=(t*tint+float3(s))*(1.0f-c.material.surface.x)
        *(float3(1.0f)-surfaceInterfaceFresnel(c,saturate(v.z)))*surfaceOuterAttenuation(c,view)/result.pdf;
      return result;
    }
    float3 h=surfaceSampleGgxNormal(v,c.transmissionAlpha,random);
    float fresnel=surfaceInterfaceProbability(c,saturate(dot(v,h)));
    float3 l;
    if(choose<fresnel)
    {
      l=reflect(-v,h);result.event=2;
      if(l.z<=0.0f)return (SurfaceSample)0;
    }
    else
    {
      if(c.material.subsurface.w>.5f || abs(c.etaI-c.etaT)<1.0e-5f)
      {
        l=reflect(-v,h);l.z=-l.z;
      }
      else l=refract(-v,h,c.etaI/c.etaT);
      if(l.z>=0.0f || dot(l,l)<1.0e-20f)return (SurfaceSample)0;
      result.event=(1.0f-c.material.transmission.x)*c.material.subsurface.x>0.0f && c.entering && c.material.subsurface.w<.5f?4:3;
    }
    result.direction=surfaceToWorld(l,c.tangent,c.bitangent,c.normal);
  }
  else if(xi.z<c.probabilities.x+c.probabilities.y+c.probabilities.z+c.transmissionProbability+c.translucentProbability)
  {
    float r=sqrt(random.x),phi=2.0f*SURFACE_PI*random.y;
    result.direction=surfaceToWorld(float3(r*cos(phi),r*sin(phi),-sqrt(max(0.0f,1.0f-random.x))),c.tangent,c.bitangent,c.normal);
    result.event=3;
  }
  else
  {
    result.direction=zeltnerSample(view,c.normal,c.material.fuzz.y,random);result.event=1;
  }
  result.direction = surfaceNormalize(result.direction, c.normal);
  if((result.event==3 || result.event==4) != (dot(result.direction,c.geometricNormal)<0.0f))return (SurfaceSample)0;
  SurfaceEval evaluated = evaluateSurface(c, view, result.direction);
  if(!(evaluated.pdf > 0.0f) || isnan(evaluated.pdf) || isinf(evaluated.pdf))
    return (SurfaceSample)0;
  result.pdf = evaluated.pdf;
  result.weight = evaluated.fCos / evaluated.pdf;
  if(!surfaceFinite(result.weight) || !any(result.weight > 0.0f)) return (SurfaceSample)0;
  return result;
}

// First-hit semantic features, independent of random lobe selection and lighting.
// Multiple anisotropic/multilayer lobes must be approximated by a single normal
// and roughness for the current NGX guide contract.
void surfaceTransportGuides(SurfaceClosure c,float3 view,inout SurfaceGuides guides)
{
  guides.diffuseAlbedo*=(1.0f-c.material.transmission.x)*(1.0f-c.material.subsurface.x);
  if(surfaceHasExtendedReflection(c))
  {
    float nv=saturate(dot(c.normal,view));
    float3 dielectric=surfaceInterfaceFresnel(c,nv)*surfaceDielectricTint(c);
    if(c.material.program.z==0)dielectric*=max(c.material.specular.x,c.material.transmission.x+(1.0f-c.material.transmission.x)*c.material.subsurface.x);
    float3 main=lerp(dielectric,surfaceFilmFresnel(c,nv,true)*(c.material.program.z==1?c.material.specular.x:1.0f),c.material.surface.x);
    float coat=c.material.coat.x*openPbrFresnelDielectric(saturate(dot(c.coatNormal,view)),c.material.coat.z/c.etaI);
    float fuzz=surfaceFuzzAlbedo(c,view);
    guides.specularAlbedo=saturate(main*surfaceOuterAttenuation(c,view)+float3(coat*(1.0f-fuzz))+fuzz*c.material.fuzzColor.xyz);
    guides.diffuseAlbedo*=1.0f-fuzz;
  }
}

SurfaceGuides surfaceGuides(SurfaceClosure c, float3 view)
{
  if(c.material.program.z>=2) return officialSurfaceGuides(c,view);
  GpuMaterial m = c.material;
  SurfaceGuides guides = (SurfaceGuides)0;
  float nDotV = saturate(dot(c.normal, view));
  float alpha = mx_average_roughness(c.alpha);
  if(m.program.z == 1)
  {
    float dielectricF = openPbrFresnelDielectric(nDotV,c.substrateIor);
    float3 dielectricComp = openPbrGgxCompensation(nDotV,alpha,float3(dielectricF));
    float dielectricAlbedo = abs(c.substrateIor-1.0f) < 0.000001f ? 0.0f :
      saturate(openPbrGgxAlbedo(nDotV,alpha,float3(mx_ior_to_f0(c.substrateIor)),float3(1)).x * dielectricComp.x);
    guides.diffuseAlbedo = openPbrDiffuseAlbedo(nDotV,m.factors.y,m.baseColor.xyz) * m.factors.x
      * (1.0f-m.surface.x) * (1.0f-dielectricAlbedo);
    float3 metalF = openPbrFresnelMetal(nDotV,m.baseColor.xyz*m.factors.x,m.specularColor.xyz);
    float3 metalAlbedo = openPbrGgxAlbedo(nDotV,alpha,m.baseColor.xyz*m.factors.x,float3(1))
      * openPbrGgxCompensation(nDotV,alpha,metalF) * m.specular.x;
    guides.specularAlbedo = lerp(m.specularColor.xyz*dielectricAlbedo,metalAlbedo,m.surface.x);
    float coatNDotV = saturate(dot(c.coatNormal,view));
    float coatAverage = mx_average_roughness(c.coatAlpha);
    float coatF = openPbrFresnelDielectric(coatNDotV,m.coat.z);
    float coatAlbedo = abs(m.coat.z-1.0f) < 0.000001f ? 0.0f : saturate(
      openPbrGgxAlbedo(coatNDotV,coatAverage,float3(mx_ior_to_f0(m.coat.z)),float3(1)).x
      * openPbrGgxCompensation(coatNDotV,coatAverage,float3(coatF)).x*m.coat.x);
    float3 attenuation = c.substrateDarkening * lerp(float3(1.0f),m.coatColor.xyz,m.coat.x) * (1.0f-coatAlbedo);
    guides.diffuseAlbedo *= attenuation;
    guides.specularAlbedo = float3(coatAlbedo)+guides.specularAlbedo*attenuation;
    float coatShare = saturate(coatAlbedo/max(surfaceLuminance(guides.specularAlbedo),0.0001f));
    guides.normal = surfaceNormalize(lerp(c.normal,c.coatNormal,coatShare),c.normal);
    guides.roughness = clamp(lerp(sqrt(alpha),sqrt(coatAverage),coatShare),SURFACE_MIN_ROUGHNESS,1.0f);
    guides.diffuseAlbedo = saturate(guides.diffuseAlbedo);
    guides.specularAlbedo = saturate(guides.specularAlbedo);
    surfaceTransportGuides(c,view,guides);
    return guides;
  }
  float baseAlbedo = saturate(mx_ggx_directional_albedo(nDotV, alpha, mx_ior_to_f0(m.specular.y), 1.0f));
  float specularLayer = saturate(baseAlbedo * m.specular.x);
  float sheenLayer = saturate(mx_imageworks_sheen_directional_albedo(nDotV, m.sheen.y) * m.sheen.x);
  guides.diffuseAlbedo = c.diffuseColor * m.factors.x * (1.0f - m.surface.x)
    * (1.0f - specularLayer) * (1.0f - sheenLayer);
  float3 dielectric = m.specularColor.xyz * specularLayer + m.sheenColor.xyz * sheenLayer * (1.0f - specularLayer);
  float3 metal = mx_ggx_directional_albedo_curve_fit(nDotV, alpha, saturate(m.baseColor.xyz * m.factors.x),
    saturate(m.specularColor.xyz * m.specular.x));
  guides.specularAlbedo = lerp(dielectric, metal, m.surface.x);
  float coatAlbedo = saturate(mx_ggx_directional_albedo(saturate(dot(c.coatNormal, view)), mx_average_roughness(c.coatAlpha),
    mx_ior_to_f0(m.coat.z), 1.0f) * m.coat.x);
  float3 coatAttenuation = lerp(float3(1.0f), m.coatColor.xyz, m.coat.x) * (1.0f - coatAlbedo);
  guides.diffuseAlbedo *= coatAttenuation;
  guides.specularAlbedo = float3(coatAlbedo) + guides.specularAlbedo * coatAttenuation;
  float coatShare = saturate(coatAlbedo / max(surfaceLuminance(guides.specularAlbedo), 0.0001f));
  guides.normal = surfaceNormalize(lerp(c.normal, c.coatNormal, coatShare), c.normal);
  guides.roughness = clamp(lerp(sqrt(alpha), m.coat.y, coatShare), SURFACE_MIN_ROUGHNESS, 1.0f);
  guides.diffuseAlbedo = saturate(guides.diffuseAlbedo);
  guides.specularAlbedo = saturate(guides.specularAlbedo);
  surfaceTransportGuides(c,view,guides);
  return guides;
}

#include "official_models.slang"
#endif // CAUSTRA_SURFACE_BSDF
