// SPDX-License-Identifier: Apache-2.0
// Sampling proposals and physical transmission adapters for native model tags.
// Ordinary surface values and EDFs use official ShaderGen code in generated/.
// The transport adapters retain native model semantics; they do not translate
// the models to Standard Surface. Provenance: third_party/materialx/ORIGIN.md.
#ifndef CAUSTRA_OFFICIAL_MODELS
#define CAUSTRA_OFFICIAL_MODELS
#include "../third_party/materialx/mx_hair_math.slang"

float officialAverage(float3 v) { return (v.x+v.y+v.z)/3.0f; }
float3 officialSchlick(float cosine,float3 f0,float3 f90)
{
  return lerp(f0,f90,pow(1.0f-saturate(cosine),5.0f));
}
float officialSheenAlbedo(float nv,float roughness)
{
  float x=saturate(nv),y=clamp(roughness,.005f,1.0f);
  float2 r=float2(13.67300f,1.0f)+float2(-68.78018f,61.57746f)*x
    +float2(799.08825f,442.78211f)*y+float2(-905.00061f,2597.49308f)*x*y
    +float2(60.28956f,121.81241f)*x*x+float2(1086.96473f,3045.55075f)*y*y;
  return saturate(r.x/r.y);
}
float3 officialSheen(float3 v,float3 l,float3 n,float weight,float3 color,float roughness,float3 base)
{
  if(weight<=0.0f)return base;
  float nv=max(dot(n,v),SURFACE_COS_EPS),nl=dot(n,l);
  float attenuation=max(0.0f,1.0f-weight*officialSheenAlbedo(nv,roughness));
  if(nl<=0.0f)return base*attenuation;
  float nh=saturate(dot(n,surfaceNormalize(v+l,n)));
  float invR=1.0f/max(roughness,.005f);
  float d=(2.0f+invR)*pow(max(0.0f,1.0f-nh*nh),invR*.5f)/(2.0f*SURFACE_PI);
  return color*(d*nl*weight/(4.0f*(nl+nv-nl*nv)))+base*attenuation;
}
float3 officialFresnel(float cosine,float3 f0,float3 f90,GpuMaterial m,bool film)
{
  float3 f=officialSchlick(cosine,f0,f90);
  if(film && m.thinFilm.x>0.0f && m.thinFilm.y>0.0f)
  {
    FilmFresnelData fd=(FilmFresnelData)0;
    fd.model=FRESNEL_MODEL_SCHLICK;fd.F0=f0;fd.F82=float3(1);fd.F90=f90;
    fd.exponent=5.0f;fd.tf_thickness=m.thinFilm.y;fd.tf_ior=m.thinFilm.z;
    f=lerp(f,film_fresnel_airy(clamp(cosine,.000001f,1.0f),fd),saturate(m.thinFilm.x));
  }
  return f;
}
// MaterialX generalized_schlick_bsdf: vector energy compensation, scalar mean
// directional-albedo layer throughput, and correlated GGX Smith masking.
float3 officialSchlickLayer(SurfaceClosure c,float3 v,float3 l,float3 n,float3 t,float2 alpha,
  float weight,float3 f0,float3 f90,float3 base,bool film=false,bool dielectric=false)
{
  if(weight<=0.0f)return base;
  float nv=dot(n,v),nl=dot(n,l);
  if(nv<=SURFACE_COS_EPS)return base;
  float3 h=surfaceNormalize(v+l,n);
  float cosine=nl>0.0f?saturate(dot(v,h)):saturate(nv);
  float3 f=dielectric?float3(openPbrFresnelDielectric(cosine,1.5f)):
    officialFresnel(cosine,f0,f90,c.material,film);
  float avgAlpha=mx_average_roughness(alpha);
  float3 comp=openPbrGgxCompensation(nv,avgAlpha,f);
  float3 albedo=openPbrGgxAlbedo(nv,avgAlpha,f0,f90)*comp;
  float attenuation=max(0.0f,1.0f-officialAverage(albedo)*weight);
  if(nl<=SURFACE_COS_EPS)return base*attenuation;
  float3 localH=surfaceToLocal(h,t,cross(n,t),n);
  float d=surfaceGgxD(localH,alpha),g=mx_ggx_smith_G(nl,nv,avgAlpha);
  return d*f*g*comp*(weight/(4.0f*nv))+base*attenuation;
}
float3 officialConductor(SurfaceClosure c,float3 v,float3 l)
{
  float nv=dot(c.normal,v),nl=dot(c.normal,l);
  if(nv<=SURFACE_COS_EPS || nl<=SURFACE_COS_EPS)return float3(0);
  float3 h=surfaceNormalize(v+l,c.normal),eta,k;
  mx_artistic_to_complex_ior(c.material.baseColor.xyz,c.material.baseColor.xyz,eta,k);
  float3 f=film_fresnel_conductor(saturate(dot(v,h)),eta,k);
  float a=mx_average_roughness(c.alpha);
  float3 comp=openPbrGgxCompensation(nv,a,f);
  return surfaceGgxD(surfaceToLocal(h,c.tangent,c.bitangent,c.normal),c.alpha)
    *mx_ggx_smith_G(nl,nv,a)*f*comp/(4.0f*nv);
}
float officialBurley(float nv,float nl,float lh,float roughness)
{
  float f90=.5f+2.0f*roughness*lh*lh;
  return lerp(1.0f,f90,pow(1.0f-saturate(nl),5.0f))
    *lerp(1.0f,f90,pow(1.0f-saturate(nv),5.0f));
}

float officialTransmissionWeight(SurfaceClosure c)
{
  GpuMaterial m=c.material;
  float nonmetal=m.program.z==4?lerp(1.0f-saturate(m.surface.x),1.0f,saturate(m.factors.w)):
    1.0f-saturate(m.surface.x);
  return nonmetal*(saturate(m.transmission.x)
    +(1.0f-saturate(m.transmission.x))*saturate(m.subsurface.x));
}
float3 officialTransmissionTint(SurfaceClosure c)
{
  GpuMaterial m=c.material;
  float t=saturate(m.transmission.x),s=(1.0f-t)*saturate(m.subsurface.x);
  // These graph transmission lobes are surface-tinted; homogeneous glTF
  // attenuation, when present, is separately integrated by the medium layer.
  float3 tint=m.program.z==4?float3(1):m.baseColor.xyz;
  float nonmetal=m.program.z==4?lerp(1.0f-saturate(m.surface.x),1.0f,saturate(m.factors.w)):
    1.0f-saturate(m.surface.x);
  return nonmetal*(t*tint+float3(s));
}
float officialSchlickThroughput(SurfaceClosure c,float3 view,float3 n,float2 alpha,float weight,
  float3 f0,float3 f90,bool film=false,bool dielectric=false)
{
  float nv=max(dot(n,view),SURFACE_COS_EPS),averageAlpha=mx_average_roughness(alpha);
  float3 f=dielectric?float3(openPbrFresnelDielectric(nv,1.5f)):
    officialFresnel(nv,f0,f90,c.material,film);
  float3 albedo=openPbrGgxAlbedo(nv,averageAlpha,f0,f90)*openPbrGgxCompensation(nv,averageAlpha,f);
  return max(0.0f,1.0f-officialAverage(albedo)*weight);
}
float3 officialOuterTransmission(SurfaceClosure c,float3 view)
{
  GpuMaterial m=c.material;float3 unit=float3(1);float throughput=1.0f;
  if(m.program.z==2)
  {
    float3 tint=lerp(unit,m.baseColor.xyz,saturate(m.factors.w));
    throughput*=officialSchlickThroughput(c,view,c.normal,c.alpha,.08f*saturate(m.specular.x),tint,unit);
    throughput*=officialSchlickThroughput(c,view,c.coatNormal,c.coatAlpha,.04f*saturate(m.coat.x),unit,unit);
  }
  else if(m.program.z==3)
  {
    float3 f0=min(unit,m.specularColor.xyz*mx_ior_to_f0(m.specular.y))*m.specular.x;
    throughput*=officialSchlickThroughput(c,view,c.normal,c.alpha,1.0f,f0,unit*m.specular.x,true);
    float sheen=max(m.sheenColor.x,max(m.sheenColor.y,m.sheenColor.z));
    throughput*=max(0.0f,1.0f-sheen*officialSheenAlbedo(saturate(dot(view,c.normal)),m.sheen.y*m.sheen.y));
    throughput*=officialSchlickThroughput(c,view,c.coatNormal,c.coatAlpha,m.coat.x,float3(.04f),unit,false,true);
  }
  else
  {
    float metallic=saturate(m.surface.x);
    float3 color90=lerp(unit,m.baseColor.xyz,metallic);
    float3 f0=lerp(color90*mx_ior_to_f0(m.specular.y),color90,metallic);
    f0=lerp(f0,m.specularColor.xyz,saturate(m.factors.w));
    color90=lerp(color90,unit,saturate(m.factors.w));
    throughput*=officialSchlickThroughput(c,view,c.normal,c.alpha,1.0f,f0,color90);
    throughput*=officialSchlickThroughput(c,view,c.coatNormal,c.coatAlpha,m.coat.x,float3(mx_ior_to_f0(m.specular.y)),unit);
  }
  return float3(throughput);
}
float officialTransmissionPdf(SurfaceClosure c,float3 view,float3 light,out float3 value)
{
  value=float3(0);float3 v=surfaceToLocal(view,c.tangent,c.bitangent,c.normal);
  float3 l=surfaceToLocal(light,c.tangent,c.bitangent,c.normal);
  if(v.z<=SURFACE_COS_EPS || l.z>=-SURFACE_COS_EPS)return 0.0f;
  bool sheet=c.material.subsurface.w>.5f;
  if(!sheet && abs(c.etaI-c.etaT)<.00001f)return 0.0f;
  float eta=c.etaT/c.etaI;
  float3 h=sheet?surfaceNormalize(v+float3(l.xy,-l.z),float3(0,0,1)):
    surfaceNormalize(v+eta*l,float3(0,0,1));
  if(h.z<0.0f)h=-h;
  float vh=dot(v,h),lh=dot(l,h);
  if(vh<=0.0f || (!sheet && lh>=0.0f))return 0.0f;
  float d=surfaceGgxD(h,c.transmissionAlpha),g1=surfaceGgxG1(v,c.transmissionAlpha);
  float g=g1*surfaceGgxG1(l,c.transmissionAlpha);
  float denominator=vh+eta*lh;
  if(!sheet && abs(denominator)<1e-12f)return 0.0f;
  float geometry=sheet?1.0f/(4.0f*v.z):abs(vh*lh)/(v.z*denominator*denominator);
  float f=openPbrFresnelDielectric(saturate(vh),eta);
  value=d*g*geometry*(1.0f-f)*officialTransmissionTint(c)*officialOuterTransmission(c,view);
  float jacobian=sheet?1.0f/(4.0f*vh):eta*eta*abs(lh)/(denominator*denominator);
  return d*g1*vh/v.z*jacobian;
}

float3 officialReflection(SurfaceClosure c,float3 view,float3 light)
{
  GpuMaterial m=c.material;float3 unit=float3(1),value=float3(0);
  float nv=dot(c.normal,view),nl=dot(c.normal,light),metallic=saturate(m.surface.x);
  bool above=nl>SURFACE_COS_EPS;
  if(m.program.z==2)
  {
    if(above)
    {
      float lh=saturate(dot(light,surfaceNormalize(light+view,c.normal)));
      value=m.baseColor.xyz*((1.0f-metallic)*(1.0f-m.subsurface.x)*nl*SURFACE_INV_PI)
        *officialBurley(nv,nl,lh,m.surface.y);
    }
    value=officialSheen(view,light,c.normal,m.sheen.x,lerp(unit,m.baseColor.xyz,m.fuzz.w),.3f,value);
    value*=1.0f-m.transmission.x;
    value=officialSchlickLayer(c,view,light,c.normal,c.tangent,c.alpha,.08f*m.specular.x,
      lerp(unit,m.baseColor.xyz,m.factors.w),unit,value);
    float3 metal=officialSchlickLayer(c,view,light,c.normal,c.tangent,c.alpha,1.0f,m.baseColor.xyz,unit,float3(0));
    value=lerp(value,metal,metallic);
    return officialSchlickLayer(c,view,light,c.coatNormal,c.coatTangent,c.coatAlpha,.04f*m.coat.x,unit,unit,value);
  }
  if(m.program.z==3)
  {
    if(above)value=m.baseColor.xyz*((1.0f-m.transmission.x)*nl*SURFACE_INV_PI);
    float3 f0=min(unit,m.specularColor.xyz*mx_ior_to_f0(m.specular.y))*m.specular.x;
    value=officialSchlickLayer(c,view,light,c.normal,c.tangent,c.alpha,1.0f,f0,unit*m.specular.x,value,true);
    float3 metal=officialSchlickLayer(c,view,light,c.normal,c.tangent,c.alpha,1.0f,m.baseColor.xyz,unit,float3(0),true);
    value=lerp(value,metal,metallic);
    float sheen=max(m.sheenColor.x,max(m.sheenColor.y,m.sheenColor.z));
    value=officialSheen(view,light,c.normal,sheen,m.sheenColor.xyz/max(sheen,.000001f),m.sheen.y*m.sheen.y,value);
    return officialSchlickLayer(c,view,light,c.coatNormal,c.coatTangent,c.coatAlpha,m.coat.x,float3(.04f),unit,value,false,true);
  }
  // UsdPreviewSurface: separate specular and metalness workflows. The latter
  // includes its native artistic-IOR conductor mixture, even for fractional metal.
  float workflow=saturate(m.factors.w);
  if(above)value=m.baseColor.xyz*((1.0f-metallic)*(1.0f-workflow)+workflow)
    *(1.0f-m.transmission.x)*nl*SURFACE_INV_PI;
  float3 color90=lerp(unit,m.baseColor.xyz,metallic);
  float3 f0=lerp(color90*mx_ior_to_f0(m.specular.y),color90,metallic);
  float3 metalness=officialSchlickLayer(c,view,light,c.normal,c.tangent,c.alpha,1.0f,f0,color90,value);
  metalness=lerp(metalness,officialConductor(c,view,light),metallic);
  float3 specular=officialSchlickLayer(c,view,light,c.normal,c.tangent,c.alpha,1.0f,m.specularColor.xyz,unit,value);
  value=lerp(metalness,specular,workflow);
  return officialSchlickLayer(c,view,light,c.coatNormal,c.coatTangent,c.coatAlpha,m.coat.x,
    float3(mx_ior_to_f0(m.specular.y)),unit,value);
}

float2 officialHairRoughness(GpuMaterial m,uint lobe)
{
  float2 r,tt,trt;
  official_chiang_hair_roughness(m.surface.y,m.specular.z,.5f,2.0f,r,tt,trt);
  return clamp(lobe==0?r:(lobe==1?tt:trt),float2(.001f),float2(1));
}
float3 officialHairAbsorption(GpuMaterial m)
{
  float3 melanin,color;
  official_deon_hair_absorption_from_melanin(m.factors.x,m.factors.y,
    float3(.657704f,.498077f,.254106f),float3(.829443f,.670320f,.349937f),melanin);
  // The official SimpleHair graph leaves beta_n at its NodeDef default, 0.2.
  official_chiang_hair_absorption_from_color(m.baseColor.xyz,.2f,color);
  return lerp(color,melanin,saturate(m.factors.w));
}
void officialHairAngles(SurfaceClosure c,float3 view,out float so,out float co,out float gammaO,out float gammaT)
{
  so=clamp(dot(view,c.tangent),-.999999f,.999999f);co=sqrt(max(0.0f,1.0f-so*so));
  float3 k=surfaceNormalize(view-c.tangent*so,c.normal);
  float cg=saturate(dot(c.normal,k));float sg=sqrt(max(0.0f,1.0f-cg*cg));
  if(dot(k,c.bitangent)>0.0f)sg=-sg;
  gammaO=asin(clamp(sg,-1.0f,1.0f));
  float eta=max(c.material.specular.y,1.00001f);
  float etaP=sqrt(max(eta*eta-so*so,0.0f))/max(co,.000001f);
  gammaT=asin(clamp(sg/etaP,-1.0f,1.0f));
}
float officialLogisticCdf(float x,float s) { return 1.0f/(1.0f+exp(clamp(-x/s,-80.0f,80.0f))); }
float officialLogisticDensity(float x,float s,float a,float b)
{
  float e=exp(-abs(x)/s);
  return e/(s*(1.0f+e)*(1.0f+e)*max(.000001f,officialLogisticCdf(b,s)-officialLogisticCdf(a,s)));
}
float officialLogisticSample(float u,float s,float a,float b)
{
  float f=lerp(officialLogisticCdf(a,s),officialLogisticCdf(b,s),clamp(u,.0000001f,.9999999f));
  return clamp(s*log(max(f,.0000001f)/max(1.0f-f,.0000001f)),a,b);
}
float officialHairMean(SurfaceClosure c,float so,uint p)
{
  float shift=p==3?0.0f:(2.0f-float(p)*3.0f)*(c.material.coat.y*SURFACE_PI-SURFACE_PI*.5f);
  return clamp(-sin(asin(so)+shift),-.99999f,.99999f);
}
float officialHairPdf(SurfaceClosure c,float3 view,float3 light)
{
  float so,co,go,gt;officialHairAngles(c,view,so,co,go,gt);
  float si=clamp(dot(light,c.tangent),-1.0f,1.0f);
  float phi=atan2(dot(light,c.normal),dot(light,c.bitangent))-atan2(dot(view,c.normal),dot(view,c.bitangent));
  float density=0.0f;
  for(uint p=0;p<4;++p)
  {
    float2 rough=officialHairRoughness(c.material,p);float mean=officialHairMean(c,so,p);
    float scale=max(.015f,sqrt(rough.x)*.551329f);
    float longitudinal=officialLogisticDensity(si-mean,scale,-1.0f-mean,1.0f-mean);
    float delta=phi-official_hair_phi(int(p),go,gt);
    delta=delta-2.0f*SURFACE_PI*floor((delta+SURFACE_PI)/(2.0f*SURFACE_PI));
    float azimuthal=p==3?.5f*SURFACE_INV_PI:
      officialLogisticDensity(delta,rough.y*.626657f,-SURFACE_PI,SURFACE_PI);
    density+=.25f*longitudinal*azimuthal;
  }
  return density;
}

SurfaceClosure makeOfficialClosure(MaterialEvaluation evaluation,MaterialContext context,float3 view)
{
  SurfaceClosure c=(SurfaceClosure)0;c.source=evaluation;c.material=evaluation.material;
  c.material.baseColor.xyz=max(c.material.baseColor.xyz,float3(0));
  c.geometricNormal=surfaceNormalize(context.geometricNormal,float3(0,0,1));
  if(dot(c.geometricNormal,view)<0.0f)c.geometricNormal=-c.geometricNormal;
  c.normal=surfaceOrientedNormal(evaluation.normal,c.geometricNormal,view);
  c.coatNormal=surfaceOrientedNormal(evaluation.coatNormal,c.geometricNormal,view);
  surfaceFrame(c.normal,evaluation.tangent,c.tangent,c.bitangent);
  surfaceFrame(c.coatNormal,evaluation.coatTangent,c.coatTangent,c.coatBitangent);
  c.etaI=1.0f;c.etaT=surfaceNominalIor(c.material);c.entering=true;
  c.substrateIor=c.material.specular.y;c.substrateDarkening=float3(1);
  c.diffuseColor=c.material.baseColor.xyz;c.emission=max(c.material.emission.xyz,float3(0));
  c.opacity=saturate(c.material.factors.z);
  uint model=c.material.program.z;
  if(model==2 || model==4)
  {
    c.coatNormal=c.normal;c.coatTangent=c.tangent;c.coatBitangent=c.bitangent;
  }
  if(model==5)
  {
    c.alpha=float2(.1f);c.coatAlpha=c.alpha;c.transmissionAlpha=c.alpha;
    c.probabilities=float3(0,1,0);return c;
  }
  c.material.surface.x=saturate(c.material.surface.x);
  c.material.surface.y=clamp(c.material.surface.y,SURFACE_MIN_ROUGHNESS,1.0f);
  c.material.transmission.x=saturate(c.material.transmission.x);
  c.material.subsurface.x=saturate(c.material.subsurface.x);
  c.alpha=surfaceAlpha(c.material.surface.y,saturate(c.material.specular.z));
  c.coatAlpha=surfaceAlpha(clamp(c.material.coat.y,SURFACE_MIN_ROUGHNESS,1.0f),0.0f);
  if(model==2)c.coatAlpha=surfaceAlpha(max(1.0f-saturate(c.material.coat.y),SURFACE_MIN_ROUGHNESS),0.0f);
  if(model==3)
  {
    float r2=c.material.surface.y*c.material.surface.y,a=saturate(c.material.specular.z);
    c.alpha=max(float2(lerp(r2,1.0f,a*a),r2),float2(SURFACE_MIN_ALPHA));
    float angle=-c.material.specular.w;
    c.tangent=cos(angle)*c.tangent+sin(angle)*c.bitangent;c.bitangent=cross(c.normal,c.tangent);
    uint mode=uint(round(max(0.0f,c.material.fuzz.z)));
    c.opacity=mode==0?1.0f:(mode==1?(c.opacity>=c.material.volume.w?1.0f:0.0f):c.opacity);
  }
  if(model==4)
  {
    float threshold=c.material.specular.w;
    bool presence=c.material.fuzz.w>.5f;
    c.opacity=presence && threshold<=0.0f?c.opacity:(c.opacity>=threshold?1.0f:0.0f);
  }
  c.transmissionAlpha=model==3?c.alpha:float2(SURFACE_MIN_ALPHA);
  float metal=c.material.surface.x;
  float diffuse=(1.0f-metal)*(1.0f-c.material.transmission.x)*(1.0f-c.material.subsurface.x)
    *max(surfaceLuminance(c.material.baseColor.xyz),.02f);
  float specular=max(.02f,metal*surfaceLuminance(c.material.baseColor.xyz)+.08f*c.material.specular.x);
  float coat=c.material.coat.x*(model==2?.04f:.1f);
  c.transmissionProbability=officialTransmissionWeight(c);
  c.probabilities=float3(diffuse,specular,coat);
  float total=diffuse+specular+coat+c.transmissionProbability;
  c.probabilities/=max(total,.000001f);c.transmissionProbability/=max(total,.000001f);
  return c;
}

SurfaceEval evaluateOfficialSurface(SurfaceClosure c,float3 view,float3 light)
{
  SurfaceEval r=(SurfaceEval)0;
  if(c.material.program.z==5)
  {
    r.fCos=surfaceGeneratedReflection(c,view,light);r.pdf=officialHairPdf(c,view,light);return r;
  }
  if(dot(view,c.geometricNormal)<=0.0f)return r;
  if(dot(light,c.geometricNormal)<0.0f)
  {
    r.pdf=c.transmissionProbability*officialTransmissionPdf(c,view,light,r.fCos);return r;
  }
  r.fCos=surfaceUsesGeneratedReflection(c)?surfaceGeneratedReflection(c,view,light):officialReflection(c,view,light);
  float diffuse=max(0.0f,dot(c.normal,light))*SURFACE_INV_PI;
  float main=surfaceGgxPdf(surfaceToLocal(view,c.tangent,c.bitangent,c.normal),surfaceToLocal(light,c.tangent,c.bitangent,c.normal),c.alpha);
  float coat=surfaceGgxPdf(surfaceToLocal(view,c.coatTangent,c.coatBitangent,c.coatNormal),surfaceToLocal(light,c.coatTangent,c.coatBitangent,c.coatNormal),c.coatAlpha);
  r.pdf=dot(c.probabilities,float3(diffuse,main,coat));return r;
}
SurfaceSample sampleOfficialSurface(SurfaceClosure c,float3 view,float3 xi)
{
  SurfaceSample r=(SurfaceSample)0;float2 u=clamp(xi.xy,float2(.0000001f),float2(.9999999f));
  if(c.material.program.z==5)
  {
    float so,co,go,gt;officialHairAngles(c,view,so,co,go,gt);
    uint p=min(uint(xi.z*4.0f),3u);float2 rough=officialHairRoughness(c.material,p);
    float mean=officialHairMean(c,so,p),scale=max(.015f,sqrt(rough.x)*.551329f);
    float si=mean+officialLogisticSample(u.x,scale,-1.0f-mean,1.0f-mean);
    float phi=atan2(dot(view,c.normal),dot(view,c.bitangent))+official_hair_phi(int(p),go,gt);
    phi+=p==3?(2.0f*u.y-1.0f)*SURFACE_PI:
      officialLogisticSample(u.y,rough.y*.626657f,-SURFACE_PI,SURFACE_PI);
    float ci=sqrt(max(0.0f,1.0f-si*si));
    r.direction=c.tangent*si+ci*(c.bitangent*cos(phi)+c.normal*sin(phi));
    r.event=dot(r.direction,c.geometricNormal)<0.0f?3u:2u;
  }
  else if(xi.z<c.probabilities.x)
  {
    float radius=sqrt(u.x),phi=2.0f*SURFACE_PI*u.y;
    r.direction=surfaceToWorld(float3(radius*cos(phi),radius*sin(phi),sqrt(1.0f-u.x)),c.tangent,c.bitangent,c.normal);r.event=1;
  }
  else if(xi.z<c.probabilities.x+c.probabilities.y+c.probabilities.z)
  {
    bool coat=xi.z>=c.probabilities.x+c.probabilities.y;
    float3 n=coat?c.coatNormal:c.normal,t=coat?c.coatTangent:c.tangent,b=coat?c.coatBitangent:c.bitangent;
    float3 v=surfaceToLocal(view,t,b,n);if(v.z<=SURFACE_COS_EPS)return r;
    float3 h=surfaceSampleGgxNormal(v,coat?c.coatAlpha:c.alpha,u),l=reflect(-v,h);
    if(l.z<=0.0f)return r;r.direction=surfaceToWorld(l,t,b,n);r.event=2;
  }
  else
  {
    float3 v=surfaceToLocal(view,c.tangent,c.bitangent,c.normal);if(v.z<=SURFACE_COS_EPS)return r;
    bool sheet=c.material.subsurface.w>.5f;
    if(!sheet && abs(c.etaI-c.etaT)<.00001f)
    {
      r.direction=-view;r.event=c.entering && c.material.subsurface.x>0.0f?4u:3u;r.delta=true;
      r.pdf=c.transmissionProbability;r.weight=officialTransmissionTint(c)*officialOuterTransmission(c,view)/max(r.pdf,.000001f);return r;
    }
    float3 h=surfaceSampleGgxNormal(v,c.transmissionAlpha,u),l;
    if(sheet){l=reflect(-v,h);l.z=-l.z;}else l=refract(-v,h,c.etaI/c.etaT);
    if(l.z>=0.0f || dot(l,l)<1e-20f)return r;
    r.direction=surfaceToWorld(l,c.tangent,c.bitangent,c.normal);
    r.event=c.entering && c.material.subsurface.x>0.0f?4u:3u;
  }
  r.direction=surfaceNormalize(r.direction,c.normal);
  if(c.material.program.z!=5 && ((r.event==3 || r.event==4)!=(dot(r.direction,c.geometricNormal)<0.0f)))return (SurfaceSample)0;
  SurfaceEval e=evaluateSurface(c,view,r.direction);
  if(e.pdf<=0.0f || isnan(e.pdf) || isinf(e.pdf))return (SurfaceSample)0;
  r.pdf=e.pdf;r.weight=e.fCos/e.pdf;
  if(!surfaceFinite(r.weight) || !any(r.weight>0.0f))return (SurfaceSample)0;
  return r;
}
SurfaceGuides officialSurfaceGuides(SurfaceClosure c,float3 view)
{
  SurfaceGuides g=(SurfaceGuides)0;GpuMaterial m=c.material;
  g.normal=c.normal;g.roughness=clamp(sqrt(mx_average_roughness(c.alpha)),SURFACE_MIN_ROUGHNESS,1.0f);
  if(m.program.z==5)
  {
    g.specularAlbedo=saturate(exp(-officialHairAbsorption(m)));
    g.roughness=clamp(m.surface.y,SURFACE_MIN_ROUGHNESS,1.0f);return g;
  }
  float nv=saturate(dot(c.normal,view));float3 f0=float3(mx_ior_to_f0(m.specular.y));
  if(m.program.z==2)f0=lerp(lerp(float3(1),m.baseColor.xyz,m.factors.w)*(.08f*m.specular.x),m.baseColor.xyz,m.surface.x);
  else if(m.program.z==3)f0=lerp(min(float3(1),m.specularColor.xyz*f0)*m.specular.x,m.baseColor.xyz,m.surface.x);
  else f0=lerp(lerp(f0,m.baseColor.xyz,m.surface.x),m.specularColor.xyz,m.factors.w);
  g.specularAlbedo=saturate(openPbrGgxAlbedo(nv,mx_average_roughness(c.alpha),f0,float3(1)));
  g.diffuseAlbedo=saturate(m.baseColor.xyz*(1.0f-m.surface.x)*(1.0f-m.transmission.x)*(1.0f-m.subsurface.x)
    *(float3(1)-g.specularAlbedo));return g;
}
#endif
