// SPDX-License-Identifier: Apache-2.0
// MaterialX v1.39.5 hex tiling, adapted from mx_hextile.glsl and
// mx_hextiledimage.glsl. Explicit ray-cone footprints replace screen derivatives.
// See third_party/materialx/ORIGIN.md for source and modifications.
#include "../third_party/materialx/mx_hsv.slang"

float2 materialHexHash(float2 p)
{
  float3 p3=frac(float3(p.x,p.y,p.x)*float3(.1031f,.1030f,.0973f));
  p3+=dot(p3,p3.yzx+33.33f);
  return frac((p3.xx+p3.yz)*p3.zy);
}
float materialHexGain(float x,float r)
{
  float a=(1.0f/clamp(r,.001f,.999f)-2.0f)*(1.0f-2.0f*x);
  return x<.5f?x/(a+1.0f):(a-x)/(a-1.0f);
}
float3 materialHexWeights(float3 luma,float3 bary,float falloff)
{
  float3 geometric=pow(max(bary,float3(0)),float3(7));
  float3 w=luma*geometric;
  // A completely black texel has no luminance distribution. Preserve black
  // with finite geometric weights rather than propagating upstream's 0/0.
  if(dot(w,float3(1))<=0.0f)w=geometric;
  w/=max(dot(w,float3(1)),1.0e-30f);
  if(falloff!=.5f) {
    w=float3(materialHexGain(w.x,falloff),materialHexGain(w.y,falloff),materialHexGain(w.z,falloff));
    w/=max(dot(w,float3(1)),1.0e-30f);
  }
  return w;
}
float4 materialHexImage(uint textureIndex,float2 uv,float2 footprint,float4 primary,float4 ranges,float4 extra,float4 luma)
{
  float2 skewed=float2(uv.x-.57735027f*uv.y,1.15470054f*uv.y)*(sqrt(3.0f)*2.0f);
  float2 f=frac(skewed),base=floor(skewed);
  float z=1.0f-f.x-f.y,s=z<=0.0f?1.0f:0.0f,s2=2.0f*s-1.0f;
  float3 bary=float3(-z*s2,s-f.y*s2,s-f.x*s2);
  float4 samples[3];
  for(uint i=0;i<3;++i)
  {
    float2 id=base+(i==0?float2(s,s):(i==1?float2(s,1.0f-s):float2(1.0f-s,s)));
    float2 center=float2(id.x+.5f*id.y,id.y/1.15470054f)/(sqrt(3.0f)*2.0f);
    float2 random=materialHexHash(id+.12345f);
    float angle=lerp(ranges.x,ranges.y,random.x*primary.x)*(3.141592653589793f/180.0f);
    float scale=lerp(1.0f,lerp(ranges.z,ranges.w,random.y),primary.y);
    float2 offset=lerp(float2(extra.x),float2(extra.y),random*primary.z);
    float co=cos(angle),si=sin(angle);float2 p=uv-center;
    float2 coord=float2(p.x*co-p.y*si,p.x*si+p.y*co)/scale+center+offset;
    float2 width=float2(abs(co)*footprint.x+abs(si)*footprint.y,abs(si)*footprint.x+abs(co)*footprint.y)/abs(scale);
    samples[i]=materialTextureSample(textureIndex,coord,width,0,0);
  }
  float3 luminance=lerp(float3(1),float3(dot(samples[0].xyz,luma.xyz),dot(samples[1].xyz,luma.xyz),dot(samples[2].xyz,luma.xyz)),extra.z);
  float3 w=materialHexWeights(luminance,bary,primary.w),aw=materialHexWeights(float3(1),bary,primary.w);
  return float4(samples[0].xyz*w.x+samples[1].xyz*w.y+samples[2].xyz*w.z,
    samples[0].w*aw.x+samples[1].w*aw.y+samples[2].w*aw.z);
}
