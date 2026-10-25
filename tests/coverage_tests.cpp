#include <caustra/renderer.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using namespace caustra;

void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}

struct Program {
  std::vector<MaterialInstruction> instructions;
  uint32_t op(MaterialOp operation,uint32_t a=0,uint32_t b=0,uint32_t c=0,Float4 value={}) {
    instructions.push_back({{uint32_t(operation),a,b,c},value});return uint32_t(instructions.size()-1);
  }
  uint32_t constant(Float4 value) {return op(MaterialOp::Constant,0,0,0,value);}
  uint32_t scalar(float value) {return constant({value,value,value,value});}
  uint32_t output(uint32_t source,MaterialParameter parameter) {
    return op(MaterialOp::Output,source,uint32_t(parameter));
  }
};

void attach(Scene& scene,uint32_t index,const Program& program) {
  auto& material=scene.materials[index];
  material.program.x=uint32_t(scene.materialInstructions.size());
  material.program.y=uint32_t(program.instructions.size());
  scene.materialInstructions.insert(scene.materialInstructions.end(),program.instructions.begin(),program.instructions.end());
}

void quad(Scene& scene,uint32_t material,Float4 a,Float4 b,Float4 c,Float4 d,Float4 normal) {
  Mesh mesh;mesh.material=material;
  const float dx=b.x-a.x,dy=b.y-a.y,dz=b.z-a.z,scale=1/std::sqrt(dx*dx+dy*dy+dz*dz);
  const Float4 tangent{dx*scale,dy*scale,dz*scale,1};
  mesh.vertices={{a,normal,{0,0,0,0},tangent},{b,normal,{1,0,0,0},tangent},
    {c,normal,{1,1,0,0},tangent},{d,normal,{0,1,0,0},tangent}};
  mesh.indices={0,1,2,0,2,3};
  scene.instances.push_back({uint32_t(scene.meshes.size())});scene.meshes.push_back(std::move(mesh));
}

void box(Scene& scene,uint32_t material,float x0,float x1,float y0,float y1,float z0,float z1) {
  // Separate faces share one instance so the medium stack sees one closed solid.
  const size_t first=scene.meshes.size();
  quad(scene,material,{x0,y0,z1,0},{x1,y0,z1,0},{x1,y1,z1,0},{x0,y1,z1,0},{0,0,1,0});
  quad(scene,material,{x1,y0,z0,0},{x0,y0,z0,0},{x0,y1,z0,0},{x1,y1,z0,0},{0,0,-1,0});
  quad(scene,material,{x1,y0,z1,0},{x1,y0,z0,0},{x1,y1,z0,0},{x1,y1,z1,0},{1,0,0,0});
  quad(scene,material,{x0,y0,z0,0},{x0,y0,z1,0},{x0,y1,z1,0},{x0,y1,z0,0},{-1,0,0,0});
  quad(scene,material,{x0,y1,z1,0},{x1,y1,z1,0},{x1,y1,z0,0},{x0,y1,z0,0},{0,1,0,0});
  quad(scene,material,{x0,y0,z0,0},{x1,y0,z0,0},{x1,y0,z1,0},{x0,y0,z1,0},{0,-1,0,0});
  auto& combined=scene.meshes[first];
  for(size_t face=first+1;face<scene.meshes.size();++face) {
    const auto offset=uint32_t(combined.vertices.size());
    combined.vertices.insert(combined.vertices.end(),scene.meshes[face].vertices.begin(),scene.meshes[face].vertices.end());
    for(auto index:scene.meshes[face].indices)combined.indices.push_back(offset+index);
  }
  scene.meshes.resize(first+1);scene.instances.resize(first+1);
}

Material unlit(Float4 emission) {
  Material material;material.baseColor={0,0,0,1};material.factors.x=0;material.specular.x=0;
  material.emission=emission;return material;
}

Texture texture() {
  Texture result;result.width=16;result.height=8;
  for(uint32_t y=0;y<result.height;++y)for(uint32_t x=0;x<result.width;++x) {
    const float alpha=((x/2+y)%3)==0?0.0f:(((x+y)%3)==0?1.0f:.35f);
    result.pixels.push_back({.2f+.65f*float(x)/15,.15f+.7f*float(y)/7,.85f,alpha});
  }
  return result;
}

Scene primaryScene(Material foreground={}) {
  Scene scene;
  foreground.baseColor={0,0,0,1};foreground.factors.x=0;foreground.specular.x=0;
  foreground.emission={1.2f,.3f,.1f,0};
  scene.materials={foreground,unlit({.1f,.7f,1.1f,0})};
  // Keep global instruction offsets and texture IDs distinct from local registers.
  scene.materialInstructions.push_back({{uint32_t(MaterialOp::Constant),0,0,0},{23,29,31,37}});
  scene.textures={{1,1,{{1,1,1,1}}},texture()};
  quad(scene,0,{-2,-2,.2f,0},{2,-2,.2f,0},{2,2,.2f,0},{-2,2,.2f,0},{0,0,1,0});
  quad(scene,1,{-3,-3,-.6f,0},{3,-3,-.6f,0},{3,3,-.6f,0},{-3,3,-.6f,0},{0,0,1,0});
  return scene;
}

Program texturedCoverage(bool normalDependent=false) {
  Program program;
  const auto uv=program.op(MaterialOp::Texcoord),scale=program.constant({5.2f,3.7f,0,0});
  const auto tiled=program.op(MaterialOp::Multiply,uv,scale),offset=program.constant({-1.2f,.6f,0,0});
  const auto shifted=program.op(MaterialOp::Add,tiled,offset);
  const auto sample=program.op(MaterialOp::Image,shifted,1,14,{1.25f,0,0,0});
  const auto colorOutput=program.output(sample,MaterialParameter::BaseColor);
  const auto strength=program.scalar(.65f);
  const auto normal=program.op(MaterialOp::Normalmap,sample,strength);
  const auto normalOutput=program.output(normal,MaterialParameter::Normal);
  uint32_t opacity;
  if(normalDependent) {
    const auto weights=program.constant({.2f,.3f,.6f,0});
    opacity=program.op(MaterialOp::Dot,normalOutput,weights,0,{3,0,0,0});
  } else opacity=program.op(MaterialOp::Swizzle,colorOutput,0,0,{3,3,3,3});
  const auto low=program.scalar(0),high=program.scalar(1);
  program.output(program.op(MaterialOp::Clamp,opacity,low,high),MaterialParameter::Opacity);
  return program;
}

float mean(const Image& image,bool central=false) {
  double sum=0;uint32_t count=0;
  const uint32_t x0=central?image.width*3/8:0,x1=central?image.width*5/8:image.width;
  const uint32_t y0=central?image.height*3/8:0,y1=central?image.height*5/8:image.height;
  for(uint32_t y=y0;y<y1;++y)for(uint32_t x=x0;x<x1;++x) {sum+=image.pixels[size_t(y)*image.width+x].x;++count;}
  return float(sum/count);
}

void compare(const Image& reference,const Image& actual,const std::string& name,bool strict) {
  require(reference.width==actual.width&&reference.height==actual.height&&reference.samples==actual.samples,
    "Coverage optimization changed image dimensions or progressive sample count");
  for(size_t index=0;index<reference.pixels.size();++index) {
    const auto a=reference.pixels[index],b=actual.pixels[index];
    for(auto value:{a.x,a.y,a.z,b.x,b.y,b.z})require(std::isfinite(value)&&value>=0,"Coverage transport produced nonfinite/negative radiance");
    auto near=[&](float left,float right) {return std::abs(left-right)<=(strict?2e-6f:2e-5f+2e-4f*std::max(left,right));};
    if(!near(a.x,b.x)||!near(a.y,b.y)||!near(a.z,b.z)||!near(a.w,b.w))
      throw std::runtime_error(name+" differs at pixel "+std::to_string(index)+": reference=("+
        std::to_string(a.x)+","+std::to_string(a.y)+","+std::to_string(a.z)+"), optimized=("+
        std::to_string(b.x)+","+std::to_string(b.y)+","+std::to_string(b.z)+")");
  }
}

RendererOptions options(bool validation,bool optimize) {
  RendererOptions result;result.validation=validation;result.optimizeMaterialCoverage=optimize;return result;
}

struct Harness {
  Renderer reference,optimized;
  RenderSettings settings;
  Camera camera{{0,0,4,0},{0,0,0,0},{0,1,0,0},35};
  explicit Harness(bool validation):reference(options(validation,false)),optimized(options(validation,true)) {
    settings.width=24;settings.height=18;settings.maxBounces=4;settings.environment={};
  }
  Image run(const Scene& scene,const std::string& name,uint32_t samples=24,bool strict=false) {
    for(auto* renderer:{&reference,&optimized}) {renderer->setSettings(settings);renderer->setScene(scene);renderer->setCamera(camera);renderer->render();}
    compare(reference.readbackRaw(),optimized.readbackRaw(),name+" first sample",strict);
    for(auto* renderer:{&reference,&optimized})renderer->render(samples-1);
    const auto actual=optimized.readbackRaw();compare(reference.readbackRaw(),actual,name+" progressive samples",strict);
    require(reference.sampleCount()==samples&&optimized.sampleCount()==samples&&reference.frameCount()==samples&&optimized.frameCount()==samples,
      "Coverage optimization changed sample or random-frame progression");
    return actual;
  }
};

void verifyPrimary(Harness& harness) {
  auto scene=primaryScene();attach(scene,0,texturedCoverage());
  harness.run(scene,"Filtered texture coverage and shared color/normal dependencies",32,true);
  // A second independently instanced plate tests repeated stochastic rejection.
  auto layered=scene;layered.instances.push_back({0});layered.instances.back().transform[11]=.3f;
  harness.run(layered,"Layered stochastic primary rejection",32,true);
  auto mirrored=primaryScene();attach(mirrored,0,texturedCoverage(true));
  mirrored.instances[0].transform={-1.4f,0,0,0, 0,.8f,0,0, 0,0,1.2f,0};
  harness.run(mirrored,"Normalmap-dependent opacity on mirrored nonuniform instance",32,true);
  for(float coverage:{0.0f,1.0f}) {
    Material material;material.factors.z=coverage;
    const auto image=harness.run(primaryScene(material),"Constant endpoint coverage",8,true);
    require(std::abs(mean(image)-(coverage==0?.1f:1.2f))<2e-6f,"Constant coverage endpoint fixture did not isolate primary acceptance");
  }
  auto overwritten=primaryScene();Program program;
  const auto first=program.output(program.scalar(.1f),MaterialParameter::Opacity);
  const auto increment=program.scalar(.45f);
  program.output(program.op(MaterialOp::Add,first,increment),MaterialParameter::Opacity);
  program.output(program.scalar(.8f),MaterialParameter::Opacity);
  attach(overwritten,0,program);harness.run(overwritten,"Last opacity write and Output-register dependencies",32,true);

  // updateCamera changes the raw accumulation while preserving random-frame
  // progression. Coverage rejection must consume the same RNG draws in both.
  auto moved=harness.camera;moved.position.x=.12f;
  for(auto* renderer:{&harness.reference,&harness.optimized}) {renderer->updateCamera(moved);renderer->render(7);}
  compare(harness.reference.readbackRaw(),harness.optimized.readbackRaw(),"Continuous camera motion",true);
  require(harness.reference.sampleCount()==7&&harness.optimized.sampleCount()==7&&
    harness.reference.frameCount()==39&&harness.optimized.frameCount()==39,"Coverage changed continuous-motion history counters");
}

void verifyNativeModes(Harness& harness) {
  for(uint32_t mode:{0u,1u,2u}) {
    Material material;material.program.z=uint32_t(MaterialModel::GltfPbr);material.fuzz.z=float((mode+1)%3);
    auto scene=primaryScene(material);auto program=texturedCoverage();
    program.output(program.scalar(float(mode)),MaterialParameter::CoatRotation);
    program.output(program.scalar(.6f),MaterialParameter::CoatAffectRoughness);
    attach(scene,0,program);harness.run(scene,"Graph-driven glTF alpha mode and cutoff",32,true);
  }
  for(bool threshold:{false,true}) {
    Material material;material.program.z=uint32_t(MaterialModel::UsdPreviewSurface);
    material.factors.w=1;material.fuzz.w=threshold?1.0f:0.0f;material.specular.w=0;
    auto scene=primaryScene(material);auto program=texturedCoverage();
    program.output(program.scalar(1),MaterialParameter::CoatAffectColor);
    program.output(program.scalar(threshold?.55f:0.0f),MaterialParameter::Rotation);
    attach(scene,0,program);harness.run(scene,"Graph-driven USD presence and threshold",32,true);
  }
}

void verifyFallbacks(Harness& harness) {
  auto longScene=primaryScene();Program longProgram;auto value=longProgram.scalar(.1f),step=longProgram.scalar(.01f);
  for(uint32_t index=0;index<36;++index)value=longProgram.op(MaterialOp::Add,value,step);
  longProgram.output(value,MaterialParameter::Opacity);attach(longScene,0,longProgram);
  harness.run(longScene,"Long coverage graph retains the complete-program fallback",32,true);

  auto hexScene=primaryScene();Program hex;
  const auto uv=hex.op(MaterialOp::Texcoord),configuration=uint32_t(hex.instructions.size());
  hex.constant({1,1,1,.75f});hex.constant({0,360,.5f,2});
  hex.constant({0,1,.5f,0});hex.constant({.2722287f,.6740818f,.0536895f,0});
  const auto sampled=hex.op(MaterialOp::HexImage,uv,1,configuration);
  const auto alpha=hex.op(MaterialOp::Swizzle,sampled,0,0,{3,3,3,3});
  hex.output(alpha,MaterialParameter::Opacity);attach(hexScene,0,hex);
  harness.run(hexScene,"Packed HexImage configuration fallback",32,true);
}

Scene shadowScene() {
  Scene scene;Material receiver;receiver.baseColor={.7f,.65f,.6f,1};receiver.specular.x=0;
  scene.materials={receiver,unlit({8,8,8,0}),unlit({})};
  scene.textures={{1,1,{{1,1,1,1}}},texture()};
  quad(scene,0,{-3,-3,0,0},{3,-3,0,0},{3,3,0,0},{-3,3,0,0},{0,0,1,0});
  quad(scene,1,{1.5f,-.7f,2,0},{1.5f,.7f,2,0},{2.9f,.7f,2,0},{2.9f,-.7f,2,0},{0,0,-1,0});
  return scene;
}

void verifyShadowAndMedia(Harness& harness) {
  const auto lit=shadowScene();const auto baseline=harness.run(lit,"Opaque receiver and off-axis light",32);
  require(mean(baseline,true)>.01f,"Shadow fixture must receive indirect-camera area-light illumination");
  auto filtered=lit;
  quad(filtered,2,{.3f,-2,.8f,0},{2.1f,-2,.8f,0},{2.1f,2,.8f,0},{.3f,2,.8f,0},{0,0,1,0});
  attach(filtered,2,texturedCoverage());harness.run(filtered,"Filtered alpha shadow and light connection",32);

  auto transparent=lit;Program zero;zero.output(zero.scalar(0),MaterialParameter::Opacity);attach(transparent,2,zero);
  const size_t plate=transparent.meshes.size();
  quad(transparent,2,{.3f,-2,.8f,0},{2.1f,-2,.8f,0},{2.1f,2,.8f,0},{.3f,2,.8f,0},{0,0,1,0});
  for(uint32_t layer=1;layer<80;++layer) {transparent.instances.push_back({uint32_t(plate)});transparent.instances.back().transform[11]=float(layer)*.005f;}
  const auto through=harness.run(transparent,"Eighty transparent shadow layers retain progress",16);
  require(mean(through,true)>.01f,"Transparent shadow layers incorrectly hide the area light");

  auto volume=filtered;Material medium;medium.volumeAbsorption={.2f,.4f,.8f,0};
  volume.materials.push_back(medium);Program interior;
  interior.output(interior.scalar(1),MaterialParameter::VolumeDensity);
  interior.output(interior.scalar(1),MaterialParameter::VolumeBoundaryOnly);
  attach(volume,3,interior);box(volume,3,.4f,2.2f,-2,2,.85f,1.6f);
  harness.run(volume,"Graph-activated interior keeps full boundary and transmittance handling",32);
}

}

int main(int argc,char**) {
  try {
    Harness harness(argc>1);
    verifyPrimary(harness);verifyNativeModes(harness);verifyFallbacks(harness);verifyShadowAndMedia(harness);
    require(harness.reference.validationErrorCount()==0&&harness.optimized.validationErrorCount()==0,"Vulkan validation errors in coverage transport");
    std::cout<<"Coverage transport equivalence passed: primary/shadow RNG, progressive and motion history, filtered UV/normal dependencies, native alpha modes, long/packed fallbacks and graph-activated media.\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
