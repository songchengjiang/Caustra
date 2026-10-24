#include <caustra/renderer.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using namespace caustra;
struct Program {
  std::vector<MaterialInstruction> instructions;
  uint32_t op(MaterialOp operation,uint32_t a=0,uint32_t b=0,uint32_t c=0,Float4 value={}) {
    instructions.push_back({{static_cast<uint32_t>(operation),a,b,c},value});
    return static_cast<uint32_t>(instructions.size()-1);
  }
  uint32_t constant(Float4 value) {return op(MaterialOp::Constant,0,0,0,value);}
  void output(uint32_t reg,MaterialParameter parameter=MaterialParameter::Emission) {op(MaterialOp::Output,reg,static_cast<uint32_t>(parameter));}
};

Scene sceneFor(Program program,std::vector<Texture> textures={}) {
  Scene scene;
  Material material;
  material.baseColor={0,0,0,1};material.specular.x=0;
  // A nonzero instruction offset catches accidental use of global register IDs.
  scene.materialInstructions.push_back({{static_cast<uint32_t>(MaterialOp::Constant),0,0,0},{99,99,99,99}});
  material.program={1,static_cast<uint32_t>(program.instructions.size()),0,0};
  scene.materialInstructions.insert(scene.materialInstructions.end(),program.instructions.begin(),program.instructions.end());
  scene.materials.push_back(material);
  scene.textures=std::move(textures);
  Mesh plane;
  plane.vertices={{{-4,-4,0,0},{0,0,1,0},{0,0,0,0},{1,0,0,1}},
                  {{4,-4,0,0},{0,0,1,0},{1,0,0,0},{1,0,0,1}},
                  {{4,4,0,0},{0,0,1,0},{1,1,0,0},{1,0,0,1}},
                  {{-4,4,0,0},{0,0,1,0},{0,1,0,0},{1,0,0,1}}};
  plane.indices={0,1,2,0,2,3};
  scene.meshes.push_back(std::move(plane));scene.instances.push_back({0});
  return scene;
}

void check(Renderer& renderer,const std::string& name,Program program,Float4 expected,std::vector<Texture> textures={}) {
  renderer.setScene(sceneFor(std::move(program),std::move(textures)));
  renderer.render(1);
  const auto image=renderer.readbackRaw();
  for(auto pixel:image.pixels) {
    if(!std::isfinite(pixel.x)||!std::isfinite(pixel.y)||!std::isfinite(pixel.z)
       ||std::abs(pixel.x-expected.x)>3e-4f||std::abs(pixel.y-expected.y)>3e-4f||std::abs(pixel.z-expected.z)>3e-4f)
      throw std::runtime_error(name+": expected ("+std::to_string(expected.x)+","+std::to_string(expected.y)+","+std::to_string(expected.z)+") but got ("+std::to_string(pixel.x)+","+std::to_string(pixel.y)+","+std::to_string(pixel.z)+")");
  }
}

void binary(Renderer& renderer,const char* name,MaterialOp operation,Float4 a,Float4 b,Float4 expected) {
  Program p;auto ra=p.constant(a),rb=p.constant(b);p.output(p.op(operation,ra,rb));check(renderer,name,std::move(p),expected);
}
void unary(Renderer& renderer,const char* name,MaterialOp operation,Float4 value,Float4 expected,Float4 options={}) {
  Program p;auto reg=p.constant(value);p.output(p.op(operation,reg,0,0,options));check(renderer,name,std::move(p),expected);
}
Texture fixture() {
  // Top row red/green, bottom row blue/white; the 1x1 mip must be gray.
  return {2,2,{{1,0,0,1},{0,1,0,1},{0,0,1,1},{1,1,1,1}}};
}
void image(Renderer& renderer,const char* name,Float4 uv,uint32_t flags,Float4 expected,float lodBias=0) {
  Program p;auto coordinate=p.constant(uv);p.output(p.op(MaterialOp::Image,coordinate,0,flags,{lodBias,0,0,0}));
  check(renderer,name,std::move(p),expected,{fixture()});
}

uint32_t hexImage(Program& program,uint32_t uv,Float4 primary={1,1,1,.75f},Float4 extra={0,1,.5f,0}) {
  // The four packed configuration registers must stay consecutive. This
  // fixture also runs behind sceneFor's nonzero global instruction offset.
  const auto configuration=static_cast<uint32_t>(program.instructions.size());
  program.constant(primary);
  program.constant({0,360,.5f,2});
  program.constant(extra);
  program.constant({.2722287f,.6740818f,.0536895f,0});
  return program.op(MaterialOp::HexImage,uv,0,configuration);
}

void verifyOfficialPatterns(Renderer& renderer) {
  unary(renderer,"MaterialX RGB to HSV",MaterialOp::RgbToHsv,{.2f,.4f,.8f,.3f},{11.0f/18.0f,.75f,.8f,0});
  unary(renderer,"MaterialX HSV to RGB",MaterialOp::HsvToRgb,{11.0f/18.0f,.75f,.8f,.3f},{.2f,.4f,.8f,0});
  unary(renderer,"MaterialX HSV hue wraps",MaterialOp::HsvToRgb,{1.25f,1,.8f,0},{.4f,.8f,0,0});
  unary(renderer,"MaterialX HSV gray threshold",MaterialOp::HsvToRgb,{.8f,.00001f,.6f,0},{.6f,.6f,.6f,0});
  unary(renderer,"MaterialX RGB gray",MaterialOp::RgbToHsv,{.6f,.6f,.6f,0},{0,0,.6f,0});
  {
    Program p;auto uv=p.constant({.23f,.67f,0,0});p.output(hexImage(p,uv));
    check(renderer,"Hex tiling preserves a constant texture",std::move(p),{.2f,.4f,.8f,0},{{1,1,{{.2f,.4f,.8f,1}}}});
  }
  {
    Program p;auto uv=p.constant({.39f,.19f,0,0});p.output(hexImage(p,uv,{1,1,1,.75f},{0,1,1,0}));
    check(renderer,"Hex all-black luminance weights stay finite",std::move(p),{},{{1,1,{{0,0,0,1}}}});
  }
  {
    Program p;auto uv=p.constant({.25f,.75f,0,0});p.output(hexImage(p,uv,{0,0,0,.75f}));
    check(renderer,"Hex identity transforms equal the source image",std::move(p),{1,0,0,0},{fixture()});
  }
  {
    Program p;auto coordinate=p.op(MaterialOp::Texcoord),scale=p.constant({7,7,0,0});
    auto tiled=p.op(MaterialOp::Multiply,coordinate,scale);p.output(hexImage(p,tiled));
    renderer.setScene(sceneFor(std::move(p),{fixture()}));renderer.render(1);
    const auto image=renderer.readbackRaw();float lowest=1,highest=0;
    for(const auto& pixel:image.pixels) {
      if(!std::isfinite(pixel.x)||!std::isfinite(pixel.y)||!std::isfinite(pixel.z))
        throw std::runtime_error("Hex tiling produced nonfinite radiance");
      lowest=std::min(lowest,pixel.x);highest=std::max(highest,pixel.x);
    }
    if(highest-lowest<.05f) throw std::runtime_error("Hex tiling lost its spatial image variation");
  }
}

void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}

void attach(Scene& scene,uint32_t material,const Program& program) {
  scene.materials[material].program.x=static_cast<uint32_t>(scene.materialInstructions.size());
  scene.materials[material].program.y=static_cast<uint32_t>(program.instructions.size());
  scene.materialInstructions.insert(scene.materialInstructions.end(),program.instructions.begin(),program.instructions.end());
}

void quad(Scene& scene,uint32_t material,Float4 a,Float4 b,Float4 c,Float4 d,Float4 normal) {
  const float x=b.x-a.x,y=b.y-a.y,z=b.z-a.z,scale=1/std::sqrt(x*x+y*y+z*z);
  const Float4 tangent{x*scale,y*scale,z*scale,1};
  Mesh mesh;mesh.material=material;
  mesh.vertices={{a,normal,{0,0,0,0},tangent},{b,normal,{1,0,0,0},tangent},
                 {c,normal,{1,1,0,0},tangent},{d,normal,{0,1,0,0},tangent}};
  mesh.indices={0,1,2,0,2,3};
  scene.instances.push_back({static_cast<uint32_t>(scene.meshes.size())});scene.meshes.push_back(std::move(mesh));
}

Scene lightingScene() {
  Scene scene;
  Material receiver;receiver.baseColor={0.7f,0.7f,0.7f,1};receiver.specular.x=0;
  Material emitter;emitter.baseColor={0,0,0,1};emitter.specular.x=0;emitter.emission={8,8,8,0};
  scene.materials={receiver,emitter};
  // The camera sees only the nonemissive receiver. The off-axis area light is
  // outside its visible field, so a test cannot pass by directly seeing emission.
  quad(scene,0,{-3,-3,0,0},{3,-3,0,0},{3,3,0,0},{-3,3,0,0},{0,0,1,0});
  quad(scene,1,{1.5f,-0.7f,2,0},{1.5f,0.7f,2,0},{2.9f,0.7f,2,0},{2.9f,-0.7f,2,0},{0,0,-1,0});
  return scene;
}

Image render(Renderer& renderer,const Scene& scene,uint32_t samples=32,Camera camera={{0,0,4,0},{0,0,0,0},{0,1,0,0},35}) {
  renderer.setScene(scene);renderer.setCamera(camera);renderer.render(samples);auto result=renderer.readbackRaw();
  for(auto p:result.pixels)
    require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&p.x>=0&&p.y>=0&&p.z>=0,"Nonfinite/negative radiance in material transport regression");
  return result;
}

Float4 mean(const Image& image,bool central=false) {
  double rgb[3]{};uint32_t count=0;
  const uint32_t x0=central?image.width*3/8:0,x1=central?image.width*5/8:image.width;
  const uint32_t y0=central?image.height*3/8:0,y1=central?image.height*5/8:image.height;
  for(uint32_t y=y0;y<y1;++y)for(uint32_t x=x0;x<x1;++x) {
    const auto p=image.pixels[size_t(y)*image.width+x];rgb[0]+=p.x;rgb[1]+=p.y;rgb[2]+=p.z;++count;
  }
  return {float(rgb[0]/count),float(rgb[1]/count),float(rgb[2]/count),0};
}

void compare(const Image& a,const Image& b,const char* message,bool central=false) {
  require(a.width==b.width&&a.height==b.height,"Transport comparison has different image dimensions");
  const uint32_t x0=central?a.width*3/8:0,x1=central?a.width*5/8:a.width;
  const uint32_t y0=central?a.height*3/8:0,y1=central?a.height*5/8:a.height;
  for(uint32_t y=y0;y<y1;++y)for(uint32_t x=x0;x<x1;++x) {
    auto p=a.pixels[size_t(y)*a.width+x],q=b.pixels[size_t(y)*b.width+x];
    if(!(std::abs(p.x-q.x)<2e-5f+2e-4f*std::max(p.x,q.x)
      && std::abs(p.y-q.y)<2e-5f+2e-4f*std::max(p.y,q.y)
      && std::abs(p.z-q.z)<2e-5f+2e-4f*std::max(p.z,q.z)))
      throw std::runtime_error(std::string(message)+" at pixel ("+std::to_string(x)+","+std::to_string(y)+
        "): reference=("+std::to_string(p.x)+","+std::to_string(p.y)+","+std::to_string(p.z)+
        "), actual=("+std::to_string(q.x)+","+std::to_string(q.y)+","+std::to_string(q.z)+")");
  }
}

Program defaultTangentProgram(uint32_t lanes=3,MaterialParameter output=MaterialParameter::Tangent,
                              MaterialOp source=MaterialOp::Tangent,uint32_t outputRegister=1) {
  Program p;
  p.op(source);
  p.op(MaterialOp::Normalize,0,0,0,{float(lanes),0,0,0});
  p.output(outputRegister,output);
  return p;
}

void verifyDefaultTangentFastPath(Renderer& renderer) {
  auto scene=lightingScene();auto& material=scene.materials[0];
  // Keep live local registers distinct from the global instruction offset.
  scene.materialInstructions.push_back({{uint32_t(MaterialOp::Constant),0,0,0},{99,99,99,99}});
  material.baseColor={0.85f,0.75f,0.6f,1};material.surface={1,0.35f,0,0};
  material.specular={1,1.5f,0.8f,0};
  auto compareProgram=[&](const Scene& input,const Program& program,uint32_t samples=64) {
    require(program.instructions.size()==3,"Tangent fast-path fixture must contain exactly three instructions");
    auto fast=input;attach(fast,0,program);
    auto generic=input;auto forced=program;
    // The same live graph plus a dead instruction forces the generic evaluator.
    // No compiler or asset rewriting is involved, so existing packages exercise
    // the same recognizer used by renderer uploads.
    forced.constant({17,19,23,29});attach(generic,0,forced);
    const auto reference=render(renderer,generic,samples);
    const auto optimized=render(renderer,fast,samples);
    compare(reference,optimized,"Three-instruction tangent program differs from the generic four-instruction control");
    return optimized;
  };
  for(uint32_t model:{0u,1u}) {
    material.program.z=model;
    for(auto& vertex:scene.meshes[0].vertices)vertex.tangent={3,0,0,1};
    const auto x=compareProgram(scene,defaultTangentProgram());
    for(auto& vertex:scene.meshes[0].vertices)vertex.tangent={0,7,0,1};
    const auto y=compareProgram(scene,defaultTangentProgram());
    const auto a=mean(x,true),b=mean(y,true);
    require(std::abs(a.x-b.x)>0.1f*std::max(a.x,b.x),
            "Default tangent fast path must preserve an observable anisotropic tangent rotation");
  }

  // Vertex tangents may be nonunit; production surfaceFrame normalizes their
  // context before either evaluator. Tilt the entire scene so that a diagonal
  // context tangent also has a nonzero Z lane. Normalize(2) must therefore not
  // be mistaken for Normalize(3), even after the BSDF projects its frame.
  material.program.z=0;
  for(auto& vertex:scene.meshes[0].vertices)vertex.tangent={3,4,0,1};
  const float c=std::cos(0.65f),s=std::sin(0.65f);
  for(auto& instance:scene.instances)instance.transform={c,0,s,0, 0,1,0,0, -s,0,c,0};
  const auto full=compareProgram(scene,defaultTangentProgram());
  const auto xy=compareProgram(scene,defaultTangentProgram(2));
  const auto fullMean=mean(full),xyMean=mean(xy);
  require(std::abs(fullMean.x-xyMean.x)>1.0e-4f,
          "Normalize lane guard fixture must change anisotropic lighting");

  // Near matches retain their actual opcode, lane count, output parameter and
  // source register. Compare each with the forced-generic form independently.
  compareProgram(scene,defaultTangentProgram(4));
  compareProgram(scene,defaultTangentProgram(0));
  compareProgram(scene,defaultTangentProgram(3,MaterialParameter::CoatTangent));
  const auto emitted=compareProgram(scene,defaultTangentProgram(3,MaterialParameter::Emission));
  require(mean(emitted).x>mean(full).x+0.01f,"Output-parameter guard fixture must have visible direct emission");
  const auto normal=compareProgram(scene,defaultTangentProgram(3,MaterialParameter::Tangent,MaterialOp::Normal));
  require(std::abs(mean(normal).x-mean(full).x)>1.0e-4f,
          "Source-opcode guard fixture must change the anisotropic frame");
  compareProgram(scene,defaultTangentProgram(3,MaterialParameter::Tangent,MaterialOp::Tangent,0));
}

void verifyMaterialTransport(Renderer& renderer) {
  const auto original=lightingScene();
  const auto constant=render(renderer,original);
  require(mean(constant).x>0.01f,"Area light failed to illuminate nonemissive receiver");
  auto generated=original;generated.materials[1].emission={};
  Program emission;emission.output(emission.constant({8,8,8,0}));attach(generated,1,emission);
  const auto graph=render(renderer,generated);
  compare(constant,graph,"Graph emission and equal CPU emission must illuminate the receiver equally (including NEE)");

  // The two images use the same sampling sequence and neutral receiver. Red +
  // blue emission is identically 8 at every UV, so channel superposition must
  // reproduce the white light's red channel, including both NEE and BSDF hits.
  auto textured=original;textured.materials[1].emission={};
  textured.textures.push_back({2,1,{{8,8,8,1},{8,8,8,1}}});
  Program imageProgram;auto uv=imageProgram.op(MaterialOp::Texcoord);
  imageProgram.output(imageProgram.op(MaterialOp::Image,uv,0,5));attach(textured,1,imageProgram);
  const auto white=render(renderer,textured);
  compare(constant,white,"A white texture emitter must match the equal constant emitter on a receiver");
  textured.textures[0].pixels={{8,0,0,1},{0,0,8,1}};
  const auto redBlue=render(renderer,textured);const auto coloredMean=mean(redBlue);
  require(coloredMean.x>0.001f&&coloredMean.z>0.001f&&coloredMean.y<1e-6f,"Red/blue emitter texture did not color receiver illumination");
  for(size_t i=0;i<white.pixels.size();++i) {
    const float sum=redBlue.pixels[i].x+redBlue.pixels[i].z,reference=white.pixels[i].x;
    require(std::abs(sum-reference)<3e-5f+3e-4f*reference,"Textured emitter violates linear channel superposition on receiver");
  }

  // A black mask between the receiver and off-axis light is outside the central
  // camera rays but intersects every central receiver-to-light segment.
  auto masked=original;Material blocker;blocker.baseColor={0,0,0,1};blocker.specular.x=0;
  masked.materials.push_back(blocker);
  quad(masked,2,{0.5f,-2,1,0},{2,-2,1,0},{2,2,1,0},{0.5f,2,1,0},{0,0,1,0});
  Program transparent;transparent.output(transparent.constant({0,0,0,0}),MaterialParameter::Opacity);attach(masked,2,transparent);
  const auto clearShadow=render(renderer,masked,64);
  const auto unmasked=render(renderer,original,64);
  compare(unmasked,clearShadow,"Zero graph opacity must transmit shadow rays to the light",true);
  auto deepShadow=original;deepShadow.materials.push_back(blocker);
  // All 80 plates intersect central receiver-to-light segments while staying
  // outside the corresponding camera rays. This catches a fixed 64-hit shadow
  // traversal budget returning black before it actually reaches the light.
  const uint32_t shadowPlateMesh=static_cast<uint32_t>(deepShadow.meshes.size());
  quad(deepShadow,2,{0.3f,-2,0.8f,0},{2.1f,-2,0.8f,0},{2.1f,2,0.8f,0},{0.3f,2,0.8f,0},{0,0,1,0});
  for(uint32_t layer=1;layer<80;++layer) {
    Instance plate;plate.mesh=shadowPlateMesh;plate.transform[11]=0.005f*layer;deepShadow.instances.push_back(plate);
  }
  attach(deepShadow,2,transparent);
  compare(unmasked,render(renderer,deepShadow,64),"80 fully transparent plates must preserve receiver illumination through shadow rays",true);
  auto opaque=masked;opaque.materialInstructions[opaque.materials[2].program.x].value={1,1,1,1};
  const auto fullShadow=render(renderer,opaque,64);
  require(mean(fullShadow,true).x<1e-6f,"Opaque mask failed to block direct and BSDF illumination");
  auto half=masked;half.materialInstructions[half.materials[2].program.x].value={0.5f,0.5f,0.5f,0.5f};
  const auto halfShadow=render(renderer,half,128);
  const float ratio=mean(halfShadow,true).x/mean(clearShadow,true).x;
  require(ratio>0.3f&&ratio<0.7f,"Half opacity must transmit approximately half the receiver illumination");

  Program flatEmission;flatEmission.output(flatEmission.constant({2,0.5f,1,0}));
  auto primary=sceneFor(flatEmission);const auto unobscured=render(renderer,primary,1);
  Material front;front.baseColor={0,0,0,1};front.specular.x=0;primary.materials.push_back(front);
  quad(primary,1,{-3,-3,1,0},{3,-3,1,0},{3,3,1,0},{-3,3,1,0},{0,0,1,0});
  attach(primary,1,transparent);
  compare(unobscured,render(renderer,primary,1),"Zero graph opacity on primary rays must reveal the emitter behind it");
  auto deepPrimary=sceneFor(flatEmission);deepPrimary.materials.push_back(front);
  // Every primary ray must traverse all 80 coverage-zero plates before seeing
  // the known emitter. A fixed 64-hit budget would incorrectly return the black
  // environment instead of its exact radiance.
  const uint32_t primaryPlateMesh=static_cast<uint32_t>(deepPrimary.meshes.size());
  quad(deepPrimary,1,{-3,-3,0.08f,0},{3,-3,0.08f,0},{3,3,0.08f,0},{-3,3,0.08f,0},{0,0,1,0});
  for(uint32_t layer=1;layer<80;++layer) {
    Instance plate;plate.mesh=primaryPlateMesh;plate.transform[11]=0.035f*layer;deepPrimary.instances.push_back(plate);
  }
  attach(deepPrimary,1,transparent);
  compare(unobscured,render(renderer,deepPrimary,1),"80 fully transparent plates must reveal the exact emitter radiance on primary rays");
  primary.materialInstructions[primary.materials[1].program.x].value={1,1,1,1};
  require(mean(render(renderer,primary,1)).x<1e-6f,"Opaque front plane must hide the emitter behind it");

  auto sided=sceneFor(flatEmission);sided.materials[0].surface.z=1;
  compare(unobscured,render(renderer,sided,1),"One-sided emitter must preserve front-side radiance");
  const Camera back{{0,0,-4,0},{0,0,0,0},{0,1,0,0},35};
  require(mean(render(renderer,sided,1,back)).x<1e-6f,"One-sided emitter must not radiate from its back face");
  sided.materials[0].surface.z=0;
  compare(unobscured,render(renderer,sided,1,back),"Two-sided control emitter must radiate equally from both faces");

  // Directional anisotropy provides an observable test of the two tangent
  // outputs, rather than checking only that their instructions are accepted.
  auto anisotropic=original;auto& receiver=anisotropic.materials[0];
  receiver.baseColor={0.8f,0.8f,0.8f,1};receiver.surface={1,0.35f,0,0};receiver.specular={1,1.5f,0.8f,0};
  const auto tangentX=render(renderer,anisotropic,64);
  Program tangent;tangent.output(tangent.constant({0,1,0,0}),MaterialParameter::Tangent);attach(anisotropic,0,tangent);
  const auto tangentY=render(renderer,anisotropic,64);
  require(std::abs(mean(tangentX,true).x-mean(tangentY,true).x)>0.1f*std::max(mean(tangentX,true).x,mean(tangentY,true).x),"Material tangent output failed to rotate anisotropic reflection");
  auto coated=original;coated.materials[0].baseColor={0,0,0,1};coated.materials[0].specular.x=0;
  coated.materials[0].coat={1,0.35f,1.5f,0.8f};
  const auto coatX=render(renderer,coated,64);
  Program coat;coat.output(coat.constant({0,1,0,0}),MaterialParameter::CoatTangent);
  coat.output(coat.constant({0.8f,0.8f,0.8f,0.8f}),MaterialParameter::CoatAnisotropy);attach(coated,0,coat);
  const auto coatY=render(renderer,coated,64);
  require(std::abs(mean(coatX,true).x-mean(coatY,true).x)>0.1f*std::max(mean(coatX,true).x,mean(coatY,true).x),"Coat tangent output failed to rotate anisotropic coat reflection");
}
}

int main() {
  try {
    caustra::Renderer renderer({true});
    caustra::RenderSettings settings;settings.width=16;settings.height=16;settings.maxBounces=1;settings.environment={};
    renderer.setSettings(settings);renderer.setCamera({{0,0,2,0},{0,0,0,0},{0,1,0,0},45});
    binary(renderer,"Add",MaterialOp::Add,{0.1f,0.2f,0.3f,0},{0.2f,0.3f,0.4f,0},{0.3f,0.5f,0.7f,0});
    binary(renderer,"Subtract",MaterialOp::Subtract,{0.9f,0.7f,0.5f,0},{0.2f,0.3f,0.4f,0},{0.7f,0.4f,0.1f,0});
    binary(renderer,"Multiply",MaterialOp::Multiply,{0.2f,0.3f,0.4f,0},{2,2,2,2},{0.4f,0.6f,0.8f,0});
    binary(renderer,"Divide",MaterialOp::Divide,{0.2f,0.4f,0.6f,0},{2,2,2,2},{0.1f,0.2f,0.3f,0});
    binary(renderer,"Power",MaterialOp::Power,{0.2f,0.4f,0.6f,0},{2,2,2,2},{0.04f,0.16f,0.36f,0});
    binary(renderer,"Min",MaterialOp::Min,{0.2f,0.8f,0.4f,0},{0.5f,0.5f,0.5f,0},{0.2f,0.5f,0.4f,0});
    binary(renderer,"Max",MaterialOp::Max,{0.2f,0.8f,0.4f,0},{0.5f,0.5f,0.5f,0},{0.5f,0.8f,0.5f,0});
    unary(renderer,"Abs",MaterialOp::Abs,{-0.2f,-0.4f,-0.6f,0},{0.2f,0.4f,0.6f,0});
    unary(renderer,"OneMinus",MaterialOp::OneMinus,{0.2f,0.4f,0.6f,0},{0.8f,0.6f,0.4f,0});
    unary(renderer,"Sin",MaterialOp::Sin,{0,0.5235987756f,1.570796327f,0},{0,0.5f,1,0});
    unary(renderer,"Cos",MaterialOp::Cos,{0,1.047197551f,1.570796327f,0},{1,0.5f,0,0});
    unary(renderer,"Floor",MaterialOp::Floor,{0.2f,1.8f,2.4f,0},{0,1,2,0});
    unary(renderer,"Ceil",MaterialOp::Ceil,{0.2f,1.8f,2.4f,0},{1,2,3,0});
    unary(renderer,"Fract",MaterialOp::Fract,{0.2f,1.8f,-0.4f,0},{0.2f,0.8f,0.6f,0});
    unary(renderer,"Normalize",MaterialOp::Normalize,{3,4,0,99},{0.6f,0.8f,0,0},{3,0,0,0});
    unary(renderer,"Length",MaterialOp::Length,{3,4,0,99},{5,5,5,0},{3,0,0,0});
    unary(renderer,"Swizzle",MaterialOp::Swizzle,{0.1f,0.2f,0.3f,0.4f},{0.3f,0.1f,0.4f,0},{2,0,3,1});
    {
      Program p;auto a=p.constant({0.2f,0.4f,0.6f,0}),b=p.constant({0.8f,0.6f,0.4f,0}),c=p.constant({0.25f,0.25f,0.25f,0.25f});
      p.output(p.op(MaterialOp::Mix,a,b,c));check(renderer,"Mix background/foreground",std::move(p),{0.35f,0.45f,0.55f,0});
    }
    {
      Program p;auto a=p.constant({0.1f,0.5f,0.9f,0}),lo=p.constant({0.2f,0.2f,0.2f,0}),hi=p.constant({0.8f,0.8f,0.8f,1});
      p.output(p.op(MaterialOp::Clamp,a,lo,hi));check(renderer,"Clamp",std::move(p),{0.2f,0.5f,0.8f,0});
    }
    {
      Program p;auto a=p.constant({1,2,3,99}),b=p.constant({2,3,4,99});p.output(p.op(MaterialOp::Dot,a,b,0,{3,0,0,0}));
      check(renderer,"Dot excludes unused fourth lane",std::move(p),{20,20,20,0});
    }
    binary(renderer,"Cross",MaterialOp::Cross,{1,0,0,0},{0,1,0,0},{0,0,1,0});
    {
      Program p;auto a=p.constant({0.2f,0,0,0}),b=p.constant({0.4f,0,0,0}),c=p.constant({0.6f,0,0,0});
      p.output(p.op(MaterialOp::Combine,a,b,c,{0,0,0,0.8f}));check(renderer,"Combine",std::move(p),{0.2f,0.4f,0.6f,0});
    }
    {
      Program p;auto uv=p.constant({0.1f,0.1f,0,0}),a=p.constant({0.2f,0.4f,0.6f,0}),b=p.constant({0.8f,0.7f,0.6f,0});
      p.output(p.op(MaterialOp::Checker,uv,a,b,{2,2,0,0}));check(renderer,"Checker even cell",std::move(p),{0.2f,0.4f,0.6f,0});
    }
    {
      Program p;auto color=p.constant({0.75f,0.75f,1,0}),strength=p.constant({1,1,1,1});p.output(p.op(MaterialOp::Normalmap,color,strength));
      check(renderer,"Tangent-space normalmap",std::move(p),{0.40824829f,0.40824829f,0.81649658f,0});
    }
    // Numeric reference from MaterialX 1.39.5's OSL-derived scalar Perlin
    // definition, not a hash/value-noise approximation.
    for(const auto& entry:std::vector<std::pair<Float4,float>>{{{0,0,0,0},0},{{0.25f,0.5f,0.75f,0},0.208695697f},{{0.1f,1.2f,-2.3f,0},0.278225404f}}) {
      Program p;auto position=p.constant(entry.first),amplitude=p.constant({1,1,1,1}),pivot=p.constant({1,1,1,1});
      p.output(p.op(MaterialOp::Noise3D,position,amplitude,pivot));const float expected=1+entry.second;
      check(renderer,"MaterialX scalar Perlin reference",std::move(p),{expected,expected,expected,0});
    }
    for(const auto& entry:std::vector<std::pair<Float4,Float4>>{
      {{0.25f,0.5f,0.75f,0},{1.208695697f,0.414293737f,1.153054936f,0}},
      {{0.1f,1.2f,-2.3f,0},{1.278225404f,0.567239840f,1.037631331f,0}}}) {
      Program p;auto position=p.constant(entry.first),amplitude=p.constant({1,1,1,1}),pivot=p.constant({1,1,1,1});
      p.output(p.op(MaterialOp::Noise3D,position,amplitude,pivot,{3,0,0,0}));
      check(renderer,"MaterialX vector Perlin reference",std::move(p),entry.second);
    }
    image(renderer,"Texture top row / bottom UV origin",{0.25f,0.75f,0,0},5,{1,0,0,0});
    image(renderer,"Texture bottom row",{0.25f,0.25f,0,0},5,{0,0,1,0});
    image(renderer,"Texture bilinear center",{0.5f,0.5f,0,0},5,{0.5f,0.5f,0.5f,0});
    image(renderer,"Texture repeat",{1.25f,0.75f,0,0},0,{1,0,0,0});
    image(renderer,"Texture clamp",{1.25f,0.75f,0,0},5,{0,1,0,0});
    image(renderer,"Texture mirror",{1.25f,0.75f,0,0},10,{0,1,0,0});
    image(renderer,"Texture black outside",{1.25f,0.75f,0,0},15,{0,0,0,0});
    image(renderer,"Texture nearest",{0.49f,0.75f,0,0},5|16,{1,0,0,0});
    image(renderer,"Texture mip chain",{0.25f,0.75f,0,0},5,{0.5f,0.5f,0.5f,0},32);
    verifyOfficialPatterns(renderer);
    {
      Program p;auto uv=p.op(MaterialOp::Texcoord),scale=p.constant({1024,1024,0,0}),tiled=p.op(MaterialOp::Multiply,uv,scale);
      p.output(p.op(MaterialOp::Image,tiled,0,0));
      check(renderer,"UV scaling propagates ray-footprint mip LOD",std::move(p),{0.5f,0.5f,0.5f,0},{fixture()});
    }
    {
      Program p;p.constant({0.2f,0.4f,0.6f,0});p.constant({0,0,0,0});
      for(uint32_t i=2;i<127;++i)p.op(MaterialOp::Add,i==2?0:i-1,1);
      p.output(126);check(renderer,"128-instruction register limit",std::move(p),{0.2f,0.4f,0.6f,0});
    }
    verifyDefaultTangentFastPath(renderer);
    verifyMaterialTransport(renderer);
    if(renderer.validationErrorCount()!=0)throw std::runtime_error("Vulkan validation errors in material programs");
    std::cout<<"Material programs passed: analytic arithmetic, normalmap, Perlin, HSV/hex-tiling references and variation, texture filtering/mips, default-tangent fast/generic equivalence and guards, graph/textured area-light transport, primary/shadow opacity, sided emission, anisotropic tangent outputs and 128 registers.\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
