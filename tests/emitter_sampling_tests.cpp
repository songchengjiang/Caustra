#include <caustra/renderer.hpp>
#include "../src/shader_types.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace caustra;
void require(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}

void verifyCdf() {
  // A float running sum in this order loses all the dim lights. Exhaust the
  // exact ticket domain and check the selector's *actual* distribution, rather
  // than merely asserting that a nominal PDF is positive.
  std::vector<double> weights{1e8,1e-8,1e-8,2e-8,1e8,1e-8};
  std::vector<GpuLight> lights(weights.size());buildLightCdf(lights,weights);
  std::vector<uint32_t> selected(lights.size());size_t index=0;
  for(uint32_t ticket=0;ticket<kLightSelectionTickets;++ticket) {
    while(float(ticket)>=lights[index].edge1.w) ++index;
    ++selected[index];
  }
  uint32_t previous=0;
  for(size_t i=0;i<lights.size();++i) {
    const auto cumulative=static_cast<uint32_t>(lights[i].edge1.w);
    require(selected[i]>0,"A positive dim emitter was starved");
    require(selected[i]==cumulative-previous,"Sampled CDF and MIS interval mass disagree");
    previous=cumulative;
  }
  require(previous==kLightSelectionTickets,"Light CDF does not cover the full ticket domain");
  require(std::abs(double(selected[0])/selected[4]-1)<1e-6,"Equal bright weights received different probabilities");
  lights.resize(3);buildLightCdf(lights,{1,2,5});
  require(std::abs(lights[0].edge1.w/float(kLightSelectionTickets)-.125f)<2e-7f,"CDF ignored nonuniform light power");
  bool rejected=false;
  try {buildLightCdf(lights,{1,0,2});} catch(const std::invalid_argument&) {rejected=true;}
  require(rejected,"A nonpositive emitter proposal was accepted");
}

struct Patch {float x0,x1,y0,y1;Float4 radiance;};
constexpr std::array<Patch,2> patches{{{-1.5f,-.2f,-.5f,.5f,{80,40,20,0}},
                                      {.2f,1.5f,-.5f,.5f,{.2f,1,.4f,0}}}};

Mesh quad(float x0,float x1,float y0,float y1,float z,uint32_t material,bool down=false) {
  Mesh mesh;mesh.material=material;
  const Float4 normal{0,0,down?-1.0f:1.0f,0};
  mesh.vertices={{{x0,y0,z,0},normal,{0,0,0,0}},{{x1,y0,z,0},normal,{1,0,0,0}},
                 {{x1,y1,z,0},normal,{1,1,0,0}},{{x0,y1,z,0},normal,{0,1,0,0}}};
  mesh.indices=down?std::vector<uint32_t>{0,2,1,0,3,2}:std::vector<uint32_t>{0,1,2,0,2,3};
  return mesh;
}

Scene lightingScene(unsigned subdivisions,bool dynamic,bool reverse) {
  Scene scene;
  Material receiver;receiver.baseColor={.5f,.5f,.5f,1};receiver.specular.x=0;
  scene.materials.push_back(receiver);scene.meshes.push_back(quad(-20,20,-20,20,0,0));scene.instances.push_back({0});
  for(unsigned p=0;p<patches.size();++p) {
    const auto& patch=patches[reverse?patches.size()-1-p:p];
    Material emitter;emitter.baseColor={0,0,0,1};emitter.specular.x=0;emitter.surface.z=1;
    if(dynamic) {
      // Zero base emission must still enter the light table. The texture path
      // exercises per-light material context, not a camera-visible emissive hit.
      const auto texture=static_cast<uint32_t>(scene.textures.size());
      scene.textures.push_back({1,1,{patch.radiance}});
      emitter.program={static_cast<uint32_t>(scene.materialInstructions.size()),3,0,0};
      scene.materialInstructions.push_back({{uint32_t(MaterialOp::Texcoord),0,0,0},{}});
      scene.materialInstructions.push_back({{uint32_t(MaterialOp::Image),0,texture,0},{}});
      scene.materialInstructions.push_back({{uint32_t(MaterialOp::Output),1,uint32_t(MaterialParameter::Emission),0},{}});
    } else emitter.emission=patch.radiance;
    const auto material=static_cast<uint32_t>(scene.materials.size());scene.materials.push_back(emitter);
    for(unsigned i=0;i<subdivisions;++i) {
      const float x0=patch.x0+(patch.x1-patch.x0)*i/subdivisions;
      const float x1=patch.x0+(patch.x1-patch.x0)*(i+1)/subdivisions;
      scene.instances.push_back({static_cast<uint32_t>(scene.meshes.size())});
      scene.meshes.push_back(quad(x0,x1,patch.y0,patch.y1,2,material,true));
    }
  }
  return scene;
}

std::array<double,3> mean(const Image& image) {
  std::array<double,3> result{};
  for(const auto p:image.pixels) {
    require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),"Nonfinite weighted emitter output");
    result[0]+=p.x;result[1]+=p.y;result[2]+=p.z;
  }
  for(auto& value:result) value/=image.pixels.size();
  return result;
}

std::array<double,3> reference(uint32_t width,uint32_t height) {
  // Independent area quadrature of Lambertian irradiance at each pixel-center
  // receiver position. The primary camera is at z=1, lights at z=2, and emitted
  // radiance is constant on each rectangle. No renderer PDF enters this integral.
  constexpr unsigned steps=96;constexpr double pi=3.14159265358979323846;
  const double halfHeight=std::tan(15*pi/360),halfWidth=halfHeight*width/height;
  std::array<double,3> result{};
  for(uint32_t y=0;y<height;++y) for(uint32_t x=0;x<width;++x) {
    const double px=(2*(x+.5)/width-1)*halfWidth,py=(1-2*(y+.5)/height)*halfHeight;
    for(const auto& patch:patches) {
      double integral=0;
      const double dx=(patch.x1-patch.x0)/steps,dy=(patch.y1-patch.y0)/steps;
      for(unsigned iy=0;iy<steps;++iy) for(unsigned ix=0;ix<steps;++ix) {
        const double lx=patch.x0+(ix+.5)*dx-px,ly=patch.y0+(iy+.5)*dy-py;
        const double distanceSquared=lx*lx+ly*ly+4;
        integral+=4/(distanceSquared*distanceSquared)*dx*dy;
      }
      const std::array rgb{patch.radiance.x,patch.radiance.y,patch.radiance.z};
      for(size_t channel=0;channel<3;++channel) result[channel]+=.5/pi*integral*rgb[channel];
    }
  }
  for(auto& value:result) value/=double(width)*height;
  return result;
}
}

int main(int argc,char** argv) {
  try {
    verifyCdf();
    if(argc>1 && std::string(argv[1])=="--cpu") {
      std::cout<<"Emitter CPU CDF: nonuniform power, exact ticket probabilities, extreme bright/dim support passed.\n";
      return 0;
    }
    Renderer renderer({argc>1});
    RenderSettings settings;settings.width=16;settings.height=12;settings.maxBounces=1;settings.environment={};
    renderer.setSettings(settings);renderer.setCamera({{0,0,1,0},{0,0,0,0},{0,1,0,0},15});
    const auto expected=reference(settings.width,settings.height);
    for(const auto [subdivisions,dynamic,reverse]:std::array<std::array<unsigned,3>,4>{{{1,0,0},{7,0,0},{7,0,1},{5,1,1}}}) {
      renderer.setScene(lightingScene(subdivisions,dynamic!=0,reverse!=0));renderer.render(4096);
      const auto result=mean(renderer.readbackRaw());
      for(size_t channel=0;channel<3;++channel) {
        std::cout<<"Emitter split="<<subdivisions<<" dynamic="<<dynamic<<" reverse="<<reverse
                 <<" channel="<<channel<<" measured="<<result[channel]<<" quadrature="<<expected[channel]<<'\n';
        if(std::abs(result[channel]/expected[channel]-1)>.035)
          throw std::runtime_error("Power-weighted MIS disagrees with independent Lambertian light integral");
      }
    }
    require(renderer.validationErrorCount()==0,"Vulkan validation errors");
    std::cout<<"Emitter sampling: split/merged, reordered, constant/dynamic emission match independent quadrature.\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
