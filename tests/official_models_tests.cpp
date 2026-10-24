#include <caustra/renderer.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace caustra;
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
Scene plane(Material material) {
  Scene scene;scene.materials.push_back(material);
  Mesh mesh;mesh.vertices={
    {{-100,-100,0,0},{0,0,1,0},{0,0,0,0},{1,0,0,1}},
    {{100,-100,0,0},{0,0,1,0},{1,0,0,0},{1,0,0,1}},
    {{100,100,0,0},{0,0,1,0},{1,1,0,0},{1,0,0,1}},
    {{-100,100,0,0},{0,0,1,0},{0,1,0,0},{1,0,0,1}}};
  mesh.indices={0,1,2,0,2,3};scene.meshes.push_back(std::move(mesh));scene.instances.push_back({0});return scene;
}
Material native(MaterialModel model) {
  Material m;m.program.z=uint32_t(model);m.baseColor={0,0,0,1};m.surface={0,.02f,0,0};
  m.specular={1,1.5f,0,0};m.factors={1,0,1,0};m.sheenColor={0,0,0,0};
  m.fuzz={0,.5f,0,0};return m;
}
std::array<double,3> render(Renderer& renderer,const Scene& scene,uint32_t samples=128) {
  renderer.setScene(scene);renderer.render(samples);auto image=renderer.readbackRaw();
  std::array<double,3> mean{};
  for(auto p:image.pixels) {
    require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&p.x>=0&&p.y>=0&&p.z>=0,
      "Native MaterialX model produced nonfinite or negative radiance");
    mean[0]+=p.x;mean[1]+=p.y;mean[2]+=p.z;
  }
  for(auto& v:mean)v/=image.pixels.size();return mean;
}
void near(double actual,double expected,double tolerance,const char* message) {
  if(std::abs(actual-expected)>tolerance) {
    std::cerr<<message<<": actual "<<actual<<", expected "<<expected<<'\n';
    throw std::runtime_error(message);
  }
}
}
int main(int argc,char**) {
  try {
    Renderer renderer({argc>1});RenderSettings settings;
    settings.width=24;settings.height=24;settings.maxBounces=1;settings.environment={1,1,1,0};renderer.setSettings(settings);
    renderer.setCamera({{0,0,4,0},{0,0,0,0},{0,1,0,0},.1f});

    // Near-normal, nearly specular furnace references follow directly from each
    // OFFICIAL nodegraph, not a reference render by the shader under test.
    // Disney's generalized Schlick color0 is ONE and its layer WEIGHT is
    // specular*0.08. It must therefore differ from glTF's IOR-F0*specular.
    auto disney=native(MaterialModel::DisneyPrincipled);disney.specular.x=.5f;
    auto disneyFurnace=render(renderer,plane(disney));
    near(disneyFurnace[0],.04,.001,"Disney native specular layer weight is not specular*0.08");
    disney.coat={.7f,1,1.5f,0};
    auto disneyCoat=render(renderer,plane(disney));
    near(disneyCoat[0],.028+.04*(1-.028),.0015,"Disney clearcoat does not use clearcoat*0.04 and native layering");

    auto gltf=native(MaterialModel::GltfPbr);gltf.specular.x=.5f;
    auto gltfFurnace=render(renderer,plane(gltf));
    near(gltfFurnace[0],.02,.001,"glTF specular extension does not scale IOR-derived F0");
    require(disneyFurnace[0]>gltfFurnace[0]*1.8,"Disney and glTF were collapsed into the same surface model");

    auto usd=native(MaterialModel::UsdPreviewSurface);usd.factors.w=1;usd.specularColor={.2f,.4f,.6f,0};
    auto specularWorkflow=render(renderer,plane(usd));
    for(size_t c=0;c<3;++c)near(specularWorkflow[c],.2*(c+1),.002,"USD native specular workflow ignores its color F0");
    usd.factors.w=0;
    auto metalnessWorkflow=render(renderer,plane(usd));
    for(auto v:metalnessWorkflow)near(v,.04,.001,"USD metalness workflow failed to use dielectric IOR F0");

    // glTF Alpha OPAQUE ignores raw alpha, MASK thresholds it, and BLEND is
    // stochastic presence. Emission isolates alpha from BSDF/refraction effects.
    settings.environment={0,0,0,0};renderer.setSettings(settings);
    auto alpha=native(MaterialModel::GltfPbr);alpha.specular.x=0;alpha.emission={2,2,2,0};
    alpha.factors.z=.25f;alpha.volume.w=.5f;alpha.fuzz.z=0;
    near(render(renderer,plane(alpha),8)[0],2,1e-5,"glTF OPAQUE incorrectly uses alpha as presence");
    alpha.fuzz.z=1;
    near(render(renderer,plane(alpha),8)[0],0,1e-6,"glTF MASK fails to reject alpha below cutoff");
    alpha.factors.z=.75f;
    near(render(renderer,plane(alpha),8)[0],2,1e-5,"glTF MASK fails to accept alpha above cutoff");
    alpha.fuzz.z=2;alpha.factors.z=.25f;
    near(render(renderer,plane(alpha),128)[0],.5,.018,"glTF BLEND does not preserve fractional alpha");

    // USD transparent opacity is NOT the same as presence opacity. At zero
    // threshold its native surface remains present and opacity drives its BTDF.
    auto usdAlpha=native(MaterialModel::UsdPreviewSurface);usdAlpha.emission={2,2,2,0};usdAlpha.factors.z=.25f;
    usdAlpha.fuzz.w=1;
    near(render(renderer,plane(usdAlpha),128)[0],.5,.018,"USD presence opacity is not stochastic coverage");
    usdAlpha.fuzz.w=0;usdAlpha.transmission.x=.75f;usdAlpha.subsurface.w=1;
    near(render(renderer,plane(usdAlpha),8)[0],2,1e-5,"USD transparent opacity was incorrectly treated as coverage");

    // Native Chiang hair is a fiber scattering closure, not a surface PBR tint.
    // The all-zero R/TT/TRT case is exactly black; high melanin attenuates
    // transmitted fiber lobes while leaving the dielectric R contribution.
    settings.environment={1,1,1,0};renderer.setSettings(settings);
    auto hair=native(MaterialModel::SimpleHair);hair.baseColor={1,1,1,1};hair.surface.y=.3f;
    hair.specular={1,1.55f,.3f,0};hair.coat.y=.5f;hair.factors={.1f,.5f,1,1};
    hair.specularColor={1,1,1,0};hair.transmissionColor={1,1,1,0};hair.sheenColor={1,1,1,0};
    auto pale=render(renderer,plane(hair),512);
    hair.factors.x=.95f;auto dark=render(renderer,plane(hair),512);
    require(pale[0]>dark[0]*1.2&&dark[0]>0,"Chiang melanin absorption has no fiber-scattering effect");
    hair.specularColor={};hair.transmissionColor={};hair.sheenColor={};
    auto blackHair=render(renderer,plane(hair),16);
    for(auto v:blackHair)near(v,0,1e-7,"Chiang zero R/TT/TRT tint still scatters energy");
    require(renderer.validationErrorCount()==0,"Vulkan validation error in native official MaterialX models");
    std::cout<<"Native MaterialX graph references passed: Disney "<<disneyFurnace[0]<<", coat "<<disneyCoat[0]
      <<", glTF "<<gltfFurnace[0]<<", USD specular "<<specularWorkflow[0]<<','<<specularWorkflow[1]<<','<<specularWorkflow[2]
      <<", hair melanin "<<pale[0]<<" -> "<<dark[0]<<'\n';return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
