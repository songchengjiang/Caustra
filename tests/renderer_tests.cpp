#include <caustra/renderer.hpp>
#include "demo_scene.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
void verify(const caustra::Image& image) {
  double energy=0;
  for(auto p:image.pixels) { require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),"Nonfinite radiance"); require(p.x>=0 && p.y>=0 && p.z>=0,"Negative radiance"); energy+=p.x+p.y+p.z; }
  require(energy>0,"Black output");
}
int main(int argc,char**) {
  try {
    caustra::Renderer renderer({argc>1});
    std::cout<<"GPU: "<<renderer.deviceName()<<'\n';
    caustra::RenderSettings settings; settings.width=48; settings.height=37; settings.maxBounces=5;
    renderer.setSettings(settings);
    auto scene=caustra::example::makeDemoScene(); renderer.setScene(scene); renderer.setCamera(caustra::example::makeDemoCamera());
    renderer.render(4); auto reference=renderer.readback(); verify(reference);
    renderer.reset(); renderer.render(1); renderer.render(3); auto split=renderer.readback();
    require(split.samples==4,"Progressive sample counter");
    for(size_t i=0;i<split.pixels.size();++i) require(split.pixels[i].x==reference.pixels[i].x && split.pixels[i].y==reference.pixels[i].y && split.pixels[i].z==reference.pixels[i].z,"Split batches changed deterministic sequence");
    auto camera=caustra::example::makeDemoCamera(); camera.position.x+=0.3f; renderer.setCamera(camera);
    require(renderer.sampleCount()==0,"Camera failed to reset accumulation"); renderer.render(2); verify(renderer.readback());
    settings.width=31; settings.height=25; renderer.setSettings(settings);
    require(renderer.sampleCount()==0,"Resize failed to reset accumulation"); renderer.render(2);
    auto resized=renderer.readback(); require(resized.pixels.size()==31*25,"Resize image dimensions"); verify(resized);
    // No emissive light: tests empty-light dummy descriptor and BSDF-only environment.
    for(auto& material:scene.materials) material.emission={};
    // Nonuniform transformed geometry exercises world-to-object normal handling.
    scene.instances.back().transform[0]*=1.4f; scene.instances.back().transform[5]*=0.8f;
    renderer.setScene(scene); require(renderer.sampleCount()==0,"Scene replacement failed to reset accumulation"); renderer.render(4); verify(renderer.readback());
    // A failed upload must preserve the previously working scene.
    bool rejected=false; try { renderer.setScene({}); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"Empty scene accepted"); renderer.render(1); verify(renderer.readback());
    // Analytic reference: a camera-filling emitter in a black environment must
    // return its exact radiance, independent of sample count and BSDF sampling.
    caustra::Scene emitter;
    caustra::Material material; material.baseColor={0,0,0,1}; material.emission={2,0.5f,1,0};
    emitter.materials.push_back(material);
    caustra::Mesh plane;
    plane.vertices={{{-10,-10,0,0},{0,0,1,0}},{{10,-10,0,0},{0,0,1,0}},{{10,10,0,0},{0,0,1,0}},{{-10,10,0,0},{0,0,1,0}}};
    plane.indices={0,1,2,0,2,3}; emitter.meshes.push_back(plane); emitter.instances.push_back({0});
    settings.environment={}; settings.maxBounces=1; renderer.setSettings(settings); renderer.setScene(emitter);
    renderer.setCamera({{0,0,2,0},{0,0,0,0},{0,1,0,0},45}); renderer.render(3);
    for(auto p:renderer.readback().pixels) require(std::abs(p.x-2)<1e-5f && std::abs(p.y-0.5f)<1e-5f && std::abs(p.z-1)<1e-5f,"Emitter radiance disagrees with analytic reference");
    require(renderer.validationErrorCount()==0,"Vulkan validation errors");
    std::cout<<"GPU integration passed: analytic emitter, finite radiance, progressive determinism, camera/scene reset, resize, nonuniform scale, no lights, rejected upload.\n";
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
