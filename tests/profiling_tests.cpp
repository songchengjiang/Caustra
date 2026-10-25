#include <caustra/renderer.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {
using namespace caustra;
void require(bool condition,const char* message) {
  if(!condition) throw std::runtime_error(message);
}
void quad(Scene& scene,uint32_t material,float z) {
  Mesh mesh;mesh.material=material;
  for(const auto p:std::array<Float4,4>{{{-10,-10,z,0},{10,-10,z,0},{10,10,z,0},{-10,10,z,0}}})
    mesh.vertices.push_back({p,{0,0,1,0},{0,0,0,0},{1,0,0,1}});
  mesh.indices={0,1,2,0,2,3};
  scene.instances.push_back({uint32_t(scene.meshes.size())});
  scene.meshes.push_back(std::move(mesh));
}
Scene emitter(bool transparentLayers) {
  Scene scene;
  Material light;light.baseColor={0,0,0,1};light.factors.x=0;light.specular.x=0;
  light.emission={3,2,1,0};scene.materials.push_back(light);
  quad(scene,0,0);
  if(transparentLayers) {
    Material clear;clear.factors.z=0;scene.materials.push_back(clear);
    quad(scene,1,.4f);quad(scene,1,.8f);
  }
  return scene;
}
void identical(const Image& reference,const Image& actual) {
  require(reference.width==actual.width && reference.height==actual.height &&
          reference.samples==actual.samples,"Profiling changed output dimensions/sample count");
  require(reference.pixels.size()==actual.pixels.size(),"Profiling changed pixel count");
  for(size_t i=0;i<actual.pixels.size();++i) {
    const auto a=reference.pixels[i],b=actual.pixels[i];
    require(a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w,"Profiling changed deterministic raw pixels");
    require(std::isfinite(b.x) && b.x>0 && std::isfinite(b.y) && std::isfinite(b.z),"Invalid analytic emitter output");
  }
}
void checkTiming(const RendererProfile& profile,bool expectedDlss) {
  if(!profile.gpuTimingSupported) {
    require(!profile.gpuTimingValid,"Unsupported timestamps exposed valid GPU timing");
    return;
  }
  require(profile.gpuTimingValid,"GPU timestamp profile unavailable on supported device");
  require(std::isfinite(profile.totalGpuMs) && profile.totalGpuMs>0,"Invalid total GPU time");
  require(std::isfinite(profile.pathTraceGpuMs) && profile.pathTraceGpuMs>0,"Invalid path trace GPU time");
  require(std::isfinite(profile.initializationGpuMs) && profile.initializationGpuMs>=0,"Invalid initialization GPU time");
  require(std::isfinite(profile.dlssGpuMs) && profile.dlssGpuMs>=0,"Invalid DLSS GPU time");
  if(expectedDlss) require(profile.dlssGpuMs>0,"Missing DLSS GPU interval");
  else require(profile.dlssGpuMs==0,"Disabled DLSS recorded GPU work");
  require(std::abs(profile.initializationGpuMs+profile.pathTraceGpuMs+profile.dlssGpuMs-profile.totalGpuMs)<1e-6,
          "GPU timestamp regions do not sum to the total");
}
void checkRays(const RendererProfile& p,uint32_t samples,bool layers) {
  require(p.rayStatisticsValid && !p.rays.overflowed,"Missing or overflowed ray statistics");
  const uint64_t paths=uint64_t(p.renderWidth)*p.renderHeight*samples;
  require(p.samples==samples && p.rays.pathSamples==paths,"Statistics did not aggregate the requested sample batch");
  const uint64_t queries=paths*(layers?3:1);
  require(p.rays.primaryRayQueries==queries,"Unexpected primary query count for analytic layers");
  require(p.rays.indirectRayQueries==0 && p.rays.shadowRayQueries==0 && p.rays.auxiliaryRayQueries==0,
          "Unexpected indirect/shadow/auxiliary queries for terminal emitter");
  require(p.rays.queryHits==queries && p.rays.queryMisses==0,"Query outcomes do not match analytic geometry");
  require(p.rays.alphaRejectedHits==paths*(layers?2:0),"Incorrect alpha rejection count");
  require(p.rays.acceptedSurfaceHits==paths,"Incorrect accepted surface count");
  require(p.rays.coverageEvaluations==queries && p.rays.fullMaterialEvaluations==paths,
          "Incorrect material evaluation count for analytic layers");
  require(p.rays.candidateVisits==0 && p.rays.boundaryContinuationQueries==0 &&
          p.rays.shadowTransparentHits==0 && p.rays.surfaceScatters==0 && p.rays.mediumScatters==0 &&
          p.rays.russianRouletteTerminations==0 && p.rays.maxScatteringDepth==0,
          "Unexpected candidate, boundary or scattering work for terminal emitter");
}
}

int main(int argc,char** argv) {
  try {
    bool validation=false,dlss=false;
    for(int i=1;i<argc;++i) {
      const std::string_view arg=argv[i];
      if(arg=="--validation") validation=true;
      else if(arg=="--dlss") dlss=true;
      else throw std::invalid_argument("Unknown profiling test option");
    }
    RenderSettings settings;settings.width=63;settings.height=47;settings.maxBounces=1;settings.environment={0,0,0,0};
    const Camera camera{{0,0,2,0},{0,0,0,0},{0,1,0,0},45};
    Image reference;
    {
      RendererOptions options;options.validation=validation;
      Renderer renderer(options);renderer.setSettings(settings);renderer.setScene(emitter(true));renderer.setCamera(camera);
      renderer.render(2);reference=renderer.readbackRaw();
      require(!renderer.profile().gpuTimingValid && !renderer.profile().rayStatisticsValid,
              "Disabled profiling exposed valid measurements");
      require(renderer.validationErrorCount()==0,"Validation errors in default renderer");
    }
    {
      RendererOptions options;options.validation=validation;options.enableGpuTiming=true;
      Renderer renderer(options);renderer.setSettings(settings);renderer.setScene(emitter(true));renderer.setCamera(camera);
      renderer.render(2);checkTiming(renderer.profile(),false);
      require(!renderer.profile().rayStatisticsValid,"Timing-only mode instrumented ray counters");
      identical(reference,renderer.readbackRaw());
      require(renderer.validationErrorCount()==0,"Validation errors in timestamp-only renderer");
    }
    {
      RendererOptions options;options.validation=validation;options.enableGpuTiming=true;options.enableRayStatistics=true;
      Renderer renderer(options);renderer.setSettings(settings);renderer.setScene(emitter(true));renderer.setCamera(camera);
      renderer.render(2);checkTiming(renderer.profile(),false);checkRays(renderer.profile(),2,true);
      identical(reference,renderer.readbackRaw());
      checkRays(renderer.profile(),2,true); // Explicit readback must not invalidate completed measurements.
      renderer.render(1);checkRays(renderer.profile(),1,true);
      renderer.reset();
      require(!renderer.profile().gpuTimingValid && !renderer.profile().rayStatisticsValid,"Reset retained stale profile");
      renderer.setScene(emitter(false));renderer.render(1);checkRays(renderer.profile(),1,false);
      if(dlss) {
        DlssSettings denoising;denoising.enabled=true;denoising.quality=DlssQuality::Quality;
        renderer.setDlssSettings(denoising);renderer.render(2);
        checkTiming(renderer.profile(),true);checkRays(renderer.profile(),2,false);
        require(renderer.profile().renderWidth<settings.width && renderer.profile().renderHeight<settings.height,
                "DLSS statistics did not use the internal trace resolution");
        denoising.enabled=false;renderer.setDlssSettings(denoising);renderer.render(1);
        checkTiming(renderer.profile(),false);checkRays(renderer.profile(),1,false);
      }
      require(renderer.validationErrorCount()==0,"Validation errors in ray-statistics renderer");
    }
    std::cout<<"GPU profiling: timestamps, exact ray counts, batch semantics and deterministic pixels passed\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"Profiling tests: "<<error.what()<<'\n';return 1;
  }
}
