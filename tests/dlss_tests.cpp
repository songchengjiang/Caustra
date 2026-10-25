#include <caustra/renderer.hpp>
#include "demo_scene.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

void require(bool c,const char* message) { if(!c) throw std::runtime_error(message); }
double energy(const caustra::Image& image) {
  double total=0;
  require(image.pixels.size()==size_t(image.width)*image.height,"Incorrect DLSS output dimensions");
  for(auto p:image.pixels) {
    require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),"Nonfinite DLSS pixel");
    total+=p.x+p.y+p.z;
  }
  require(total>0,"DLSS returned black image"); return total;
}

namespace {
constexpr float reprojectionBackground=.05f,reprojectionForeground=1.0f;

caustra::Scene reprojectionScene() {
  caustra::Scene scene;
  for(float emission:{reprojectionBackground,reprojectionForeground}) {
    caustra::Material material;
    material.baseColor={0,0,0,1};material.factors.x=0;material.specular.x=0;
    material.emission={emission,emission,emission,0};
    scene.materials.push_back(material);
  }
  auto quad=[&](float left,float bottom,float right,float top,uint32_t material) {
    caustra::Mesh mesh;mesh.material=material;
    mesh.vertices={
      {{left,bottom,0,0},{0,0,1,0},{0,0,0,0},{1,0,0,1}},
      {{right,bottom,0,0},{0,0,1,0},{1,0,0,0},{1,0,0,1}},
      {{right,top,0,0},{0,0,1,0},{1,1,0,0},{1,0,0,1}},
      {{left,top,0,0},{0,0,1,0},{0,1,0,0},{1,0,0,1}}};
    mesh.indices={0,1,2,0,2,3};
    scene.instances.push_back({static_cast<uint32_t>(scene.meshes.size())});
    scene.meshes.push_back(std::move(mesh));
  };
  // A coplanar mosaic gives foreground/background identical depth, normal,
  // roughness and albedo. Zero scattering leaves a noise-free analytic EDF
  // image, so changed-region errors measure reconstruction artifacts directly.
  // The network can also recover this image from the current frame: this
  // fixture does NOT establish motion-vector sign or scaling correctness.
  quad(-100,-100,100,-.7f,0);quad(-100,.7f,100,100,0);
  quad(-100,-.7f,-1,.7f,0);quad(1,-.7f,100,.7f,0);
  quad(-1,-.7f,1,.7f,1);
  return scene;
}

struct ProjectedRectangle {
  double left,right,top,bottom;
  int classify(double x,double y,double margin) const {
    if(std::abs(x-left)<margin || std::abs(x-right)<margin ||
       std::abs(y-top)<margin || std::abs(y-bottom)<margin) return -1;
    return x>left && x<right && y>top && y<bottom?1:0;
  }
};

ProjectedRectangle reprojectionRectangle(const caustra::Camera& camera,uint32_t width,uint32_t height) {
  // The parallel camera looks straight down -Z; all mosaic vertices have Z=0.
  // Film aspect is the OUTPUT aspect, including when the raw RR input has a
  // slightly different integer-rounded aspect ratio.
  const double halfHeight=camera.position.z*std::tan(camera.verticalFov*3.14159265358979323846/360);
  const double sx=width/(2*halfHeight*(320.0/240.0)),sy=height/(2*halfHeight);
  return {width*.5+(-1-camera.position.x)*sx,width*.5+(1-camera.position.x)*sx,
          height*.5-(.7-camera.position.y)*sy,height*.5-(-.7-camera.position.y)*sy};
}

struct ReprojectionError {double mean=0,maximum=0;size_t pixels=0;};
ReprojectionError reprojectionError(const caustra::Image& image,const ProjectedRectangle& current,
                                   const ProjectedRectangle* previous=nullptr) {
  ReprojectionError error;
  for(uint32_t y=0;y<image.height;++y) for(uint32_t x=0;x<image.width;++x) {
    const int expected=current.classify(x+.5,y+.5,2);
    if(expected<0) continue;
    if(previous) {
      const int before=previous->classify(x+.5,y+.5,2);
      if(before<0 || before==expected) continue;
    }
    const auto pixel=image.pixels[size_t(y)*image.width+x];
    require(std::isfinite(pixel.x)&&std::isfinite(pixel.y)&&std::isfinite(pixel.z),
      "Nonfinite temporal reprojection output");
    const double value=(pixel.x-reprojectionBackground)/(reprojectionForeground-reprojectionBackground);
    const double difference=std::abs(value-expected);
    error.mean+=difference;error.maximum=std::max(error.maximum,difference);++error.pixels;
  }
  require(error.pixels>=64,"Temporal reprojection fixture did not cover enough changed pixels");
  error.mean/=error.pixels;
  return error;
}

void verifyTemporalMovingEdges(caustra::Renderer& renderer) {
  caustra::RenderSettings settings;settings.width=320;settings.height=240;
  settings.maxBounces=1;settings.environment={0,0,0,0};
  renderer.setSettings(settings);renderer.setScene(reprojectionScene());
  for(auto quality:{caustra::DlssQuality::Native,caustra::DlssQuality::Quality}) {
    caustra::DlssSettings dlss;dlss.enabled=true;dlss.quality=quality;dlss.preset=caustra::DlssPreset::E;
    renderer.setDlssSettings(dlss);
    caustra::Camera camera{{0,0,4,0},{0,0,0,0},{0,1,0,0},60};
    renderer.setCamera(camera);renderer.render(24);
    auto previous=reprojectionRectangle(camera,settings.width,settings.height);
    auto lastPrevious=previous;
    double movingMean=0,movingWorst=0;uint32_t frames=0;
    for(const auto delta:{std::array{.14f,0.f},std::array{0.f,.12f},std::array{-.14f,-.12f}}) {
      for(uint32_t step=0;step<8;++step) {
        camera.position.x+=delta[0];camera.target.x+=delta[0];
        camera.position.y+=delta[1];camera.target.y+=delta[1];
        renderer.updateCamera(camera);renderer.render();
        const auto current=reprojectionRectangle(camera,settings.width,settings.height);
        const auto error=reprojectionError(renderer.readback(),current,&previous);
        movingMean+=error.mean;movingWorst=std::max(movingWorst,error.mean);++frames;
        // Validate the analytic fixture independently of the temporal network,
        // at the actual native/upscaled input dimensions and current jitter.
        const auto raw=renderer.readbackRaw();
        const auto rawError=reprojectionError(raw,reprojectionRectangle(camera,raw.width,raw.height));
        require(rawError.maximum<1e-4,"Temporal reprojection reference must be a noise-free analytic image");
        lastPrevious=previous;previous=current;
      }
    }
    renderer.render(8);
    const auto stopped=reprojectionError(renderer.readback(),previous,&lastPrevious);
    movingMean/=frames;
    std::cout<<"Temporal EDF moving edges "<<(quality==caustra::DlssQuality::Native?"native":"quality")
      <<": changed-region mean="<<movingMean<<", worst-frame="<<movingWorst
      <<", stopped mean="<<stopped.mean<<'\n';
    // Errors are fractions of the known foreground/background radiance
    // contrast, not whole-image averages that would dilute a narrow trail.
    // The two-pixel exclusion permits reconstruction-filter edge support;
    // 5% mean / 10% worst-frame allow transient neural reconstruction while
    // detecting visible retained silhouettes in this controlled EDF fixture.
    // After eight stationary frames the last changed strips have a 2% budget.
    // This is an image-artifact regression, not a proof of MV correctness or
    // coverage of noisy reflections, refraction, SSS, or general DLSS ghosting.
    require(movingMean<.05,"DLSS moving EDF edges leave persistent trails in changed regions");
    require(movingWorst<.10,"DLSS moving EDF edges severely misplace the current silhouette");
    require(stopped.mean<.02,"DLSS EDF edge history failed to settle after camera motion stopped");
  }
}
}

int main() {
  try {
    caustra::Renderer renderer({true});
    caustra::RenderSettings render; render.width=320; render.height=240;
    renderer.setSettings(render); renderer.setScene(caustra::example::makeDemoScene()); renderer.setCamera(caustra::example::makeDemoCamera());
    caustra::DlssSettings dlss; dlss.enabled=true;
    renderer.setDlssSettings(dlss); std::cout<<renderer.dlssStatus()<<'\n';
    renderer.render(8); auto native=renderer.readback(); auto raw=renderer.readbackRaw();
    energy(native); energy(raw);
    require(native.width==320 && native.height==240 && native.samples==8,"Native RR dimensions or frame counter");
    double delta=0; for(size_t i=0;i<raw.pixels.size();++i) delta+=std::abs(native.pixels[i].x-raw.pixels[i].x);
    require(delta>0.001,"DLSS output is identical to raw input");
    auto orbitCamera=caustra::example::makeDemoCamera();
    for(uint32_t frame=0;frame<8;++frame) {
      orbitCamera.position.x+=0.04f;
      renderer.updateCamera(orbitCamera);
      require(renderer.sampleCount()==0 && renderer.frameCount()==8+frame,"Moving RR camera reset temporal history or kept stale raw accumulation");
      renderer.render(1); energy(renderer.readback()); energy(renderer.readbackRaw());
      require(renderer.frameCount()==9+frame,"RR temporal frame sequence did not advance during motion");
    }
    renderer.updateCamera(orbitCamera); renderer.render(2);
    require(renderer.sampleCount()==3 && renderer.frameCount()==18,"Stationary camera must resume progressive accumulation without losing RR history");
    renderer.reset(); require(renderer.frameCount()==0,"Explicit reset must discard RR history");
    renderer.render(2); energy(renderer.readback()); require(renderer.sampleCount()==2,"RR reset failed");
    dlss.quality=caustra::DlssQuality::Quality; dlss.preset=caustra::DlssPreset::E;
    renderer.setDlssSettings(dlss); renderer.render(4); auto quality=renderer.readback(); auto inner=renderer.readbackRaw();
    energy(quality); energy(inner); require(quality.width==320 && quality.height==240 && inner.width<320,"RR quality upscaling dimensions");
    for(auto mode:{caustra::DlssQuality::Balanced,caustra::DlssQuality::Performance}) {
      dlss.quality=mode; dlss.preset=caustra::DlssPreset::D;
      renderer.setDlssSettings(dlss); renderer.render(2); energy(renderer.readback());
      require(renderer.readbackRaw().width<320,"RR balanced/performance must use lower internal resolution");
    }
    const auto beforeFailure=renderer.sampleCount();
    auto invalid=dlss; invalid.preset=static_cast<caustra::DlssPreset>(99);
    bool rejected=false; try { renderer.setDlssSettings(invalid); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected && renderer.sampleCount()==beforeFailure,"Rejected RR config must preserve current state");
    auto tooSmall=render; tooSmall.width=16;
    rejected=false; try { renderer.setSettings(tooSmall); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected && renderer.sampleCount()==beforeFailure,"Rejected RR dimensions must preserve current state");
    auto camera=caustra::example::makeDemoCamera(); camera.position.x+=0.2f; renderer.setCamera(camera);
    require(renderer.frameCount()==0,"Camera cut must discard RR history"); renderer.render(2); energy(renderer.readback());
    dlss.quality=caustra::DlssQuality::Quality; dlss.preset=caustra::DlssPreset::E;
    renderer.setDlssSettings(dlss); renderer.render(2);
    auto previousRaw=renderer.readbackRaw();
    uint32_t previousWidth=render.width,previousHeight=render.height;
    // Rebuild both directions without changing quality: a stale output image or
    // internal guide buffer must not survive a viewport resolution change.
    for(const auto& dimensions:{std::pair{512u,384u},std::pair{256u,160u},std::pair{384u,256u}}) {
      render.width=dimensions.first; render.height=dimensions.second;
      renderer.setSettings(render);
      require(renderer.sampleCount()==0 && renderer.frameCount()==0,"RR resize must reset accumulation and temporal history");
      renderer.render(2);
      auto resized=renderer.readback(); auto resizedRaw=renderer.readbackRaw();
      energy(resized); energy(resizedRaw);
      require(resized.width==render.width && resized.height==render.height && resized.samples==2,"RR resized output dimensions or sample counter");
      require(resizedRaw.width>0 && resizedRaw.height>0 && resizedRaw.width<render.width && resizedRaw.height<render.height && resizedRaw.samples==2,"RR resized internal dimensions or sample counter");
      require((render.width>previousWidth ? resizedRaw.width>previousRaw.width : resizedRaw.width<previousRaw.width) &&
              (render.height>previousHeight ? resizedRaw.height>previousRaw.height : resizedRaw.height<previousRaw.height),"RR internal resolution did not follow output resize");
      previousWidth=render.width; previousHeight=render.height; previousRaw=std::move(resizedRaw);
    }
    dlss.enabled=false; renderer.setDlssSettings(dlss); renderer.render(2); auto disabled=renderer.readback(); auto disabledRaw=renderer.readbackRaw();
    require(disabled.pixels[123].x==disabledRaw.pixels[123].x,"Disabled RR must return raw output");
    for(auto mode:{caustra::DlssQuality::Native,caustra::DlssQuality::Quality,caustra::DlssQuality::Performance}) {
      dlss.enabled=true; dlss.quality=mode;
      renderer.setDlssSettings(dlss);
      require(renderer.dlssSettings().enabled && renderer.sampleCount()==0 && renderer.frameCount()==0,"Re-enabling RR must activate it and reset history");
      renderer.render(3); auto restored=renderer.readback(); energy(restored);
      require(restored.width==render.width && restored.height==render.height,"Re-enabled RR output dimensions");
      dlss.enabled=false; renderer.setDlssSettings(dlss); renderer.render(2);
      energy(renderer.readback());
      // Settings changes while disabled must not drop the retained NGX session.
      render.width+=16; renderer.setSettings(render); renderer.render(1); energy(renderer.readback());
    }
    verifyTemporalMovingEdges(renderer);
    require(renderer.validationErrorCount()==0,"Vulkan validation errors in DLSS integration");
    std::cout<<"DLSS RR passed: native, quality/balanced/performance upscale, Default/D/E presets, moving-camera history, quantitative analytic EDF moving edges, reset, camera cut, output/internal resize up and down, invalid config and repeated off/on switching.\n";
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
