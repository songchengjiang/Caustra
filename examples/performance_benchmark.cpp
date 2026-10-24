#include <caustra/renderer.hpp>
#include "demo_scene.hpp"
#include "orbit_camera.hpp"
#if CAUSTRA_HAS_ASSETS
#include <caustra/asset.hpp>
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
struct Options {
  caustra::RenderSettings settings;
  uint32_t frames=30,warmup=5;
  std::string mode="off";
  std::filesystem::path asset;
  bool moving=true,validation=false,help=false;
};
uint32_t number(std::string_view value,bool zero=false) {
  uint32_t n=0;const auto parsed=std::from_chars(value.data(),value.data()+value.size(),n);
  if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size() || (!zero && !n))
    throw std::invalid_argument("Expected a positive integer (warmup also accepts zero)");
  return n;
}
Options arguments(int argc,char** argv) {
  Options o;o.settings.environment={.025f,.03f,.04f,0};
  for(int i=1;i<argc;++i) {
    const std::string_view arg=argv[i];
    if(arg=="--help" || arg=="-h") {o.help=true;continue;}
    if(arg=="--static") {o.moving=false;continue;}
    if(arg=="--validation") {o.validation=true;continue;}
    if(arg!="--asset" && arg!="--width" && arg!="--height" && arg!="--bounces" && arg!="--frames" && arg!="--warmup" && arg!="--dlss")
      throw std::invalid_argument("Unknown benchmark option: "+std::string(arg));
    if(++i==argc) throw std::invalid_argument("Missing option value");
    const std::string_view value=argv[i];
    if(arg=="--asset") o.asset=argv[i];
    else if(arg=="--dlss") o.mode=value;
    else if(arg=="--width") o.settings.width=number(value);
    else if(arg=="--height") o.settings.height=number(value);
    else if(arg=="--bounces") o.settings.maxBounces=number(value);
    else if(arg=="--frames") o.frames=number(value);
    else o.warmup=number(value,true);
  }
  if(o.frames>100000 || o.warmup>100000) throw std::invalid_argument("Frames and warmup must not exceed 100000");
  if(o.mode!="off" && o.mode!="native" && o.mode!="quality" && o.mode!="balanced" && o.mode!="performance" && o.mode!="all")
    throw std::invalid_argument("DLSS mode: off/native/quality/balanced/performance/all");
  caustra::validateSettings(o.settings);return o;
}
double milliseconds(Clock::time_point a,Clock::time_point b) {
  return std::chrono::duration<double,std::milli>(b-a).count();
}
double quantile(std::vector<double> values,double fraction) {
  std::sort(values.begin(),values.end());
  const double index=fraction*(values.size()-1);const auto lower=size_t(index);
  return values[lower]+(values[std::min(lower+1,values.size()-1)]-values[lower])*(index-lower);
}
void toneMap(const caustra::Image& image,std::vector<uint32_t>& display) {
  static const auto lut=[] {
    std::array<uint8_t,8192> values{};
    for(size_t i=0;i<values.size();++i) {
      const double x=double(i)/(values.size()-1);
      values[i]=uint8_t(std::lround(255*(x<=.0031308?12.92*x:1.055*std::pow(x,1/2.4)-.055)));
    }
    return values;
  }();
  auto channel=[&](float value) {
    value=std::max(0.f,value);const float mapped=value/(1+value);
    return uint32_t(lut[std::min(size_t(8191),size_t(mapped*8191+.5f))]);
  };
  display.resize(image.pixels.size());
  for(size_t i=0;i<display.size();++i) {
    const auto p=image.pixels[i];
    if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) throw std::runtime_error("Nonfinite benchmark output");
    display[i]=(channel(p.x)<<16)|(channel(p.y)<<8)|channel(p.z);
  }
}
}

int main(int argc,char** argv) {
  try {
    const auto options=arguments(argc,argv);
    if(options.help) {
      std::cout<<"Caustra interaction benchmark (1 spp per frame; moving camera by default)\n"
        "--asset FILE --width N --height N --bounces N --frames N --warmup N\n"
        "--dlss off/native/quality/balanced/performance/all --static --validation\n"
        "CSV reports CPU wall time around render, HDR readback and viewer-equivalent tone mapping.\n"
        "Render includes GPU completion and optional RR; window painting is excluded.\n";
      return 0;
    }
    auto scene=caustra::example::makeDemoScene();auto camera=caustra::example::makeDemoCamera();
    if(!options.asset.empty()) {
#if CAUSTRA_HAS_ASSETS
      auto asset=caustra::loadAsset(options.asset);scene=std::move(asset.scene);camera=asset.camera;
#else
      throw std::invalid_argument("Build with CAUSTRA_BUILD_ASSETS=ON to benchmark a compiled asset");
#endif
    }
    std::cout<<std::unitbuf<<std::fixed<<std::setprecision(3);
    const auto start=Clock::now();caustra::Renderer renderer({options.validation});const auto initialized=Clock::now();
    renderer.setSettings(options.settings);renderer.setScene(scene);renderer.setCamera(camera);
    std::cerr<<"Device: "<<renderer.deviceName()<<"; init "<<milliseconds(start,initialized)
      <<" ms; upload "<<milliseconds(initialized,Clock::now())<<" ms; "<<(options.moving?"orbit":"static")<<"; 1 spp/frame\n";
    std::vector<std::string> modes=options.mode=="all"?std::vector<std::string>{"off","quality","performance"}:std::vector<std::string>{options.mode};
    std::cout<<"mode,width,height,bounces,frames,render_ms,readback_ms,tonemap_ms,total_ms,p95_ms,equivalent_fps,checksum\n";
    for(const auto& mode:modes) {
      caustra::DlssSettings dlss;dlss.enabled=mode!="off";dlss.preset=caustra::DlssPreset::E;
      dlss.quality=mode=="quality"?caustra::DlssQuality::Quality:mode=="balanced"?caustra::DlssQuality::Balanced:
        mode=="performance"?caustra::DlssQuality::Performance:caustra::DlssQuality::Native;
      renderer.setDlssSettings(dlss);
      caustra::example::OrbitCamera orbit;orbit.setHome(camera);renderer.setCamera(orbit.camera());
      std::vector<double> render,readback,tone,total;std::vector<uint32_t> display;uint64_t checksum=0;
      for(uint32_t frame=0;frame<options.warmup+options.frames;++frame) {
        if(options.moving) {orbit.orbit(1.5f,.2f);renderer.updateCamera(orbit.camera());}
        const auto a=Clock::now();renderer.render(1);const auto b=Clock::now();
        const auto image=renderer.readback();const auto c=Clock::now();toneMap(image,display);const auto d=Clock::now();
        for(auto pixel:display) checksum+=pixel;
        if(frame>=options.warmup) {
          render.push_back(milliseconds(a,b));readback.push_back(milliseconds(b,c));
          tone.push_back(milliseconds(c,d));total.push_back(milliseconds(a,d));
        }
      }
      if(renderer.validationErrorCount()) throw std::runtime_error("Vulkan validation errors during benchmark");
      const auto median=quantile(total,.5);
      std::cout<<mode<<','<<options.settings.width<<','<<options.settings.height<<','<<options.settings.maxBounces<<','<<options.frames<<','
        <<quantile(render,.5)<<','<<quantile(readback,.5)<<','<<quantile(tone,.5)<<','<<median<<','<<quantile(total,.95)<<','<<1000/median<<','<<checksum<<'\n';
    }
    return 0;
  } catch(const std::exception& error) {std::cerr<<"Caustra benchmark: "<<error.what()<<'\n';return 1;}
}
