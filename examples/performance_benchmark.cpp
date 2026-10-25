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
  std::vector<std::string> variants{"original"};
  std::vector<uint32_t> depths;
  std::filesystem::path asset;
  bool moving=true,validation=false,help=false,skipInvalidMedia=false,renderOnly=false;
};
uint32_t number(std::string_view value,bool zero=false) {
  uint32_t n=0;const auto parsed=std::from_chars(value.data(),value.data()+value.size(),n);
  if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size() || (!zero && !n))
    throw std::invalid_argument("Expected a positive integer (warmup also accepts zero)");
  return n;
}
std::vector<std::string> list(std::string_view value) {
  std::vector<std::string> result;
  for(size_t begin=0;begin<value.size();) {
    const auto end=value.find(',',begin);
    const auto item=value.substr(begin,end==std::string_view::npos?value.size()-begin:end-begin);
    if(item.empty()) throw std::invalid_argument("List entries must not be empty");
    result.emplace_back(item);
    if(end==std::string_view::npos) return result;
    begin=end+1;
    if(begin==value.size()) throw std::invalid_argument("List entries must not be empty");
  }
  if(result.empty()) throw std::invalid_argument("List must not be empty");
  return result;
}
Options arguments(int argc,char** argv) {
  Options o;o.settings.environment={.025f,.03f,.04f,0};
  for(int i=1;i<argc;++i) {
    const std::string_view arg=argv[i];
    if(arg=="--help" || arg=="-h") {o.help=true;continue;}
    if(arg=="--static") {o.moving=false;continue;}
    if(arg=="--validation") {o.validation=true;continue;}
    if(arg=="--skip-invalid-media") {o.skipInvalidMedia=true;continue;}
    if(arg=="--render-only") {o.renderOnly=true;continue;}
    if(arg!="--asset" && arg!="--width" && arg!="--height" && arg!="--bounces" && arg!="--frames" && arg!="--warmup" && arg!="--dlss" && arg!="--variants" && arg!="--depths")
      throw std::invalid_argument("Unknown benchmark option: "+std::string(arg));
    if(++i==argc) throw std::invalid_argument("Missing option value");
    const std::string_view value=argv[i];
    if(arg=="--asset") o.asset=argv[i];
    else if(arg=="--dlss") o.mode=value;
    else if(arg=="--variants") o.variants=value=="all"?std::vector<std::string>{"original","no-graphs","opaque"}:list(value);
    else if(arg=="--depths") {for(const auto& depth:list(value)) o.depths.push_back(number(depth));}
    else if(arg=="--width") o.settings.width=number(value);
    else if(arg=="--height") o.settings.height=number(value);
    else if(arg=="--bounces") o.settings.maxBounces=number(value);
    else if(arg=="--frames") o.frames=number(value);
    else o.warmup=number(value,true);
  }
  if(o.frames>100000 || o.warmup>100000) throw std::invalid_argument("Frames and warmup must not exceed 100000");
  if(o.mode!="off" && o.mode!="native" && o.mode!="quality" && o.mode!="balanced" && o.mode!="performance" && o.mode!="all")
    throw std::invalid_argument("DLSS mode: off/native/quality/balanced/performance/all");
  for(const auto& variant:o.variants)
    if(variant!="original" && variant!="no-graphs" && variant!="opaque" && variant!="no-opacity")
      throw std::invalid_argument("Scene variants: original/no-graphs/opaque/no-opacity/all (comma-separated)");
  caustra::validateSettings(o.settings);
  if(o.depths.empty()) o.depths.push_back(o.settings.maxBounces);
  for(auto depth:o.depths) {auto settings=o.settings;settings.maxBounces=depth;caustra::validateSettings(settings);}
  return o;
}

struct SceneCounts {uint64_t sharedTriangles=0,instancedTriangles=0;};
SceneCounts counts(const caustra::Scene& scene) {
  SceneCounts result;
  for(const auto& mesh:scene.meshes) result.sharedTriangles+=mesh.indices.size()/3;
  for(const auto& instance:scene.instances) result.instancedTriangles+=scene.meshes.at(instance.mesh).indices.size()/3;
  return result;
}
caustra::Scene diagnosticVariant(const caustra::Scene& source,const std::string& variant) {
  auto scene=source;
  if(variant=="no-opacity") {
    // Keep every graph instruction, texture lookup and downstream register
    // reference. Only suppress the coverage write: identity swizzle preserves
    // the Output instruction's value and footprint without changing opacity.
    for(auto& material:scene.materials) material.factors.z=1;
    for(auto& instruction:scene.materialInstructions)
      if(instruction.code.x==uint32_t(caustra::MaterialOp::Output) &&
         instruction.code.z==uint32_t(caustra::MaterialParameter::Opacity)) {
        instruction.code={uint32_t(caustra::MaterialOp::Swizzle),instruction.code.y,0,0};
        instruction.value={0,1,2,3};
      }
    return scene;
  }
  scene.materialInstructions.clear();
  for(auto& material:scene.materials) {
    if(variant=="no-graphs") {
      // Deliberate diagnostic approximation, not a MaterialX fallback. Preserve
      // the native surface model, POD parameters, textures and interior flags.
      material.program.x=material.program.y=0;
    } else {
      // The opaque baseline measures the same acceleration structures with a
      // simple shader. Keep POD emission and emitter sidedness, but no graphs,
      // coverage, transmission, subsurface or participating media.
      caustra::Material opaque;opaque.specular.x=0;
      opaque.emission=material.emission;opaque.surface.z=material.surface.z;
      material=opaque;
    }
  }
  return scene;
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
        "--variants original,no-graphs,opaque,no-opacity (all = first three); --depths 1,6,24\n"
        "--skip-invalid-media   Use the viewer's medium boundary recovery policy\n"
        "--render-only          Time render only; check the last output outside timing\n"
        "CSV reports CPU wall time around render, HDR readback and viewer-equivalent tone mapping.\n"
        "Render includes GPU completion and optional RR; window painting is excluded.\n"
        "Scene variants are diagnostic approximations with identical geometry, not supported material fallbacks.\n";
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
    caustra::RendererOptions rendererOptions;rendererOptions.validation=options.validation;
    if(options.skipInvalidMedia) rendererOptions.mediumErrorPolicy=caustra::MediumErrorPolicy::SkipInvalidPath;
    const auto start=Clock::now();caustra::Renderer renderer(rendererOptions);const auto initialized=Clock::now();
    renderer.setSettings(options.settings);renderer.setCamera(camera);
    std::cerr<<"Device: "<<renderer.deviceName()<<"; init "<<milliseconds(start,initialized)
      <<" ms; "<<(options.moving?"orbit":"static")<<"; 1 spp/frame; "
      <<(options.skipInvalidMedia?"recover invalid media":"strict media")<<'\n';
    if(options.variants.size()!=1 || options.variants[0]!="original")
      std::cerr<<"Diagnostic scene variants intentionally approximate materials and may change ray paths; geometry is unchanged.\n";
    const auto geometry=counts(scene);
    std::vector<std::string> modes=options.mode=="all"?std::vector<std::string>{"off","quality","performance"}:std::vector<std::string>{options.mode};
    std::cout<<"variant,meshes,instances,shared_triangles,instanced_triangles,mode,width,height,bounces,frames,render_only,render_ms,readback_ms,tonemap_ms,total_ms,p95_ms,equivalent_fps,checksum,invalid_boundaries,near_skips\n";
    for(const auto& variant:options.variants) {
      const auto uploadStart=Clock::now();
      if(variant=="original") renderer.setScene(scene);
      else {auto modified=diagnosticVariant(scene,variant);renderer.setScene(modified);}
      std::cerr<<"Variant "<<variant<<": "<<scene.meshes.size()<<" meshes, "<<scene.instances.size()
        <<" instances, "<<geometry.sharedTriangles<<" shared / "<<geometry.instancedTriangles
        <<" instanced triangles; variant preparation + upload "<<milliseconds(uploadStart,Clock::now())<<" ms\n";
    for(auto depth:options.depths) for(const auto& mode:modes) {
      auto settings=options.settings;settings.maxBounces=depth;renderer.setSettings(settings);
      caustra::DlssSettings dlss;dlss.enabled=mode!="off";dlss.preset=caustra::DlssPreset::E;
      dlss.quality=mode=="quality"?caustra::DlssQuality::Quality:mode=="balanced"?caustra::DlssQuality::Balanced:
        mode=="performance"?caustra::DlssQuality::Performance:caustra::DlssQuality::Native;
      renderer.setDlssSettings(dlss);
      caustra::example::OrbitCamera orbit;orbit.setHome(camera);renderer.setCamera(orbit.camera());
      const auto diagnosticsBefore=renderer.diagnostics();
      // First-frame material/interior initialization is not part of warmup or
      // measured frames. Every configuration then restarts the same trajectory.
      const auto firstStart=Clock::now();renderer.render(1);
      std::cerr<<"Variant "<<variant<<", depth "<<depth<<", "<<mode
        <<": first frame "<<milliseconds(firstStart,Clock::now())<<" ms (excluded)\n";
      renderer.setCamera(orbit.camera());
      std::vector<double> render,readback,tone,total;std::vector<uint32_t> display;uint64_t checksum=0;
      for(uint32_t frame=0;frame<options.warmup+options.frames;++frame) {
        if(options.moving) {orbit.orbit(1.5f,.2f);renderer.updateCamera(orbit.camera());}
        const auto a=Clock::now();renderer.render(1);const auto b=Clock::now();
        auto c=b,d=b;
        if(!options.renderOnly) {
          const auto image=renderer.readback();c=Clock::now();toneMap(image,display);d=Clock::now();
          for(auto pixel:display) checksum+=pixel;
        }
        if(frame>=options.warmup) {
          render.push_back(milliseconds(a,b));readback.push_back(milliseconds(b,c));
          tone.push_back(milliseconds(c,d));total.push_back(milliseconds(a,d));
        }
      }
      if(options.renderOnly) {toneMap(renderer.readback(),display);for(auto pixel:display) checksum+=pixel;}
      if(renderer.validationErrorCount()) throw std::runtime_error("Vulkan validation errors during benchmark");
      const auto diagnosticsAfter=renderer.diagnostics();
      const auto median=quantile(total,.5);
      std::cout<<variant<<','<<scene.meshes.size()<<','<<scene.instances.size()<<','<<geometry.sharedTriangles<<','<<geometry.instancedTriangles<<','
        <<mode<<','<<settings.width<<','<<settings.height<<','<<depth<<','<<options.frames<<','<<options.renderOnly<<','
        <<quantile(render,.5)<<','<<quantile(readback,.5)<<','<<quantile(tone,.5)<<','<<median<<','<<quantile(total,.95)<<','<<1000/median<<','<<checksum<<','
        <<diagnosticsAfter.invalidMediumBoundaries-diagnosticsBefore.invalidMediumBoundaries<<','
        <<diagnosticsAfter.skippedNearMediumBoundaries-diagnosticsBefore.skippedNearMediumBoundaries<<'\n';
    }
    }
    return 0;
  } catch(const std::exception& error) {std::cerr<<"Caustra benchmark: "<<error.what()<<'\n';return 1;}
}
