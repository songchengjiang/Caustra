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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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
  std::filesystem::path output;
  std::filesystem::path profileOutput;
  bool moving=true,validation=false,help=false,skipInvalidMedia=false,renderOnly=false;
  bool gpuTiming=false,rayStatistics=false;
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
    if(arg=="--gpu-timing") {o.gpuTiming=true;continue;}
    if(arg=="--ray-stats") {o.rayStatistics=true;o.gpuTiming=true;continue;}
    if(arg!="--asset" && arg!="--width" && arg!="--height" && arg!="--bounces" && arg!="--frames" && arg!="--warmup" && arg!="--dlss" && arg!="--variants" && arg!="--depths" && arg!="--output" && arg!="--profile-output")
      throw std::invalid_argument("Unknown benchmark option: "+std::string(arg));
    if(++i==argc) throw std::invalid_argument("Missing option value");
    const std::string_view value=argv[i];
    if(arg=="--asset") o.asset=argv[i];
    else if(arg=="--output") o.output=argv[i];
    else if(arg=="--profile-output") o.profileOutput=argv[i];
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

struct ImageMetrics {
  uint32_t width=0,height=0;
  uint64_t nonzeroPixels=0;
  double meanLuminance=0,maxRgb=0;
};
ImageMetrics imageMetrics(const caustra::Image& image) {
  if(image.pixels.empty()) throw std::runtime_error("Empty benchmark output");
  ImageMetrics result;result.width=image.width;result.height=image.height;
  for(const auto& p:image.pixels) {
    if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
      throw std::runtime_error("Nonfinite benchmark output");
    const double maximum=std::max({double(p.x),double(p.y),double(p.z)});
    result.maxRgb=std::max(result.maxRgb,maximum);
    result.nonzeroPixels+=maximum>0;
    result.meanLuminance+=.2126*double(p.x)+.7152*double(p.y)+.0722*double(p.z);
  }
  result.meanLuminance/=double(image.pixels.size());
  if(!result.nonzeroPixels) throw std::runtime_error("Black benchmark output");
  return result;
}

// Raw frame records are retained until all measured work finishes. File IO and
// image validation never enter a measured render/readback/tonemap interval.
struct ProfileFrame {
  std::string variant,mode;
  uint32_t bounces=0,frame=0,trajectoryFrame=0;
  double renderMs=0,readbackMs=0,tonemapMs=0,totalMs=0;
  caustra::RendererProfile gpu;
};
struct ProfileConfiguration {
  std::string variant,mode,dlssStatus;
  uint32_t bounces=0;
  size_t begin=0,end=0;
  ImageMetrics output,raw;
  double preparationUploadMs=0,firstFrameMs=0;
  uint64_t invalidBoundaries=0,nearSkips=0;
};
struct CounterField {
  const char* name;
  uint64_t caustra::RayStatistics::*member;
};
constexpr std::array counterFields{
  CounterField{"path_samples",&caustra::RayStatistics::pathSamples},
  CounterField{"primary_ray_queries",&caustra::RayStatistics::primaryRayQueries},
  CounterField{"indirect_ray_queries",&caustra::RayStatistics::indirectRayQueries},
  CounterField{"shadow_ray_queries",&caustra::RayStatistics::shadowRayQueries},
  CounterField{"auxiliary_ray_queries",&caustra::RayStatistics::auxiliaryRayQueries},
  CounterField{"boundary_continuation_queries",&caustra::RayStatistics::boundaryContinuationQueries},
  CounterField{"query_hits",&caustra::RayStatistics::queryHits},
  CounterField{"query_misses",&caustra::RayStatistics::queryMisses},
  CounterField{"candidate_visits",&caustra::RayStatistics::candidateVisits},
  CounterField{"coverage_evaluations",&caustra::RayStatistics::coverageEvaluations},
  CounterField{"full_material_evaluations",&caustra::RayStatistics::fullMaterialEvaluations},
  CounterField{"alpha_rejected_hits",&caustra::RayStatistics::alphaRejectedHits},
  CounterField{"shadow_transparent_hits",&caustra::RayStatistics::shadowTransparentHits},
  CounterField{"accepted_surface_hits",&caustra::RayStatistics::acceptedSurfaceHits},
  CounterField{"surface_scatters",&caustra::RayStatistics::surfaceScatters},
  CounterField{"medium_scatters",&caustra::RayStatistics::mediumScatters},
  CounterField{"russian_roulette_terminations",&caustra::RayStatistics::russianRouletteTerminations},
  CounterField{"max_scattering_depth",&caustra::RayStatistics::maxScatteringDepth}
};
uint64_t rayQueries(const caustra::RayStatistics& rays) {
  return rays.primaryRayQueries+rays.indirectRayQueries+rays.shadowRayQueries+rays.auxiliaryRayQueries;
}
void jsonString(std::ostream& stream,std::string_view value) {
  constexpr char hex[]="0123456789abcdef";
  stream<<'"';
  for(unsigned char c:value) {
    if(c=='"' || c=='\\') stream<<'\\'<<char(c);
    else if(c<0x20) stream<<"\\u00"<<hex[c>>4]<<hex[c&15];
    else stream<<char(c);
  }
  stream<<'"';
}
void jsonDistribution(std::ostream& stream,const std::vector<double>& values,bool includeSum=false) {
  if(values.empty()) {stream<<"null";return;}
  const double sum=std::accumulate(values.begin(),values.end(),0.0);
  stream<<"{\"count\":"<<values.size()<<",\"min\":"<<*std::min_element(values.begin(),values.end())
    <<",\"mean\":"<<sum/double(values.size())<<",\"p50\":"<<quantile(values,.5)
    <<",\"p95\":"<<quantile(values,.95)<<",\"max\":"<<*std::max_element(values.begin(),values.end());
  if(includeSum) stream<<",\"sum\":"<<sum;
  stream<<'}';
}
void jsonImage(std::ostream& stream,const ImageMetrics& image) {
  stream<<"{\"width\":"<<image.width<<",\"height\":"<<image.height
    <<",\"finite\":true,\"nonzero_pixels\":"<<image.nonzeroPixels
    <<",\"mean_luminance\":"<<image.meanLuminance<<",\"max_rgb\":"<<image.maxRgb<<'}';
}
void writeProfileReports(const Options& options,const std::string& device,const caustra::Scene& scene,
                         const SceneCounts& geometry,const caustra::Camera& camera,
                         double rendererInitializationMs,
                         const std::vector<ProfileFrame>& frames,
                         const std::vector<ProfileConfiguration>& configurations) {
  if(options.profileOutput.empty()) return;
  if(!options.profileOutput.parent_path().empty())
    std::filesystem::create_directories(options.profileOutput.parent_path());
  auto csvPath=options.profileOutput;csvPath+="_frames.csv";
  auto jsonPath=options.profileOutput;jsonPath+="_summary.json";
  std::ofstream csv(csvPath),json(jsonPath);
  if(!csv || !json) throw std::runtime_error("Cannot create benchmark profile reports");
  csv<<std::fixed<<std::setprecision(9);
  json<<std::boolalpha<<std::setprecision(17);
  csv<<"variant,mode,bounces,frame,trajectory_frame,output_width,output_height,render_width,render_height,samples,"
    "cpu_render_ms,cpu_readback_ms,cpu_tonemap_ms,cpu_total_ms,gpu_timing_supported,gpu_timing_valid,"
    "initialization_gpu_ms,path_trace_gpu_ms,dlss_gpu_ms,total_gpu_ms,ray_statistics_valid,ray_statistics_overflowed,ray_queries";
  for(const auto& field:counterFields) csv<<','<<field.name;
  csv<<'\n';
  for(const auto& frame:frames) {
    const auto& gpu=frame.gpu;
    csv<<frame.variant<<','<<frame.mode<<','<<frame.bounces<<','<<frame.frame<<','<<frame.trajectoryFrame
      <<','<<options.settings.width<<','<<options.settings.height<<','<<gpu.renderWidth<<','<<gpu.renderHeight
      <<','<<gpu.samples<<','<<frame.renderMs<<','<<frame.readbackMs<<','<<frame.tonemapMs<<','<<frame.totalMs
      <<','<<gpu.gpuTimingSupported<<','<<gpu.gpuTimingValid;
    for(double value:{gpu.initializationGpuMs,gpu.pathTraceGpuMs,gpu.dlssGpuMs,gpu.totalGpuMs}) {
      csv<<',';if(gpu.gpuTimingValid) csv<<value;
    }
    csv<<','<<gpu.rayStatisticsValid<<','<<gpu.rays.overflowed<<',';
    if(gpu.rayStatisticsValid && !gpu.rays.overflowed) csv<<rayQueries(gpu.rays);
    for(const auto& field:counterFields) {csv<<',';if(gpu.rayStatisticsValid) csv<<gpu.rays.*field.member;}
    csv<<'\n';
  }
  json<<"{\n  \"schema_version\":1,\n  \"device\":";jsonString(json,device);
  json<<",\n  \"asset\":";jsonString(json,options.asset.generic_string());
  json<<",\n  \"gpu_timing_requested\":"<<options.gpuTiming
    <<",\n  \"ray_statistics_requested\":"<<options.rayStatistics
    <<",\n  \"validation\":"<<options.validation<<",\n  \"moving_camera\":"<<options.moving
    <<",\n  \"render_only\":"<<options.renderOnly<<",\n  \"skip_invalid_media\":"<<options.skipInvalidMedia
    <<",\n  \"samples_per_frame\":1,\n  \"warmup_frames\":"<<options.warmup
    <<",\n  \"measured_frames_per_configuration\":"<<options.frames
    <<",\n  \"output_width\":"<<options.settings.width<<",\n  \"output_height\":"<<options.settings.height
    <<",\n  \"renderer_initialization_cpu_ms\":"<<rendererInitializationMs
    <<",\n  \"environment_radiance\":["<<options.settings.environment.x<<','<<options.settings.environment.y
    <<','<<options.settings.environment.z<<"],\n  \"orbit_step\":{\"dx_pixels\":1.5,\"dy_pixels\":0.2},\n  \"camera\":{\"position\":["
    <<camera.position.x<<','<<camera.position.y<<','<<camera.position.z<<"],\"target\":["
    <<camera.target.x<<','<<camera.target.y<<','<<camera.target.z<<"],\"up\":["
    <<camera.up.x<<','<<camera.up.y<<','<<camera.up.z<<"],\"vertical_fov\":"<<camera.verticalFov<<"},\n"
    "  \"scene\":{\"meshes\":"<<scene.meshes.size()<<",\"instances\":"<<scene.instances.size()
    <<",\"materials\":"<<scene.materials.size()<<",\"material_instructions\":"<<scene.materialInstructions.size()
    <<",\"textures\":"<<scene.textures.size()<<",\"shared_triangles\":"<<geometry.sharedTriangles
    <<",\"instanced_triangles\":"<<geometry.instancedTriangles<<"},\n"
    "  \"notes\":[\"CPU render time includes synchronous GPU completion, diagnostics and optional profiling readback.\","
    "\"GPU times are intervals within the compute command stream, not isolated RT traversal or shading times.\","
    "\"candidate_visits counts only shader-visible RayQuery candidates, not hardware BVH box or triangle tests.\","
    "\"boundary_continuation_queries is a subset of query classes and must not be added to ray_queries.\","
    "\"Counter insertion can change shader occupancy and execution time; compare against a separate run without --ray-stats.\","
    "\"All configurations exclude one initialization frame, then restart the same camera and sample sequence.\","
    "\"invalid_boundaries and near_skips include the initialization frame and warmup; ray counter distributions include only measured frames.\","
    "\"Warmup frames are excluded. Last-frame image validation and file IO occur outside timing.\"],\n"
    "  \"configurations\":[\n";
  bool firstConfiguration=true;
  for(const auto& config:configurations) {
    if(!firstConfiguration) json<<",\n";firstConfiguration=false;
    json<<"    {\"variant\":";jsonString(json,config.variant);json<<",\"mode\":";jsonString(json,config.mode);
    json<<",\"bounces\":"<<config.bounces<<",\"dlss_preset\":\"E\",\"dlss_status\":";
    jsonString(json,config.dlssStatus);
    const auto& gpu=frames.at(config.begin).gpu;
    size_t validTiming=0,validCounters=0,overflowFrames=0;
    for(size_t i=config.begin;i<config.end;++i) {
      validTiming+=frames[i].gpu.gpuTimingValid;
      validCounters+=frames[i].gpu.rayStatisticsValid && !frames[i].gpu.rays.overflowed;
      overflowFrames+=frames[i].gpu.rays.overflowed;
    }
    json<<",\"render_width\":"<<gpu.renderWidth<<",\"render_height\":"<<gpu.renderHeight
      <<",\"preparation_upload_cpu_ms\":"<<config.preparationUploadMs<<",\"first_frame_cpu_ms\":"<<config.firstFrameMs
      <<",\"frame_count\":"<<config.end-config.begin<<",\"gpu_timing_supported\":"<<gpu.gpuTimingSupported
      <<",\"valid_gpu_timing_frames\":"<<validTiming<<",\"valid_ray_statistics_frames\":"<<validCounters
      <<",\"counter_overflow_frames\":"<<overflowFrames
      <<",\"invalid_boundaries\":"<<config.invalidBoundaries<<",\"near_skips\":"<<config.nearSkips;
    const auto values=[&](auto get,auto valid) {
      std::vector<double> result;
      for(size_t i=config.begin;i<config.end;++i) if(valid(frames[i])) result.push_back(double(get(frames[i])));
      return result;
    };
    const auto any=[](const ProfileFrame&){return true;};
    const auto timed=[](const ProfileFrame& f){return f.gpu.gpuTimingValid;};
    const auto counted=[](const ProfileFrame& f){return f.gpu.rayStatisticsValid && !f.gpu.rays.overflowed;};
    json<<",\"timings_ms\":{\"cpu_render\":";
    jsonDistribution(json,values([](const auto& f){return f.renderMs;},any));
    json<<",\"cpu_readback\":";jsonDistribution(json,values([](const auto& f){return f.readbackMs;},any));
    json<<",\"cpu_tonemap\":";jsonDistribution(json,values([](const auto& f){return f.tonemapMs;},any));
    json<<",\"cpu_total\":";jsonDistribution(json,values([](const auto& f){return f.totalMs;},any));
    json<<",\"gpu_initialization\":";jsonDistribution(json,values([](const auto& f){return f.gpu.initializationGpuMs;},timed));
    json<<",\"gpu_path_trace\":";jsonDistribution(json,values([](const auto& f){return f.gpu.pathTraceGpuMs;},timed));
    json<<",\"gpu_dlss\":";jsonDistribution(json,values([](const auto& f){return f.gpu.dlssGpuMs;},timed));
    json<<",\"gpu_total\":";jsonDistribution(json,values([](const auto& f){return f.gpu.totalGpuMs;},timed));
    json<<"},\"ray_counts_per_frame\":{\"ray_queries\":";
    jsonDistribution(json,values([](const auto& f){return rayQueries(f.gpu.rays);},counted),true);
    for(const auto& field:counterFields) {
      json<<',';jsonString(json,field.name);json<<':';
      jsonDistribution(json,values([&](const auto& f){return f.gpu.rays.*field.member;},counted),
        field.member!=&caustra::RayStatistics::maxScatteringDepth);
    }
    json<<"},\"last_output\":";jsonImage(json,config.output);
    json<<",\"last_raw_output\":";jsonImage(json,config.raw);json<<'}';
  }
  json<<"\n  ]\n}\n";
  csv.close();json.close();
  if(!csv || !json) throw std::runtime_error("Cannot finish benchmark profile reports");
  std::cerr<<"Saved profile frames: "<<csvPath.string()<<"; summary: "<<jsonPath.string()<<'\n';
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
        "--output PREFIX        Save final output and raw HDR PFM per configuration outside timing\n"
        "--gpu-timing           Collect GPU initialization/path-trace/DLSS command intervals\n"
        "--ray-stats            Collect instrumented shader counters; implies --gpu-timing\n"
        "--profile-output PREFIX Write PREFIX_frames.csv and PREFIX_summary.json (also works for CPU-only baselines)\n"
        "CSV reports CPU wall time around render, HDR readback and viewer-equivalent tone mapping.\n"
        "Render includes GPU completion and optional RR; window painting is excluded.\n"
        "Compare --ray-stats against a separate run without counters to measure instrumentation overhead.\n"
        "GPU path-trace time includes traversal, material evaluation and integration; it does not isolate RT hardware.\n"
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
    rendererOptions.enableGpuTiming=options.gpuTiming;
    rendererOptions.enableRayStatistics=options.rayStatistics;
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
    const bool collectProfile=options.gpuTiming || options.rayStatistics || !options.profileOutput.empty();
    std::vector<ProfileFrame> profileFrames;
    std::vector<ProfileConfiguration> profileConfigurations;
    if(options.rayStatistics)
      std::cerr<<"Ray counters use an instrumented shader; timings may differ from the production shader.\n";
    std::cout<<"variant,meshes,instances,shared_triangles,instanced_triangles,mode,width,height,bounces,frames,render_only,render_ms,readback_ms,tonemap_ms,total_ms,p95_ms,equivalent_fps,checksum,invalid_boundaries,near_skips\n";
    for(const auto& variant:options.variants) {
      const auto uploadStart=Clock::now();
      if(variant=="original") renderer.setScene(scene);
      else {auto modified=diagnosticVariant(scene,variant);renderer.setScene(modified);}
      const auto preparationUploadMs=milliseconds(uploadStart,Clock::now());
      std::cerr<<"Variant "<<variant<<": "<<scene.meshes.size()<<" meshes, "<<scene.instances.size()
        <<" instances, "<<geometry.sharedTriangles<<" shared / "<<geometry.instancedTriangles
        <<" instanced triangles; variant preparation + upload "<<preparationUploadMs<<" ms\n";
    for(auto depth:options.depths) for(const auto& mode:modes) {
      auto settings=options.settings;settings.maxBounces=depth;renderer.setSettings(settings);
      caustra::DlssSettings dlss;dlss.enabled=mode!="off";dlss.preset=caustra::DlssPreset::E;
      dlss.quality=mode=="quality"?caustra::DlssQuality::Quality:mode=="balanced"?caustra::DlssQuality::Balanced:
        mode=="performance"?caustra::DlssQuality::Performance:caustra::DlssQuality::Native;
      renderer.setDlssSettings(dlss);
      ProfileConfiguration profileConfig;profileConfig.variant=variant;profileConfig.mode=mode;
      profileConfig.bounces=depth;profileConfig.preparationUploadMs=preparationUploadMs;
      profileConfig.begin=profileFrames.size();
      if(collectProfile) profileFrames.reserve(profileFrames.size()+options.frames);
      caustra::example::OrbitCamera orbit;orbit.setHome(camera);renderer.setCamera(orbit.camera());
      const auto diagnosticsBefore=renderer.diagnostics();
      // First-frame material/interior initialization is not part of warmup or
      // measured frames. Every configuration then restarts the same trajectory.
      const auto firstStart=Clock::now();renderer.render(1);
      profileConfig.firstFrameMs=milliseconds(firstStart,Clock::now());
      std::cerr<<"Variant "<<variant<<", depth "<<depth<<", "<<mode
        <<": first frame "<<profileConfig.firstFrameMs<<" ms (excluded)\n";
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
          if(collectProfile) profileFrames.push_back({variant,mode,depth,frame-options.warmup,frame,
            milliseconds(a,b),milliseconds(b,c),milliseconds(c,d),milliseconds(a,d),renderer.profile()});
        }
      }
      if(options.renderOnly || !options.output.empty() || collectProfile) {
        const auto image=renderer.readback();
        if(options.renderOnly) {toneMap(image,display);for(auto pixel:display) checksum+=pixel;}
        if(collectProfile) profileConfig.output=imageMetrics(image);
        std::filesystem::path output;
        if(!options.output.empty()) {
          output=options.output;
          output+="_"+variant+"_"+mode+"_b"+std::to_string(depth);
          if(!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
          image.savePfm(output.string()+".pfm");
          image.savePpm(output.string()+".ppm");
          std::cerr<<"Saved final output: "<<output.string()<<".{pfm,ppm}; raw HDR: "<<output.string()<<"_raw.pfm\n";
        }
        if(dlss.enabled && (collectProfile || !output.empty())) {
          const auto raw=renderer.readbackRaw();
          if(collectProfile) profileConfig.raw=imageMetrics(raw);
          if(!output.empty()) raw.savePfm(output.string()+"_raw.pfm");
        } else {
          if(collectProfile) profileConfig.raw=profileConfig.output;
          if(!output.empty()) image.savePfm(output.string()+"_raw.pfm");
        }
      }
      if(renderer.validationErrorCount()) throw std::runtime_error("Vulkan validation errors during benchmark");
      const auto diagnosticsAfter=renderer.diagnostics();
      if(collectProfile) {
        profileConfig.end=profileFrames.size();
        profileConfig.dlssStatus=renderer.dlssStatus();
        profileConfig.invalidBoundaries=diagnosticsAfter.invalidMediumBoundaries-diagnosticsBefore.invalidMediumBoundaries;
        profileConfig.nearSkips=diagnosticsAfter.skippedNearMediumBoundaries-diagnosticsBefore.skippedNearMediumBoundaries;
        profileConfigurations.push_back(profileConfig);
        if(options.gpuTiming) {
          std::vector<double> gpuPath,gpuDlss,gpuTotal;
          for(size_t i=profileConfig.begin;i<profileConfig.end;++i) if(profileFrames[i].gpu.gpuTimingValid) {
            gpuPath.push_back(profileFrames[i].gpu.pathTraceGpuMs);
            gpuDlss.push_back(profileFrames[i].gpu.dlssGpuMs);
            gpuTotal.push_back(profileFrames[i].gpu.totalGpuMs);
          }
          std::cerr<<"GPU timing, "<<variant<<", depth "<<depth<<", "<<mode<<": ";
          if(gpuPath.empty()) std::cerr<<"unavailable on the selected queue\n";
          else std::cerr<<"path trace "<<quantile(gpuPath,.5)<<" ms, DLSS "<<quantile(gpuDlss,.5)
            <<" ms, total "<<quantile(gpuTotal,.5)<<" ms (medians)\n";
        }
      }
      const auto median=quantile(total,.5);
      std::cout<<variant<<','<<scene.meshes.size()<<','<<scene.instances.size()<<','<<geometry.sharedTriangles<<','<<geometry.instancedTriangles<<','
        <<mode<<','<<settings.width<<','<<settings.height<<','<<depth<<','<<options.frames<<','<<options.renderOnly<<','
        <<quantile(render,.5)<<','<<quantile(readback,.5)<<','<<quantile(tone,.5)<<','<<median<<','<<quantile(total,.95)<<','<<1000/median<<','<<checksum<<','
        <<diagnosticsAfter.invalidMediumBoundaries-diagnosticsBefore.invalidMediumBoundaries<<','
        <<diagnosticsAfter.skippedNearMediumBoundaries-diagnosticsBefore.skippedNearMediumBoundaries<<'\n';
    }
    }
    writeProfileReports(options,renderer.deviceName(),scene,geometry,camera,milliseconds(start,initialized),
      profileFrames,profileConfigurations);
    return 0;
  } catch(const std::exception& error) {std::cerr<<"Caustra benchmark: "<<error.what()<<'\n';return 1;}
}
