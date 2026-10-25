#include <caustra/renderer.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace caustra;

void require(bool condition,const char* message) {
  if(!condition) throw std::runtime_error(message);
}

void face(Mesh& mesh,Float4 a,Float4 b,Float4 c,Float4 d,Float4 normal) {
  const auto first=static_cast<uint32_t>(mesh.vertices.size());
  mesh.vertices.insert(mesh.vertices.end(),{{a,normal,{0,0,0,0}},{b,normal,{1,0,0,0}},
    {c,normal,{1,1,0,0}},{d,normal,{0,1,0,0}}});
  mesh.indices.insert(mesh.indices.end(),{first,first+1,first+2,first,first+2,first+3});
}

void add(Scene& scene,Mesh mesh) {
  scene.instances.push_back({static_cast<uint32_t>(scene.meshes.size())});
  scene.meshes.push_back(std::move(mesh));
}

void box(Scene& scene,float x0,float x1,float y0,float y1,float z0,float z1) {
  Mesh mesh;
  face(mesh,{x0,y0,z1,0},{x1,y0,z1,0},{x1,y1,z1,0},{x0,y1,z1,0},{0,0,1,0});
  face(mesh,{x1,y0,z0,0},{x0,y0,z0,0},{x0,y1,z0,0},{x1,y1,z0,0},{0,0,-1,0});
  face(mesh,{x1,y0,z1,0},{x1,y0,z0,0},{x1,y1,z0,0},{x1,y1,z1,0},{1,0,0,0});
  face(mesh,{x0,y0,z0,0},{x0,y0,z1,0},{x0,y1,z1,0},{x0,y1,z0,0},{-1,0,0,0});
  face(mesh,{x0,y1,z1,0},{x1,y1,z1,0},{x1,y1,z0,0},{x0,y1,z0,0},{0,1,0,0});
  face(mesh,{x0,y0,z0,0},{x1,y0,z0,0},{x1,y0,z1,0},{x0,y0,z1,0},{0,-1,0,0});
  add(scene,std::move(mesh));
}

Material fog(Float4 absorption={},Float4 emission={}) {
  Material result;result.volume={1,0,1,0};
  result.volumeAbsorption=absorption;result.volumeEmission=emission;
  return result;
}

void emitter(Scene& scene,float z=-1) {
  Material light;light.baseColor={0,0,0,1};light.specular.x=0;light.emission={1,1,1,0};
  Mesh plane;plane.material=static_cast<uint32_t>(scene.materials.size());scene.materials.push_back(light);
  face(plane,{-10,-10,z,0},{10,-10,z,0},{10,10,z,0},{-10,10,z,0},{0,0,1,0});
  add(scene,std::move(plane));
}

std::array<double,3> average(const Image& image) {
  require(!image.pixels.empty(),"Missing medium recovery image");
  std::array<double,3> result{};
  for(const auto pixel:image.pixels) {
    require(std::isfinite(pixel.x)&&std::isfinite(pixel.y)&&std::isfinite(pixel.z)&&
      pixel.x>=0&&pixel.y>=0&&pixel.z>=0,"Medium recovery produced nonfinite or negative radiance");
    result[0]+=pixel.x;result[1]+=pixel.y;result[2]+=pixel.z;
  }
  for(auto& channel:result) channel/=image.pixels.size();
  return result;
}

void close(std::array<double,3> actual,std::array<double,3> expected,double tolerance,const char* message) {
  for(size_t channel=0;channel<3;++channel) if(std::abs(actual[channel]-expected[channel])>tolerance)
    throw std::runtime_error(std::string(message)+" channel "+std::to_string(channel)+": got "+
      std::to_string(actual[channel])+", expected "+std::to_string(expected[channel]));
}

RenderSettings settings() {
  RenderSettings result;result.width=16;result.height=8;result.maxBounces=8;result.environment={};return result;
}

RendererOptions options(bool validation,MediumErrorPolicy policy,float tolerance=.001f) {
  RendererOptions result;result.validation=validation;result.mediumErrorPolicy=policy;
  result.mediumBoundaryTolerance=tolerance;return result;
}

void verifyOptionsValidation() {
  require(RendererOptions{}.mediumErrorPolicy==MediumErrorPolicy::Strict,
    "The rendering core must default to strict medium error handling");
  for(const float tolerance:std::array{-1.0f,std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity()}) {
    bool rejected=false;
    try {Renderer invalid(options(false,MediumErrorPolicy::SkipInvalidPath,tolerance));}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"Invalid medium tolerances must be rejected before Vulkan initialization");
  }
  auto invalid=options(false,MediumErrorPolicy::Strict);
  invalid.mediumErrorPolicy=static_cast<MediumErrorPolicy>(42);
  bool rejected=false;
  try {Renderer renderer(invalid);} catch(const std::invalid_argument&) {rejected=true;}
  require(rejected,"Unknown medium error policies must be rejected before Vulkan initialization");
}

Image render(Renderer& renderer,const Scene& scene,const Camera& camera,uint32_t samples=1) {
  renderer.setScene(scene);renderer.setCamera(camera);renderer.render(samples);return renderer.readbackRaw();
}

void expectBoundaryError(Renderer& renderer,const Scene& scene,const Camera& camera) {
  bool rejected=false;
  try {render(renderer,scene,camera);} catch(const std::runtime_error& error) {
    rejected=std::string(error.what()).find("Invalid medium boundary nesting")!=std::string::npos;
  }
  require(rejected,"Strict mode must continue reporting invalid medium boundary nesting");
}

Scene thinSlab() {
  Scene scene;scene.materials.push_back(fog({1000,2000,4000,0}));
  box(scene,-3,3,-3,3,0,.0002f);emitter(scene);return scene;
}

// The fixed camera containment probe travels toward +Z, while primary rays look
// toward -Z. The intentionally open face therefore creates a path-local unmatched
// exit without corrupting the camera cache. Only the left half sees that face.
Scene unmatchedExit(float distance) {
  Scene scene;scene.materials.push_back(fog());
  Mesh plane;
  face(plane,{-10,-10,-distance,0},{-10,10,-distance,0},{0,10,-distance,0},
    {0,-10,-distance,0},{0,0,-1,0});
  add(scene,std::move(plane));emitter(scene);return scene;
}

void verifyHalves(const Image& image,bool recoverNear) {
  average(image);
  for(uint32_t y=0;y<image.height;++y) for(uint32_t x=0;x<image.width;++x) {
    const auto pixel=image.pixels[size_t(y)*image.width+x];
    const float expected=x<image.width/2&&!recoverNear?0.0f:1.0f;
    require(std::abs(pixel.x-expected)<1e-5f&&std::abs(pixel.y-expected)<1e-5f&&
      std::abs(pixel.z-expected)<1e-5f,
      "Recovery must preserve valid pixels and only discard or skip the invalid boundary path");
  }
}

void verifyValidThinMedia(Renderer& strict,Renderer& tolerant) {
  const Camera camera={{0,0,.01f,0},{0,0,-1,0},{0,1,0,0},.001f};
  const auto scene=thinSlab();
  const auto strictResult=average(render(strict,scene,camera,2));
  const auto before=tolerant.diagnostics();
  const auto tolerantResult=average(render(tolerant,scene,camera,2));
  close(strictResult,{std::exp(-.2),std::exp(-.4),std::exp(-.8)},2e-5,
    "Valid sub-millimeter media must retain their physical Beer attenuation");
  close(tolerantResult,strictResult,1e-6,"Tolerance must not erase valid thin medium boundaries");
  const auto after=tolerant.diagnostics();
  require(after.invalidMediumBoundaries==before.invalidMediumBoundaries&&
    after.skippedNearMediumBoundaries==before.skippedNearMediumBoundaries,
    "Valid thin media must not be counted as invalid or skipped");
}

void verifyNearBoundary(Renderer& strict,Renderer& zeroTolerance,Renderer& tolerant) {
  const Camera camera={{0,0,0,0},{0,0,-1,0},{0,1,0,0},45};
  const auto scene=unmatchedExit(.0005f);
  expectBoundaryError(strict,scene,camera);
  const auto zeroBefore=zeroTolerance.diagnostics();
  verifyHalves(render(zeroTolerance,scene,camera,2),false);
  const auto zeroAfter=zeroTolerance.diagnostics();
  require(zeroAfter.invalidMediumBoundaries>zeroBefore.invalidMediumBoundaries&&
    zeroAfter.skippedNearMediumBoundaries==zeroBefore.skippedNearMediumBoundaries,
    "Zero tolerance must discard invalid paths without treating them as near contacts");
  require(zeroTolerance.sampleCount()==2&&zeroTolerance.frameCount()==2,
    "Invalid paths must not reset sample or frame progress in tolerant mode");

  const auto before=tolerant.diagnostics();
  verifyHalves(render(tolerant,scene,camera,2),true);
  const auto after=tolerant.diagnostics();
  require(after.skippedNearMediumBoundaries>before.skippedNearMediumBoundaries,
    "A near invalid exit within one millimeter must be counted and skipped");
  require(tolerant.sampleCount()==2&&tolerant.frameCount()==2,
    "Near-boundary recovery must preserve progressive sample and frame counts");

  // A genuinely unsupported exit farther away is still discarded; the configured
  // tolerance must not make arbitrary non-nested or open media transparent.
  const auto farBefore=tolerant.diagnostics();
  verifyHalves(render(tolerant,unmatchedExit(.01f),camera,2),false);
  const auto farAfter=tolerant.diagnostics();
  require(farAfter.invalidMediumBoundaries>farBefore.invalidMediumBoundaries&&
    farAfter.skippedNearMediumBoundaries==farBefore.skippedNearMediumBoundaries,
    "A distant invalid exit must terminate only its path instead of being skipped");
}

void verifyNonNested(Renderer& strict,Renderer& tolerant) {
  Scene scene;scene.materials.push_back(fog({}, {.1f,.2f,.3f,0}));
  box(scene,-2,2,-2,2,0,2);box(scene,-1,1,-1,1,-1,1);
  const Camera camera={{0,0,4,0},{0,0,-1,0},{0,1,0,0},.05f};
  expectBoundaryError(strict,scene,camera);
  const auto before=tolerant.diagnostics();
  const auto result=average(render(tolerant,scene,camera,2));
  require(result[0]>.1&&result[1]>.2&&result[2]>.3,
    "Stopping an invalid path must preserve radiance accumulated before the bad crossing");
  require(tolerant.diagnostics().invalidMediumBoundaries>before.invalidMediumBoundaries,
    "Non-nested media must remain observable through recovery diagnostics");
  require(tolerant.sampleCount()==2&&tolerant.frameCount()==2,
    "Non-nested media must not prevent later frame submissions");
}

void verifyCoincidentContactBudget(Renderer& tolerant) {
  auto scene=unmatchedExit(.0005f);
  // Distinct instances share the identical open plane and ray parameter.
  // Excluding only the last instance can alternate between them forever;
  // recovery must bound those contacts and discard that connection safely.
  scene.instances.push_back({0});
  const Camera camera={{0,0,0,0},{0,0,-1,0},{0,1,0,0},45};
  const auto before=tolerant.diagnostics();
  verifyHalves(render(tolerant,scene,camera,2),false);
  const auto after=tolerant.diagnostics();
  require(after.skippedNearMediumBoundaries>before.skippedNearMediumBoundaries&&
    after.invalidMediumBoundaries>before.invalidMediumBoundaries,
    "Coincident invalid contacts must hit a bounded recovery limit instead of looping forever");
  require(tolerant.sampleCount()==2&&tolerant.frameCount()==2,
    "Exhausting a path's contact budget must still allow later frames to render");
}

void verifyCameraFailure(Renderer& strict,Renderer& tolerant) {
  Scene scene;scene.materials.push_back(fog());
  // One front-facing +Z probe entry with no corresponding exit. Primary rays
  // could otherwise see the emitter, so black proves the partial cache is never
  // interpreted as a valid air stack. A second submission reuses that cache.
  Mesh plane;
  face(plane,{-10,-10,1,0},{-10,10,1,0},{10,10,1,0},{10,-10,1,0},{0,0,-1,0});
  add(scene,std::move(plane));emitter(scene);
  const Camera camera={{0,0,0,0},{0,0,-1,0},{0,1,0,0},.05f};
  expectBoundaryError(strict,scene,camera);
  const auto before=tolerant.diagnostics();
  close(average(render(tolerant,scene,camera)),{0,0,0},1e-8,
    "Failed camera containment must reject paths without publishing a partial medium stack");
  tolerant.render();
  close(average(tolerant.readbackRaw()),{0,0,0},1e-8,
    "Failed camera containment must remain safe when the cached result is reused");
  require(tolerant.sampleCount()==2&&tolerant.frameCount()==2,
    "Failed containment must not turn into a later invalid-cache exception");
  require(tolerant.diagnostics().invalidMediumBoundaries>before.invalidMediumBoundaries,
    "Failed camera containment must be included in recovery diagnostics");
}

void verifyLifetimeAndRecovery(Renderer& tolerant) {
  const auto before=tolerant.diagnostics();
  tolerant.reset();
  const auto reset=tolerant.diagnostics();
  require(reset.invalidMediumBoundaries==before.invalidMediumBoundaries&&
    reset.skippedNearMediumBoundaries==before.skippedNearMediumBoundaries,
    "Reset must preserve lifetime recovery diagnostics");
  const Camera camera={{0,0,.01f,0},{0,0,-1,0},{0,1,0,0},.001f};
  close(average(render(tolerant,thinSlab(),camera,2)),
    {std::exp(-.2),std::exp(-.4),std::exp(-.8)},2e-5,
    "A valid scene must render correctly after failed paths and failed camera containment");
  const auto after=tolerant.diagnostics();
  require(after.invalidMediumBoundaries==before.invalidMediumBoundaries&&
    after.skippedNearMediumBoundaries==before.skippedNearMediumBoundaries,
    "Scene and camera changes must preserve accumulated diagnostics without inventing new failures");
}

void verifyOverflowRemainsFatal(Renderer& tolerant) {
  Scene scene;scene.materials.push_back(fog());
  for(int index=0;index<17;++index) {
    const float radius=1+float(index)*.1f;box(scene,-radius,radius,-radius,radius,-radius,radius);
  }
  bool rejected=false;
  try {render(tolerant,scene,{{0,0,0,0},{0,0,-3,0},{0,1,0,0},.05f});}
  catch(const std::runtime_error& error) {
    rejected=std::string(error.what()).find("Medium nesting exceeds")!=std::string::npos;
  }
  require(rejected,"Tolerant boundary handling must still reject fixed-array stack overflow");
}

}

int main(int argc,char**) {
  try {
    const bool validation=argc>1;
    verifyOptionsValidation();
    Renderer strict(options(validation,MediumErrorPolicy::Strict));
    Renderer zeroTolerance(options(validation,MediumErrorPolicy::SkipInvalidPath,0));
    Renderer tolerant(options(validation,MediumErrorPolicy::SkipInvalidPath));
    strict.setSettings(settings());zeroTolerance.setSettings(settings());tolerant.setSettings(settings());
    verifyValidThinMedia(strict,tolerant);
    verifyNearBoundary(strict,zeroTolerance,tolerant);
    verifyNonNested(strict,tolerant);
    verifyCoincidentContactBudget(tolerant);
    verifyCameraFailure(strict,tolerant);
    verifyLifetimeAndRecovery(tolerant);
    verifyOverflowRemainsFatal(tolerant);
    verifyLifetimeAndRecovery(tolerant);
    require(strict.validationErrorCount()==0&&zeroTolerance.validationErrorCount()==0&&
      tolerant.validationErrorCount()==0,"Vulkan validation errors in medium recovery");
    std::cout<<"Medium recovery passed: strict errors, near-contact tolerance, valid thin Beer media, "
      "non-nested paths, coincident-contact budgets, valid pixels, failed camera caches, lifetime diagnostics, recovery and fatal overflow.\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
