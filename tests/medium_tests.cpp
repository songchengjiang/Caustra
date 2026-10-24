#include <caustra/renderer.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace caustra;
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
void face(Mesh& mesh,Float4 a,Float4 b,Float4 c,Float4 d,Float4 normal) {
  uint32_t first=static_cast<uint32_t>(mesh.vertices.size());
  float x=b.x-a.x,y=b.y-a.y,z=b.z-a.z,s=1/std::sqrt(x*x+y*y+z*z);
  Float4 tangent{x*s,y*s,z*s,1};
  mesh.vertices.insert(mesh.vertices.end(),{{a,normal,{0,0,0,0},tangent},{b,normal,{1,0,0,0},tangent},
    {c,normal,{1,1,0,0},tangent},{d,normal,{0,1,0,0},tangent}});
  mesh.indices.insert(mesh.indices.end(),{first,first+1,first+2,first,first+2,first+3});
}
void add(Scene& scene,Mesh mesh) {
  scene.instances.push_back({static_cast<uint32_t>(scene.meshes.size())});scene.meshes.push_back(std::move(mesh));
}
void box(Scene& scene,uint32_t material,float x0,float x1,float y0,float y1,float z0,float z1) {
  Mesh m;m.material=material;
  face(m,{x0,y0,z1,0},{x1,y0,z1,0},{x1,y1,z1,0},{x0,y1,z1,0},{0,0,1,0});
  face(m,{x1,y0,z0,0},{x0,y0,z0,0},{x0,y1,z0,0},{x1,y1,z0,0},{0,0,-1,0});
  face(m,{x1,y0,z1,0},{x1,y0,z0,0},{x1,y1,z0,0},{x1,y1,z1,0},{1,0,0,0});
  face(m,{x0,y0,z0,0},{x0,y0,z1,0},{x0,y1,z1,0},{x0,y1,z0,0},{-1,0,0,0});
  face(m,{x0,y1,z1,0},{x1,y1,z1,0},{x1,y1,z0,0},{x0,y1,z0,0},{0,1,0,0});
  face(m,{x0,y0,z0,0},{x1,y0,z0,0},{x1,y0,z1,0},{x0,y0,z1,0},{0,-1,0,0});
  add(scene,std::move(m));
}
void emitter(Scene& scene,float z,float radius=10) {
  Material light;light.baseColor={0,0,0,1};light.specular.x=0;light.emission={1,1,1,0};
  uint32_t material=static_cast<uint32_t>(scene.materials.size());scene.materials.push_back(light);
  Mesh plane;plane.material=material;
  face(plane,{-radius,-radius,z,0},{radius,-radius,z,0},{radius,radius,z,0},{-radius,radius,z,0},{0,0,1,0});add(scene,std::move(plane));
}
Material fog(Float4 absorption={},Float4 scattering={},Float4 emission={}) {
  Material m;m.volume={1,0,1,0};m.volumeAbsorption=absorption;m.volumeScattering=scattering;m.volumeEmission=emission;
  return m;
}
Scene slab(Material medium,bool light=true,float thickness=1) {
  Scene s;s.materials.push_back(medium);box(s,0,-3,3,-3,3,0,thickness);
  if(light)emitter(s,-1);return s;
}
std::array<double,3> average(const Image& image) {
  std::array<double,3> result{};
  for(auto p:image.pixels) {
    require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&p.x>=0&&p.y>=0&&p.z>=0,"Nonfinite or negative medium radiance");
    result[0]+=p.x;result[1]+=p.y;result[2]+=p.z;
  }
  for(auto& c:result)c/=image.pixels.size();return result;
}
std::array<double,3> render(Renderer& renderer,const Scene& scene,uint32_t samples=1,Camera camera={{0,0,4,0},{0,0,-1,0},{0,1,0,0},0.05f}) {
  renderer.setScene(scene);renderer.setCamera(camera);renderer.render(samples);return average(renderer.readbackRaw());
}
void close(std::array<double,3> actual,std::array<double,3> expected,double tolerance,const char* message) {
  for(size_t c=0;c<3;++c)if(std::abs(actual[c]-expected[c])>tolerance)
    throw std::runtime_error(std::string(message)+" channel "+std::to_string(c)+": got "+std::to_string(actual[c])+", expected "+std::to_string(expected[c]));
}
}

int main(int argc,char**) {
  try {
    Renderer renderer({argc>1});
    RenderSettings settings;settings.width=8;settings.height=8;settings.maxBounces=64;settings.environment={};renderer.setSettings(settings);
    auto absorbing=fog({0.2f,0.5f,1,0});
    close(render(renderer,slab(absorbing)),{std::exp(-.2),std::exp(-.5),std::exp(-1.)},2e-5,"RGB Beer-Lambert slab");
    absorbing.volume.x=2;
    close(render(renderer,slab(absorbing)),{std::exp(-.4),std::exp(-1.),std::exp(-2.)},2e-5,"Volume density must scale optical depth");
    // A flagged constant interior must retain the same physical Beer law as
    // the homogeneous path. The grid estimator samples free flight, so compare
    // against analytic RGB expectations over a deterministic Monte Carlo run.
    auto constantGrid=fog({.2f,.5f,1,0});constantGrid.program.w=1;
    close(render(renderer,slab(constantGrid),1024),{std::exp(-.2),std::exp(-.5),std::exp(-1.)},.015,
      "Constant heterogeneous grid must preserve the homogeneous Beer integral");

    // The graph defines sigma_a(z)=(.2,.5,1)*z inside [0,1]. Its exact optical
    // depth is (.1,.25,.5); midpoint integration is exact for this linear field.
    // Freezing the graph at the entry, exit or camera cannot pass this reference.
    auto layered=slab(fog());layered.materials[0].program={0,5,0,1};
    layered.materialInstructions={
      {{uint32_t(MaterialOp::Position),0,0,0},{}},
      {{uint32_t(MaterialOp::Swizzle),0,0,0},{2,2,2,2}},
      {{uint32_t(MaterialOp::Constant),0,0,0},{.2f,.5f,1,0}},
      {{uint32_t(MaterialOp::Multiply),1,2,0},{}},
      {{uint32_t(MaterialOp::Output),3,uint32_t(MaterialParameter::VolumeAbsorption),0},{}}};
    close(render(renderer,layered,1024),{std::exp(-.1),std::exp(-.25),std::exp(-.5)},.015,
      "Spatial volume absorption must integrate each traversed grid cell");
    close(render(renderer,layered,1024,{{0,0,.5f,0},{0,0,-1,0},{0,1,0,0},.05f}),
      {std::exp(-.025),std::exp(-.0625),std::exp(-.125)},.015,
      "Camera-inside DDA must start at its interior cell and integrate the remaining layer");

    // World- and object-position graphs have different values after translation.
    // Inverse instance transforms must also keep world ray distances correct for
    // nonuniform/mirrored geometry, rather than treating object distances as t.
    layered.instances[0].transform[11]=2;
    close(render(renderer,layered,1024),{std::exp(-.5),std::exp(-1.25),std::exp(-2.5)},.015,
      "World-position grid fields must include their instance translation");
    layered.materialInstructions[0].code.x=uint32_t(MaterialOp::ObjectPosition);
    close(render(renderer,layered,1024),{std::exp(-.1),std::exp(-.25),std::exp(-.5)},.015,
      "Object-position grid fields must be independent of instance translation");
    layered.instances[0].transform[10]=-2;layered.instances[0].transform[11]=4;
    close(render(renderer,layered,1024,{{0,0,6,0},{0,0,-1,0},{0,1,0,0},.05f}),
      {std::exp(-.2),std::exp(-.5),std::exp(-1.)},.015,
      "Mirrored nonuniform heterogeneous grids must integrate in world distance");

    auto microscopicGrid=slab(fog(),true,1.0e-9f);
    microscopicGrid.materials[0].program={0,5,0,1};
    microscopicGrid.materialInstructions=layered.materialInstructions;
    microscopicGrid.materialInstructions[2].value={.4e18f,1.0e18f,2.0e18f,0};
    close(render(renderer,microscopicGrid,1024,{{0,0,1e-7f,0},{0,0,-1,0},{0,1,0,0},.001f}),
      {std::exp(-.2),std::exp(-.5),std::exp(-1.)},.015,
      "Heterogeneous grid coordinates must retain a real sub-1e-8 slab thickness");

    // Emission is integrated over every cell analytically, even on a free-flight
    // sample that is subsequently absorbed. Homogeneous constant cells give a
    // direct source-integral reference, independent of the random walk sampler.
    auto gridGlow=fog({.2f,.5f,1,0},{},{.3f,.4f,.5f,0});gridGlow.program.w=1;
    close(render(renderer,slab(gridGlow,false),16),
      {.3/.2*(1-std::exp(-.2)),.4/.5*(1-std::exp(-.5)),.5*(1-std::exp(-1.))},2e-5,
      "Heterogeneous cell source integrals must preserve homogeneous emission");

    Material graphVolume;graphVolume.volumeAbsorption={.2f,.5f,1,0};graphVolume.program={0,3,0,0};
    auto graphSlab=slab(graphVolume);
    graphSlab.materialInstructions={
      {{uint32_t(MaterialOp::Constant),0,0,0},{1,1,1,1}},
      {{uint32_t(MaterialOp::Output),0,uint32_t(MaterialParameter::VolumeDensity),0},{}},
      {{uint32_t(MaterialOp::Output),0,uint32_t(MaterialParameter::VolumeBoundaryOnly),0},{}}};
    close(render(renderer,graphSlab),{std::exp(-.2),std::exp(-.5),std::exp(-1.)},2e-5,
      "Per-material medium precomputation must evaluate graph outputs that activate a raw-zero volume");
    close(render(renderer,graphSlab,1,{{0,0,.5f,0},{0,0,-1,0},{0,1,0,0},.05f}),
      {std::exp(-.1),std::exp(-.25),std::exp(-.5)},2e-5,
      "Camera containment must use graph-activated material metadata and shared coefficients");
    Scene manyMaterials;manyMaterials.materials.resize(65);manyMaterials.materials[64]=fog({.2f,.5f,1,0});
    box(manyMaterials,64,-3,3,-3,3,0,1);emitter(manyMaterials,-1);
    close(render(renderer,manyMaterials),{std::exp(-.2),std::exp(-.5),std::exp(-1.)},2e-5,
      "Material-medium initialization must cover materials beyond the first compute workgroup");
    auto glowing=fog({0.2f,0.5f,1,0},{},{0.3f,0.4f,0.5f,0});
    close(render(renderer,slab(glowing,false)),{.3/.2*(1-std::exp(-.2)),.4/.5*(1-std::exp(-.5)),.5*(1-std::exp(-1.))},2e-5,"Homogeneous emission integral");
    close(render(renderer,slab(glowing,false),1,{{0,0,.5f,0},{0,0,-1,0},{0,1,0,0},.05f}),
      {.3/.2*(1-std::exp(-.1)),.4/.5*(1-std::exp(-.25)),.5*(1-std::exp(-.5))},2e-5,
      "Camera-inside cached medium must retain RGB absorption and emission coefficients");
    auto zeroExtinction=fog({}, {}, {0.1f,0.2f,0.3f,0});
    close(render(renderer,slab(zeroExtinction,false)),{.1,.2,.3},2e-5,"Zero-extinction emission must integrate linearly");

    // These are physical coefficient controls, using an invisible boundary so
    // boundary BSDF approximations cannot hide a wrong material-to-VDF mapping.
    Material native=fog();native.volume.x=0;native.program.z=1;
    native.transmission={1,2,.2f,0};native.transmissionColor={.8f,.5f,.3f,0};
    close(render(renderer,slab(native)),{std::sqrt(.8),std::sqrt(.5),std::sqrt(.3)},2e-5,"OpenPBR color must specify extinction over attenuation depth");
    native.transmissionScatter={.4f,.1f,.05f,0};
    // Specified sigmaS=(.2,.05,.025); absorption before the specified gray
    // correction has a negative red component. This reference is the resulting
    // physical medium, evaluated independently via the explicit volume terminal.
    auto transControl=fog({0,.38500181f,.66541463f,0},{.2f,.05f,.025f,0});transControl.volume.y=.2f;
    auto transmissionMapped=render(renderer,slab(native),256);
    close(transmissionMapped,render(renderer,slab(transControl),256),3e-4,"OpenPBR scattering/depth and gray absorption correction must match physical volume coefficients");
    native.transmission.y=0;native.transmissionScatter={100,200,300,0};
    close(render(renderer,slab(native)),{1,1,1},2e-5,"Zero OpenPBR attenuation depth must create no interior absorption or scattering");

    Material calibrated=fog();calibrated.volume.x=0;calibrated.subsurface={1,1,.3f,0};
    calibrated.subsurfaceColor={.25f,.5f,.8f,0};calibrated.subsurfaceRadius={.4f,.6f,1,0};
    // High-precision van-de-Hulst inversion of observed albedo (.25,.5,.8),
    // with MFP (.4,.6,1), gives these independent physical coefficients.
    auto sssControl=fog({.59085466f,.10581166f,.00660713f,0},{1.90914534f,1.56085501f,.99339287f,0});sssControl.volume.y=.3f;
    auto calibrationReference=render(renderer,slab(sssControl),256);
    close(render(renderer,slab(calibrated),256),calibrationReference,3e-4,"Standard Surface SSS observed albedo and MFP must map to the calibrated physical medium");
    calibrated.program.z=1;
    close(render(renderer,slab(calibrated),256),calibrationReference,3e-4,"OpenPBR SSS observed albedo and MFP must map to the calibrated physical medium");
    calibrated.subsurfaceColor={0,0,0,0};calibrated.subsurfaceRadius={1,.5f,.25f,0};calibrated.subsurface.y=.5f;
    close(render(renderer,slab(calibrated)),{std::exp(-2.),std::exp(-4.),std::exp(-8.)},2e-5,"Black observed SSS color must absorb with reciprocal RGB mean free path");

    auto thinVolume=fog({.2f,.5f,1,0});thinVolume.subsurface.w=1;
    thinVolume.transmission={1,1,0,0};thinVolume.transmissionColor={.01f,.01f,.01f,0};
    thinVolume.subsurface.x=1;thinVolume.subsurfaceColor={.2f,.3f,.4f,0};
    close(render(renderer,slab(thinVolume)),{std::exp(-.2),std::exp(-.5),std::exp(-1.)},2e-5,"An independent volume terminal must survive thin-walled surface without adding thin-surface interior coefficients");

    // The film collapses to a small deterministic footprint straddling cube
    // face diagonals/edges, where adjacent triangles can disagree by a few ULPs
    // about one physical crossing. Duplicates must not push/pop twice.
    settings.environment={1,1,1,0};renderer.setSettings(settings);
    Scene diagonal;diagonal.materials={fog({.2f,.5f,1,0})};box(diagonal,0,-1,1,-1,1,-1,1);
    float diagonalLength=2*std::sqrt(2*4.73f*4.73f+4.19f*4.19f)/4.73f;
    close(render(renderer,diagonal,64,{{4.73f,4.73f,4.19f,0},{0,0,0,0},{0,0,1,0},.000001f}),
      {std::exp(-.2*diagonalLength),std::exp(-.5*diagonalLength),std::exp(-diagonalLength)},5e-5,
      "Coincident closed-cube triangle crossings must count as one physical boundary");
    float grazingLength=2*std::sqrt(2*100.f*100.f+.002f*.002f)/100.f;
    close(render(renderer,diagonal,64,{{100,100,1.001f,0},{0,0,.999f,0},{0,0,1,0},.0000001f}),
      {std::exp(-.2*grazingLength),std::exp(-.5*grazingLength),std::exp(-grazingLength)},1e-4,
      "Grazing cube edges must retain correct optical thickness through jittered rays");
    settings.environment={};renderer.setSettings(settings);
    auto thinBoundary=fog({200000,500000,1000000,0});
    close(render(renderer,slab(thinBoundary,true,.000001f),1,{{0,0,.0001f,0},{0,0,-1,0},{0,1,0,0},.05f}),
      {std::exp(-.2),std::exp(-.5),std::exp(-1.)},2e-5,
      "Numerical boundary deduplication must preserve an opposite-facing exit of a sub-epsilon thin volume");
    float oneUlp=std::nextafter(.5f,1.f)-.5f;
    Scene oneUlpVolume;oneUlpVolume.materials={fog({.2f/oneUlp,.5f/oneUlp,1.f/oneUlp,0})};
    box(oneUlpVolume,0,-3,3,-3,3,.5f-oneUlp,.5f);emitter(oneUlpVolume,-1);
    close(render(renderer,oneUlpVolume,1,{{0,0,1,0},{0,0,-1,0},{0,1,0,0},.000001f}),
      {std::exp(-.2),std::exp(-.5),std::exp(-1.)},2e-5,
      "Distinct front/back ray parameters one ULP apart must retain their finite optical depth");
    Scene edgeTouch;edgeTouch.materials={fog({1,2,3,0})};
    box(edgeTouch,0,2.5f,4.1f,.1f,1.7f,-3.5f,-1.9f);
    settings.environment={1,1,1,0};renderer.setSettings(settings);
    // Arrive at XMIN/ZMAX while moving out of X and into Z. The entry/exit
    // pair is co-located and the continuation is entirely outside the box.
    close(render(renderer,edgeTouch,64,{{8,9,12,0},{2.5f,1.154149f,-1.9f,0},{0,1,0,0},.0000001f}),
      {1,1,1},2e-5,"A zero-length null-volume edge touch must not create an interior medium");
    // This is the XMIN/ZMIN contact reported by a ground-to-light shadow ray
    // in the gallery. Adjacent faces can reconstruct the same edge at unequal
    // ray parameters; exact-t-only tangent detection does not cover this case.
    // Probe both neighboring representable targets as well as the edge itself.
    for(float x:std::array{std::nextafter(2.5f,0.f),2.5f,std::nextafter(2.5f,4.f)})
      for(float z:std::array{std::nextafter(-3.5f,-4.f),-3.5f,std::nextafter(-3.5f,-3.f)})
        close(render(renderer,edgeTouch,64,{{2.634398f,.000168f,-3.634558f,0},
          {x,.208113f,z,0},{0,1,0,0},.0000001f}),{1,1,1},2e-4,
          "Unequal-t reports at a shared cube edge must not leave an unmatched medium transition");
    Scene nestedTangent=edgeTouch;
    // White thermal equilibrium: the inner absorber emits sigmaA*Lwhite,
    // while the enclosing medium only scatters. Thus the exact radiance is
    // white even when a scattered path subsequently enters the inner cube.
    nestedTangent.materials[0].volumeEmission={1,2,3,0};
    auto outerPhase=fog({}, {.02f,.03f,.05f,0});outerPhase.volume.y=.35f;
    nestedTangent.materials.push_back(outerPhase);box(nestedTangent,1,-20,20,-20,20,-20,20);
    close(render(renderer,nestedTangent,512,{{8,9,12,0},{2.5f,1.154149f,-1.9f,0},{0,1,0,0},.0000001f}),
      {1,1,1},.06,"A zero-length inner boundary touch must preserve the enclosing scattering medium");
    settings.environment={};renderer.setSettings(settings);

    Material thinSurface;thinSurface.program.z=1;thinSurface.baseColor={0,0,0,1};thinSurface.surface.y=0;
    thinSurface.transmission.x=1;thinSurface.subsurface.w=1;thinSurface.specular={1,1.5f,0,0};
    auto twoThinSurfaces=render(renderer,slab(thinSurface),1024);
    // A lossless thin 1.5-IOR sheet has R_eff=2*.04/(1+.04)=1/13.
    // Two such sheets including inter-sheet bounces transmit (1-R)/(1+R)=6/7.
    close(twoThinSurfaces,{6./7,6./7,6./7},.015,"Two thin dielectric surfaces must retain both effective Fresnel interfaces");
    thinSurface.volume.x=1;
    close(render(renderer,slab(thinSurface),1024),twoThinSurfaces,2e-5,
      "An independent zero-extinction volume must not erase the second thin-sheet optical interface");

    Scene nested;nested.materials={fog({.2f,.4f,.8f,0}),fog({.8f,.3f,.1f,0})};
    box(nested,0,-4,4,-4,4,-2,2);box(nested,1,-1,1,-1,1,0,1);emitter(nested,-3);
    Camera inside{{0,0,.5f,0},{0,0,-3,0},{0,1,0,0},.05f};
    close(render(renderer,nested,1,inside),{std::exp(-.8*.5-.2*2),std::exp(-.3*.5-.4*2),std::exp(-.1*.5-.8*2)},2e-5,"Camera-inside nested media must initialize inner/outer optical depths");
    close(render(renderer,nested),{std::exp(-.8-.2*3),std::exp(-.3-.4*3),std::exp(-.1-.8*3)},2e-5,"Nested media must restore the outer coefficients after inner exit");
    nested.instances[1].transform[0]=-1;
    close(render(renderer,nested,1,inside),{std::exp(-.8*.5-.2*2),std::exp(-.3*.5-.4*2),std::exp(-.1*.5-.8*2)},2e-5,"Mirrored medium winding must preserve physical inside/outside");

    Scene sharedMaterial;sharedMaterial.materials={fog({.2f,.5f,1,0})};
    box(sharedMaterial,0,-4,4,-4,4,-2,2);box(sharedMaterial,0,-1,1,-1,1,0,1);emitter(sharedMaterial,-3);
    close(render(renderer,sharedMaterial),{std::exp(-.2*4),std::exp(-.5*4),std::exp(-4.)},2e-5,
      "Nested instances sharing one material must retain distinct boundary identities in the compact stack");
    close(render(renderer,sharedMaterial,1,inside),{std::exp(-.2*2.5),std::exp(-.5*2.5),std::exp(-2.5)},2e-5,
      "Shared-material camera containment must cache instance IDs rather than material IDs");
    Scene reordered=nested;
    std::swap(reordered.materials[0],reordered.materials[1]);
    reordered.meshes[0].material=1;reordered.meshes[1].material=0;
    close(render(renderer,reordered,1,inside),{std::exp(-.8*.5-.2*2),std::exp(-.3*.5-.4*2),std::exp(-.1*.5-.8*2)},2e-5,
      "Compact instance stacks must look up the actual material index after material ordering changes");

    // Keep one uploaded scene while moving between outside, inner and outer
    // regions. Cache invalidation must follow camera origin changes through
    // both camera APIs, while reset/rotation retain the same physical medium.
    // The interior absorption comes from a graph instead of the material POD,
    // so an IDs-only cache that bypasses graph outputs cannot pass this control.
    nested.materials[1].volumeAbsorption={};nested.materials[1].program={0,2,0,0};
    nested.materialInstructions={
      {{uint32_t(MaterialOp::Constant),0,0,0},{.8f,.3f,.1f,0}},
      {{uint32_t(MaterialOp::Output),0,uint32_t(MaterialParameter::VolumeAbsorption),0},{}}};
    settings.environment={1,1,1,0};renderer.setSettings(settings);
    renderer.setScene(nested);
    renderer.setCamera({{0,0,4,0},{0,0,-3,0},{0,1,0,0},.05f});renderer.render();
    close(average(renderer.readbackRaw()),{std::exp(-.8-.2*3),std::exp(-.3-.4*3),std::exp(-.1-.8*3)},2e-5,
      "Fresh camera-medium cache must preserve constant graph interior outputs");
    renderer.updateCamera(inside);renderer.render();
    const std::array<double,3> insideExpected={std::exp(-.8*.5-.2*2),std::exp(-.3*.5-.4*2),std::exp(-.1*.5-.8*2)};
    close(average(renderer.readbackRaw()),insideExpected,2e-5,
      "Interactive camera movement must refresh the shared containment cache");
    renderer.reset();renderer.render(2);
    close(average(renderer.readbackRaw()),insideExpected,2e-5,
      "Accumulation reset must preserve valid camera-inside containment and coefficients");
    renderer.updateCamera({{0,0,1.5f,0},{0,0,-3,0},{0,1,0,0},.05f});renderer.render();
    close(average(renderer.readbackRaw()),{std::exp(-.8-.2*2.5),std::exp(-.3-.4*2.5),std::exp(-.1-.8*2.5)},2e-5,
      "Leaving the inner medium must refresh the camera cache to its outer containing region");
    Camera reverseInside=inside;reverseInside.target={0,0,3,0};
    renderer.setCamera(reverseInside);renderer.render();
    close(average(renderer.readbackRaw()),{std::exp(-.8*.5-.2),std::exp(-.3*.5-.4),std::exp(-.1*.5-.8)},2e-5,
      "Camera cuts must refresh cached containment independently of the previous direction");
    reverseInside.verticalFov=.08f;renderer.updateCamera(reverseInside);renderer.render();
    close(average(renderer.readbackRaw()),{std::exp(-.8*.5-.2),std::exp(-.3*.5-.4),std::exp(-.1*.5-.8)},2e-5,
      "Camera rotation and film changes must use direction-independent cached containment");
    nested.materialInstructions[0].value={.4f,.7f,.9f,0};renderer.setScene(nested);renderer.render();
    close(average(renderer.readbackRaw()),{std::exp(-.4*.5-.2),std::exp(-.7*.5-.4),std::exp(-.9*.5-.8)},2e-5,
      "Scene material changes must invalidate fully evaluated camera-medium coefficients");
    settings.environment={};renderer.setSettings(settings);

    // With a tiny scattering coefficient the full RGB transport approaches the
    // analytic unscattered solution. This invokes channel-mixture free flight,
    // unlike the deterministic pure-absorption fast path above.
    auto spectral=fog({.2f,.5f,1,0},{.0001f,.0002f,.0003f,0});
    close(render(renderer,slab(spectral),512),{std::exp(-.2001),std::exp(-.5002),std::exp(-1.0003)},.025,"RGB free-flight mixture must retain spectral Beer attenuation");

    settings.environment={1,1,1,0};renderer.setSettings(settings);
    auto conservative=fog({}, {.2f,1,3,0});
    close(render(renderer,slab(conservative,false),512),{1,1,1},.06,"Conservative scattering must preserve a uniform white environment");
    conservative.volume.y=.7f;
    close(render(renderer,slab(conservative,false),512),{1,1,1},.06,"HG sampling/PDF must preserve the white furnace");
    // Large entry t makes short positive flights disappear when added to the
    // entry parameter. Translating the scene also stresses representable world
    // points at both sides of a volume; neither may create a second entry.
    auto distantVolume=slab(conservative,false);
    close(render(renderer,distantVolume,512,{{0,0,4096,0},{0,0,-1,0},{0,1,0,0},.0001f}),
      {1,1,1},.06,"Short free flights after a large entry distance must remain inside the current volume");
    for(auto& instance:distantVolume.instances) {
      instance.transform[3]=8192;instance.transform[7]=4096;instance.transform[11]=-16384;
    }
    close(render(renderer,distantVolume,512,{{8192,4096,-12288,0},{8192,4096,-16385,0},{0,1,0,0},.0001f}),
      {1,1,1},.06,"World-translated medium collisions must stay strictly inside their entry/exit planes");

    // Gallery regression: subtracting the instance translation can round a
    // phase collision one world-space ULP inside YMIN back onto its local
    // plane. A shadow or continuation ray pointing further inward can then
    // report a second zero-t entry for the medium already on the stack.
    // Only R/B scatter and only G is emitted, so the exact image is black;
    // NEE still traces its shadow segments before multiplying the RGB terms.
    settings.environment={};settings.maxBounces=1;renderer.setSettings(settings);
    Scene translatedPhase;translatedPhase.materials={fog({}, {10000000,0,10000000,0})};
    box(translatedPhase,0,-.8f,.8f,-.8f,.8f,-.8f,.8f);
    translatedPhase.instances[0].transform[3]=3.3f;
    translatedPhase.instances[0].transform[7]=.9f;
    translatedPhase.instances[0].transform[11]=-2.7f;
    Material greenEmitter;greenEmitter.baseColor={0,0,0,1};greenEmitter.specular.x=0;
    greenEmitter.emission={0,1,0,0};translatedPhase.materials.push_back(greenEmitter);
    Mesh phaseLight;phaseLight.material=1;
    face(phaseLight,{1.7495f,2.4f,-1.2628f,0},{1.7695f,2.4f,-1.2628f,0},
      {1.7695f,2.4f,-1.2428f,0},{1.7495f,2.4f,-1.2428f,0},{0,-1,0,0});
    add(translatedPhase,std::move(phaseLight));
    close(render(renderer,translatedPhase,128,{{3.843227f,.025f,-3.277152f,0},
      {3.843227f,.1f,-3.277152f,0},{0,0,1,0},.000001f}),{0,0,0},1e-7,
      "Phase NEE and continuation from a translated volume boundary must not create a zero-t duplicate entry");

    settings.environment={};settings.maxBounces=64;renderer.setSettings(settings);
    auto phase=fog({}, {1,1,1,0});phase.volume.y=-.7f;
    Scene backlit;backlit.materials.push_back(phase);box(backlit,0,-3,3,-3,3,0,2);emitter(backlit,-1,.75f);
    auto backward=render(renderer,backlit,512);
    backlit.materials[0].volume.y=.7f;
    auto forward=render(renderer,backlit,512);
    require(forward[0]>backward[0]*1.2,"Forward HG scattering must increase finite area back-light visibility (phase NEE plus MIS)");

    settings.environment={};settings.maxBounces=1;renderer.setSettings(settings);
    Scene receiver;Material matte;matte.baseColor={.7f,.7f,.7f,1};matte.specular.x=0;receiver.materials.push_back(matte);
    Mesh ground;ground.material=0;face(ground,{-3,-3,0,0},{3,-3,0,0},{3,3,0,0},{-3,3,0,0},{0,0,1,0});add(receiver,std::move(ground));
    Material area;area.emission={8,8,8,0};area.baseColor={0,0,0,1};area.specular.x=0;receiver.materials.push_back(area);
    Mesh light;light.material=1;face(light,{1.5f,-.7f,2,0},{1.5f,.7f,2,0},{2.9f,.7f,2,0},{2.9f,-.7f,2,0},{0,0,-1,0});add(receiver,std::move(light));
    auto reference=render(renderer,receiver,64);
    receiver.materials.push_back(fog({1,0,0,0}));box(receiver,2,-10,10,-10,10,.5f,1.5f);
    auto attenuated=render(renderer,receiver,64);
    require(reference[1]>.01&&std::abs(attenuated[1]-reference[1])<1e-5,"Zero-extinction green channel must preserve receiver illumination");
    require(attenuated[0]/attenuated[1]<.2,"Area-light shadow segments must include medium Beer attenuation as well as primary attenuation");

    // Reproduce the same near-edge contact through surface NEE (source=5),
    // rather than relying only on a pinhole ray or an exact equal-t contact.
    // The tiny emitter straddles the reported tangent direction from a matte
    // ground point below the volume. It gives both real short chords and
    // numerically split edge contacts across a deterministic set of samples.
    Scene edgeShadow;edgeShadow.materials={matte,area};
    Mesh edgeGround;edgeGround.material=0;
    face(edgeGround,{-10,0,-10,0},{-10,0,10,0},{10,0,10,0},{10,0,-10,0},{0,1,0,0});
    add(edgeShadow,std::move(edgeGround));
    constexpr float lightX=1.083341f,lightZ=-2.081657f,lightRadius=.0002f;
    edgeShadow.materials[1].emission={100000000,100000000,100000000,0};
    Mesh edgeLight;edgeLight.material=1;
    face(edgeLight,{lightX-lightRadius,2.4f,lightZ-lightRadius,0},
      {lightX+lightRadius,2.4f,lightZ-lightRadius,0},
      {lightX+lightRadius,2.4f,lightZ+lightRadius,0},
      {lightX-lightRadius,2.4f,lightZ+lightRadius,0},{0,-1,0,0});
    add(edgeShadow,std::move(edgeLight));
    for(bool reflected:std::array{false,true}) {
      Scene shadow=edgeShadow;
      if(reflected) {
        // Reflect the light/camera across the box midpoint to cover XMIN/ZMAX
        // as well as XMIN/ZMIN, including the emitter's mirrored winding.
        shadow.instances[1].transform[10]=-1;shadow.instances[1].transform[11]=-5.4f;
      }
      const float cameraZ=reflected?-1.765442f:-3.634558f;
      const Camera edgeGroundCamera={{2.634398f,.05f,cameraZ,0},
        {2.634398f,0,cameraZ,0},{0,0,1,0},.000001f};
      auto edgeShadowReference=render(renderer,shadow,512,edgeGroundCamera);
      shadow.materials.push_back(fog({1,0,3,0},{.1f,0,.2f,0}));
      box(shadow,2,2.5f,4.1f,.1f,1.7f,-3.5f,-1.9f);
      auto edgeShadowVolume=render(renderer,shadow,512,edgeGroundCamera);
      require(edgeShadowReference[1]>.01,"Near-edge shadow regression must receive finite area-light illumination");
      require(std::abs(edgeShadowVolume[1]-edgeShadowReference[1])<1e-5,
        "A zero-extinction channel must retain surface NEE through near-edge volume contacts");
      close(edgeShadowVolume,edgeShadowReference,.001,
        "Near-edge NEE must preserve illumination without inventing a finite interior segment");
    }

    settings.maxBounces=64;renderer.setSettings(settings);
    Material sss;sss.baseColor={0,0,0,1};sss.specular={1,1,0,0};sss.subsurface={1,.15f,0,0};sss.subsurfaceColor={.8f,.8f,.8f,0};
    auto thinWalk=render(renderer,slab(sss,true,.1f),1024);
    auto thickWalk=render(renderer,slab(sss,true,.5f),1024);
    require(thinWalk[0]>.05&&thickWalk[0]>0,"SSS random walk must carry back-light through a finite solid");
    require(thickWalk[0]<thinWalk[0]*.85,"SSS transport must respond to physical slab thickness");

    // A per-cell SSS graph must survive precomputation and produce finite,
    // nonzero transport. Its observed color varies with local depth; replacing
    // that graph by a bright literal gives a distinct physical random walk.
    auto spatialSss=slab(sss,true,.3f);spatialSss.materials[0].program={0,7,0,1};
    spatialSss.materialInstructions={
      {{uint32_t(MaterialOp::ObjectPosition),0,0,0},{}},
      {{uint32_t(MaterialOp::Swizzle),0,0,0},{2,2,2,2}},
      {{uint32_t(MaterialOp::Constant),0,0,0},{.5f,.3f,.2f,0}},
      {{uint32_t(MaterialOp::Multiply),1,2,0},{}},
      {{uint32_t(MaterialOp::Constant),0,0,0},{.2f,.3f,.4f,0}},
      {{uint32_t(MaterialOp::Add),3,4,0},{}},
      {{uint32_t(MaterialOp::Output),5,uint32_t(MaterialParameter::SubsurfaceColor),0},{}}};
    auto variableSss=render(renderer,spatialSss,1024);
    for(auto channel:variableSss)require(channel>0,"A spatial SSS graph must carry nonzero finite light through its solid");
    spatialSss.materialInstructions[5].code={uint32_t(MaterialOp::Constant),0,0,0};
    spatialSss.materialInstructions[5].value={.9f,.9f,.9f,0};
    auto brightGridSss=render(renderer,spatialSss,1024);
    require(std::abs(brightGridSss[0]-variableSss[0])>.01,
      "Changing the interior SSS graph must change transport instead of reusing frozen surface coefficients");

    Scene overlapping;overlapping.materials={fog()};
    box(overlapping,0,-2,2,-2,2,0,2);box(overlapping,0,-1,1,-1,1,-1,1);
    bool invalidNesting=false;
    try {render(renderer,overlapping);}catch(const std::exception& e) {
      invalidNesting=std::string(e.what()).find("Invalid medium boundary nesting")!=std::string::npos;
    }
    require(invalidNesting,"Numerical boundary deduplication must still reject genuinely overlapping non-nested media");

    // Solid contacts use the same edge filter as null volumes. Matched IOR
    // makes every boundary transmit, so an actual non-nested overlap must
    // still fail instead of being hidden by that contact classification.
    auto overlappingSolids=overlapping;
    Material matchedSolid;matchedSolid.factors.x=0;matchedSolid.specular={0,1,0,0};
    matchedSolid.transmission.x=1;overlappingSolids.materials[0]=matchedSolid;
    invalidNesting=false;
    try {render(renderer,overlappingSolids);}catch(const std::exception& e) {
      invalidNesting=std::string(e.what()).find("Invalid medium boundary nesting")!=std::string::npos;
    }
    require(invalidNesting,"Solid-glass edge handling must still reject genuinely overlapping non-nested solids");

    Scene excessive;excessive.materials={fog()};
    for(int i=0;i<17;++i) {float r=1+float(i)*.1f;box(excessive,0,-r,r,-r,r,-r,r);}
    bool rejected=false;
    try {render(renderer,excessive,1,{{0,0,0,0},{0,0,-3,0},{0,1,0,0},.05f});}catch(const std::exception&) {rejected=true;}
    require(rejected,"Medium nesting overflow must fail explicitly, never silently truncate GPU arrays");
    close(render(renderer,slab(fog({.2f,.5f,1,0}))),{std::exp(-.2),std::exp(-.5),std::exp(-1.)},2e-5,"Renderer must recover after explicit medium error");
    require(renderer.validationErrorCount()==0,"Vulkan validation errors in medium transport");
    std::cout<<"Medium transport passed: Beer/emission, nested/camera-inside/mirrored media, camera-cache invalidation, RGB channel sampling, HG furnace, medium shadows, SSS random walk, heterogeneous Beer/emission/position transforms/SSS and explicit stack overflow.\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
