#include <caustra/renderer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }

caustra::Mesh quad(float x0,float x1,float y0,float y1,float z,uint32_t material) {
  caustra::Mesh mesh;
  mesh.material=material;
  mesh.vertices={
    {{x0,y0,z,0},{0,0,1,0},{0,0,0,0},{1,0,0,1}},
    {{x1,y0,z,0},{0,0,1,0},{1,0,0,0},{1,0,0,1}},
    {{x1,y1,z,0},{0,0,1,0},{1,1,0,0},{1,0,0,1}},
    {{x0,y1,z,0},{0,0,1,0},{0,1,0,0},{1,0,0,1}}};
  mesh.indices={0,1,2,0,2,3};
  return mesh;
}

caustra::Scene planeScene(caustra::Material material) {
  caustra::Scene scene;
  scene.materials={material};
  scene.meshes={quad(-100,100,-100,100,0,0)};
  scene.instances={caustra::Instance{0}};
  return scene;
}

caustra::Scene slabScene(caustra::Material material) {
  caustra::Scene scene;scene.materials={material};
  caustra::Mesh mesh;
  auto face=[&](std::array<caustra::Float4,4> p,caustra::Float4 n) {
    auto first=uint32_t(mesh.vertices.size());
    const std::array<caustra::Float4,4> uv{{{0,0,0,0},{1,0,0,0},{1,1,0,0},{0,1,0,0}}};
    for(size_t i=0;i<4;++i)mesh.vertices.push_back({p[i],n,uv[i],{}});
    for(auto i:{0u,1u,2u,0u,2u,3u})mesh.indices.push_back(first+i);
  };
  constexpr float e=100;
  face({{{-e,-e,0,0},{e,-e,0,0},{e,e,0,0},{-e,e,0,0}}},{0,0,1,0});
  face({{{-e,e,-1,0},{e,e,-1,0},{e,-e,-1,0},{-e,-e,-1,0}}},{0,0,-1,0});
  face({{{-e,-e,-1,0},{-e,-e,0,0},{-e,e,0,0},{-e,e,-1,0}}},{-1,0,0,0});
  face({{{e,e,-1,0},{e,e,0,0},{e,-e,0,0},{e,-e,-1,0}}},{1,0,0,0});
  face({{{-e,-e,-1,0},{e,-e,-1,0},{e,-e,0,0},{-e,-e,0,0}}},{0,-1,0,0});
  face({{{-e,e,0,0},{e,e,0,0},{e,e,-1,0},{-e,e,-1,0}}},{0,1,0,0});
  scene.meshes.push_back(std::move(mesh));scene.instances.push_back({0});return scene;
}

std::array<double,3> centroid(const caustra::Image& image) {
  std::array<double,3> moment{},energy{};
  for(uint32_t y=0;y<image.height;++y)for(uint32_t x=0;x<image.width;++x) {
    const auto p=image.pixels[size_t(y)*image.width+x];
    const std::array v{double(p.x),double(p.y),double(p.z)};
    for(size_t c=0;c<3;++c){energy[c]+=v[c];moment[c]+=v[c]*x;}
  }
  for(size_t c=0;c<3;++c){require(energy[c]>0,"Dispersive target is black");moment[c]/=energy[c];}
  return moment;
}

std::array<double,3> mean(const caustra::Image& image) {
  std::array<double,3> result{};
  for(auto pixel:image.pixels) {
    require(std::isfinite(pixel.x)&&std::isfinite(pixel.y)&&std::isfinite(pixel.z),"Nonfinite Standard Surface output");
    require(pixel.x>=0&&pixel.y>=0&&pixel.z>=0,"Negative Standard Surface radiance");
    result[0]+=pixel.x;result[1]+=pixel.y;result[2]+=pixel.z;
  }
  for(auto& channel:result) channel/=image.pixels.size();
  return result;
}

caustra::Image render(caustra::Renderer& renderer,const caustra::Scene& scene,uint32_t samples) {
  renderer.setScene(scene);renderer.render(samples);auto image=renderer.readbackRaw();mean(image);return image;
}
}

int main(int argc,char**) {
  try {
    caustra::Renderer renderer({argc>1});
    caustra::RenderSettings settings;settings.width=32;settings.height=32;settings.maxBounces=1;settings.environment={1,1,1,0};
    renderer.setSettings(settings);
    caustra::Camera camera;camera.position={0,0,4,0};camera.target={0,0,0,0};camera.up={0,1,0,0};camera.verticalFov=20;
    renderer.setCamera(camera);

    // A single isolated surface in a uniform environment measures directional
    // hemispherical reflectance. There are no area-light MIS or further bounces.
    caustra::Material material;material.baseColor={.8f,.6f,.3f,1};material.specular.x=0;
    auto lambert=mean(render(renderer,planeScene(material),4));
    require(std::abs(lambert[0]-.8)<.0001&&std::abs(lambert[1]-.6)<.0001&&std::abs(lambert[2]-.3)<.0001,"Lambert white furnace is not equal to its albedo");
    material.factors.x=.25f;
    auto weighted=mean(render(renderer,planeScene(material),4));
    require(std::abs(weighted[0]-.2)<.0001,"Standard Surface base weight was ignored or applied twice");
    material.factors.x=1;material.factors.y=.7f;
    auto roughDiffuse=mean(render(renderer,planeScene(material),128));
    require(roughDiffuse[0]<lambert[0]*.8&&roughDiffuse[0]>lambert[0]*.4,"Oren-Nayar diffuse roughness has the wrong response");

    material.factors={0,0,1,0};material.specular={1,1.3f,0,0};material.surface.y=.35f;
    auto lowIor=mean(render(renderer,planeScene(material),128));
    material.specular.y=2;
    auto highIor=mean(render(renderer,planeScene(material),128));
    require(highIor[0]>lowIor[0]*2,"Dielectric IOR does not control reflectivity");

    material.factors.x=1;material.surface.x=1;material.baseColor={.95f,.64f,.22f,1};material.specular.z=.65f;
    auto metal=mean(render(renderer,planeScene(material),128));
    require(metal[0]>metal[1]&&metal[1]>metal[2]&&metal[0]<1.2,"Complex-IOR metal furnace has wrong tint or excessive energy");
    material.surface.x=.35f;material.coat={.8f,.3f,1.8f,0};material.coatColor={.6f,1,.7f,0};
    material.sheen={.7f,.2f,0,0};material.sheenColor={1,.6f,.3f,0};
    auto layeredScene=planeScene(material);
    layeredScene.materialInstructions={
      {{uint32_t(caustra::MaterialOp::Constant),0,0,0},{.4f,.2f,1,0}},
      {{uint32_t(caustra::MaterialOp::Output),0,uint32_t(caustra::MaterialParameter::CoatNormal),0},{}}};
    layeredScene.materials[0].program={0,2,0,0};
    auto layered=mean(render(renderer,layeredScene,256));
    for(auto channel:layered) require(channel>0&&channel<1.25,"Layered coat/sheen furnace is invalid");

    // Opposing 82-degree normals expose reflected GGX proposals below their
    // own shading hemisphere but above Ng. Such proposals must be absorption:
    // their density is zero in surfaceGgxPdf, even if another lobe has a value.
    // Reference directly integrates the official ShaderGen provider over the
    // geometric hemisphere, bypassing closure construction and the sampler.
    // Reproduce with generated_material_tests --integrate-tilted:
    // 65536 / 262144 directions agree within 5e-6 in each channel.
    auto tiltedMaterial=caustra::Material{};tiltedMaterial.baseColor={.8f,.6f,.3f,1};tiltedMaterial.surface.y=.8f;
    tiltedMaterial.specular.x=0;tiltedMaterial.coat={1,.8f,2.5f,0};
    auto tiltedScene=planeScene(tiltedMaterial);
    tiltedScene.materialInstructions={
      {{uint32_t(caustra::MaterialOp::Constant),0,0,0},{.99f,0,.14106736f,0}},
      {{uint32_t(caustra::MaterialOp::Output),0,uint32_t(caustra::MaterialParameter::Normal),0},{}},
      {{uint32_t(caustra::MaterialOp::Constant),0,0,0},{-.99f,0,.14106736f,0}},
      {{uint32_t(caustra::MaterialOp::Output),2,uint32_t(caustra::MaterialParameter::CoatNormal),0},{}}};
    tiltedScene.materials[0].program={0,4,0,0};
    camera.verticalFov=1;renderer.setCamera(camera);
    auto tilted=mean(render(renderer,tiltedScene,1024));
    std::cout<<"Tilted base/coat furnace RGB: "<<tilted[0]<<','<<tilted[1]<<','<<tilted[2]<<'\n';
    require(std::abs(tilted[0]-.435242762593)<.0035&&std::abs(tilted[1]-.342283699616)<.0035&&std::abs(tilted[2]-.202845129575)<.0035,
      "Tilted base/coat GGX sampler disagrees with the independently integrated complete BSDF");
    camera.verticalFov=20;renderer.setCamera(camera);

    // Rotate anisotropy relative to an off-axis area light. Uniform lighting alone
    // cannot expose a forgotten tangent rotation, so this uses actual direct MIS.
    settings.environment={0,0,0,0};renderer.setSettings(settings);
    material=caustra::Material{};material.baseColor={.8f,.6f,.3f,1};material.surface={1,.35f,0,0};material.specular.z=.8f;
    auto anisotropicScene=planeScene(material);
    caustra::Material emitter;emitter.emission={12,12,12,0};anisotropicScene.materials.push_back(emitter);
    anisotropicScene.meshes.push_back(quad(1.0f,1.3f,-.8f,.8f,2,1));anisotropicScene.instances.push_back(caustra::Instance{1});
    auto unrotated=render(renderer,anisotropicScene,64);
    anisotropicScene.materials[0].specular.w=.25f;
    auto rotated=render(renderer,anisotropicScene,64);
    double difference=0;
    for(size_t i=0;i<rotated.pixels.size();++i) difference+=std::abs(rotated.pixels[i].x-unrotated.pixels[i].x);
    require(difference/rotated.pixels.size()>.001,"Anisotropy rotation has no lighting effect");

    // OpenPBR is a separate model, not a renamed Standard Surface. Its compensated
    // rough diffuse preserves a white furnace, and specular_weight scales metal.
    settings.environment={1,1,1,0};renderer.setSettings(settings);
    material=caustra::Material{};material.program.z=1;material.baseColor={1,1,1,1};material.specular.x=0;material.factors.y=.8f;
    auto openPbrWhite=mean(render(renderer,planeScene(material),128));
    require(std::abs(openPbrWhite[0]-1)<.03,"OpenPBR compensated rough diffuse lost white-furnace energy");
    material.factors.y=0;material.baseColor={.8f,.6f,.3f,1};material.surface={1,.35f,0,0};material.specular={.25f,1.5f,.5f,0};
    auto weakMetal=mean(render(renderer,planeScene(material),128));
    material.specular.x=.75f;
    auto strongMetal=mean(render(renderer,planeScene(material),128));
    require(std::abs(strongMetal[0]-weakMetal[0]*3)<.001,"OpenPBR specular weight does not scale the complete metal lobe");
    material.surface.x=0;material.specular.x=1;material.baseColor={.5f,.3f,.1f,1};material.coat={1,.3f,1.6f,0};
    // OpenPBR's default coat_darkening is 1; explicitly disable it for the
    // control so this comparison measures two different material closures.
    material.factors.w=0;
    auto noDarkening=mean(render(renderer,planeScene(material),128));
    material.factors.w=1;
    auto darkened=mean(render(renderer,planeScene(material),128));
    std::cout<<"OpenPBR coat darkening (0 -> 1), mean red: "<<noDarkening[0]<<" -> "<<darkened[0]<<'\n';
    require(darkened[0]<noDarkening[0]*.9&&darkened[0]>0,"OpenPBR coat darkening graph is missing");

    // Explicit tangent outputs must affect native OpenPBR's anisotropy frame.
    settings.environment={0,0,0,0};renderer.setSettings(settings);
    anisotropicScene.materials[0].program={0,0,1,0};anisotropicScene.materials[0].specular.w=0;
    auto nativeTangentX=render(renderer,anisotropicScene,64);
    anisotropicScene.materialInstructions={
      {{uint32_t(caustra::MaterialOp::Constant),0,0,0},{0,1,0,0}},
      {{uint32_t(caustra::MaterialOp::Output),0,21,0},{}}};
    anisotropicScene.materials[0].program.y=2;
    auto nativeTangentY=render(renderer,anisotropicScene,64);
    difference=0;
    for(size_t i=0;i<nativeTangentY.pixels.size();++i) difference+=std::abs(nativeTangentY.pixels[i].x-nativeTangentX.pixels[i].x);
    require(difference/nativeTangentY.pixels.size()>.001,"Explicit OpenPBR tangent does not rotate the highlight");

    auto& coatMaterial=anisotropicScene.materials[0];coatMaterial.baseColor={0,0,0,1};coatMaterial.surface.x=0;coatMaterial.specular.x=0;
    coatMaterial.coat={1,.35f,1.6f,.8f};coatMaterial.program.y=0;
    auto nativeCoatTangentX=render(renderer,anisotropicScene,64);
    anisotropicScene.materialInstructions[1].code.z=22;coatMaterial.program.y=2;
    auto nativeCoatTangentY=render(renderer,anisotropicScene,64);
    difference=0;
    for(size_t i=0;i<nativeCoatTangentY.pixels.size();++i) difference+=std::abs(nativeCoatTangentY.pixels[i].x-nativeCoatTangentX.pixels[i].x);
    require(difference/nativeCoatTangentY.pixels.size()>.001,"Explicit OpenPBR coat tangent does not rotate the coat highlight");

    material=caustra::Material{};material.program.z=1;material.emission={2,3,4,0};material.baseColor={0,0,0,1};material.specular.x=0;
    material.coat={1,.2f,1.5f,0};material.coatColor={.5f,.6f,.7f,0};
    auto faceEmission=mean(render(renderer,planeScene(material),4));
    require(std::abs(faceEmission[0]-.96)<.005&&std::abs(faceEmission[1]-1.728)<.005&&std::abs(faceEmission[2]-2.688)<.005,"OpenPBR normal-incidence coated emission is incorrect");
    camera.position={4,0,.4f,0};camera.verticalFov=4;renderer.setCamera(camera);
    auto grazingEmission=mean(render(renderer,planeScene(material),4));
    require(grazingEmission[0]<faceEmission[0]*.65,"OpenPBR coated EDF lacks grazing-angle attenuation");

    // The incoherent parallel sheet sums its internal reflections: R=2F/(1+F).
    // With IOR 1.5, R=1/13, so a tinted sheet has R+(1-R)*tint in a white furnace.
    settings.environment={1,1,1,0};settings.maxBounces=1;renderer.setSettings(settings);
    camera={{0,0,4,0},{0,0,0,0},{0,1,0,0},1};renderer.setCamera(camera);
    material=caustra::Material{};material.factors.x=0;material.transmission.x=1;
    material.surface.y=.02f;material.subsurface.w=1;material.transmissionColor={.2f,.5f,.9f,0};
    auto sheet=mean(render(renderer,planeScene(material),512));
    const std::array<double,3> expectedSheet{1.0/13+12.0/13*.2,1.0/13+12.0/13*.5,1.0/13+12.0/13*.9};
    for(size_t c=0;c<3;++c)require(std::abs(sheet[c]-expectedSheet[c])<.008,"Thin-walled reflection/transmission does not conserve the parallel-sheet Fresnel energy");

    // OpenPBR specular weight multiplies F0, then the resulting IOR drives both
    // reflection and refraction. F0=.04*3=.12 gives sheet R=.24/1.12=3/14.
    auto nativeInterface=material;nativeInterface.program.z=1;nativeInterface.specular.x=3;
    nativeInterface.transmissionColor={0,0,0,0};
    auto weightedInterface=mean(render(renderer,planeScene(nativeInterface),512));
    require(std::abs(weightedInterface[0]-3.0/14.0)<.006,"OpenPBR specular weight does not modulate the dielectric interface IOR");

    // A closed unit-thickness slab cancels entry/exit radiance eta factors.
    settings.maxBounces=12;renderer.setSettings(settings);
    material.subsurface.w=0;material.transmissionColor={1,1,1,0};
    auto clearSlab=mean(render(renderer,slabScene(material),256));
    require(std::abs(clearSlab[0]-1)<.035,"Closed glass slab loses energy or applies radiance eta twice");
    material.transmission.y=1;material.transmissionColor={.25f,.5f,.8f,0};
    auto absorbed=mean(render(renderer,slabScene(material),512));
    const std::array<double,3> attenuation{.25,.5,.8};
    for(size_t c=0;c<3;++c) {
      const double expected=.04+.9216*attenuation[c]/(1-.0016*attenuation[c]*attenuation[c]);
      require(std::abs(absorbed[c]-expected)<.025,"Glass slab does not match Beer-Lambert attenuation with Fresnel internal reflections");
    }

    // At this camera-inside angle both broad slab faces are beyond the critical
    // angle. A matched-index control exits, while glass undergoes total reflection.
    camera={{0,0,-.5f,0},{.9f,0,-.06411f,0},{0,1,0,0},1};renderer.setCamera(camera);
    material.transmission.y=0;material.transmissionColor={1,1,1,0};
    auto tir=mean(render(renderer,slabScene(material),256));
    material.specular.y=1;
    auto matched=mean(render(renderer,slabScene(material),64));
    require(tir[0]<.02&&matched[0]>.98,"Camera-inside dielectric boundary fails total internal reflection or matched-IOR null transmission");

    // OpenPBR tints exterior reflection only. From inside a clear n=1.5 slab,
    // white outside radiance is n^2=2.25 in every channel regardless of the
    // authored exterior specular tint, including paths with internal reflection.
    camera.target={0,0,0,0};renderer.setCamera(camera);
    auto internalTint=material;internalTint.program.z=1;internalTint.specular={1,1.5f,0,0};internalTint.specularColor={.2f,.5f,.9f,0};
    auto insideColored=mean(render(renderer,slabScene(internalTint),512));
    internalTint.specularColor={1,1,1,0};
    auto insideWhite=mean(render(renderer,slabScene(internalTint),512));
    for(size_t c=0;c<3;++c)require(std::abs(insideColored[c]-insideWhite[c])<.02&&std::abs(insideWhite[c]-2.25)<.04,
      "OpenPBR exterior specular color incorrectly tints internal reflection");

    // Reference Zeltner Gaussian-fit directional albedo at N.V=1, roughness=.5:
    // 0.0248686480157271 (MaterialX 1.39.5). Pure fuzz samples its exact LTC PDF.
    camera={{0,0,4,0},{0,0,0,0},{0,1,0,0},1};renderer.setCamera(camera);
    settings.maxBounces=1;renderer.setSettings(settings);
    material=caustra::Material{};material.program.z=1;material.baseColor={0,0,0,1};material.factors.x=0;material.specular.x=0;
    material.fuzz={1,.5f,0,0};material.fuzzColor={.8f,.4f,.1f,0};
    auto fuzz=mean(render(renderer,planeScene(material),64));
    require(std::abs(fuzz[0]-.0248686480157271*.8)<.0002&&std::abs(fuzz[1]-fuzz[0]*.5)<.00001,"Native OpenPBR Zeltner fuzz disagrees with the reference directional albedo");

    // Canonical OpenPBR thin-SSS albedos: E_R=S*(1-g)/2, E_T=S*(1+g)/2.
    // Black blockers isolate each hemisphere while preserving the white source
    // environment in the other; IOR=1 removes the outer Fresnel layer.
    material=caustra::Material{};material.program.z=1;material.specular={0,1,0,0};material.factors.x=0;
    material.subsurface={1,1,.3f,1};material.subsurfaceColor={.8f,.4f,.2f,0};
    caustra::Material black;black.baseColor={0,0,0,1};black.specular.x=0;
    auto hemisphereScene=planeScene(material);hemisphereScene.materials.push_back(black);
    hemisphereScene.meshes.push_back(quad(-1000,1000,-1000,1000,-1,1));hemisphereScene.instances.push_back({1});
    camera.position={0,0,.5f,0};renderer.setCamera(camera);
    auto thinSssReflection=mean(render(renderer,hemisphereScene,512));
    hemisphereScene.meshes[1]=quad(-1000,1000,-1000,1000,1,1);
    auto thinSssTransmission=mean(render(renderer,hemisphereScene,512));
    const std::array<double,3> sssColor{.8,.4,.2};
    for(size_t c=0;c<3;++c) {
      require(std::abs(thinSssReflection[c]-sssColor[c]*.35)<.006,"OpenPBR thin-SSS reflection differs from single-color S*(1-g)/2");
      require(std::abs(thinSssTransmission[c]-sssColor[c]*.65)<.006,"OpenPBR thin-SSS transmission differs from single-color S*(1+g)/2");
    }
    camera.position={0,0,4,0};renderer.setCamera(camera);

    material=caustra::Material{};material.program.z=1;material.surface={1,.25f,0,0};material.baseColor={.8f,.6f,.3f,1};material.thinFilm={1,300,1.35f,0};
    auto film300=mean(render(renderer,planeScene(material),256));
    material.thinFilm.y=600;
    auto film600=mean(render(renderer,planeScene(material),256));
    double filmDifference=0;for(size_t c=0;c<3;++c)filmDifference+=std::abs(film300[c]-film600[c]);
    require(filmDifference>.02,"Spectral Airy thin film has no wavelength-dependent thickness response");

    // A narrow white emitter viewed obliquely through a finite slab separates
    // hero wavelengths by refractive lateral displacement. No texture color is used.
    settings.width=64;settings.height=32;settings.maxBounces=8;settings.environment={};renderer.setSettings(settings);
    camera={{0,0,3,0},{.9f,0,-1,0},{0,1,0,0},16};renderer.setCamera(camera);
    material=caustra::Material{};material.factors.x=0;material.transmission.x=1;material.surface.y=.02f;
    auto spectralScene=slabScene(material);caustra::Material white;white.emission={4,4,4,0};white.baseColor={0,0,0,1};white.specular.x=0;
    spectralScene.materials.push_back(white);spectralScene.meshes.push_back(quad(1.0f,1.2f,-100,100,-3,1));spectralScene.instances.push_back({1});
    auto neutralCentroid=centroid(render(renderer,spectralScene,256));
    spectralScene.materials[0].thinFilm.w=.2f;
    auto dispersedCentroid=centroid(render(renderer,spectralScene,512));
    require(std::abs(neutralCentroid[0]-neutralCentroid[2])<.03,"Nondispersive glass introduced spectral separation");
    require(std::abs(dispersedCentroid[0]-dispersedCentroid[2])>.08,"Hero-channel Cauchy dispersion does not separate refracted wavelengths");
    spectralScene.materials[0].thinFilm.w=0;spectralScene.materials[0].program.z=1;
    spectralScene.materials[0].specular={0,2.4f,0,0};
    auto zeroWeight=render(renderer,spectralScene,16);
    spectralScene.materials[0].specular.y=1;
    auto zeroWeightMatched=render(renderer,spectralScene,16);
    double zeroWeightDifference=0;
    for(size_t i=0;i<zeroWeight.pixels.size();++i)zeroWeightDifference+=std::abs(zeroWeight.pixels[i].x-zeroWeightMatched.pixels[i].x);
    require(zeroWeightDifference/zeroWeight.pixels.size()<1e-6,"OpenPBR zero specular weight still bends light like the unmodulated IOR");
    std::cout<<"Transport checks: sheet "<<sheet[0]<<", clear slab "<<clearSlab[0]<<", Beer RGB "<<absorbed[0]<<','<<absorbed[1]<<','<<absorbed[2]
      <<", TIR "<<tir[0]<<", matched IOR "<<matched[0]<<", fuzz "<<fuzz[0]<<", thin SSS R/T "<<thinSssReflection[0]<<'/'<<thinSssTransmission[0]<<", film difference "<<filmDifference
      <<", dispersion red/blue centroid "<<dispersedCentroid[0]<<'/'<<dispersedCentroid[2]<<'\n';

    require(renderer.validationErrorCount()==0,"Vulkan validation errors in Standard Surface integration");
    std::cout<<"Surface models passed: Standard Surface furnace/layers/aniso; native OpenPBR compensated diffuse, metal weight, coat darkening, explicit base/coat tangent and coated EDF.\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
