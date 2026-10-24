#include <caustra/asset.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {
void require(bool condition,const char* message) {
  if(!condition) throw std::runtime_error(message);
}

caustra::Float4 instanceCenter(const caustra::Scene& scene,size_t index) {
  const auto& instance=scene.instances[index];
  const auto& vertices=scene.meshes[instance.mesh].vertices;
  require(!vertices.empty(),"Transport fixture has no vertices");
  auto minimum=vertices.front().position,maximum=minimum;
  for(const auto& v:vertices) {
    minimum.x=std::min(minimum.x,v.position.x);maximum.x=std::max(maximum.x,v.position.x);
    minimum.y=std::min(minimum.y,v.position.y);maximum.y=std::max(maximum.y,v.position.y);
    minimum.z=std::min(minimum.z,v.position.z);maximum.z=std::max(maximum.z,v.position.z);
  }
  const float x=(minimum.x+maximum.x)*.5f,y=(minimum.y+maximum.y)*.5f,z=(minimum.z+maximum.z)*.5f;
  const auto& t=instance.transform;
  return {t[0]*x+t[1]*y+t[2]*z+t[3],t[4]*x+t[5]*y+t[6]*z+t[7],t[8]*x+t[9]*y+t[10]*z+t[11],0};
}

void checkTransportImage(caustra::Renderer& renderer) {
  const auto image=renderer.readbackRaw();
  require(image.width==64 && image.height==64 && image.samples==64 && image.pixels.size()==64*64,
    "Transport regression returned an incomplete image");
  for(const auto& pixel:image.pixels)
    require(std::isfinite(pixel.x) && std::isfinite(pixel.y) && std::isfinite(pixel.z) &&
      pixel.x>=0 && pixel.y>=0 && pixel.z>=0,"Transport regression produced invalid HDR radiance");
  require(renderer.validationErrorCount()==0,"Transport regression reported a Vulkan validation error");
}

void transportSweep(caustra::Renderer& renderer,const caustra::Scene& scene,const char* variant) {
  renderer.setScene(scene);
  // This reported camera ray clips a faceted solid silhouette. A world-space
  // normal dot product can misclassify its first entry as an unmatched exit;
  // traversal can also report opposite-facing facets at the same contact.
  // Keep a range of finite footprints around the original narrow repro: the
  // old path reliably failed at sample 25 for the .001-degree footprint.
  for(float fov:std::array{.00001f,.0001f,.001f,.01f,.1f,1.f}) {
    std::cout<<"Transport "<<variant<<", FOV "<<fov<<'\n';
    renderer.setCamera({{5.796277f,7.531106f,-11.909305f,0},
      {.397207f,1.453877f,-.139794f,0},{0,1,0,0},fov});
    renderer.render(64);
    checkTransportImage(renderer);
  }
  // Rejecting a degenerate silhouette must not turn ordinary exits from a
  // camera genuinely inside the same solid into skipped boundary crossings.
  auto center=instanceCenter(scene,7);
  auto target=center;target.x+=1;target.y+=.2f;target.z+=.3f;
  std::cout<<"Transport "<<variant<<", camera inside\n";
  renderer.setCamera({center,target,{0,1,0,0},30});
  renderer.render(64);
  checkTransportImage(renderer);
}

void testTransport(const caustra::Scene& scene) {
  require(scene.instances.size()>7,"Gallery lost the solid silhouette regression instance");
  caustra::Renderer renderer({true});
  caustra::RenderSettings settings;settings.width=64;settings.height=64;settings.maxBounces=24;
  renderer.setSettings(settings);
  transportSweep(renderer,scene,"gallery");

  // Reflect the local X coordinate around the instance origin. The closed
  // sphere stays in place while the TLAS determinant becomes negative.
  auto mirrored=scene;
  for(size_t row=0;row<3;++row) mirrored.instances[7].transform[row*4]*=-1;
  transportSweep(renderer,mirrored,"mirrored solid");

  // Run the same contact through a solid random-walk SSS closure as well as
  // the gallery's authored transmission material. Clone only this instance's
  // mesh/material so every other gallery object retains its authored graph.
  auto subsurface=scene;
  const auto meshIndex=subsurface.instances[7].mesh;
  caustra::Material material;material.program.z=scene.materials[scene.meshes[meshIndex].material].program.z;
  material.baseColor={.8f,.5f,.2f,1};material.subsurface={1,.2f,.1f,0};
  material.subsurfaceColor={.8f,.65f,.4f,0};material.subsurfaceRadius={1,.5f,.25f,0};
  auto mesh=subsurface.meshes[meshIndex];mesh.material=static_cast<uint32_t>(subsurface.materials.size());
  subsurface.materials.push_back(material);
  subsurface.instances[7].mesh=static_cast<uint32_t>(subsurface.meshes.size());
  subsurface.meshes.push_back(std::move(mesh));
  transportSweep(renderer,subsurface,"solid SSS");
  std::cout<<"Feature transport passed: silhouette footprint sweep, mirrored solid, SSS and camera-inside exits.\n";
}
}

int main(int argc,char** argv) {
  try {
    const bool transport=argc==3 && std::string_view(argv[2])=="--transport";
    require(argc==2 || transport,"Usage: caustra_feature_asset_tests gallery.caustra [--transport]");
    const auto asset=caustra::loadAsset(argv[1]);
    require(asset.hasCamera,"Gallery camera is missing");
    bool glass=false,frosted=false,tinted=false,thinWalled=false,subsurface=false,fuzz=false,film=false,volume=false;
    for(const auto& m:asset.scene.materials) {
      if(m.transmission.x>0) {
        glass=glass || m.subsurface.w==0;
        frosted=frosted || m.surface.y>=0.1f;
        tinted=tinted || (std::min({m.transmissionColor.x,m.transmissionColor.y,m.transmissionColor.z})<0.9f && m.transmission.y>0);
        thinWalled=thinWalled || m.subsurface.w==1;
      }
      subsurface=subsurface || (m.subsurface.x>0 && m.subsurface.y>0 && m.subsurfaceRadius.x>0);
      fuzz=fuzz || (m.fuzz.x>0 && m.fuzz.y>0 && m.program.z==1);
      film=film || (m.thinFilm.x>0 && m.thinFilm.y>0 && m.thinFilm.z>1);
      volume=volume || (m.volume.x>0 && m.volume.z==1 && m.volumeScattering.x>0);
    }
    require(glass && frosted && tinted && thinWalled,"Gallery lost glass, absorption, roughness or thin-walled settings");
    require(subsurface && fuzz && film && volume,"Gallery lost subsurface, native fuzz, thin-film or volume coefficients");
    bool displaced=false;
    for(const auto& mesh:asset.scene.meshes) {
      const auto& color=asset.scene.materials[mesh.material].baseColor;
      if(std::abs(color.x-0.38f)>1e-5f || std::abs(color.y-0.18f)>1e-5f || std::abs(color.z-0.07f)>1e-5f) continue;
      if(mesh.indices.size()/3<3200) continue;
      // The displaced fixture is a planar square with tessellated interior
      // vertices. Interior vertices must move away from its original plane;
      // normals must also change, proving this is geometry rather than bump.
      float minHeight=1e30f,maxHeight=-1e30f;
      size_t interior=0;bool changedNormal=false;
      for(const auto& v:mesh.vertices) {
        if(v.texcoord.x>0.05f && v.texcoord.x<0.95f && v.texcoord.y>0.05f && v.texcoord.y<0.95f) {
          minHeight=std::min(minHeight,v.position.z);maxHeight=std::max(maxHeight,v.position.z);++interior;
          changedNormal=changedNormal || std::abs(v.normal.x)>0.02f || std::abs(v.normal.y)>0.02f;
        }
      }
      // Blender's source displacement tile lies in local XY. The imported
      // instance carries the Y-up conversion, leaving local vertex Z intact.
      displaced=displaced || (interior>100 && maxHeight-minHeight>0.02f && changedNormal);
    }
    require(displaced,"Gallery has no tessellated displacement geometry");
    std::cout<<"Feature asset passed: solid/tinted/frosted/thin glass, SSS, OpenPBR fuzz, thin film, participating volume and geometric displacement.\n";
    if(transport) testTransport(asset.scene);
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"Feature asset test: "<<error.what()<<'\n';return 1;
  }
}
