#include <caustra/renderer.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace caustra;
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}

void face(Mesh& mesh,Float4 a,Float4 b,Float4 c,Float4 d,Float4 normal,bool reverse=false) {
  const auto first=static_cast<uint32_t>(mesh.vertices.size());
  if(reverse)normal={-normal.x,-normal.y,-normal.z,0};
  mesh.vertices.insert(mesh.vertices.end(),{{a,normal,{0,0,0,0}},{b,normal,{1,0,0,0}},
                                          {c,normal,{1,1,0,0}},{d,normal,{0,1,0,0}}});
  for(auto index:reverse?std::array{0u,2u,1u,0u,3u,2u}:std::array{0u,1u,2u,0u,2u,3u})
    mesh.indices.push_back(first+index);
}

void cube(Mesh& mesh,float r,bool reverse) {
  face(mesh,{-r,-r,r,0},{r,-r,r,0},{r,r,r,0},{-r,r,r,0},{0,0,1,0},reverse);
  face(mesh,{r,-r,-r,0},{-r,-r,-r,0},{-r,r,-r,0},{r,r,-r,0},{0,0,-1,0},reverse);
  face(mesh,{r,-r,r,0},{r,-r,-r,0},{r,r,-r,0},{r,r,r,0},{1,0,0,0},reverse);
  face(mesh,{-r,-r,-r,0},{-r,-r,r,0},{-r,r,r,0},{-r,r,-r,0},{-1,0,0,0},reverse);
  face(mesh,{-r,r,r,0},{r,r,r,0},{r,r,-r,0},{-r,r,-r,0},{0,1,0,0},reverse);
  face(mesh,{-r,-r,-r,0},{r,-r,-r,0},{r,-r,r,0},{-r,-r,r,0},{0,-1,0,0},reverse);
}

std::array<double,3> average(const Image& image) {
  std::array<double,3> result{};
  for(const auto p:image.pixels) {
    require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&p.x>=0&&p.y>=0&&p.z>=0,
            "Nonfinite or negative surface-offset result");
    result[0]+=p.x;result[1]+=p.y;result[2]+=p.z;
  }
  for(auto& value:result)value/=image.pixels.size();
  return result;
}

Instance located(uint32_t mesh,float x=21.375f,float angle=0) {
  Instance instance{mesh};const float c=std::cos(angle),s=std::sin(angle);
  instance.transform={c,0,s,x,0,1,0,.828f,-s,0,c,6.508f};return instance;
}

void hollowShell(Renderer& renderer,float wall,bool rough,float translateX,float angle) {
  Scene scene;Material glass;glass.baseColor={0,0,0,1};glass.transmission.x=1;
  glass.specular={1,rough?1.5f:1.0f,0,0};glass.surface.y=rough?.17f:.02f;
  glass.transmission.y=.0005f;
  glass.transmissionColor=rough?Float4{1,1,1,0}:Float4{float(std::exp(-.25)),float(std::exp(-.125)),float(std::exp(-.0625)),0};
  scene.materials.push_back(glass);
  Mesh mesh;cube(mesh,.25f,false);cube(mesh,.25f-wall,true);
  scene.meshes.push_back(std::move(mesh));scene.instances.push_back(located(0,translateX,angle));
  const auto& transform=scene.instances[0].transform;
  const Float4 center{transform[3],transform[7],transform[11],0};
  const Float4 camera{center.x+transform[2],center.y+transform[6],center.z+transform[10],0};
  renderer.setScene(scene);renderer.setCamera({camera,center,{0,1,0,0},.01f});renderer.render(rough?512:32);
  const auto result=average(renderer.readbackRaw());
  const std::array sigma{500.0,250.0,125.0};
  for(size_t channel=0;channel<3;++channel) {
    const double expected=rough?1.0:std::exp(-sigma[channel]*2*wall);
    std::cout<<"Shell wall="<<wall<<" rough="<<rough<<" x="<<translateX<<" channel="<<channel
             <<" measured="<<result[channel]<<" expected="<<expected<<'\n';
    require(std::abs(result[channel]-expected)<.025,
            "Thin closed shell lost a boundary or changed its physical transmission");
  }
}

Scene shadowScene(unsigned blocker) {
  Scene scene;Material receiver;receiver.baseColor={.7f,.7f,.7f,1};receiver.specular.x=0;
  Material light;light.baseColor={0,0,0,1};light.specular.x=0;light.emission={8,8,8,0};light.surface.z=1;
  Material black;black.baseColor={0,0,0,1};black.specular.x=0;scene.materials={receiver,light,black};
  Mesh floor;face(floor,{-1,-1,0,0},{1,-1,0,0},{1,1,0,0},{-1,1,0,0},{0,0,1,0});
  scene.meshes.push_back(std::move(floor));scene.instances.push_back(located(0));
  Mesh emitter;emitter.material=1;
  face(emitter,{1,-.1f,.4f,0},{1,-.1f,.6f,0},{1,.1f,.6f,0},{1,.1f,.4f,0},{-1,0,0,0});
  scene.meshes.push_back(std::move(emitter));scene.instances.push_back(located(1));
  if(blocker) {
    Mesh occluder;occluder.material=2;
    if(blocker==1) {
      // A real blocker 100 um from the receiver. The old ~440 um upward spawn
      // jumps above its top, although the exact receiver-to-light rays hit it.
      face(occluder,{.0001f,-.2f,-.001f,0},{.0001f,-.2f,.00015f,0},
                    {.0001f,.2f,.00015f,0},{.0001f,.2f,-.001f,0},{-1,0,0,0});
    } else {
      // A second fixture isolates the *emitter* endpoint: a blocker 100 um in
      // front of its plane must survive target offset and shadow TMax trimming.
      face(occluder,{.9999f,-.2f,.3f,0},{.9999f,-.2f,.7f,0},
                    {.9999f,.2f,.7f,0},{.9999f,.2f,.3f,0},{-1,0,0,0});
    }
    scene.meshes.push_back(std::move(occluder));scene.instances.push_back(located(2));
  }
  return scene;
}

void verifyNee(Renderer& renderer) {
  RenderSettings settings;settings.width=8;settings.height=8;settings.maxBounces=1;settings.environment={};
  renderer.setSettings(settings);
  renderer.setCamera({{21.375f,.828f,7.508f,0},{21.375f,.828f,6.508f,0},{0,1,0,0},.000001f});
  for(unsigned blocker=0;blocker<3;++blocker) {
    renderer.setScene(shadowScene(blocker));renderer.render(512);
    const auto result=average(renderer.readbackRaw());
    std::cout<<"NEE blocker="<<blocker<<" radiance="<<result[0]<<'\n';
    if(blocker==0) require(result[0]>.01,"NEE offset fixture has no reference illumination");
    else for(const auto value:result)require(value<1e-7,"A surface/target offset skipped a nearby real shadow blocker");
  }
}

}

int main(int argc,char**) {
  try {
    Renderer renderer({argc>1});
    RenderSettings settings;settings.width=8;settings.height=8;settings.maxBounces=24;settings.environment={1,1,1,0};
    renderer.setSettings(settings);
    // Outer and inward-wound inner boxes form one valid closed hollow solid.
    // A skipped front wall leaves its instance on the stack in the air cavity;
    // the back wall then reports a duplicate entry. This catches the old spawn
    // deterministically, unlike a lone slab whose missing exit may just miss.
    hollowShell(renderer,.0002f,false,21.375f,0);
    hollowShell(renderer,.000321f,false,21.375f,.3f);
    hollowShell(renderer,.0002f,false,1000.0f,0);
    hollowShell(renderer,.000321f,true,21.375f,.3f);
    verifyNee(renderer);
    require(renderer.validationErrorCount()==0,"Vulkan validation errors");
    std::cout<<"Surface offsets preserve closed thin shells, reflection/transmission, and near-source/near-target NEE occlusion.\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
