#include "viewer_scene.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

namespace caustra::example {
namespace {
constexpr float pi = std::numbers::pi_v<float>;
Float4 vector(float x, float y, float z) { return {x,y,z,0}; }
Float4 add(Float4 a, Float4 b) { return vector(a.x+b.x, a.y+b.y, a.z+b.z); }
Float4 scale(Float4 a, float s) { return vector(a.x*s, a.y*s, a.z*s); }
Float4 cross(Float4 a, Float4 b) {
  return vector(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
Float4 normalize(Float4 a) { return scale(a, 1 / std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z)); }
void triangle(Mesh& mesh, uint32_t a, uint32_t b, uint32_t c) {
  const auto& p = mesh.vertices[a].position;
  const auto& q = mesh.vertices[b].position;
  const auto& r = mesh.vertices[c].position;
  const auto normal = cross(vector(q.x-p.x,q.y-p.y,q.z-p.z), vector(r.x-p.x,r.y-p.y,r.z-p.z));
  const auto shadingNormal = mesh.vertices[a].normal;
  if (normal.x*shadingNormal.x + normal.y*shadingNormal.y + normal.z*shadingNormal.z < 0)
    std::swap(b,c);
  mesh.indices.insert(mesh.indices.end(), {a,b,c});
}
void quad(Mesh& mesh, Float4 a, Float4 b, Float4 c, Float4 d, Float4 normal) {
  const auto first = static_cast<uint32_t>(mesh.vertices.size());
  for (const auto p : {a,b,c,d}) mesh.vertices.push_back({p,normal});
  triangle(mesh, first, first+1, first+2);
  triangle(mesh, first, first+2, first+3);
}
void addMesh(Scene& scene, Mesh mesh) {
  const auto index = static_cast<uint32_t>(scene.meshes.size());
  scene.meshes.push_back(std::move(mesh));
  scene.instances.push_back({index});
}

Mesh knot(uint32_t material) {
  Mesh mesh;
  mesh.material = material;
  constexpr uint32_t segments = 256, sides = 20;
  constexpr float radius = 0.64f, modulation = 0.24f, tube = 0.115f;
  float minY = std::numeric_limits<float>::max();
  for (uint32_t i = 0; i < segments; ++i) {
    const float t = 2*pi*float(i)/float(segments);
    const float radial = radius + modulation*std::cos(3*t);
    const float derivative = -3*modulation*std::sin(3*t);
    const auto center = vector(radial*std::cos(2*t), radial*std::sin(2*t), modulation*std::sin(3*t));
    const auto tangent = normalize(vector(derivative*std::cos(2*t)-2*radial*std::sin(2*t),
                                          derivative*std::sin(2*t)+2*radial*std::cos(2*t),
                                          3*modulation*std::cos(3*t)));
    // The XY derivative is always nonzero, so this frame is periodic and stable.
    const auto normal = normalize(cross(tangent, vector(0,0,1)));
    const auto binormal = cross(tangent, normal);
    for (uint32_t j = 0; j < sides; ++j) {
      const float angle = 2*pi*float(j)/float(sides);
      const auto n = add(scale(normal,std::cos(angle)), scale(binormal,std::sin(angle)));
      const auto position = add(center,scale(n,tube));
      minY = std::min(minY,position.y);
      mesh.vertices.push_back({position,n});
    }
  }
  // Sit on the pedestal without requiring a model file or a baked transform.
  for (auto& vertex : mesh.vertices) vertex.position.y += 0.22f-minY;
  for (uint32_t i = 0; i < segments; ++i) {
    for (uint32_t j = 0; j < sides; ++j) {
      const auto a = i*sides+j, b = ((i+1)%segments)*sides+j;
      const auto c = ((i+1)%segments)*sides+(j+1)%sides, d = i*sides+(j+1)%sides;
      triangle(mesh,a,b,c);
      triangle(mesh,a,c,d);
    }
  }
  return mesh;
}

Mesh cylinder(uint32_t material, float radius, float bottom, float top) {
  Mesh mesh;
  mesh.material = material;
  constexpr uint32_t sides = 96;
  for (uint32_t i = 0; i < sides; ++i) {
    const float a = 2*pi*float(i)/float(sides), b = 2*pi*float(i+1)/float(sides);
    const auto first = static_cast<uint32_t>(mesh.vertices.size());
    const auto na = vector(std::cos(a),0,std::sin(a)), nb = vector(std::cos(b),0,std::sin(b));
    mesh.vertices.push_back({vector(radius*na.x,bottom,radius*na.z),na});
    mesh.vertices.push_back({vector(radius*nb.x,bottom,radius*nb.z),nb});
    mesh.vertices.push_back({vector(radius*nb.x,top,radius*nb.z),nb});
    mesh.vertices.push_back({vector(radius*na.x,top,radius*na.z),na});
    triangle(mesh,first,first+1,first+2);
    triangle(mesh,first,first+2,first+3);
    const auto cap = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({vector(0,top,0),vector(0,1,0)});
    mesh.vertices.push_back({vector(radius*na.x,top,radius*na.z),vector(0,1,0)});
    mesh.vertices.push_back({vector(radius*nb.x,top,radius*nb.z),vector(0,1,0)});
    triangle(mesh,cap,cap+1,cap+2);
    const auto base = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({vector(0,bottom,0),vector(0,-1,0)});
    mesh.vertices.push_back({vector(radius*na.x,bottom,radius*na.z),vector(0,-1,0)});
    mesh.vertices.push_back({vector(radius*nb.x,bottom,radius*nb.z),vector(0,-1,0)});
    triangle(mesh,base,base+1,base+2);
  }
  return mesh;
}

Mesh light(uint32_t material, Float4 center, Float4 aim, float halfWidth, float halfHeight) {
  Mesh mesh;
  mesh.material = material;
  const auto normal = normalize(vector(aim.x-center.x,aim.y-center.y,aim.z-center.z));
  const auto u = scale(normalize(cross(vector(0,1,0),normal)),halfWidth);
  const auto v = scale(cross(normal,normalize(u)),halfHeight);
  quad(mesh,add(add(center,scale(u,-1)),scale(v,-1)),add(add(center,u),scale(v,-1)),
       add(add(center,u),v),add(add(center,scale(u,-1)),v),normal);
  return mesh;
}
}

Scene makeViewerScene() {
  Scene scene;
  scene.materials = {
    {{0.93f,0.48f,0.20f,1}, {}, {1,0.23f,0,0}}, // copper sculpture
    {{0.10f,0.12f,0.15f,1}, {}, {0.25f,0.38f,0,0}}, // graphite pedestal
    {{0.22f,0.24f,0.27f,1}, {}, {0,0.8f,0,0}}, // studio ground
    {{1,1,1,1}, {30,27,22,0}, {0,0.5f,0,0}},
    {{1,1,1,1}, {12,16,24,0}, {0,0.5f,0,0}},
    {{1,1,1,1}, {26,19,12,0}, {0,0.5f,0,0}},
  };
  addMesh(scene,knot(0));
  addMesh(scene,cylinder(1,1.12f,0.015f,0.22f));
  Mesh ground;
  ground.material = 2;
  quad(ground,vector(-40,0,40),vector(40,0,40),vector(40,0,-40),vector(-40,0,-40),vector(0,1,0));
  addMesh(scene,std::move(ground));
  const auto aim = vector(0,1.1f,0);
  // Elevated softboxes stay above the framing used for object inspection.
  addMesh(scene,light(3,vector(-3,7,3),aim,1.8f,1.8f));
  addMesh(scene,light(4,vector(4,6,1),aim,1.1f,1.8f));
  addMesh(scene,light(5,vector(0,6,-4),aim,1.7f,1.4f));
  return scene;
}

} // namespace caustra::example
