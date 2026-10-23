#include "demo_scene.hpp"

#include <cmath>
#include <numbers>
#include <utility>

namespace caustra::example {
namespace {
Float4 vector(float x, float y, float z) { return {x, y, z, 0}; }
void triangle(Mesh& mesh, uint32_t a, uint32_t b, uint32_t c) {
  const auto& p = mesh.vertices[a].position;
  const auto& q = mesh.vertices[b].position;
  const auto& r = mesh.vertices[c].position;
  const Float4 normal = mesh.vertices[a].normal;
  const float ux = q.x-p.x, uy = q.y-p.y, uz = q.z-p.z;
  const float vx = r.x-p.x, vy = r.y-p.y, vz = r.z-p.z;
  if ((uy*vz-uz*vy)*normal.x + (uz*vx-ux*vz)*normal.y + (ux*vy-uy*vx)*normal.z < 0) std::swap(b, c);
  mesh.indices.insert(mesh.indices.end(), {a, b, c});
}
void quad(Mesh& mesh, Float4 a, Float4 b, Float4 c, Float4 d, Float4 normal) {
  const auto first = static_cast<uint32_t>(mesh.vertices.size());
  for (const auto& p : {a,b,c,d}) mesh.vertices.push_back({p, normal});
  triangle(mesh, first, first+1, first+2);
  triangle(mesh, first, first+2, first+3);
}
Mesh sphere(uint32_t material) {
  Mesh mesh;
  mesh.material = material;
  constexpr uint32_t sectors = 48, rings = 24;
  mesh.vertices.push_back({vector(0,1,0), vector(0,1,0)});
  for (uint32_t ring = 1; ring < rings; ++ring) {
    const float phi = std::numbers::pi_v<float> * float(ring) / float(rings);
    for (uint32_t sector = 0; sector < sectors; ++sector) {
      const float theta = 2 * std::numbers::pi_v<float> * float(sector) / float(sectors);
      const auto p = vector(std::sin(phi)*std::cos(theta), std::cos(phi), std::sin(phi)*std::sin(theta));
      mesh.vertices.push_back({p, p});
    }
  }
  const auto bottom = static_cast<uint32_t>(mesh.vertices.size());
  mesh.vertices.push_back({vector(0,-1,0), vector(0,-1,0)});
  for (uint32_t sector = 0; sector < sectors; ++sector) {
    const auto next = (sector+1) % sectors;
    triangle(mesh, 0, 1+sector, 1+next);
    for (uint32_t ring = 0; ring < rings-2; ++ring) {
      const auto a = 1 + ring*sectors + sector, b = 1 + ring*sectors + next;
      const auto c = a + sectors, d = b + sectors;
      triangle(mesh, a, b, d);
      triangle(mesh, a, d, c);
    }
    triangle(mesh, bottom, 1+(rings-2)*sectors+sector, 1+(rings-2)*sectors+next);
  }
  return mesh;
}
Mesh box(uint32_t material) {
  Mesh mesh;
  mesh.material = material;
  constexpr float a = -0.5f, b = 0.5f;
  quad(mesh, vector(a,a,b), vector(b,a,b), vector(b,b,b), vector(a,b,b), vector(0,0,1));
  quad(mesh, vector(b,a,a), vector(a,a,a), vector(a,b,a), vector(b,b,a), vector(0,0,-1));
  quad(mesh, vector(a,a,a), vector(a,a,b), vector(a,b,b), vector(a,b,a), vector(-1,0,0));
  quad(mesh, vector(b,a,b), vector(b,a,a), vector(b,b,a), vector(b,b,b), vector(1,0,0));
  quad(mesh, vector(a,b,b), vector(b,b,b), vector(b,b,a), vector(a,b,a), vector(0,1,0));
  quad(mesh, vector(a,a,a), vector(b,a,a), vector(b,a,b), vector(a,a,b), vector(0,-1,0));
  return mesh;
}
void addMesh(Scene& scene, Mesh mesh) {
  const auto index = static_cast<uint32_t>(scene.meshes.size());
  scene.meshes.push_back(std::move(mesh));
  scene.instances.push_back({index});
}
}

Scene makeDemoScene() {
  Scene scene;
  scene.materials = {
    {{0.73f,0.73f,0.73f,1}, {}, {0,0.7f,0,0}},
    {{0.63f,0.07f,0.05f,1}, {}, {0,0.7f,0,0}},
    {{0.07f,0.48f,0.13f,1}, {}, {0,0.7f,0,0}},
    {{0.95f,0.74f,0.35f,1}, {}, {1,0.22f,0,0}},
    {{1,1,1,1}, {16,14,11,0}, {0,0.5f,0,0}}
  };
  Mesh room;
  room.material = 0;
  quad(room, vector(-1.2f,0,1.2f), vector(1.2f,0,1.2f), vector(1.2f,0,-1.2f), vector(-1.2f,0,-1.2f), vector(0,1,0));
  quad(room, vector(-1.2f,2.4f,-1.2f), vector(1.2f,2.4f,-1.2f), vector(1.2f,2.4f,1.2f), vector(-1.2f,2.4f,1.2f), vector(0,-1,0));
  quad(room, vector(-1.2f,0,-1.2f), vector(1.2f,0,-1.2f), vector(1.2f,2.4f,-1.2f), vector(-1.2f,2.4f,-1.2f), vector(0,0,1));
  addMesh(scene, std::move(room));
  Mesh left;
  left.material = 1;
  quad(left, vector(-1.2f,0,1.2f), vector(-1.2f,0,-1.2f), vector(-1.2f,2.4f,-1.2f), vector(-1.2f,2.4f,1.2f), vector(1,0,0));
  addMesh(scene, std::move(left));
  Mesh right;
  right.material = 2;
  quad(right, vector(1.2f,0,-1.2f), vector(1.2f,0,1.2f), vector(1.2f,2.4f,1.2f), vector(1.2f,2.4f,-1.2f), vector(-1,0,0));
  addMesh(scene, std::move(right));
  Mesh light;
  light.material = 4;
  quad(light, vector(-0.45f,2.38f,-0.5f), vector(0.45f,2.38f,-0.5f), vector(0.45f,2.38f,0.4f), vector(-0.45f,2.38f,0.4f), vector(0,-1,0));
  addMesh(scene, std::move(light));
  addMesh(scene, sphere(3));
  scene.instances.back().transform = {0.52f,0,0,-0.53f, 0,0.52f,0,0.53f, 0,0,0.52f,0.2f};
  addMesh(scene, box(0));
  const float angle = -0.32f, c = std::cos(angle), s = std::sin(angle);
  scene.instances.back().transform = {0.7f*c,0,0.7f*s,0.48f, 0,1.05f,0,0.525f, -0.7f*s,0,0.7f*c,-0.35f};
  return scene;
}

Camera makeDemoCamera() {
  return {{0,1.18f,4.3f,0}, {0,1.15f,-0.15f,0}, {0,1,0,0}, 38};
}
} // namespace caustra::example
