#include <caustra/asset.hpp>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace caustra {
namespace {
class Reader {
public:
  explicit Reader(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    if(!input) throw std::runtime_error("Cannot open Caustra asset: "+path.string());
    const auto size=input.tellg();
    if(size<0 || uint64_t(size)>uint64_t(2)*1024*1024*1024) fail("Asset exceeds the 2 GiB package limit");
    bytes_.resize(static_cast<size_t>(size));input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes_.data()),static_cast<std::streamsize>(bytes_.size()));
    if(!input) fail("Cannot read Caustra asset");
  }
  [[noreturn]] static void fail(const char* message) {throw std::runtime_error(message);}
  size_t remaining() const {return bytes_.size()-cursor_;}
  void requireBytes(uint64_t count) const {if(count>remaining()) fail("Truncated or invalid Caustra asset");}
  uint32_t integer() {
    requireBytes(4);uint32_t result=0;
    for(unsigned shift=0;shift<32;shift+=8) result|=uint32_t(bytes_[cursor_++])<<shift;
    return result;
  }
  float number() {
    const auto result=std::bit_cast<float>(integer());
    if(!std::isfinite(result)) fail("Nonfinite value in Caustra asset");
    return result;
  }
  Float4 vector() {return {number(),number(),number(),number()};}
private:
  std::vector<uint8_t> bytes_;
  size_t cursor_=0;
};
}

Asset loadAsset(const std::filesystem::path& path) {
  static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
  Reader r(path);
  if(r.integer()!=0x53554143u || r.integer()!=0x00415254u) Reader::fail("Not a Caustra asset package");
  const auto version=r.integer();
  if(version!=1 && version!=2) Reader::fail("Unsupported Caustra asset version");
  const auto flags=r.integer();if(flags&~1u) Reader::fail("Unsupported Caustra asset flags");
  const auto materialCount=r.integer(),meshCount=r.integer(),instanceCount=r.integer(),instructionCount=r.integer(),textureCount=r.integer();
  r.requireBytes(84);
  Asset result;result.hasCamera=(flags&1)!=0;
  result.camera.position=r.vector();result.camera.target=r.vector();result.camera.up=r.vector();result.camera.verticalFov=r.number();
  result.boundsMin=r.vector();result.boundsMax=r.vector();
  r.requireBytes(uint64_t(materialCount)*(version==1?176:384)+uint64_t(meshCount)*12+uint64_t(instanceCount)*52+uint64_t(instructionCount)*32+uint64_t(textureCount)*8);
  result.scene.materials.resize(materialCount);
  for(auto& m:result.scene.materials) {
    m.baseColor=r.vector();m.emission=r.vector();m.surface=r.vector();m.specular=r.vector();m.specularColor=r.vector();
    m.coat=r.vector();m.coatColor=r.vector();m.sheen=r.vector();m.sheenColor=r.vector();m.factors=r.vector();
    if(version>=2) {
      m.transmission=r.vector();m.transmissionColor=r.vector();m.transmissionScatter=r.vector();
      m.subsurface=r.vector();m.subsurfaceColor=r.vector();m.subsurfaceRadius=r.vector();
      m.thinFilm=r.vector();m.fuzz=r.vector();m.fuzzColor=r.vector();
      m.volume=r.vector();m.volumeAbsorption=r.vector();m.volumeScattering=r.vector();m.volumeEmission=r.vector();
    }
    m.program={r.integer(),r.integer(),r.integer(),r.integer()};
    if(uint64_t(m.program.x)+m.program.y>instructionCount || m.program.y>maxMaterialInstructions ||
       m.program.z>uint32_t(MaterialModel::SimpleHair) || m.program.w>3)
      Reader::fail("Invalid Caustra material program range");
  }
  result.scene.materialInstructions.resize(instructionCount);
  for(auto& instruction:result.scene.materialInstructions) {
    instruction.code={r.integer(),r.integer(),r.integer(),r.integer()};instruction.value=r.vector();
  }
  result.scene.textures.resize(textureCount);
  for(auto& texture:result.scene.textures) {
    texture.width=r.integer();texture.height=r.integer();
    if(!texture.width || !texture.height) Reader::fail("Invalid Caustra texture dimensions");
    const uint64_t pixels=uint64_t(texture.width)*texture.height;
    if(pixels>r.remaining()/16) Reader::fail("Invalid Caustra texture size");
    texture.pixels.resize(static_cast<size_t>(pixels));for(auto& pixel:texture.pixels) pixel=r.vector();
  }
  result.scene.meshes.resize(meshCount);
  for(auto& mesh:result.scene.meshes) {
    mesh.material=r.integer();const auto vertices=r.integer(),indices=r.integer();
    if(mesh.material>=materialCount || !vertices || !indices || indices%3) Reader::fail("Invalid Caustra mesh header");
    r.requireBytes(uint64_t(vertices)*64+uint64_t(indices)*4);
    mesh.vertices.resize(vertices);mesh.indices.resize(indices);
    for(auto& vertex:mesh.vertices) {vertex.position=r.vector();vertex.normal=r.vector();vertex.texcoord=r.vector();vertex.tangent=r.vector();}
    for(auto& index:mesh.indices) {index=r.integer();if(index>=vertices) Reader::fail("Invalid Caustra vertex index");}
  }
  result.scene.instances.resize(instanceCount);
  for(auto& instance:result.scene.instances) {
    instance.mesh=r.integer();if(instance.mesh>=meshCount) Reader::fail("Invalid Caustra instance mesh");
    for(auto& value:instance.transform) value=r.number();
  }
  if(r.remaining()) Reader::fail("Trailing bytes in Caustra asset package");
  const std::array lo{result.boundsMin.x,result.boundsMin.y,result.boundsMin.z};
  const std::array hi{result.boundsMax.x,result.boundsMax.y,result.boundsMax.z};
  for(unsigned i=0;i<3;++i) {
    if(lo[i]>hi[i]) Reader::fail("Invalid Caustra asset bounds");
  }
  validateScene(result.scene);
  validateCamera(result.camera);
  return result;
}
} // namespace caustra
