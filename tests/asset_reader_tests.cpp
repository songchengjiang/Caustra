#include <caustra/asset.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
void require(bool condition,const char* message) {
  if(!condition) throw std::runtime_error(message);
}

std::vector<uint8_t> readBytes(const std::filesystem::path& path) {
  std::ifstream input(path,std::ios::binary|std::ios::ate);
  require(bool(input),"Cannot open compiled Blender fixture");
  const auto size=input.tellg();require(size>=120,"Compiled Blender fixture has no complete header");
  std::vector<uint8_t> bytes(static_cast<size_t>(size));input.seekg(0);
  input.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
  require(bool(input),"Cannot read compiled Blender fixture");
  return bytes;
}

uint32_t integerAt(const std::vector<uint8_t>& bytes,size_t offset) {
  require(offset<=bytes.size() && bytes.size()-offset>=4,"Fixture offset is outside package");
  uint32_t value=0;
  for(unsigned i=0;i<4;++i) value|=uint32_t(bytes[offset+i])<<(8*i);
  return value;
}

void setInteger(std::vector<uint8_t>& bytes,size_t offset,uint32_t value) {
  require(offset<=bytes.size() && bytes.size()-offset>=4,"Mutation offset is outside package");
  for(unsigned i=0;i<4;++i) bytes[offset+i]=static_cast<uint8_t>(value>>(8*i));
}

class MalformedPackages {
public:
  MalformedPackages() {
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    directory_=std::filesystem::temp_directory_path()/("caustra_asset_reader_"+std::to_string(stamp));
    require(std::filesystem::create_directory(directory_),"Cannot create asset reader test directory");
  }
  ~MalformedPackages() {
    // Every path is a single file created by this test. Never recursively delete.
    std::error_code error;
    for(const auto& path:files_) std::filesystem::remove(path,error);
    std::filesystem::remove(directory_,error);
  }
  template<class Mutation>
  void reject(const char* name,const std::vector<uint8_t>& original,Mutation mutation) {
    auto bytes=original;mutation(bytes);
    auto path=directory_/(std::string(name)+".caustra");files_.push_back(path);
    {
      std::ofstream output(path,std::ios::binary|std::ios::trunc);
      output.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
      output.close();require(bool(output),"Cannot write malformed asset test fixture");
    }
    bool rejected=false;
    try {static_cast<void>(caustra::loadAsset(path));}
    catch(const std::exception&) {rejected=true;}
    require(std::filesystem::remove(path),"Cannot remove malformed asset test fixture");
    if(!rejected) throw std::runtime_error(std::string("loadAsset accepted malformed package: ")+name);
    ++checks_;
  }
  unsigned checks() const {return checks_;}
  void acceptLegacy(const std::vector<uint8_t>& bytes) {
    const auto path=directory_/"legacy_v1.caustra";files_.push_back(path);
    std::ofstream output(path,std::ios::binary|std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    output.close();require(bool(output),"Cannot write legacy asset fixture");
    const auto asset=caustra::loadAsset(path);
    require(!asset.scene.materials.empty(),"Legacy material data was lost");
    for(const auto& m:asset.scene.materials)
      require(m.transmission.x==0 && m.subsurface.x==0 && m.thinFilm.y==0 && m.volume.x==0,"Legacy material defaults changed");
    require(std::filesystem::remove(path),"Cannot remove legacy asset fixture");
  }
private:
  std::filesystem::path directory_;
  std::vector<std::filesystem::path> files_;
  unsigned checks_=0;
};

void verifyFixture(const caustra::Asset& asset) {
  require(asset.hasCamera,"Blender fixture camera was not imported");
  require(!asset.scene.meshes.empty() && !asset.scene.instances.empty(),"Blender fixture geometry was not imported");
  require(!asset.scene.materials.empty(),"Blender fixture materials were not imported");
  for(const auto& material:asset.scene.materials)
    require(material.program.z==1,"Blender material lost its native OpenPBR model");
  require(asset.scene.textures.size()==2,"Blender texture count must be two");
  bool varyingTexture=false;
  for(const auto& texture:asset.scene.textures) {
    require(texture.width>0 && texture.height>0 && texture.pixels.size()==size_t(texture.width)*texture.height,"Invalid Blender texture dimensions");
    bool nonblack=false;
    for(const auto& pixel:texture.pixels) {
      require(std::isfinite(pixel.x) && std::isfinite(pixel.y) && std::isfinite(pixel.z) && std::isfinite(pixel.w),"Nonfinite Blender texture pixel");
      nonblack=nonblack || std::max({pixel.x,pixel.y,pixel.z})>0;
      const auto& first=texture.pixels.front();
      varyingTexture=varyingTexture || pixel.x!=first.x || pixel.y!=first.y || pixel.z!=first.z;
    }
    require(nonblack,"Blender texture became black during packaging");
  }
  require(varyingTexture,"Blender fixture textures unexpectedly became constants");
  bool image=false,normalmap=false;
  for(const auto& instruction:asset.scene.materialInstructions) {
    image=image || instruction.code.x==static_cast<uint32_t>(caustra::MaterialOp::Image);
    normalmap=normalmap || instruction.code.x==static_cast<uint32_t>(caustra::MaterialOp::Normalmap);
  }
  require(image && normalmap,"Blender MaterialX graph was flattened into constant PBR values");
  const auto& camera=asset.camera;
  const auto dx=double(camera.position.x)-camera.target.x;
  const auto dy=double(camera.position.y)-camera.target.y;
  const auto dz=double(camera.position.z)-camera.target.z;
  require(std::sqrt(dx*dx+dy*dy+dz*dz)>1,"Blender orbit camera target remained one unit ahead of the camera");
  require(camera.verticalFov>0 && camera.verticalFov<180,"Invalid Blender camera field of view");
}
} // namespace

int main(int argc,char** argv) {
  try {
    require(argc==2,"Usage: caustra_asset_reader_tests compiled-blender.caustra");
    const std::filesystem::path path=argv[1];
    verifyFixture(caustra::loadAsset(path));
    const auto bytes=readBytes(path);
    // Version 2: header[120], Material[384], Instruction[32], then textures.
    const auto materialCount=integerAt(bytes,16),instructionCount=integerAt(bytes,28),textureCount=integerAt(bytes,32);
    require(materialCount>0 && instructionCount>0 && textureCount==2,"Fixture must exercise each asset record type");
    require(integerAt(bytes,8)==2,"The compiler must write version 2 assets");
    constexpr size_t firstMaterial=120,firstProgram=firstMaterial+368;
    const size_t instructions=firstMaterial+size_t(materialCount)*384;
    const size_t textures=instructions+size_t(instructionCount)*32;
    require(textures+8<=bytes.size(),"Fixture texture data is missing");
    MalformedPackages packages;
    auto legacy=std::vector<uint8_t>(bytes.begin(),bytes.begin()+120);
    setInteger(legacy,8,1);
    for(uint32_t i=0;i<materialCount;++i) {
      const auto row=bytes.begin()+firstMaterial+size_t(i)*384;
      legacy.insert(legacy.end(),row,row+160);
      legacy.insert(legacy.end(),row+368,row+384);
    }
    legacy.insert(legacy.end(),bytes.begin()+instructions,bytes.end());
    packages.acceptLegacy(legacy);
    packages.reject("truncated_header",bytes,[](auto& b) {b.resize(119);});
    packages.reject("truncated_payload",bytes,[](auto& b) {b.pop_back();});
    packages.reject("bad_magic",bytes,[](auto& b) {b[0]^=0xff;});
    packages.reject("unsupported_version",bytes,[](auto& b) {setInteger(b,8,0xffffffffu);});
    packages.reject("unsupported_flags",bytes,[](auto& b) {setInteger(b,12,0x80000000u);});
    packages.reject("material_count_overflow",bytes,[](auto& b) {setInteger(b,16,0xffffffffu);});
    packages.reject("instruction_count_overflow",bytes,[](auto& b) {setInteger(b,28,0xffffffffu);});
    packages.reject("program_range_overflow",bytes,[&](auto& b) {setInteger(b,firstProgram,0xffffffffu);});
    packages.reject("unsupported_material_model",bytes,[&](auto& b) {setInteger(b,firstProgram+8,0xffffffffu);});
    packages.reject("invalid_opcode",bytes,[&](auto& b) {setInteger(b,instructions,0xffffffffu);});
    packages.reject("nonfinite_material",bytes,[&](auto& b) {setInteger(b,firstMaterial,0x7fc00000u);});
    packages.reject("nonfinite_medium",bytes,[&](auto& b) {setInteger(b,firstMaterial+20*16,0x7fc00000u);});
    packages.reject("negative_absorption",bytes,[&](auto& b) {setInteger(b,firstMaterial+20*16,0xbf800000u);});
    packages.reject("nonfinite_camera",bytes,[](auto& b) {setInteger(b,36,0x7f800000u);});
    packages.reject("texture_size_overflow",bytes,[&](auto& b) {setInteger(b,textures,0xffffffffu);setInteger(b,textures+4,0xffffffffu);});
    packages.reject("nonfinite_texture",bytes,[&](auto& b) {setInteger(b,textures+8,0x7fc00000u);});
    packages.reject("trailing_bytes",bytes,[](auto& b) {b.push_back(0);});
    std::cout<<"CPU asset reader passed: native Blender OpenPBR graphs, textures, camera and "<<packages.checks()<<" malformed packages.\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"Asset reader test: "<<error.what()<<'\n';return 1;
  }
}
