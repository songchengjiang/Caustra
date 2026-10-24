#include <caustra/renderer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace caustra {
namespace {
bool finite(const Float4& v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
}
bool nonnegativeRgb(const Float4& v) { return v.x >= 0 && v.y >= 0 && v.z >= 0; }
struct D3 { double x, y, z; };
D3 subtract(const Float4& a, const Float4& b) {
  return {double(a.x) - b.x, double(a.y) - b.y, double(a.z) - b.z};
}
D3 cross(D3 a, D3 b) {
  return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
double lengthSquared(D3 a) { return a.x*a.x + a.y*a.y + a.z*a.z; }
void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}
}

void validateScene(const Scene& scene) {
  require(!scene.materials.empty(), "Scene must contain at least one material.");
  require(!scene.meshes.empty(), "Scene must contain at least one mesh.");
  require(!scene.instances.empty(), "Scene must contain at least one instance.");
  constexpr auto maxCount = std::numeric_limits<uint32_t>::max();
  require(scene.materials.size() <= maxCount && scene.meshes.size() <= maxCount &&
          scene.instances.size() <= 0x00ffffffu, "Scene exceeds GPU index limits.");

  for (size_t i = 0; i < scene.materials.size(); ++i) {
    const auto& m = scene.materials[i];
    const auto prefix = "Material " + std::to_string(i) + ": ";
    require(finite(m.baseColor) && finite(m.emission) && finite(m.surface), prefix + "all components must be finite.");
    require(nonnegativeRgb(m.baseColor) && m.baseColor.x <= 1 && m.baseColor.y <= 1 && m.baseColor.z <= 1,
            prefix + "base color must be in [0, 1].");
    require(nonnegativeRgb(m.emission), prefix + "emission must be nonnegative.");
    require(m.surface.x >= 0 && m.surface.x <= 1 && m.surface.y >= 0 && m.surface.y <= 1,
            prefix + "metallic and roughness must be in [0, 1].");
    require(m.surface.z==0 || m.surface.z==1,prefix+"one-sided emission flag must be 0 or 1.");
    require(finite(m.specular) && finite(m.specularColor) && finite(m.coat) && finite(m.coatColor) &&
            finite(m.sheen) && finite(m.sheenColor) && finite(m.factors),prefix+"surface parameters must be finite.");
    auto unit=[](float value) {return value>=0 && value<=1;};
    const bool inactiveZeroIor=m.program.z==0 && m.specular.x==0 && m.transmission.x==0 && m.subsurface.x==0 && m.specular.y==0;
    require(m.specular.x>=0 && (m.program.z==1 || unit(m.specular.x)) && (m.specular.y>=1 || inactiveZeroIor) && m.specular.y<=10 && unit(m.specular.z),prefix+"invalid specular weight, IOR or anisotropy.");
    require(unit(m.coat.x) && unit(m.coat.y) && m.coat.z>=1 && m.coat.z<=10 && unit(m.coat.w) && unit(m.sheen.x) && unit(m.sheen.y),prefix+"invalid coat or sheen parameters.");
    require(unit(m.factors.x) && unit(m.factors.y) && unit(m.factors.z) && unit(m.factors.w),prefix+"invalid base weight, diffuse roughness, opacity or coat darkening.");
    for(const auto& value:{m.transmission,m.transmissionColor,m.transmissionScatter,m.subsurface,
                           m.subsurfaceColor,m.subsurfaceRadius,m.thinFilm,m.fuzz,m.fuzzColor,
                           m.volume,m.volumeAbsorption,m.volumeScattering,m.volumeEmission})
      require(finite(value),prefix+"transport and volume parameters must be finite.");
    require(unit(m.transmission.x) && m.transmission.y>=0 && std::abs(m.transmission.z)<=1 && std::abs(m.transmission.w)<=1,
            prefix+"invalid transmission weight, depth, anisotropy or extra roughness.");
    require(unit(m.subsurface.x) && m.subsurface.y>=0 && std::abs(m.subsurface.z)<=1 && (m.subsurface.w==0 || m.subsurface.w==1),
            prefix+"invalid subsurface weight, radius scale, anisotropy or thin-walled flag.");
    require(unit(m.thinFilm.x) && m.thinFilm.y>=0 && m.thinFilm.z>=1 && m.thinFilm.z<=10 && m.thinFilm.w>=0,
            prefix+"invalid thin-film coverage, thickness, IOR or inverse Abbe number.");
    require(unit(m.fuzz.x) && unit(m.fuzz.y) && unit(m.fuzz.w),prefix+"invalid fuzz or coat color-affect parameters.");
    require(m.volume.x>=0 && std::abs(m.volume.y)<=1 && (m.volume.z==0 || m.volume.z==1) && unit(m.volume.w),
            prefix+"invalid volume density, anisotropy, boundary-only flag or coat roughness-affect parameter.");
    for(const auto& color:{m.transmissionColor,m.subsurfaceColor,m.fuzzColor})
      require(nonnegativeRgb(color) && color.x<=1 && color.y<=1 && color.z<=1,prefix+"transport colors must be in [0, 1].");
    for(const auto& coefficient:{m.transmissionScatter,m.subsurfaceRadius,m.volumeAbsorption,m.volumeScattering,m.volumeEmission})
      require(nonnegativeRgb(coefficient),prefix+"medium coefficients and subsurface radii must be nonnegative.");
    require(m.program.z<=static_cast<uint32_t>(MaterialModel::SimpleHair) && m.program.w<=3 &&
            ((m.program.w&2)==0 || (m.program.w&1)!=0),prefix+"unknown surface model or program flags.");
    if(m.program.z==uint32_t(MaterialModel::GltfPbr))
      require(m.fuzz.z>=0 && m.fuzz.z<=2 && m.fuzz.z==std::floor(m.fuzz.z),prefix+"glTF alpha mode must be OPAQUE (0), MASK (1) or BLEND (2).");
    if(m.program.z==uint32_t(MaterialModel::UsdPreviewSurface))
      require((m.factors.w==0 || m.factors.w==1) && (m.fuzz.w==0 || m.fuzz.w==1) && unit(m.specular.w),prefix+"invalid USD workflow, opacity mode or threshold.");
    for(const auto& color:{m.specularColor,m.coatColor,m.sheenColor})
      require(nonnegativeRgb(color) && color.x<=1 && color.y<=1 && color.z<=1,prefix+"surface colors must be in [0, 1].");
    require(m.program.y<=maxMaterialInstructions && uint64_t(m.program.x)+m.program.y<=scene.materialInstructions.size(),prefix+"material program range exceeds instruction buffer or register limit.");
    std::array<bool,maxMaterialInstructions> spatial{};
    bool hasVolume=m.volume.x>0 || m.volume.z>0;
    bool potentialInteriorLobe=m.transmission.x>0 || m.subsurface.x>0;
    bool thinWalledKnown=true, thinWalled=m.subsurface.w==1;
    bool opacityOutput=false, spatialInterior=false, spatialIor=false, spatialThin=false, hasSubsurface=m.subsurface.x>0, spatialCoatColor=false;
    for(uint32_t j=0;j<m.program.y;++j) {
      const auto& instruction=scene.materialInstructions[m.program.x+j];
      const auto& c=instruction.code;require(finite(instruction.value),prefix+"instruction values must be finite.");
      bool varies=false;
      auto argument=[&](uint32_t index) {require(index<j,prefix+"instruction must reference an earlier local register.");varies=varies || spatial[index];};
      switch(static_cast<MaterialOp>(c.x)) {
        case MaterialOp::Constant: break;
        case MaterialOp::Texcoord: case MaterialOp::Position:
        case MaterialOp::ObjectPosition: case MaterialOp::Normal: case MaterialOp::Tangent: varies=true;break;
        case MaterialOp::Add: case MaterialOp::Subtract: case MaterialOp::Multiply: case MaterialOp::Divide:
        case MaterialOp::Power: case MaterialOp::Min: case MaterialOp::Max: case MaterialOp::Dot:
        case MaterialOp::Cross: argument(c.y);argument(c.z);break;
        case MaterialOp::Normalmap:
          argument(c.y);argument(c.z);
          varies=true; // Even constant texels use the hit's normal/tangent frame.
          break;
        case MaterialOp::Mix: case MaterialOp::Clamp: case MaterialOp::Combine:
        case MaterialOp::Checker: case MaterialOp::Noise3D: argument(c.y);argument(c.z);argument(c.w);break;
        case MaterialOp::Abs: case MaterialOp::OneMinus: case MaterialOp::Sin: case MaterialOp::Cos:
        case MaterialOp::Floor: case MaterialOp::Ceil: case MaterialOp::Fract: case MaterialOp::Normalize:
        case MaterialOp::Length: case MaterialOp::RgbToHsv: case MaterialOp::HsvToRgb: argument(c.y);break;
        case MaterialOp::HexImage:
          argument(c.y);require(c.z<scene.textures.size(),prefix+"hex image texture index is out of range.");
          require(uint64_t(c.w)+3<j,prefix+"hex image configuration must contain four earlier registers.");
          for(uint32_t k=0;k<4;++k) argument(c.w+k);
          break;
        case MaterialOp::Swizzle:
          argument(c.y);
          for(float lane:{instruction.value.x,instruction.value.y,instruction.value.z,instruction.value.w})
            require(lane>=0 && lane<=3 && lane==std::floor(lane),prefix+"swizzle lanes must be integers 0..3.");
          break;
        case MaterialOp::Image:
          argument(c.y);require(c.z<scene.textures.size(),prefix+"image texture index is out of range.");
          require(c.w<=31,prefix+"invalid texture wrapping/filter flags.");break;
        case MaterialOp::Output:
          argument(c.y);require(c.z<=uint32_t(MaterialParameter::CoatAffectRoughness),prefix+"unknown surface output parameter.");
          opacityOutput=opacityOutput || c.z==uint32_t(MaterialParameter::Opacity);
          if(c.z==25 || c.z==31) potentialInteriorLobe=true;
          if(c.z==44 || c.z==49) hasVolume=true;
          if(c.z==uint32_t(MaterialParameter::ThinWalled)) {
            spatialThin=spatialThin || varies;
            const auto& source=scene.materialInstructions[m.program.x+c.y];
            // Honor a final literal bool. More complex control graphs may
            // create a solid, so require uniform interior inputs unless the
            // frontend folds the flag into the material's POD settings.
            thinWalledKnown=source.code.x==uint32_t(MaterialOp::Constant) && (source.value.x==0 || source.value.x==1);
            if(thinWalledKnown) thinWalled=source.value.x==1;
          }
          hasSubsurface=hasSubsurface || c.z==31;
          if((c.z==12 || c.z==51) && varies) spatialCoatColor=true;
          if(((m.program.z==1 && c.z==7) || c.z==9 || (c.z>=25 && c.z<=36 && c.z!=30) || (c.z>=43 && c.z<=49)) && varies) spatialInterior=true;
          if(((m.program.z==1 && c.z==7) || c.z==9 || c.z==43) && varies) spatialIor=true;
          break;
        default: require(false,prefix+"unknown material opcode.");
      }
      spatial[j]=varies;
    }
    const bool hasInterior=hasVolume || (m.program.z!=5 && (potentialInteriorLobe || (m.program.w&1)) && (!thinWalledKnown || !thinWalled));
    const bool opaqueNative=(m.program.z==3 && m.fuzz.z==0) || (m.program.z==4 && m.fuzz.w==0 && m.specular.w==0);
    require(!hasInterior || opaqueNative || (m.factors.z==1 && !opacityOutput),prefix+"solid interior boundaries require full coverage.");
    require(!hasInterior || !spatialInterior || (m.program.w&1),prefix+"spatial interior coefficients require the heterogeneous-medium flag.");
    require(!hasInterior || (!spatialIor && !spatialThin),prefix+"interior IOR, dispersion and thin-walled state must be spatially uniform.");
    require(!hasInterior || m.program.z!=0 || !hasSubsurface || !spatialCoatColor || (m.program.w&1),
            prefix+"Standard Surface subsurface coat color controls must be spatially uniform.");
  }
  require(scene.textures.size()<=maxCount && scene.materialInstructions.size()<=maxCount,"Material resources exceed 32-bit GPU indexing.");
  for(const auto& texture:scene.textures) {
    require(texture.width>0 && texture.height>0 && texture.width<=16384 && texture.height<=16384,"Texture dimensions must be in [1, 16384].");
    require(texture.pixels.size()==uint64_t(texture.width)*texture.height,"Texture pixel count does not match its dimensions.");
    for(const auto& pixel:texture.pixels) require(finite(pixel),"Texture pixels must be finite.");
  }
  uint64_t totalVertices = 0, totalIndices = 0;
  for (size_t i = 0; i < scene.meshes.size(); ++i) {
    const auto& mesh = scene.meshes[i];
    const auto prefix = "Mesh " + std::to_string(i) + ": ";
    require(mesh.material < scene.materials.size(), prefix + "material index is out of range.");
    require(!mesh.vertices.empty(), prefix + "vertices are empty.");
    require(!mesh.indices.empty() && mesh.indices.size() % 3 == 0, prefix + "indices must be a nonempty triangle list.");
    totalVertices += mesh.vertices.size();
    totalIndices += mesh.indices.size();
    require(totalVertices <= maxCount && totalIndices <= maxCount, "Combined mesh data exceeds 32-bit GPU indices.");
    for (const auto& vertex : mesh.vertices)
      require(finite(vertex.position) && finite(vertex.normal) && finite(vertex.texcoord) && finite(vertex.tangent), prefix + "vertex components must be finite.");
    if(scene.materials[mesh.material].program.w&1) {
      Float4 low=mesh.vertices.front().position,high=low;
      for(const auto& vertex:mesh.vertices) {
        low.x=std::min(low.x,vertex.position.x);low.y=std::min(low.y,vertex.position.y);low.z=std::min(low.z,vertex.position.z);
        high.x=std::max(high.x,vertex.position.x);high.y=std::max(high.y,vertex.position.y);high.z=std::max(high.z,vertex.position.z);
      }
      require(high.x>low.x && high.y>low.y && high.z>low.z,prefix+"heterogeneous interior requires nonzero object bounds on all three axes.");
    }
    for (size_t triangle = 0; triangle < mesh.indices.size(); triangle += 3) {
      const auto a = mesh.indices[triangle], b = mesh.indices[triangle + 1], c = mesh.indices[triangle + 2];
      require(a < mesh.vertices.size() && b < mesh.vertices.size() && c < mesh.vertices.size(),
              prefix + "triangle index is out of range.");
      const auto& p = mesh.vertices[a].position;
      require(lengthSquared(cross(subtract(mesh.vertices[b].position, p), subtract(mesh.vertices[c].position, p))) > 0,
              prefix + "triangle " + std::to_string(triangle / 3) + " is degenerate.");
    }
  }
  for (size_t i = 0; i < scene.instances.size(); ++i) {
    const auto& instance = scene.instances[i];
    const auto prefix = "Instance " + std::to_string(i) + ": ";
    require(instance.mesh < scene.meshes.size(), prefix + "mesh index is out of range.");
    const auto& t = instance.transform;
    for (float value : t) require(std::isfinite(value), prefix + "transform must be finite.");
    const double determinant = double(t[0]) * (double(t[5])*t[10] - double(t[6])*t[9])
                             - double(t[1]) * (double(t[4])*t[10] - double(t[6])*t[8])
                             + double(t[2]) * (double(t[4])*t[9] - double(t[5])*t[8]);
    double scale = 0;
    for (size_t row = 0; row < 3; ++row)
      for (size_t col = 0; col < 3; ++col) scale = std::max(scale, std::abs(double(t[row*4 + col])));
    require(scale > 0 && std::abs(determinant) > 1e-12 * scale * scale * scale,
            prefix + "transform is singular or numerically unstable.");
    for (const auto& vertex : scene.meshes[instance.mesh].vertices) {
      for (size_t row = 0; row < 3; ++row) {
        const auto& p = vertex.position;
        const double world = double(t[row*4])*p.x + double(t[row*4+1])*p.y + double(t[row*4+2])*p.z + t[row*4+3];
        require(std::isfinite(world) && std::abs(world) <= std::numeric_limits<float>::max(),
                prefix + "transformed vertex exceeds float range.");
      }
    }
  }
}

void validateCamera(const Camera& camera) {
  require(finite(camera.position) && finite(camera.target) && finite(camera.up) && std::isfinite(camera.verticalFov),
          "Camera components must be finite.");
  require(camera.verticalFov > 0 && camera.verticalFov < 179, "Camera vertical FOV must be in (0, 179) degrees.");
  const D3 direction = subtract(camera.target, camera.position);
  const D3 up{camera.up.x, camera.up.y, camera.up.z};
  require(lengthSquared(direction) > 0, "Camera position and target must differ.");
  require(lengthSquared(up) > 0 && lengthSquared(cross(direction, up)) > 1e-12 * lengthSquared(direction) * lengthSquared(up),
          "Camera up must be nonzero and must not be parallel to the viewing direction.");
}

void validateSettings(const RenderSettings& settings) {
  require(settings.width > 0 && settings.height > 0 && settings.width <= 16384 && settings.height <= 16384,
          "Render dimensions must each be in [1, 16384].");
  require(settings.maxBounces >= 1 && settings.maxBounces <= 64, "Bounce count must be in [1, 64].");
  require(finite(settings.environment) && nonnegativeRgb(settings.environment), "Environment radiance must be finite and nonnegative.");
}

void validateDlssSettings(const DlssSettings& settings) {
  require(settings.quality==DlssQuality::Native || settings.quality==DlssQuality::Quality ||
          settings.quality==DlssQuality::Balanced || settings.quality==DlssQuality::Performance,
          "Invalid DLSS quality mode.");
  require(settings.preset==DlssPreset::Default || settings.preset==DlssPreset::D || settings.preset==DlssPreset::E,
          "DLSS RR preset must be Default, D or E.");
}

} // namespace caustra
