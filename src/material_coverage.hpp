#pragma once

#include <caustra/renderer.hpp>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace caustra {

// Private, derived GPU programs: public Material IR and asset packages retain
// their original SSA instructions. The small evaluator shares the full VM's
// arithmetic/footprint implementation; larger slices use that full evaluator.
inline constexpr uint32_t kCoverageRegisterCount = 32;
inline constexpr uint32_t kCoverageFullProgram = 1u;
struct CoveragePrograms {
  // first derived instruction, instruction count, flags, reserved
  std::vector<Uint4> ranges;
  std::vector<MaterialInstruction> instructions;
};

namespace coverage_detail {

inline bool affectsCoverage(uint32_t parameter, uint32_t model) {
  if(parameter == uint32_t(MaterialParameter::Opacity)) return true;
  // Native models reuse these POD lanes for their alpha modes/cutoffs. A raw
  // IR caller may write those lanes, so they must participate in the slice.
  if(model == uint32_t(MaterialModel::GltfPbr))
    return parameter == uint32_t(MaterialParameter::CoatRotation) ||
           parameter == uint32_t(MaterialParameter::CoatAffectRoughness);
  if(model == uint32_t(MaterialModel::UsdPreviewSurface))
    return parameter == uint32_t(MaterialParameter::Rotation) ||
           parameter == uint32_t(MaterialParameter::CoatAffectColor);
  return false;
}

// Enumerate only operands that are register references. In particular, an
// Image's texture index/filter flags and an Output's parameter are not refs.
template<class Visit>
inline void dependencies(const MaterialInstruction& instruction, Visit&& visit) {
  const auto& c = instruction.code;
  switch(static_cast<MaterialOp>(c.x)) {
    case MaterialOp::Constant: case MaterialOp::Texcoord: case MaterialOp::Position:
    case MaterialOp::ObjectPosition: case MaterialOp::Normal: case MaterialOp::Tangent:
      break;
    case MaterialOp::Add: case MaterialOp::Subtract: case MaterialOp::Multiply:
    case MaterialOp::Divide: case MaterialOp::Power: case MaterialOp::Min:
    case MaterialOp::Max: case MaterialOp::Dot: case MaterialOp::Cross:
    case MaterialOp::Normalmap:
      visit(c.y);visit(c.z);break;
    case MaterialOp::Mix: case MaterialOp::Clamp: case MaterialOp::Combine:
    case MaterialOp::Checker: case MaterialOp::Noise3D:
      visit(c.y);visit(c.z);visit(c.w);break;
    case MaterialOp::Abs: case MaterialOp::OneMinus: case MaterialOp::Sin:
    case MaterialOp::Cos: case MaterialOp::Floor: case MaterialOp::Ceil:
    case MaterialOp::Fract: case MaterialOp::Normalize: case MaterialOp::Length:
    case MaterialOp::Swizzle: case MaterialOp::RgbToHsv: case MaterialOp::HsvToRgb:
    case MaterialOp::Image: case MaterialOp::Output:
      visit(c.y);break;
    case MaterialOp::HexImage:
      visit(c.y);
      for(uint32_t lane = 0; lane < 4; ++lane) visit(c.w + lane);
      break;
    default:
      throw std::invalid_argument("Unknown opcode in material coverage program");
  }
}

inline bool remap(MaterialInstruction& instruction, const std::array<uint32_t,maxMaterialInstructions>& map) {
  auto& c = instruction.code;
  switch(static_cast<MaterialOp>(c.x)) {
    case MaterialOp::Constant: case MaterialOp::Texcoord: case MaterialOp::Position:
    case MaterialOp::ObjectPosition: case MaterialOp::Normal: case MaterialOp::Tangent:
      break;
    case MaterialOp::Add: case MaterialOp::Subtract: case MaterialOp::Multiply:
    case MaterialOp::Divide: case MaterialOp::Power: case MaterialOp::Min:
    case MaterialOp::Max: case MaterialOp::Dot: case MaterialOp::Cross:
    case MaterialOp::Normalmap:
      c.y=map[c.y];c.z=map[c.z];break;
    case MaterialOp::Mix: case MaterialOp::Clamp: case MaterialOp::Combine:
    case MaterialOp::Checker: case MaterialOp::Noise3D:
      c.y=map[c.y];c.z=map[c.z];c.w=map[c.w];break;
    case MaterialOp::Abs: case MaterialOp::OneMinus: case MaterialOp::Sin:
    case MaterialOp::Cos: case MaterialOp::Floor: case MaterialOp::Ceil:
    case MaterialOp::Fract: case MaterialOp::Normalize: case MaterialOp::Length:
    case MaterialOp::Swizzle: case MaterialOp::RgbToHsv: case MaterialOp::HsvToRgb:
    case MaterialOp::Image: case MaterialOp::Output:
      c.y=map[c.y];break;
    case MaterialOp::HexImage: {
      const auto first=map[c.w];
      // All four original adjacent registers are dependencies and normally
      // remain adjacent. Retain an explicit safe fallback if that changes.
      for(uint32_t lane=1;lane<4;++lane) if(map[c.w+lane]!=first+lane) return false;
      c.y=map[c.y];c.w=first;break;
    }
    default: return false;
  }
  return true;
}

} // namespace coverage_detail

// Called after validateScene. Slice roots are the last writes affecting
// materialOpacity, with transitive SSA dependencies in their original order.
// No arithmetic is folded, reordered or substituted, including Output values
// that later instructions reference and texture-coordinate filter footprints.
inline CoveragePrograms buildCoveragePrograms(const Scene& scene) {
  CoveragePrograms result;
  result.ranges.reserve(scene.materials.size());
  constexpr uint32_t absent=std::numeric_limits<uint32_t>::max();
  for(const auto& material:scene.materials) {
    const auto count=material.program.y;
    if(count>maxMaterialInstructions || uint64_t(material.program.x)+count>scene.materialInstructions.size())
      throw std::invalid_argument("Material coverage source range exceeds instruction buffer");
    std::array<uint32_t,uint32_t(MaterialParameter::CoatAffectRoughness)+1> last;
    last.fill(absent);
    for(uint32_t index=0;index<count;++index) {
      const auto& c=scene.materialInstructions[material.program.x+index].code;
      if(c.x==uint32_t(MaterialOp::Output) && coverage_detail::affectsCoverage(c.z,material.program.z))
        last[c.z]=index;
    }
    std::array<bool,maxMaterialInstructions> needed{};
    for(auto index:last) if(index!=absent) needed[index]=true;
    for(uint32_t reverse=count;reverse>0;--reverse) {
      const auto index=reverse-1;
      if(!needed[index]) continue;
      coverage_detail::dependencies(scene.materialInstructions[material.program.x+index],[&](uint32_t source) {
        if(source>=index) throw std::invalid_argument("Material coverage instructions require earlier registers");
        needed[source]=true;
      });
    }
    uint32_t derivedCount=0;
    std::array<uint32_t,maxMaterialInstructions> map;
    map.fill(absent);
    for(uint32_t index=0;index<count;++index) if(needed[index]) map[index]=derivedCount++;
    if(derivedCount>kCoverageRegisterCount) {
      result.ranges.push_back({0,0,kCoverageFullProgram,0});
      continue;
    }
    std::vector<MaterialInstruction> derived;
    derived.reserve(derivedCount);
    bool compact=true;
    for(uint32_t index=0;index<count;++index) if(needed[index]) {
      auto instruction=scene.materialInstructions[material.program.x+index];
      if(!coverage_detail::remap(instruction,map)) {compact=false;break;}
      derived.push_back(instruction);
    }
    if(!compact) {
      result.ranges.push_back({0,0,kCoverageFullProgram,0});
      continue;
    }
    if(uint64_t(result.instructions.size())+derivedCount>std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument("Material coverage programs exceed 32-bit GPU indexing");
    result.ranges.push_back({static_cast<uint32_t>(result.instructions.size()),derivedCount,0,0});
    result.instructions.insert(result.instructions.end(),derived.begin(),derived.end());
  }
  return result;
}

} // namespace caustra
