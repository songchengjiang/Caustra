#include "material_coverage.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace caustra;
size_t checks=0;
void check(bool condition,const char* message) {
  ++checks;if(!condition) throw std::runtime_error(message);
}
struct Program {
  std::vector<MaterialInstruction> code;
  uint32_t op(MaterialOp operation,uint32_t a=0,uint32_t b=0,uint32_t c=0,Float4 value={}) {
    code.push_back({{uint32_t(operation),a,b,c},value});return uint32_t(code.size()-1);
  }
  uint32_t constant(float value) {return op(MaterialOp::Constant,0,0,0,{value,value,value,value});}
  uint32_t output(uint32_t source,MaterialParameter parameter=MaterialParameter::Opacity) {
    return op(MaterialOp::Output,source,uint32_t(parameter));
  }
  Scene scene(MaterialModel model=MaterialModel::StandardSurface) const {
    Scene result;result.materials.emplace_back();result.materialInstructions=code;
    result.materials[0].program={0,uint32_t(code.size()),uint32_t(model),0};return result;
  }
};
const MaterialInstruction& instruction(const CoveragePrograms& programs,size_t index,size_t material=0) {
  return programs.instructions.at(programs.ranges.at(material).x+index);
}
void compact(const CoveragePrograms& result,uint32_t count) {
  check(result.ranges.size()==1,"Missing material coverage range");
  check(result.ranges[0].y==count && result.ranges[0].z==0,"Unexpected compact coverage program");
}
void testNoCoverageOutput() {
  Program p;const auto value=p.constant(.75f);p.output(value,MaterialParameter::BaseColor);
  p.output(value,MaterialParameter::Normal);p.output(value,MaterialParameter::CoatRotation);
  const auto scene=p.scene();const auto derived=buildCoveragePrograms(scene);compact(derived,0);
  check(derived.instructions.empty(),"Unrelated outputs must not execute for coverage");
  check(scene.materialInstructions.size()==p.code.size() && scene.materials[0].program.y==p.code.size(),
    "Coverage construction must not rewrite public material programs");
  Scene raw;raw.materials.resize(2);const auto empty=buildCoveragePrograms(raw);
  check(empty.ranges.size()==2 && empty.instructions.empty(),"POD-only coverage must require no instructions");
}
void testLastWriteAndOutputDependency() {
  Program p;auto old=p.constant(.1f);p.output(old);auto final=p.constant(.8f);p.output(final);
  auto result=buildCoveragePrograms(p.scene());compact(result,2);
  check(instruction(result,0).value.x==.8f && instruction(result,1).code.y==0,
    "Only the final independent opacity write and its source must survive");

  Program dependent;auto a=dependent.constant(.2f);auto intermediate=dependent.output(a,MaterialParameter::BaseColor);
  dependent.constant(99);auto b=dependent.constant(.5f);auto sum=dependent.op(MaterialOp::Add,intermediate,b);
  dependent.output(sum);result=buildCoveragePrograms(dependent.scene());compact(result,5);
  check(instruction(result,1).code.x==uint32_t(MaterialOp::Output) &&
    instruction(result,1).code.z==uint32_t(MaterialParameter::BaseColor),
    "An Output used as a value must keep its original pass-through semantics");
  check(instruction(result,3).code.y==1 && instruction(result,3).code.z==2,
    "SSA dependencies must be remapped through discarded instructions");
}
void testSourcesAndUnary() {
  for(auto op:{MaterialOp::Constant,MaterialOp::Texcoord,MaterialOp::Position,
      MaterialOp::ObjectPosition,MaterialOp::Normal,MaterialOp::Tangent}) {
    Program p;p.constant(99);auto value=p.op(op,77,88,99,{.2f,.3f,.4f,.5f});p.output(value);
    const auto result=buildCoveragePrograms(p.scene());compact(result,2);
    check(instruction(result,0).code.y==77 && instruction(result,0).code.z==88 &&
      instruction(result,0).code.w==99,"Source operand payloads must not become dependencies");
  }
  for(auto op:{MaterialOp::Abs,MaterialOp::OneMinus,MaterialOp::Sin,MaterialOp::Cos,
      MaterialOp::Floor,MaterialOp::Ceil,MaterialOp::Fract,MaterialOp::Normalize,
      MaterialOp::Length,MaterialOp::Swizzle,MaterialOp::RgbToHsv,MaterialOp::HsvToRgb}) {
    Program p;p.constant(99);auto input=p.constant(.2f);auto value=p.op(op,input,23,31,{0,1,2,3});p.output(value);
    const auto result=buildCoveragePrograms(p.scene());compact(result,3);
    check(instruction(result,1).code.y==0 && instruction(result,1).code.z==23 &&
      instruction(result,1).code.w==31,"Unary instruction payloads or source changed");
    check(instruction(result,1).value.w==3,"Unary options must retain their original bytes");
  }
}
void testBinaryAndTernary() {
  for(auto op:{MaterialOp::Add,MaterialOp::Subtract,MaterialOp::Multiply,MaterialOp::Divide,
      MaterialOp::Power,MaterialOp::Min,MaterialOp::Max,MaterialOp::Dot,MaterialOp::Cross,
      MaterialOp::Normalmap}) {
    Program p;p.constant(99);auto a=p.constant(.2f);p.constant(98);auto b=p.constant(.5f);
    auto value=p.op(op,a,b,31,{3,0,0,0});p.output(value);
    const auto result=buildCoveragePrograms(p.scene());compact(result,4);
    check(instruction(result,2).code.y==0 && instruction(result,2).code.z==1 &&
      instruction(result,2).code.w==31,"Binary operands must compact without changing payload fields");
  }
  for(auto op:{MaterialOp::Mix,MaterialOp::Clamp,MaterialOp::Combine,MaterialOp::Checker,MaterialOp::Noise3D}) {
    Program p;p.constant(99);auto a=p.constant(.2f);p.constant(98);auto b=p.constant(.5f);
    auto c=p.constant(.7f);auto value=p.op(op,a,b,c,{3,4,5,6});p.output(value);
    const auto result=buildCoveragePrograms(p.scene());compact(result,5);
    const auto& actual=instruction(result,3);
    check(actual.code.y==0 && actual.code.z==1 && actual.code.w==2,
      "All ternary dependencies must survive and compact");
    check(actual.value.x==3 && actual.value.w==6,"Ternary options must be copied unchanged");
  }
}
void testTexturePayloadAndHexConfiguration() {
  Program image;image.constant(99);auto uv=image.op(MaterialOp::Texcoord);
  auto sample=image.op(MaterialOp::Image,uv,17,31,{1.25f,0,0,0});image.output(sample);
  const auto result=buildCoveragePrograms(image.scene());compact(result,3);
  const auto& actual=instruction(result,1);
  check(actual.code.y==0 && actual.code.z==17 && actual.code.w==31 && actual.value.x==1.25f,
    "Texture indices, sampling flags and LOD bias must not be remapped as registers");

  Program hex;hex.constant(99);auto coordinate=hex.op(MaterialOp::Texcoord);
  const auto configuration=hex.constant(.1f);hex.constant(.2f);hex.constant(.3f);hex.constant(.4f);
  hex.constant(98);auto tiled=hex.op(MaterialOp::HexImage,coordinate,27,configuration);hex.output(tiled);
  const auto tiledResult=buildCoveragePrograms(hex.scene());compact(tiledResult,7);
  const auto& tile=instruction(tiledResult,5);
  check(tile.code.y==0 && tile.code.z==27 && tile.code.w==1,
    "HexImage must keep the texture index and remap four adjacent configuration registers");
  for(uint32_t lane=0;lane<4;++lane)
    check(instruction(tiledResult,1+lane).value.x==.1f*float(lane+1),"HexImage configuration order changed");
}
void testNativeAlphaControls() {
  Program p;
  auto opacity=p.constant(.2f);p.output(opacity);
  auto mode=p.constant(2);p.output(mode,MaterialParameter::CoatRotation);
  auto cutoff=p.constant(.5f);p.output(cutoff,MaterialParameter::CoatAffectRoughness);
  auto threshold=p.constant(.4f);p.output(threshold,MaterialParameter::Rotation);
  auto presence=p.constant(1);p.output(presence,MaterialParameter::CoatAffectColor);
  auto anisotropy=p.constant(.7f);p.output(anisotropy,MaterialParameter::Anisotropy);
  const auto standard=buildCoveragePrograms(p.scene());compact(standard,2);
  const auto gltf=buildCoveragePrograms(p.scene(MaterialModel::GltfPbr));compact(gltf,6);
  check(instruction(gltf,3).code.z==uint32_t(MaterialParameter::CoatRotation) &&
    instruction(gltf,5).code.z==uint32_t(MaterialParameter::CoatAffectRoughness),
    "glTF graph writes to alpha mode and cutoff must participate in coverage");
  const auto usd=buildCoveragePrograms(p.scene(MaterialModel::UsdPreviewSurface));compact(usd,6);
  check(instruction(usd,3).code.z==uint32_t(MaterialParameter::Rotation) &&
    instruction(usd,5).code.z==uint32_t(MaterialParameter::CoatAffectColor),
    "USD threshold uses Rotation, not the neighboring Anisotropy parameter");

  Program controlsOnly;auto value=controlsOnly.constant(.5f);controlsOnly.output(value,MaterialParameter::Rotation);
  const auto usdOnly=buildCoveragePrograms(controlsOnly.scene(MaterialModel::UsdPreviewSurface));compact(usdOnly,2);
  const auto standardOnly=buildCoveragePrograms(controlsOnly.scene());compact(standardOnly,0);
}
void testCapacityAndFallback() {
  for(uint32_t size:{kCoverageRegisterCount,kCoverageRegisterCount+1,maxMaterialInstructions}) {
    Program p;auto value=p.constant(.2f);
    while(p.code.size()<size-1) value=p.op(MaterialOp::Abs,value);
    p.output(value);const auto result=buildCoveragePrograms(p.scene());
    if(size<=kCoverageRegisterCount) compact(result,size);
    else check(result.ranges[0].z==kCoverageFullProgram && result.ranges[0].y==0 && result.instructions.empty(),
      "Large coverage slices must explicitly select the exact full evaluator");
  }
  Program dead;auto value=dead.constant(.2f);
  for(uint32_t index=1;index<maxMaterialInstructions-1;++index) value=dead.op(MaterialOp::Abs,value);
  dead.output(value,MaterialParameter::BaseColor);compact(buildCoveragePrograms(dead.scene()),0);
}
void testMaterialRangesAndInputGuards() {
  Program p;auto value=p.constant(.3f);p.output(value);
  auto scene=p.scene();
  scene.materialInstructions.insert(scene.materialInstructions.begin(),MaterialInstruction{{0,0,0,0},{99,99,99,99}});
  scene.materials[0].program.x=1;scene.materials.push_back(scene.materials[0]);
  const auto result=buildCoveragePrograms(scene);
  check(result.ranges.size()==2 && result.ranges[0].x==0 && result.ranges[1].x==2 &&
    result.ranges[0].y==2 && result.ranges[1].y==2,"Shared source programs need independent correct ranges");
  check(instruction(result,0,1).value.x==.3f && instruction(result,1,1).code.y==0,
    "Nonzero material source ranges must retain local register addressing");
  auto invalid=scene;invalid.materials[0].program.x=UINT32_MAX;
  bool rejected=false;try {buildCoveragePrograms(invalid);} catch(const std::invalid_argument&) {rejected=true;}
  check(rejected,"Invalid source ranges must fail explicitly");
  invalid=p.scene();invalid.materialInstructions.back().code.y=1;
  rejected=false;try {buildCoveragePrograms(invalid);} catch(const std::invalid_argument&) {rejected=true;}
  check(rejected,"Forward coverage references must fail explicitly");
}
} // namespace

int main() {
  try {
    testNoCoverageOutput();testLastWriteAndOutputDependency();testSourcesAndUnary();
    testBinaryAndTernary();testTexturePayloadAndHexConfiguration();testNativeAlphaControls();
    testCapacityAndFallback();testMaterialRangesAndInputGuards();
    std::cout<<"Material coverage: "<<checks<<" checks passed (SSA slicing, native alpha controls and fallback).\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
