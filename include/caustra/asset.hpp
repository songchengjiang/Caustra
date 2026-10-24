#pragma once
#include <caustra/renderer.hpp>

namespace caustra {
// Optional portable asset adapter. USD/MaterialX compilation happens outside the
// rendering core; this reader only depends on the public scene types and C++.
struct Asset {
  Scene scene;
  Camera camera;
  Float4 boundsMin, boundsMax;
  bool hasCamera=false;
};
Asset loadAsset(const std::filesystem::path& path);
} // namespace caustra
