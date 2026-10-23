#pragma once

#include <caustra/renderer.hpp>

namespace caustra::example {
// Entirely procedural: no asset files, importers, textures, or model loaders.
Scene makeDemoScene();
Camera makeDemoCamera();
} // namespace caustra::example
