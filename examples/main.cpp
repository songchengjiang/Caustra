#include <caustra/renderer.hpp>
#include "demo_scene.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
struct Arguments {
  caustra::RenderSettings settings;
  caustra::DlssSettings dlss;
  uint32_t samples = 64;
  std::filesystem::path output = "render";
  bool validation = false;
  bool selfTest = false;
  bool saveRaw = false;
  bool help = false;
};
uint32_t positiveInteger(std::string_view value, std::string_view name) {
  uint32_t result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result == 0)
    throw std::invalid_argument(std::string(name) + " requires a positive 32-bit integer.");
  return result;
}
Arguments parseArguments(int argc, char** argv) {
  Arguments arguments;
  arguments.settings.environment = {0.015f,0.015f,0.015f,0};
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument = argv[i];
    if (argument == "--help" || argument == "-h") { arguments.help = true; continue; }
    if (argument == "--validation") { arguments.validation = true; continue; }
    if (argument == "--self-test") { arguments.selfTest = true; continue; }
    if (argument == "--save-raw") { arguments.saveRaw = true; continue; }
    if (argument != "--width" && argument != "--height" && argument != "--samples" &&
        argument != "--bounces" && argument != "--output" && argument != "--dlss" && argument != "--dlss-preset")
      throw std::invalid_argument("Unknown option: " + std::string(argument));
    if (++i == argc) throw std::invalid_argument("Missing value after " + std::string(argument));
    const std::string_view value = argv[i];
    if (argument == "--dlss") {
      arguments.dlss.enabled = value != "off";
      if(value == "off" || value == "native") arguments.dlss.quality = caustra::DlssQuality::Native;
      else if(value == "quality") arguments.dlss.quality = caustra::DlssQuality::Quality;
      else if(value == "balanced") arguments.dlss.quality = caustra::DlssQuality::Balanced;
      else if(value == "performance") arguments.dlss.quality = caustra::DlssQuality::Performance;
      else throw std::invalid_argument("--dlss requires off, native, quality, balanced or performance.");
    } else if(argument == "--dlss-preset") {
      if(value == "default") arguments.dlss.preset = caustra::DlssPreset::Default;
      else if(value == "d") arguments.dlss.preset = caustra::DlssPreset::D;
      else if(value == "e") arguments.dlss.preset = caustra::DlssPreset::E;
      else throw std::invalid_argument("--dlss-preset requires default, d or e.");
    } else if (argument == "--output") {
      if (value.empty()) throw std::invalid_argument("--output must not be empty.");
      arguments.output = std::filesystem::path(argv[i]);
    } else if (argument == "--width") arguments.settings.width = positiveInteger(value, argument);
    else if (argument == "--height") arguments.settings.height = positiveInteger(value, argument);
    else if (argument == "--samples") arguments.samples = positiveInteger(value, argument);
    else arguments.settings.maxBounces = positiveInteger(value, argument);
  }
  return arguments;
}
void printHelp() {
  std::cout << "Caustra - Vulkan hardware ray-query path tracer\n"
            << "Usage: caustra_demo [options]\n\n"
            << "  --width N       Image width, 1..16384 (default 640)\n"
            << "  --height N      Image height, 1..16384 (default 480)\n"
            << "  --samples N     Samples per pixel (default 64)\n"
            << "  --bounces N     Maximum path depth, 1..64 (default 8)\n"
            << "  --dlss MODE     off/native/quality/balanced/performance (default off)\n"
            << "  --dlss-preset P RR model preset: default/d/e (default default)\n"
            << "  --save-raw      Also save progressive raw HDR at internal resolution\n"
            << "  --output PATH   Output stem or .ppm/.pfm path (default render)\n"
            << "  --validation    Enable Vulkan validation layer; fail if unavailable\n"
            << "  --self-test     Check dimensions, accumulation, finite/nonblack HDR\n"
            << "  --help, -h      Show this help\n\n"
            << "Always writes PATH.ppm (Reinhard + sRGB) and PATH.pfm (linear HDR).\n"
            << "With DLSS, each sample advances one noisy 1-spp temporal frame.\n"
            << "The demo creates a Cornell room, area light, metal sphere, and box\n"
            << "in C++; no model files, textures, window, or display server required.\n";
}
void verifyImage(const caustra::Image& image, const Arguments& arguments, const caustra::Renderer& renderer) {
  if (image.width != arguments.settings.width || image.height != arguments.settings.height ||
      image.samples != arguments.samples || renderer.sampleCount() != arguments.samples ||
      image.pixels.size() != size_t(image.width) * image.height)
    throw std::runtime_error("GPU self-test: image dimensions or sample count are incorrect.");
  double luminance = 0;
  for (const auto& pixel : image.pixels) {
    if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) || !std::isfinite(pixel.z) || !std::isfinite(pixel.w))
      throw std::runtime_error("GPU self-test: nonfinite image pixel.");
    if (pixel.x < 0 || pixel.y < 0 || pixel.z < 0)
      throw std::runtime_error("GPU self-test: negative radiance.");
    luminance += 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
  }
  if (luminance <= 0) throw std::runtime_error("GPU self-test: image is completely black.");
  if (renderer.validationErrorCount() != 0)
    throw std::runtime_error("GPU self-test: Vulkan reported validation errors.");
  std::cout << "GPU self-test passed; mean linear luminance: " << luminance / double(image.pixels.size()) << '\n';
}
}

int main(int argc, char** argv) {
  try {
    auto arguments = parseArguments(argc, argv);
    if (arguments.help) { printHelp(); return 0; }
    caustra::validateSettings(arguments.settings);
    const auto scene = caustra::example::makeDemoScene();
    const auto camera = caustra::example::makeDemoCamera();
    caustra::validateScene(scene);
    caustra::validateCamera(camera);
    auto stem = arguments.output;
    if (stem.extension() == ".ppm" || stem.extension() == ".pfm") stem.replace_extension();
    if (stem.filename().empty()) throw std::invalid_argument("--output requires a file stem.");
    if (!stem.parent_path().empty()) std::filesystem::create_directories(stem.parent_path());
    auto ppm = stem; ppm += ".ppm";
    auto pfm = stem; pfm += ".pfm";

    caustra::Renderer renderer({arguments.validation});
    renderer.setSettings(arguments.settings);
    renderer.setDlssSettings(arguments.dlss);
    renderer.setScene(scene);
    renderer.setCamera(camera);
    std::cout << "Device: " << renderer.deviceName() << '\n'
              << "Render: " << arguments.settings.width << 'x' << arguments.settings.height
              << ", " << arguments.samples << (arguments.dlss.enabled?" temporal frames":" spp") << ", " << arguments.settings.maxBounces << " bounces\n"
              << renderer.dlssStatus() << '\n';
    const auto start = std::chrono::steady_clock::now();
    auto lastProgress = start;
    while (renderer.sampleCount() < arguments.samples) {
      renderer.render(std::min(8u, arguments.samples - renderer.sampleCount()));
      const auto now = std::chrono::steady_clock::now();
      if (now - lastProgress >= std::chrono::seconds(1) || renderer.sampleCount() == arguments.samples) {
        std::cout << "Samples: " << renderer.sampleCount() << '/' << arguments.samples << '\n';
        lastProgress = now;
      }
    }
    const auto image = renderer.readback();
    if (arguments.selfTest) verifyImage(image, arguments, renderer);
    if (renderer.validationErrorCount() != 0) throw std::runtime_error("Vulkan validation errors occurred during rendering.");
    image.savePpm(ppm);
    image.savePfm(pfm);
    if(arguments.saveRaw) {
      const auto raw = renderer.readbackRaw();
      auto rawPpm=stem; rawPpm+="_raw.ppm";
      auto rawPfm=stem; rawPfm+="_raw.pfm";
      raw.savePpm(rawPpm); raw.savePfm(rawPfm);
      std::cout << "Saved raw " << raw.width << 'x' << raw.height << " comparison output\n";
    }
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Saved " << ppm.string() << " and " << pfm.string() << " in " << seconds << " s\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Caustra: " << error.what() << '\n';
    return 1;
  }
}
