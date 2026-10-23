#include <caustra/renderer.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace caustra {
namespace {
void validateImage(const Image& image) {
  if (image.width == 0 || image.height == 0 ||
      uint64_t(image.width) * image.height != image.pixels.size())
    throw std::invalid_argument("Image dimensions must be positive and match its pixel count.");
  for (const auto& p : image.pixels)
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(p.w))
      throw std::invalid_argument("Cannot save an image containing nonfinite pixel values.");
}
uint8_t displayChannel(float value, double exposureScale) {
  const double x = std::max(0.0, double(value) * exposureScale);
  // Reinhard in display-linear RGB, then the standard sRGB transfer function.
  const double mapped = x / (1.0 + x);
  const double srgb = mapped <= 0.0031308 ? 12.92 * mapped : 1.055 * std::pow(mapped, 1.0 / 2.4) - 0.055;
  return static_cast<uint8_t>(std::lround(std::clamp(srgb, 0.0, 1.0) * 255));
}
}

void Image::savePpm(const std::filesystem::path& path, float exposure) const {
  validateImage(*this);
  if (!std::isfinite(exposure) || exposure < -100 || exposure > 100)
    throw std::invalid_argument("Image exposure must be finite and in [-100, 100] stops.");
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("Cannot open image output: " + path.string());
  stream << "P6\n" << width << ' ' << height << "\n255\n";
  const double exposureScale = std::exp2(double(exposure));
  for (const auto& p : pixels) {
    const uint8_t rgb[]{displayChannel(p.x, exposureScale), displayChannel(p.y, exposureScale), displayChannel(p.z, exposureScale)};
    stream.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
  }
  stream.close();
  if (!stream) throw std::runtime_error("Failed while writing image: " + path.string());
}

void Image::savePfm(const std::filesystem::path& path) const {
  validateImage(*this);
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("Cannot open image output: " + path.string());
  stream << "PF\n" << width << ' ' << height << '\n';
  stream << (std::endian::native == std::endian::little ? "-1.0\n" : "1.0\n");
  // PFM stores the bottom row first and preserves the original linear HDR values.
  for (uint32_t row = height; row > 0; --row) {
    for (uint32_t x = 0; x < width; ++x) {
      const auto& p = pixels[size_t(row - 1) * width + x];
      const float rgb[]{p.x, p.y, p.z};
      stream.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
    }
  }
  stream.close();
  if (!stream) throw std::runtime_error("Failed while writing image: " + path.string());
}

} // namespace caustra
