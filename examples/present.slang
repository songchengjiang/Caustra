// Display conversion belongs to the viewer, not the path-tracing kernel.
[[vk::binding(0, 0)]] StructuredBuffer<float4> hdr;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> display;
struct PresentConstants { uint sourceWidth, sourceHeight, width, height, outputBgra; };
[[vk::push_constant]] ConstantBuffer<PresentConstants> pc;

float3 displayTransform(float3 color) {
  color = max(color, 0.0f);
  color = color / (1.0f + color);
  return select(color <= 0.0031308f, 12.92f * color,
                1.055f * pow(color, float3(1.0f / 2.4f)) - 0.055f);
}

void storeDisplay(uint2 pixel, float3 color) {
  color = displayTransform(color);
  // CopyImage preserves bytes, including sRGB encoding; prepare the native
  // channel order here instead of requesting a format-converting image blit.
  if (pc.outputBgra != 0) color = color.bgr;
  display[pixel] = float4(color, 1.0f);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= pc.width || id.y >= pc.height) return;
  if (pc.sourceWidth == pc.width && pc.sourceHeight == pc.height) {
    // Integer addressing is essential for the ordinary one-to-one case. A
    // reciprocal-based floating-point remap can otherwise leak a tiny fraction
    // of an arbitrarily bright neighboring HDR sample into this pixel.
    storeDisplay(id.xy, hdr[id.y * pc.width + id.x].rgb);
    return;
  }
  // During interactive resizing the previous HDR frame may have a different
  // size. Bilinear filtering matches the old stretched viewport without a CPU
  // image transfer; ordinary frames address the same output-resolution pixels.
  float2 p = (float2(id.xy) + 0.5f) * float2(pc.sourceWidth, pc.sourceHeight)
           / float2(pc.width, pc.height) - 0.5f;
  int2 base = int2(floor(p));
  float2 f = frac(p);
  int2 limit = int2(pc.sourceWidth - 1, pc.sourceHeight - 1);
  int2 a = clamp(base, int2(0), limit);
  int2 b = clamp(base + 1, int2(0), limit);
  float3 top = lerp(hdr[a.y * pc.sourceWidth + a.x].rgb,
                    hdr[a.y * pc.sourceWidth + b.x].rgb, f.x);
  float3 bottom = lerp(hdr[b.y * pc.sourceWidth + a.x].rgb,
                       hdr[b.y * pc.sourceWidth + b.x].rgb, f.x);
  storeDisplay(id.xy, lerp(top, bottom, f.y));
}
