#include <caustra/renderer.hpp>
#include "demo_scene.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool c,const char* message) { if(!c) throw std::runtime_error(message); }
double energy(const caustra::Image& image) {
  double total=0;
  require(image.pixels.size()==size_t(image.width)*image.height,"Incorrect DLSS output dimensions");
  for(auto p:image.pixels) {
    require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),"Nonfinite DLSS pixel");
    total+=p.x+p.y+p.z;
  }
  require(total>0,"DLSS returned black image"); return total;
}
int main() {
  try {
    caustra::Renderer renderer({true});
    caustra::RenderSettings render; render.width=320; render.height=240;
    renderer.setSettings(render); renderer.setScene(caustra::example::makeDemoScene()); renderer.setCamera(caustra::example::makeDemoCamera());
    caustra::DlssSettings dlss; dlss.enabled=true;
    renderer.setDlssSettings(dlss); std::cout<<renderer.dlssStatus()<<'\n';
    renderer.render(8); auto native=renderer.readback(); auto raw=renderer.readbackRaw();
    energy(native); energy(raw);
    require(native.width==320 && native.height==240 && native.samples==8,"Native RR dimensions or frame counter");
    double delta=0; for(size_t i=0;i<raw.pixels.size();++i) delta+=std::abs(native.pixels[i].x-raw.pixels[i].x);
    require(delta>0.001,"DLSS output is identical to raw input");
    renderer.reset(); renderer.render(2); energy(renderer.readback()); require(renderer.sampleCount()==2,"RR reset failed");
    dlss.quality=caustra::DlssQuality::Quality; dlss.preset=caustra::DlssPreset::E;
    renderer.setDlssSettings(dlss); renderer.render(4); auto quality=renderer.readback(); auto inner=renderer.readbackRaw();
    energy(quality); energy(inner); require(quality.width==320 && quality.height==240 && inner.width<320,"RR quality upscaling dimensions");
    for(auto mode:{caustra::DlssQuality::Balanced,caustra::DlssQuality::Performance}) {
      dlss.quality=mode; dlss.preset=caustra::DlssPreset::D;
      renderer.setDlssSettings(dlss); renderer.render(2); energy(renderer.readback());
      require(renderer.readbackRaw().width<320,"RR balanced/performance must use lower internal resolution");
    }
    const auto beforeFailure=renderer.sampleCount();
    auto invalid=dlss; invalid.preset=static_cast<caustra::DlssPreset>(99);
    bool rejected=false; try { renderer.setDlssSettings(invalid); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected && renderer.sampleCount()==beforeFailure,"Rejected RR config must preserve current state");
    auto tooSmall=render; tooSmall.width=16;
    rejected=false; try { renderer.setSettings(tooSmall); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected && renderer.sampleCount()==beforeFailure,"Rejected RR dimensions must preserve current state");
    auto camera=caustra::example::makeDemoCamera(); camera.position.x+=0.2f; renderer.setCamera(camera); renderer.render(2); energy(renderer.readback());
    render.width=384; render.height=256; renderer.setSettings(render); renderer.render(2); auto resized=renderer.readback(); energy(resized); require(resized.width==384 && resized.height==256,"RR resize failed");
    dlss.enabled=false; renderer.setDlssSettings(dlss); renderer.render(2); auto disabled=renderer.readback(); auto disabledRaw=renderer.readbackRaw();
    require(disabled.pixels[123].x==disabledRaw.pixels[123].x,"Disabled RR must return raw output");
    require(renderer.validationErrorCount()==0,"Vulkan validation errors in DLSS integration");
    std::cout<<"DLSS RR passed: native, quality/balanced/performance upscale, Default/D/E presets, reset, camera change, resize, invalid config and disable.\n";
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
