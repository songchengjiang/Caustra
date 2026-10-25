// Interactive presentation belongs to the example; the renderer stays window-system independent.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <windowsx.h>
#include <caustra/renderer.hpp>
#if CAUSTRA_HAS_ASSETS
#include <caustra/asset.hpp>
#endif
#include "orbit_camera.hpp"
#include "viewer_scene.hpp"
#include "vulkan_presenter.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {
using Clock=std::chrono::steady_clock;
constexpr int toolbarHeight=72, footerHeight=72;
constexpr int idQuality=101, idSamples=102, idPause=103, idReset=104, idSave=105, idAuto=106;
struct Options {
  uint32_t width=960, height=640, frames=0, fpsLimit=60, maxBounces=6;
  int quality=-1; // auto: try DLSS quality, visibly fall back to raw path tracing
  bool validation=false, smoke=false, autoOrbit=false, help=false, strictMedia=false;
  float mediumTolerance=0.001f;
  std::filesystem::path screenshot;
  std::filesystem::path asset;
};
uint32_t number(std::string_view value) {
  uint32_t n=0; auto r=std::from_chars(value.data(),value.data()+value.size(),n);
  if(r.ec!=std::errc{} || r.ptr!=value.data()+value.size() || !n) throw std::invalid_argument("Expected a positive integer");
  return n;
}
float mediumTolerance(std::string_view value) {
  float tolerance=0;
  auto r=std::from_chars(value.data(),value.data()+value.size(),tolerance);
  if(r.ec!=std::errc{} || r.ptr!=value.data()+value.size() || !std::isfinite(tolerance) || tolerance<0)
    throw std::invalid_argument("Medium tolerance must be a finite nonnegative distance in meters");
  return tolerance;
}
Options arguments(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string_view a=argv[i];
    if(a=="--help" || a=="-h") {o.help=true; continue;}
    if(a=="--validation") {o.validation=true; continue;}
    if(a=="--strict-media") {o.strictMedia=true; continue;}
    if(a=="--smoke-test") {o.smoke=true; continue;}
    if(a=="--auto-orbit") {o.autoOrbit=true; continue;}
    if(a!="--width" && a!="--height" && a!="--frames" && a!="--fps-limit" && a!="--bounces" && a!="--dlss" && a!="--screenshot" && a!="--asset" && a!="--medium-tolerance") throw std::invalid_argument("Unknown option: "+std::string(a));
    if(++i==argc) throw std::invalid_argument("Missing option value");
    std::string_view v=argv[i];
    if(a=="--dlss") {
      if(v=="auto") o.quality=-1;
      else if(v=="off") o.quality=0;
      else if(v=="native") o.quality=1;
      else if(v=="quality") o.quality=2;
      else if(v=="balanced") o.quality=3;
      else if(v=="performance") o.quality=4;
      else throw std::invalid_argument("DLSS mode: auto/off/native/quality/balanced/performance");
    } else if(a=="--screenshot") o.screenshot=argv[i];
    else if(a=="--asset") o.asset=argv[i];
    else if(a=="--medium-tolerance") o.mediumTolerance=mediumTolerance(v);
    else if(a=="--width") o.width=number(v);
    else if(a=="--height") o.height=number(v);
    else if(a=="--frames") o.frames=number(v);
    else if(a=="--bounces") o.maxBounces=number(v);
    else o.fpsLimit=number(v);
  }
  if(o.width<320 || o.width>3840 || o.height<240 || o.height>2160 || o.fpsLimit>240) throw std::invalid_argument("Width 320..3840, height 240..2160, FPS limit 1..240");
  if(o.maxBounces>64) throw std::invalid_argument("Path depth must be in 1..64");
  if(o.smoke && !o.frames) o.frames=24;
  if(o.smoke && o.frames<20) throw std::invalid_argument("Smoke test needs at least 20 frames");
  return o;
}
void help() {
  std::cout<<"Caustra real-time path tracing viewer (Windows)\n"
    "Left drag: orbit | Right/middle drag: pan | Wheel: zoom\n"
    "Space: pause | R/Home: reset | D: toggle DLSS | A: auto orbit | S: save PPM/PFM | Esc: close\n"
    "--width N --height N    Initial viewport dimensions (default 960x640)\n"
    "--dlss MODE            auto/off/native/quality/balanced/performance (default auto)\n"
    "--fps-limit N          Presentation rate cap (default 60)\n"
    "--bounces N            Scattering depth 1..64 (default 6; use 24 for glass/SSS)\n"
    "--medium-tolerance M   Treat mismatched near boundaries as zero contact (meters; default 0.001)\n"
    "--strict-media         Fail on invalid medium nesting instead of skipping affected paths\n"
    "--validation           Vulkan validation and nonzero exit on errors\n"
    "--auto-orbit           Rotate the inspection camera continuously\n"
    "--frames N             Close after N presented frames\n"
    "--screenshot FILE.bmp  Save final window client including controls\n"
    "--asset FILE.caustra   Inspect a compiled Blender USD/MaterialX asset\n"
    "--smoke-test           Exercise window input; with DLSS, also test combo/key off-on cycles\n";
}
std::wstring wide(const std::string& s) {
  int n=MultiByteToWideChar(CP_UTF8,0,s.data(),static_cast<int>(s.size()),nullptr,0);
  std::wstring out(n,L' '); MultiByteToWideChar(CP_UTF8,0,s.data(),static_cast<int>(s.size()),out.data(),n); return out;
}
void require(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}

// GDI is used only for the native toolbar/footer. The child viewport is owned
// exclusively by the Vulkan swapchain and excluded from these copies.
class BackBuffer {
public:
  ~BackBuffer() {
    if(dc_) {
      SelectObject(dc_,original_);
      DeleteObject(bitmap_);
      DeleteDC(dc_);
    }
  }
  BackBuffer()=default;
  BackBuffer(const BackBuffer&)=delete;
  BackBuffer& operator=(const BackBuffer&)=delete;
  HDC prepare(HDC windowDc,int width,int height) {
    if(width<=0 || height<=0) return nullptr;
    if(width==width_ && height==height_) return dc_;
    auto bitmap=CreateCompatibleBitmap(windowDc,width,height);
    if(!bitmap) return nullptr;
    if(!dc_) {
      dc_=CreateCompatibleDC(windowDc);
      if(!dc_) {DeleteObject(bitmap);return nullptr;}
      original_=SelectObject(dc_,bitmap);
    } else {
      SelectObject(dc_,bitmap);
      DeleteObject(bitmap_);
    }
    bitmap_=bitmap;width_=width;height_=height;
    return dc_;
  }
private:
  HDC dc_{};
  HBITMAP bitmap_{};
  HGDIOBJ original_{};
  int width_=0,height_=0;
};

class Viewer {
public:
  explicit Viewer(Options options):options_(std::move(options)),autoOrbit_(options_.autoOrbit) {}
  ~Viewer() {
    renderer_=nullptr;
    presenter_.reset();
    if(window_ && IsWindow(window_)) DestroyWindow(window_);
    for(auto font:{font_,titleFont_}) if(font) DeleteObject(font);
  }
  int run() {
    // Per-monitor v2 when available; using physical pixels keeps mouse/render mapping stable.
    using DpiFn=BOOL(WINAPI*)(HANDLE);
    auto dpi=reinterpret_cast<DpiFn>(GetProcAddress(GetModuleHandleW(L"user32.dll"),"SetProcessDpiAwarenessContext"));
    if(dpi) dpi(reinterpret_cast<HANDLE>(-4));
    else SetProcessDPIAware();
    WNDCLASSW wc{}; wc.lpfnWndProc=windowProc; wc.hInstance=GetModuleHandleW(nullptr); wc.lpszClassName=L"CaustraRealtimeViewer"; wc.hCursor=LoadCursorW(nullptr,IDC_ARROW);
    require(RegisterClassW(&wc)!=0 || GetLastError()==ERROR_CLASS_ALREADY_EXISTS,"RegisterClass failed");
    WNDCLASSW viewportClass{};viewportClass.lpfnWndProc=viewportProc;viewportClass.hInstance=wc.hInstance;
    viewportClass.lpszClassName=L"CaustraVulkanViewport";viewportClass.hCursor=wc.hCursor;
    require(RegisterClassW(&viewportClass)!=0 || GetLastError()==ERROR_CLASS_ALREADY_EXISTS,"Register viewport class failed");
    RECT bounds{0,0,static_cast<LONG>(std::max(options_.width,880u)),static_cast<LONG>(options_.height+toolbarHeight+footerHeight)};
    AdjustWindowRect(&bounds,WS_OVERLAPPEDWINDOW|WS_CLIPCHILDREN,FALSE);
    window_=CreateWindowExW(0,wc.lpszClassName,L"Caustra | Starting Vulkan path tracer",WS_OVERLAPPEDWINDOW|WS_CLIPCHILDREN,CW_USEDEFAULT,CW_USEDEFAULT,bounds.right-bounds.left,bounds.bottom-bounds.top,nullptr,nullptr,wc.hInstance,this);
    require(window_!=nullptr,"CreateWindow failed");
    createControls(); ShowWindow(window_,SW_SHOW); UpdateWindow(window_);

    caustra::RendererOptions rendererOptions;
    rendererOptions.validation=options_.validation;
    rendererOptions.mediumErrorPolicy=options_.strictMedia?caustra::MediumErrorPolicy::Strict:caustra::MediumErrorPolicy::SkipInvalidPath;
    rendererOptions.mediumBoundaryTolerance=options_.mediumTolerance;
    presenter_=std::make_unique<caustra::example::VulkanPresenter>(viewportWindow_,rendererOptions);
    renderer_=&presenter_->renderer();
    settings_.maxBounces=options_.maxBounces; settings_.environment={0.025f,0.03f,0.04f,0};
    if(options_.asset.empty()) renderer_->setScene(caustra::example::makeViewerScene());
    else {
#if CAUSTRA_HAS_ASSETS
      auto asset=caustra::loadAsset(options_.asset);
      orbit_.setHome(asset.camera);renderer_->setScene(asset.scene);
      notice_="MaterialX asset: "+options_.asset.filename().string();
      std::cout<<"Asset: "<<asset.scene.meshes.size()<<" meshes, "<<asset.scene.materials.size()<<" materials, "<<asset.scene.textures.size()<<" textures\n";
#else
      throw std::runtime_error("Asset reader disabled. Build with CAUSTRA_BUILD_ASSETS=ON.");
#endif
    }
    renderer_->setCamera(orbit_.camera());
    resizeRenderer();
    const int requested=options_.quality<0?2:options_.quality;
    try {applyQuality(requested);} catch(const std::exception& e) {
      if(options_.quality>=0) throw;
      notice_="DLSS unavailable; using native path tracing"; std::cerr<<e.what()<<'\n';
      SendMessageW(quality_,CB_SETCURSEL,0,0);
    }
    smokeDlssQuality_=activeQuality_;
    SetWindowTextW(window_,(L"Caustra | "+wide(renderer_->deviceName())+L" | Real-time Path Trace").c_str());
    std::cout<<"Device: "<<renderer_->deviceName()<<"\n"<<renderer_->dlssStatus()<<'\n';
    std::cout<<"Medium boundaries: "<<(options_.strictMedia?"strict":"skip invalid paths")
      <<"; near-contact tolerance: "<<options_.mediumTolerance<<" m\n";
    lastFrame_=lastStats_=Clock::now();
    const auto presentationStart=Clock::now();
    auto nextFrame=Clock::now();
    MSG message{};
    while(running_) {
      while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE)) {
        if(message.message==WM_QUIT) {running_=false; break;}
        if(message.message==WM_KEYDOWN && message.hwnd!=quality_ && message.hwnd!=samples_) {
          const bool repeat=(message.lParam&(1LL<<30))!=0;
          if(!repeat && key(static_cast<UINT>(message.wParam))) continue;
        }
        TranslateMessage(&message); DispatchMessageW(&message);
      }
      if(!running_) break;
      if(minimized_) {WaitMessage(); continue;}
      const auto now=Clock::now();
      if(paused_) {
        if(needsPresent_ && frames_) republishCached();
        MsgWaitForMultipleObjectsEx(0,nullptr,40,QS_ALLINPUT,MWMO_INPUTAVAILABLE);continue;
      }
      if(now<nextFrame) {
        MsgWaitForMultipleObjectsEx(0,nullptr,1,QS_ALLINPUT,MWMO_INPUTAVAILABLE); continue;
      }
      // Mutations invalidate HDR/history, so commit them only when this loop
      // will immediately render the replacement frame.
      applyPendingConfiguration(now);
      float dt=std::clamp(std::chrono::duration<float>(now-lastFrame_).count(),0.0f,0.1f);
      lastFrame_=now;
      if(autoOrbit_) {orbit_.orbit(dt*70,0); cameraDirty_=true;}
      if(cameraCutPending_) {renderer_->setCamera(orbit_.camera());cameraCutPending_=false;cameraDirty_=false;}
      else if(cameraDirty_) {renderer_->updateCamera(orbit_.camera()); cameraDirty_=false;}
      const auto renderStart=Clock::now();
      renderer_->render(samplesPerFrame_);
      const auto rendered=Clock::now();
      updateMediumDiagnostics(rendered);
      const auto area=viewport();
      if(!presenter_->present(uint32_t(area.right),uint32_t(area.bottom-area.top))) continue;
      needsPresent_=false;
      const auto presented=Clock::now();
      ++frames_; ++statsFrames_;
      auto finish=Clock::now();
      if(finish-lastStats_>std::chrono::milliseconds(500)) {
        fps_=statsFrames_/std::chrono::duration<float>(finish-lastStats_).count(); statsFrames_=0; lastStats_=finish;
      }
      nextFrame=now+std::chrono::microseconds(1000000/options_.fpsLimit);
      InvalidateRect(window_,nullptr,FALSE); UpdateWindow(window_);
      const auto finished=Clock::now();
      if(options_.frames && frames_>5 && frameTimes_.size()<1024) {
        auto ms=[](auto a,auto b) {return std::chrono::duration<double,std::milli>(b-a).count();};
        frameTimes_.push_back({ms(renderStart,rendered),ms(rendered,presented),ms(renderStart,finished)});
      }
      if(options_.smoke) smokeStep();
      if(options_.frames && frames_>=options_.frames) {
        if(options_.smoke) verifySmoke();
        if(!options_.screenshot.empty()) capture(options_.screenshot);
        running_=false;
      }
    }
    require(renderer_->validationErrorCount()==0,"Vulkan validation errors occurred");
    std::cout<<"Presented "<<frames_<<" frames; "<<settings_.width<<'x'<<settings_.height<<"; Vulkan errors: "<<renderer_->validationErrorCount()<<'\n';
    std::cout<<"Medium diagnostics: invalid boundaries="<<mediumDiagnostics_.invalidMediumBoundaries
      <<", near-contact skips="<<mediumDiagnostics_.skippedNearMediumBoundaries<<'\n';
    if(options_.frames && !frameTimes_.empty()) {
      std::array<double,3> median{};
      for(size_t stage=0;stage<median.size();++stage) {
        std::vector<double> values;values.reserve(frameTimes_.size());
        for(const auto& frame:frameTimes_) values.push_back(frame[stage]);
        std::sort(values.begin(),values.end());median[stage]=(values[(values.size()-1)/2]+values[values.size()/2])*.5;
      }
      const auto elapsed=std::chrono::duration<double>(Clock::now()-presentationStart).count();
      std::cout<<std::fixed<<std::setprecision(3)<<"Median frame work (ms): render="<<median[0]<<", present="<<median[1]<<", total="<<median[2]
        <<"; run FPS="<<frames_/elapsed<<" (includes FPS cap and idle time)\n";
    }
    return 0;
  }
private:
  Options options_;
  HWND window_{},viewportWindow_{},quality_{},samples_{},pauseButton_{},autoButton_{};
  HFONT font_{},titleFont_{};
  bool running_=true,minimized_=false,paused_=false,autoOrbit_=false,cameraDirty_=false,resizePending_=false,rendererSized_=false;
  bool needsPresent_=false,cameraCutPending_=false;
  int activeQuality_=0,pendingQuality_=-1,dragButton_=0;
  POINT lastMouse_{};
  std::unique_ptr<caustra::example::VulkanPresenter> presenter_;
  caustra::Renderer* renderer_{};
  caustra::example::OrbitCamera orbit_;
  caustra::RenderSettings settings_;
  // CPU copies exist only at explicit smoke checkpoints, never per frame.
  caustra::Image probeImage_;
  std::vector<uint32_t> probeLdr_;
  std::vector<std::array<double,3>> frameTimes_;
  BackBuffer backBuffer_;
  uint32_t samplesPerFrame_=1,frames_=0,statsFrames_=0;
  float fps_=0;
  Clock::time_point lastFrame_,lastStats_,resizeTime_;
  Clock::time_point lastMediumWarning_{};
  caustra::RendererDiagnostics mediumDiagnostics_{},reportedMediumDiagnostics_{};
  bool mediumWarningReported_=false;
  std::string notice_;
  std::wstring resolutionStatus_;
  uint64_t initialHash_=0;
  uint32_t idleSamples_=0;
  int smokeDlssQuality_=0;
  unsigned smokeActions_=0,smokeDlssTransitions_=0;

  void updateMediumDiagnostics(Clock::time_point now) {
    mediumDiagnostics_=renderer_->diagnostics();
    const bool changed=mediumDiagnostics_.invalidMediumBoundaries!=reportedMediumDiagnostics_.invalidMediumBoundaries
      || mediumDiagnostics_.skippedNearMediumBoundaries!=reportedMediumDiagnostics_.skippedNearMediumBoundaries;
    if(changed && (!mediumWarningReported_ || now-lastMediumWarning_>=std::chrono::seconds(5))) {
      std::cerr<<"Medium recovery: invalid boundaries="<<mediumDiagnostics_.invalidMediumBoundaries
        <<", near-contact skips="<<mediumDiagnostics_.skippedNearMediumBoundaries<<"; rendering continues\n";
      reportedMediumDiagnostics_=mediumDiagnostics_;
      lastMediumWarning_=now;mediumWarningReported_=true;
    }
  }

  RECT viewport() const {
    RECT r{}; GetClientRect(window_,&r); r.top=toolbarHeight; r.bottom=std::max(r.top+1,r.bottom-footerHeight); return r;
  }
  static LRESULT CALLBACK windowProc(HWND hwnd,UINT message,WPARAM wParam,LPARAM lParam) {
    auto* self=reinterpret_cast<Viewer*>(GetWindowLongPtrW(hwnd,GWLP_USERDATA));
    if(message==WM_NCCREATE) {
      self=static_cast<Viewer*>(reinterpret_cast<CREATESTRUCTW*>(lParam)->lpCreateParams);
      self->window_=hwnd; SetWindowLongPtrW(hwnd,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(self));
    }
    if(!self) return DefWindowProcW(hwnd,message,wParam,lParam);
    return self->message(message,wParam,lParam);
  }
  static LRESULT CALLBACK viewportProc(HWND hwnd,UINT message,WPARAM wp,LPARAM lp) {
    auto* self=reinterpret_cast<Viewer*>(GetWindowLongPtrW(hwnd,GWLP_USERDATA));
    if(message==WM_NCCREATE) {
      self=static_cast<Viewer*>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams);
      SetWindowLongPtrW(hwnd,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(self));
    }
    if(message==WM_ERASEBKGND) return 1;
    if(message==WM_PAINT) {
      // DWM retains the last presented swapchain image while paused/exposed.
      // Validate the native region without allowing GDI to touch its pixels.
      PAINTSTRUCT ps{};BeginPaint(hwnd,&ps);EndPaint(hwnd,&ps);
      if(self) self->needsPresent_=true;
      return 0;
    }
    if(self) {
      switch(message) {
        case WM_SIZE: self->needsPresent_=true;return 0;
        case WM_LBUTTONDOWN:case WM_RBUTTONDOWN:case WM_MBUTTONDOWN:
        case WM_LBUTTONUP:case WM_RBUTTONUP:case WM_MBUTTONUP:case WM_MOUSEMOVE: {
          POINT p{GET_X_LPARAM(lp),GET_Y_LPARAM(lp)};
          MapWindowPoints(hwnd,self->window_,&p,1);
          return SendMessageW(self->window_,message,wp,MAKELPARAM(p.x,p.y));
        }
        case WM_MOUSEWHEEL: return SendMessageW(self->window_,message,wp,lp);
      }
    }
    return DefWindowProcW(hwnd,message,wp,lp);
  }
  LRESULT message(UINT msg,WPARAM wp,LPARAM lp) {
    switch(msg) {
      case WM_CLOSE: running_=false; return 0;
      case WM_DESTROY: running_=false; PostQuitMessage(0); return 0;
      case WM_ERASEBKGND: return 1;
      case WM_GETMINMAXINFO: {
        auto* m=reinterpret_cast<MINMAXINFO*>(lp); m->ptMinTrackSize={896,400}; return 0;
      }
      case WM_SIZE:
        minimized_=wp==SIZE_MINIMIZED;
        if(!minimized_) {resizePending_=true;needsPresent_=true;resizeTime_=Clock::now();layoutControls();}
        return 0;
      case WM_DPICHANGED: {
        auto* r=reinterpret_cast<RECT*>(lp); SetWindowPos(window_,nullptr,r->left,r->top,r->right-r->left,r->bottom-r->top,SWP_NOZORDER|SWP_NOACTIVATE); return 0;
      }
      case WM_PAINT: paint(); return 0;
      case WM_LBUTTONDOWN: case WM_RBUTTONDOWN: case WM_MBUTTONDOWN:
        if(GET_Y_LPARAM(lp)>=toolbarHeight && GET_Y_LPARAM(lp)<viewport().bottom) {
          SetFocus(window_); SetCapture(window_); dragButton_=msg==WM_LBUTTONDOWN?1:2; lastMouse_={GET_X_LPARAM(lp),GET_Y_LPARAM(lp)};
        } return 0;
      case WM_LBUTTONUP: case WM_RBUTTONUP: case WM_MBUTTONUP:
        dragButton_=0; if(GetCapture()==window_) ReleaseCapture(); return 0;
      case WM_CAPTURECHANGED: dragButton_=0; return 0;
      case WM_CANCELMODE: dragButton_=0; if(GetCapture()==window_) ReleaseCapture(); return 0;
      case WM_MOUSEMOVE:
        if(dragButton_) {
          const int x=GET_X_LPARAM(lp),y=GET_Y_LPARAM(lp);
          if(dragButton_==1) orbit_.orbit(float(x-lastMouse_.x),float(y-lastMouse_.y));
          else orbit_.pan(float(x-lastMouse_.x),float(y-lastMouse_.y),float(viewport().bottom-toolbarHeight));
          lastMouse_={x,y}; cameraDirty_=true;
        } return 0;
      case WM_MOUSEWHEEL: orbit_.zoom(float(GET_WHEEL_DELTA_WPARAM(wp))/WHEEL_DELTA); cameraDirty_=true; return 0;
      case WM_COMMAND: {
        bool refocus=false;
        switch(LOWORD(wp)) {
          case idQuality: if(HIWORD(wp)==CBN_SELCHANGE) {pendingQuality_=int(SendMessageW(quality_,CB_GETCURSEL,0,0));refocus=true;} break;
          case idSamples: if(HIWORD(wp)==CBN_SELCHANGE) {samplesPerFrame_=1u<<SendMessageW(samples_,CB_GETCURSEL,0,0);refocus=true;} break;
          case idPause: togglePause(); refocus=true; break;
          case idReset: resetCamera(); refocus=true; break;
          case idSave: save(); refocus=true; break;
          case idAuto: toggleAuto(); refocus=true; break;
        } if(refocus) SetFocus(window_); return 0;
      }
    }
    return DefWindowProcW(window_,msg,wp,lp);
  }
  bool key(UINT keyCode) {
    switch(keyCode) {
      case VK_ESCAPE: PostMessageW(window_,WM_CLOSE,0,0); return true;
      case VK_SPACE: togglePause(); return true;
      case 'R': case VK_HOME: resetCamera(); return true;
      case 'D': pendingQuality_=(pendingQuality_>=0?pendingQuality_:activeQuality_)?0:2; return true;
      case 'A': toggleAuto(); return true;
      case 'S': save(); return true;
      default: return false;
    }
  }
  void togglePause() {paused_=!paused_;needsPresent_=true;SetWindowTextW(pauseButton_,paused_?L"Resume":L"Pause");InvalidateRect(window_,nullptr,FALSE);}
  void toggleAuto() {autoOrbit_=!autoOrbit_;SetWindowTextW(autoButton_,autoOrbit_?L"Auto: On":L"Auto: Off");}
  void resetCamera() {
    orbit_.reset(); autoOrbit_=false;SetWindowTextW(autoButton_,L"Auto: Off");
    // Preserve the retained output while paused; apply the cut before the next
    // actual render, together with deferred resolution/DLSS changes.
    cameraCutPending_=true;
    cameraDirty_=false; notice_.clear();
  }
  void createControls() {
    const auto area=viewport();
    viewportWindow_=CreateWindowExW(0,L"CaustraVulkanViewport",L"",WS_CHILD|WS_VISIBLE|WS_CLIPSIBLINGS,
      0,area.top,area.right,area.bottom-area.top,window_,nullptr,GetModuleHandleW(nullptr),this);
    require(viewportWindow_!=nullptr,"Create Vulkan viewport failed");
    font_=CreateFontW(-15,0,0,0,FW_NORMAL,FALSE,FALSE,FALSE,DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,L"Segoe UI");
    titleFont_=CreateFontW(-23,0,0,0,FW_SEMIBOLD,FALSE,FALSE,FALSE,DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,L"Segoe UI");
    auto control=[&](const wchar_t* cls,const wchar_t* label,int id,DWORD style,int width) {
      auto hwnd=CreateWindowExW(0,cls,label,WS_CHILD|WS_VISIBLE|WS_TABSTOP|style,0,22,width,200,window_,reinterpret_cast<HMENU>(INT_PTR(id)),GetModuleHandleW(nullptr),nullptr);
      require(hwnd!=nullptr,"Create control failed"); SendMessageW(hwnd,WM_SETFONT,reinterpret_cast<WPARAM>(font_),TRUE); return hwnd;
    };
    quality_=control(L"COMBOBOX",L"",idQuality,CBS_DROPDOWNLIST|WS_VSCROLL,180);
    for(auto label:{L"DLSS: Off",L"DLSS: Native",L"DLSS: Quality",L"DLSS: Balanced",L"DLSS: Performance"}) SendMessageW(quality_,CB_ADDSTRING,0,reinterpret_cast<LPARAM>(label));
    SendMessageW(quality_,CB_SETCURSEL,0,0);
    samples_=control(L"COMBOBOX",L"",idSamples,CBS_DROPDOWNLIST,90);
    for(auto label:{L"1 spp",L"2 spp",L"4 spp",L"8 spp"}) SendMessageW(samples_,CB_ADDSTRING,0,reinterpret_cast<LPARAM>(label));
    SendMessageW(samples_,CB_SETCURSEL,0,0);
    pauseButton_=control(L"BUTTON",L"Pause",idPause,BS_PUSHBUTTON,72);
    control(L"BUTTON",L"Reset",idReset,BS_PUSHBUTTON,72);
    control(L"BUTTON",L"Save",idSave,BS_PUSHBUTTON,72);
    autoButton_=control(L"BUTTON",autoOrbit_?L"Auto: On":L"Auto: Off",idAuto,BS_PUSHBUTTON,90);
    layoutControls();
  }
  void layoutControls() {
    if(viewportWindow_) {
      const auto area=viewport();
      MoveWindow(viewportWindow_,0,area.top,area.right,area.bottom-area.top,TRUE);
    }
    if(!quality_) return;
    RECT r{};GetClientRect(window_,&r); int x=std::max(210,int(r.right)-650);
    for(auto item:std::array<std::pair<HWND,int>,6>{{{quality_,180},{samples_,80},{pauseButton_,70},{GetDlgItem(window_,idReset),70},{GetDlgItem(window_,idSave),70},{autoButton_,90}}}) {
      MoveWindow(item.first,x,24,item.second,item.first==quality_||item.first==samples_?200:28,TRUE); x+=item.second+12;
    }
  }
  void applyQuality(int mode) {
    caustra::DlssSettings settings; settings.enabled=mode!=0; settings.preset=caustra::DlssPreset::E;
    settings.quality=mode<2?caustra::DlssQuality::Native:mode==2?caustra::DlssQuality::Quality:mode==3?caustra::DlssQuality::Balanced:caustra::DlssQuality::Performance;
    renderer_->setDlssSettings(settings); activeQuality_=mode; SendMessageW(quality_,CB_SETCURSEL,mode,0);
    updateResolutionStatus();
  }
  void applyPendingConfiguration(Clock::time_point now) {
    if(paused_) return;
    if(pendingQuality_>=0) {
      const int requestedMode=std::exchange(pendingQuality_,-1);
      try {applyQuality(requestedMode);notice_.clear();}
      catch(const std::exception& e) {notice_=e.what();SendMessageW(quality_,CB_SETCURSEL,activeQuality_,0);}
    }
    if(resizePending_ && now-resizeTime_>std::chrono::milliseconds(180)) resizeRenderer();
  }
  bool republishCached() {
    if(!presenter_ || !frames_ || minimized_ || IsIconic(window_)) return false;
    const auto area=viewport();
    const bool shown=presenter_->present(uint32_t(area.right),uint32_t(area.bottom-area.top));
    needsPresent_=!shown;
    return shown;
  }
  void updateResolutionStatus() {
    if(renderer_->dlssSettings().enabled) resolutionStatus_=wide(renderer_->dlssStatus());
    else resolutionStatus_=L"Path Trace: "+std::to_wstring(settings_.width)+L" x "+std::to_wstring(settings_.height)+L" (native pixels)";
    auto r=viewport();
    std::cout<<"Viewport: "<<r.right<<'x'<<r.bottom-r.top<<"; output: "<<settings_.width<<'x'<<settings_.height<<"; "<<renderer_->dlssStatus()<<'\n';
  }
  void resizeRenderer() {
    if(!renderer_ || paused_) return;
    auto r=viewport();
    // The output follows physical viewport pixels. DLSS alone determines the
    // path-tracing input size; a hidden pixel cap must not rescale its output.
    auto next=settings_; next.width=uint32_t(std::max<LONG>(32,r.right));next.height=uint32_t(std::max<LONG>(32,r.bottom-r.top));
    if(next.width!=settings_.width || next.height!=settings_.height || !rendererSized_) {
      renderer_->setSettings(next);settings_=next;rendererSized_=true;updateResolutionStatus();
    }
    resizePending_=false;
  }
  void paint() {
    PAINTSTRUCT ps{}; HDC windowDc=BeginPaint(window_,&ps); RECT client{};GetClientRect(window_,&client);
    HDC dc=backBuffer_.prepare(windowDc,client.right,client.bottom);
    if(!dc) {EndPaint(window_,&ps);return;}
    auto fill=[&](RECT r,COLORREF color) {auto brush=CreateSolidBrush(color);FillRect(dc,&r,brush);DeleteObject(brush);};
    fill({0,0,client.right,toolbarHeight},RGB(22,27,35));fill({0,client.bottom-footerHeight,client.right,client.bottom},RGB(22,27,35));
    SetBkMode(dc,TRANSPARENT);SetTextColor(dc,RGB(95,218,200));auto old=SelectObject(dc,titleFont_);TextOutW(dc,18,12,L"CAUSTRA",7);
    SelectObject(dc,font_);SetTextColor(dc,RGB(151,163,183));TextOutW(dc,20,42,L"PATH TRACE / LIVE",17);
    std::wostringstream stats;
    stats<<std::fixed<<std::setprecision(1)<<fps_<<L" FPS  |  Output "<<settings_.width<<L" x "<<settings_.height<<L"  |  "<<(renderer_?renderer_->sampleCount():0)<<L" spp"<<(paused_?L"  PAUSED":L"");
    auto text=stats.str();RECT badge{16,client.bottom-footerHeight+5,client.right-16,client.bottom-footerHeight+25};
    SetTextColor(dc,RGB(235,241,247));DrawTextW(dc,text.c_str(),-1,&badge,DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    RECT resolutionBadge{16,client.bottom-footerHeight+26,client.right-16,client.bottom-footerHeight+46};
    SetTextColor(dc,RGB(151,163,183));DrawTextW(dc,resolutionStatus_.c_str(),-1,&resolutionBadge,DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS);
    RECT status{16,client.bottom-footerHeight+47,client.right-12,client.bottom-5};
    auto hint=notice_.empty()?std::wstring(L"Left drag: orbit   |   Right drag: pan   |   Wheel: zoom   |   Space: pause   |   R: reset   |   S: save"):wide(notice_);
    const bool mediumWarning=mediumDiagnostics_.invalidMediumBoundaries || mediumDiagnostics_.skippedNearMediumBoundaries;
    if(mediumWarning) hint=L"Media: invalid="+std::to_wstring(mediumDiagnostics_.invalidMediumBoundaries)
      +L", near skips="+std::to_wstring(mediumDiagnostics_.skippedNearMediumBoundaries)+L"  |  "+hint;
    SetTextColor(dc,notice_.empty() && !mediumWarning?RGB(151,163,183):RGB(255,198,108));DrawTextW(dc,hint.c_str(),-1,&status,DT_SINGLELINE|DT_END_ELLIPSIS);
    SelectObject(dc,old);
    // WS_CLIPCHILDREN keeps these native chrome copies out of the GPU viewport.
    const auto& dirty=ps.rcPaint;
    BitBlt(windowDc,dirty.left,dirty.top,dirty.right-dirty.left,dirty.bottom-dirty.top,dc,dirty.left,dirty.top,SRCCOPY);
    EndPaint(window_,&ps);
  }
  void save() {
    if(!renderer_ || !frames_) return;
    try {
      std::filesystem::create_directories("renders");std::ostringstream stem;stem<<"renders/viewer_"<<std::setfill('0')<<std::setw(6)<<frames_;
      const auto image=renderer_->readback();
      image.savePpm(stem.str()+".ppm");image.savePfm(stem.str()+".pfm");notice_="Saved "+stem.str()+".ppm / .pfm";
    } catch(const std::exception& e) {notice_=e.what();}
    InvalidateRect(window_,nullptr,FALSE);
  }
  uint64_t imageHash() const {
    uint64_t hash=1469598103934665603ULL;for(auto p:probeLdr_) {hash^=p&0xffffffu;hash*=1099511628211ULL;}return hash;
  }
  void probeOutput() {
    probeImage_=renderer_->readback();
    probeLdr_=presenter_->readbackLdr();
    require(probeImage_.width==presenter_->width() && probeImage_.height==presenter_->height() &&
      probeLdr_.size()==probeImage_.pixels.size(),"GPU display capture does not match HDR output dimensions");
    auto channel=[](float value) {
      const double positive=std::max(0.0,double(value)),mapped=positive/(1+positive);
      const double encoded=mapped<=.0031308?12.92*mapped:1.055*std::pow(mapped,1/2.4)-.055;
      return int(std::lround(encoded*255));
    };
    int maxDifference=0;double energy=0;size_t worstPixel=0;
    for(size_t i=0;i<probeImage_.pixels.size();++i) {
      const auto p=probeImage_.pixels[i];
      require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&std::isfinite(p.w),"Nonfinite HDR output at display checkpoint");
      energy+=p.x+p.y+p.z;
      const auto ldr=probeLdr_[i];
      const auto difference=std::max({std::abs(channel(p.x)-int((ldr>>16)&255u)),
        std::abs(channel(p.y)-int((ldr>>8)&255u)),std::abs(channel(p.z)-int(ldr&255u))});
      if(difference>maxDifference) {maxDifference=difference;worstPixel=i;}
    }
    require(energy>0,"Black HDR output at display checkpoint");
    if(maxDifference>2) {
      const auto p=probeImage_.pixels[worstPixel];const auto ldr=probeLdr_[worstPixel];
      std::cerr<<"GPU tone map mismatch: maximum="<<maxDifference<<" at ("<<worstPixel%probeImage_.width<<','<<worstPixel/probeImage_.width
        <<"), HDR=("<<p.x<<','<<p.y<<','<<p.z<<"), GPU RGB=("<<((ldr>>16)&255u)<<','<<((ldr>>8)&255u)<<','<<(ldr&255u)
        <<"), expected RGB=("<<channel(p.x)<<','<<channel(p.y)<<','<<channel(p.z)<<"), HDR extent="<<probeImage_.width<<'x'<<probeImage_.height
        <<", display extent="<<presenter_->width()<<'x'<<presenter_->height()<<'\n';
    }
    require(maxDifference<=2,"GPU tone mapping differs from HDR Reinhard/sRGB reference (channel, gamma or orientation mismatch)");
  }
  void verifyDlssTransition(int mode) {
    probeOutput();
    const auto settings=renderer_->dlssSettings();
    const auto quality=mode<2?caustra::DlssQuality::Native:mode==2?caustra::DlssQuality::Quality:mode==3?caustra::DlssQuality::Balanced:caustra::DlssQuality::Performance;
    require(activeQuality_==mode && SendMessageW(quality_,CB_GETCURSEL,0,0)==mode,"DLSS window selection did not switch");
    require(settings.enabled==(mode!=0) && settings.quality==quality && settings.preset==caustra::DlssPreset::E,"DLSS renderer configuration did not switch");
    require(renderer_->sampleCount()==samplesPerFrame_ && probeImage_.samples==samplesPerFrame_,"DLSS switch did not reset accumulated samples");
    require(probeImage_.width==settings_.width && probeImage_.height==settings_.height && probeImage_.pixels.size()==size_t(probeImage_.width)*probeImage_.height,"DLSS switch returned incorrect image dimensions");
    ++smokeDlssTransitions_;
  }
  void smokeStep() {
    if(frames_==1) {probeOutput();initialHash_=imageHash();}
    if(frames_==3) {
      probeOutput();
      require(renderer_->sampleCount()==samplesPerFrame_,"Orbit must reset stale raw accumulation");
      require(initialHash_!=imageHash(),"Same-sized viewport did not change after orbit");
    }
    if(frames_==16) idleSamples_=renderer_->sampleCount();
    if(frames_==17) require(renderer_->sampleCount()==idleSamples_+samplesPerFrame_,"Idle viewport failed to accumulate");
    if(frames_==18) verifyPausedResizeRestore();
    if(frames_==19) require(renderer_->sampleCount()==samplesPerFrame_,"Deferred paused camera cut did not apply on resume");
    if(smokeDlssQuality_) {
      auto selectQuality=[&](int mode) {
        SendMessageW(quality_,CB_SETCURSEL,mode,0);
        SendMessageW(window_,WM_COMMAND,MAKEWPARAM(idQuality,CBN_SELCHANGE),reinterpret_cast<LPARAM>(quality_));
      };
      if(frames_==4) selectQuality(0);
      if(frames_==5) {verifyDlssTransition(0);selectQuality(smokeDlssQuality_);}
      if(frames_==6) verifyDlssTransition(smokeDlssQuality_);
      // Queue keyboard input so it passes through the same shortcut dispatch as a user press.
      if(frames_==10 || frames_==11) {
        if(frames_==11) verifyDlssTransition(0);
        require(PostMessageW(window_,WM_KEYDOWN,'D',1)!=0,"Could not queue DLSS shortcut");
      }
      if(frames_==12) verifyDlssTransition(2);
    }
    const int x=viewport().right/2,y=(viewport().bottom-toolbarHeight)/2;
    auto drag=[&](UINT down,UINT up,WPARAM mask,int dx,int dy) {
      const auto before=orbit_.camera();
      SendMessageW(viewportWindow_,down,mask,MAKELPARAM(x,y));SendMessageW(viewportWindow_,WM_MOUSEMOVE,mask,MAKELPARAM(x+dx,y+dy));SendMessageW(viewportWindow_,up,0,MAKELPARAM(x+dx,y+dy));
      const auto after=orbit_.camera();
      require(before.position.x!=after.position.x || before.position.y!=after.position.y || before.position.z!=after.position.z,"Mouse drag did not change camera");
      ++smokeActions_;
    };
    if(frames_==2) drag(WM_LBUTTONDOWN,WM_LBUTTONUP,MK_LBUTTON,95,24);
    if(frames_==4) {SendMessageW(viewportWindow_,WM_MOUSEWHEEL,MAKEWPARAM(0,WHEEL_DELTA),0);++smokeActions_;}
    if(frames_==6) drag(WM_RBUTTONDOWN,WM_RBUTTONUP,MK_RBUTTON,24,-12);
    if(frames_==8) {
      RECT r{},client{};GetWindowRect(window_,&r);GetClientRect(window_,&client);
      const int borderWidth=r.right-r.left-client.right,borderHeight=r.bottom-r.top-client.bottom;
      SetWindowPos(window_,nullptr,0,0,std::max(r.right-r.left+64,LONG(1280+borderWidth)),
        std::max(r.bottom-r.top+32,LONG(840+toolbarHeight+footerHeight+borderHeight)),SWP_NOMOVE|SWP_NOZORDER|SWP_NOACTIVATE);
      resizeRenderer();
      const auto area=viewport();
      require(uint64_t(area.right)*uint64_t(area.bottom-area.top)>1000000,"Resize regression must exceed the former hidden pixel cap");
      require(settings_.width==uint32_t(area.right) && settings_.height==uint32_t(area.bottom-area.top),"Viewport output was silently downscaled after resize");
      ++smokeActions_;
    }
    if(frames_==10) {SendMessageW(window_,WM_COMMAND,idPause,0);require(paused_,"Pause input failed");SendMessageW(window_,WM_COMMAND,idPause,0);require(!paused_,"Resume input failed");++smokeActions_;}
    if(frames_==12) {SendMessageW(window_,WM_COMMAND,idReset,0);++smokeActions_;}
    if(frames_==14) drag(WM_LBUTTONDOWN,WM_LBUTTONUP,MK_LBUTTON,-40,15);
  }
  void verifyPausedResizeRestore() {
    require(!paused_,"Paused presentation smoke must start from a running viewer");
    const auto samples=renderer_->sampleCount();
    const auto history=renderer_->frameCount();
    const auto originalSettings=settings_;
    const auto originalDlss=renderer_->dlssSettings();
    RECT original{};GetWindowRect(window_,&original);
    const int originalWidth=original.right-original.left,originalHeight=original.bottom-original.top;
    auto pump=[&] {
      MSG message{};
      while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE)) {
        require(message.message!=WM_QUIT,"Window unexpectedly closed during paused resize smoke");
        TranslateMessage(&message);DispatchMessageW(&message);
      }
    };
    auto unchanged=[&] {
      require(renderer_->sampleCount()==samples && renderer_->frameCount()==history,"Paused presentation reset or advanced rendering history");
      require(settings_.width==originalSettings.width && settings_.height==originalSettings.height,"Paused resize invalidated retained HDR output");
      require(renderer_->dlssSettings().enabled==originalDlss.enabled,"Paused DLSS selection invalidated retained HDR output");
    };
    unsigned checkpoint=0;
    auto display=[&] {
      ++checkpoint;
      pump();
      // Exercise the same deferred configuration/presentation helpers as the
      // real paused loop, without waiting for debounce or rendering samples.
      applyPendingConfiguration(Clock::now()+std::chrono::seconds(1));
      resizeRenderer();unchanged();
      // Native surface extents may settle after the size message; an
      // OUT_OF_DATE acquire/present is retried by the real paused loop too.
      const auto deadline=Clock::now()+std::chrono::seconds(2);
      bool shown=republishCached();
      while(!shown && Clock::now()<deadline) {
        MsgWaitForMultipleObjectsEx(0,nullptr,10,QS_ALLINPUT,MWMO_INPUTAVAILABLE);
        pump();shown=republishCached();
      }
      if(!shown) {
        const auto area=viewport();
        std::cerr<<"Paused presentation checkpoint="<<checkpoint<<", minimized="<<minimized_
          <<", iconic="<<IsIconic(window_)<<", viewport="<<area.right<<'x'<<area.bottom-area.top
          <<", GPU extent="<<presenter_->width()<<'x'<<presenter_->height()<<", frames="<<frames_<<'\n';
      }
      require(shown,"Paused viewport could not republish retained GPU output");
      const auto area=viewport();const auto pixels=presenter_->readbackLdr();
      require(presenter_->width()==uint32_t(area.right) && presenter_->height()==uint32_t(area.bottom-area.top) &&
        pixels.size()==size_t(presenter_->width())*presenter_->height(),"Paused GPU capture does not match resized viewport");
      unchanged();
    };
    togglePause();
    // Queue an actual camera cut and, when available, a different DLSS mode.
    // Neither may destroy the cached output before resuming the render loop.
    resetCamera();pendingQuality_=activeQuality_?0:activeQuality_;
    SetWindowPos(window_,nullptr,0,0,originalWidth+32,originalHeight+24,SWP_NOMOVE|SWP_NOZORDER|SWP_NOACTIVATE);display();
    SetWindowPos(window_,nullptr,0,0,originalWidth-48,originalHeight-32,SWP_NOMOVE|SWP_NOZORDER|SWP_NOACTIVATE);display();
    ShowWindow(window_,SW_MINIMIZE);pump();
    require(minimized_ && !republishCached(),"Minimized viewer attempted to present");unchanged();
    WINDOWPLACEMENT placement{sizeof(WINDOWPLACEMENT)};
    require(GetWindowPlacement(window_,&placement)!=0,"Could not query minimized window placement");
    placement.flags=0;placement.showCmd=SW_SHOWNORMAL;
    require(SetWindowPlacement(window_,&placement)!=0,"Could not restore window placement");
    display();
    SetWindowPos(window_,nullptr,0,0,originalWidth,originalHeight,SWP_NOMOVE|SWP_NOZORDER|SWP_NOACTIVATE);display();
    pendingQuality_=-1;SendMessageW(quality_,CB_SETCURSEL,activeQuality_,0);
    togglePause();
    require(!paused_ && cameraCutPending_,"Resume lost its deferred camera cut");
    std::cout<<"Paused GPU presentation smoke passed: resize up/down, minimize/restore, retained sample/history counts and deferred configuration.\n";
  }
  void verifySmoke() {
    probeOutput();
    require(smokeActions_==7,"Viewer input smoke sequence incomplete");
    if(smokeDlssQuality_) require(smokeDlssTransitions_==4,"DLSS window off-on smoke sequence incomplete");
    require(initialHash_!=imageHash(),"Viewport did not change after orbit/pan/zoom");
    require(renderer_->validationErrorCount()==0,"Viewer Vulkan validation errors");
    const auto area=viewport();
    require(probeImage_.width==uint32_t(area.right) && probeImage_.height==uint32_t(area.bottom-area.top) &&
      probeImage_.pixels.size()==size_t(probeImage_.width)*probeImage_.height,"Resized readback does not match physical viewport pixels");
    verifyRepaint();
    if(smokeDlssQuality_) std::cout<<"DLSS window smoke passed: two off-on cycles via combo selection and D shortcut.\n";
    std::cout<<"Viewer smoke passed: window, child viewport input, orbit, pan, zoom, resize, pause/resume, reset, stable GPU repaint, finite nonblack output and GPU tone mapping reference.\n";
  }
  void verifyRepaint() {
    // Presenting the retained GPU output and repainting native chrome must not
    // restart path tracing, change the image, or leak native GDI objects.
    const auto area=viewport();
    const auto expected=probeLdr_;
    const auto samples=renderer_->sampleCount();
    const auto objects=GetGuiResources(GetCurrentProcess(),GR_GDIOBJECTS);
    for(int i=0;i<32;++i) {
      RedrawWindow(window_,nullptr,nullptr,RDW_INVALIDATE|RDW_UPDATENOW|RDW_NOERASE|RDW_ALLCHILDREN);
      require(presenter_->present(uint32_t(area.right),uint32_t(area.bottom-area.top)),"Retained GPU frame could not be presented again");
    }
    const auto actual=presenter_->readbackLdr();
    const auto objectsAfter=GetGuiResources(GetCurrentProcess(),GR_GDIOBJECTS);
    require(actual==expected,"Retained GPU repaint changed image pixels");
    require(renderer_->sampleCount()==samples,"Retained GPU repaint unexpectedly traced new samples");
    require(objectsAfter==objects,"Cached repaint leaked GDI objects");
  }
  void capture(const std::filesystem::path& path) {
    require(path.extension()==".bmp","--screenshot expects .bmp");
    if(!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    RECT r{};GetClientRect(window_,&r);HDC src=GetDC(window_),memory=CreateCompatibleDC(src);
    HBITMAP bitmap=CreateCompatibleBitmap(src,r.right,r.bottom);auto old=SelectObject(memory,bitmap);
    BitBlt(memory,0,0,r.right,r.bottom,src,0,0,SRCCOPY);
    // Native BitBlt cannot reliably capture a Vulkan surface. Compose its
    // explicit GPU readback only for this requested screenshot.
    const auto display=presenter_->readbackLdr();
    const auto area=viewport();
    BITMAPINFO viewportInfo{};viewportInfo.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
    viewportInfo.bmiHeader.biWidth=LONG(presenter_->width());viewportInfo.bmiHeader.biHeight=-LONG(presenter_->height());
    viewportInfo.bmiHeader.biPlanes=1;viewportInfo.bmiHeader.biBitCount=32;viewportInfo.bmiHeader.biCompression=BI_RGB;
    SetStretchBltMode(memory,COLORONCOLOR);
    require(StretchDIBits(memory,0,area.top,area.right,area.bottom-area.top,0,0,presenter_->width(),presenter_->height(),
      display.data(),&viewportInfo,DIB_RGB_COLORS,SRCCOPY)!=GDI_ERROR,"GPU viewport screenshot composition failed");
    SelectObject(memory,old);
    BITMAPINFO info{};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=r.right;info.bmiHeader.biHeight=-r.bottom;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;info.bmiHeader.biCompression=BI_RGB;
    std::vector<uint32_t> pixels(size_t(r.right)*r.bottom);int lines=GetDIBits(memory,bitmap,0,r.bottom,pixels.data(),&info,DIB_RGB_COLORS);
    DeleteObject(bitmap);DeleteDC(memory);ReleaseDC(window_,src);require(lines==r.bottom,"Window screenshot capture failed");
    BITMAPFILEHEADER file{};file.bfType=0x4d42;file.bfOffBits=sizeof(file)+sizeof(BITMAPINFOHEADER);file.bfSize=file.bfOffBits+DWORD(pixels.size()*4);
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(&file),sizeof(file));out.write(reinterpret_cast<const char*>(&info.bmiHeader),sizeof(info.bmiHeader));out.write(reinterpret_cast<const char*>(pixels.data()),pixels.size()*4);out.close();require(bool(out),"Window screenshot write failed");
    std::cout<<"Screenshot: "<<path.string()<<'\n';
  }
};
}

int main(int argc,char** argv) {
  std::cout<<std::unitbuf;
  bool smoke=false;
  try {
    auto options=arguments(argc,argv);smoke=options.smoke;
    if(options.help) {help();return 0;}
    Viewer viewer(std::move(options));return viewer.run();
  } catch(const std::exception& e) {
    std::cerr<<"Caustra viewer: "<<e.what()<<'\n';
    if(!smoke) MessageBoxW(nullptr,wide(e.what()).c_str(),L"Caustra viewer",MB_OK|MB_ICONERROR);
    return 1;
  }
}
