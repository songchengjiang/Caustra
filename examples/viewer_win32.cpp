// Interactive presentation belongs to the example; the renderer stays window-system independent.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <windowsx.h>
#include <caustra/renderer.hpp>
#include "orbit_camera.hpp"
#include "viewer_scene.hpp"

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
constexpr int toolbarHeight=72, footerHeight=38;
constexpr int idQuality=101, idSamples=102, idPause=103, idReset=104, idSave=105, idAuto=106;
struct Options {
  uint32_t width=960, height=640, frames=0, fpsLimit=60;
  int quality=-1; // auto: try DLSS quality, visibly fall back to raw path tracing
  bool validation=false, smoke=false, autoOrbit=false, help=false;
  std::filesystem::path screenshot;
};
uint32_t number(std::string_view value) {
  uint32_t n=0; auto r=std::from_chars(value.data(),value.data()+value.size(),n);
  if(r.ec!=std::errc{} || r.ptr!=value.data()+value.size() || !n) throw std::invalid_argument("Expected a positive integer");
  return n;
}
Options arguments(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string_view a=argv[i];
    if(a=="--help" || a=="-h") {o.help=true; continue;}
    if(a=="--validation") {o.validation=true; continue;}
    if(a=="--smoke-test") {o.smoke=true; continue;}
    if(a=="--auto-orbit") {o.autoOrbit=true; continue;}
    if(a!="--width" && a!="--height" && a!="--frames" && a!="--fps-limit" && a!="--dlss" && a!="--screenshot") throw std::invalid_argument("Unknown option: "+std::string(a));
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
    else if(a=="--width") o.width=number(v);
    else if(a=="--height") o.height=number(v);
    else if(a=="--frames") o.frames=number(v);
    else o.fpsLimit=number(v);
  }
  if(o.width<320 || o.width>3840 || o.height<240 || o.height>2160 || o.fpsLimit>240) throw std::invalid_argument("Width 320..3840, height 240..2160, FPS limit 1..240");
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
    "--validation           Vulkan validation and nonzero exit on errors\n"
    "--auto-orbit           Rotate the inspection camera continuously\n"
    "--frames N             Close after N presented frames\n"
    "--screenshot FILE.bmp  Save final window client including controls\n"
    "--smoke-test           Exercise window input; with DLSS, also test combo/key off-on cycles\n";
}
std::wstring wide(const std::string& s) {
  int n=MultiByteToWideChar(CP_UTF8,0,s.data(),static_cast<int>(s.size()),nullptr,0);
  std::wstring out(n,L' '); MultiByteToWideChar(CP_UTF8,0,s.data(),static_cast<int>(s.size()),out.data(),n); return out;
}
void require(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}

// Compose offscreen so neither the background clear nor partially drawn frames
// ever reach the window. Reuse GDI objects until the client dimensions change.
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
    renderer_.reset();
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
    RECT bounds{0,0,static_cast<LONG>(std::max(options_.width,880u)),static_cast<LONG>(options_.height+toolbarHeight+footerHeight)};
    AdjustWindowRect(&bounds,WS_OVERLAPPEDWINDOW|WS_CLIPCHILDREN,FALSE);
    window_=CreateWindowExW(0,wc.lpszClassName,L"Caustra | Starting Vulkan path tracer",WS_OVERLAPPEDWINDOW|WS_CLIPCHILDREN,CW_USEDEFAULT,CW_USEDEFAULT,bounds.right-bounds.left,bounds.bottom-bounds.top,nullptr,nullptr,wc.hInstance,this);
    require(window_!=nullptr,"CreateWindow failed");
    createControls(); ShowWindow(window_,SW_SHOW); UpdateWindow(window_);

    renderer_=std::make_unique<caustra::Renderer>(caustra::RendererOptions{options_.validation});
    settings_.maxBounces=6; settings_.environment={0.025f,0.03f,0.04f,0};
    renderer_->setScene(caustra::example::makeViewerScene());
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
    lastFrame_=lastStats_=Clock::now();
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
      if(pendingQuality_>=0) {
        int requestedMode=std::exchange(pendingQuality_,-1);
        try {applyQuality(requestedMode); notice_.clear();}
        catch(const std::exception& e) { notice_=e.what(); SendMessageW(quality_,CB_SETCURSEL,activeQuality_,0); }
      }
      if(resizePending_ && now-resizeTime_>std::chrono::milliseconds(180)) resizeRenderer();
      if(paused_ || now<nextFrame) {
        MsgWaitForMultipleObjectsEx(0,nullptr,paused_?40:1,QS_ALLINPUT,MWMO_INPUTAVAILABLE); continue;
      }
      float dt=std::clamp(std::chrono::duration<float>(now-lastFrame_).count(),0.0f,0.1f);
      lastFrame_=now;
      if(autoOrbit_) {orbit_.orbit(dt*70,0); cameraDirty_=true;}
      if(cameraDirty_) {renderer_->updateCamera(orbit_.camera()); cameraDirty_=false;}
      renderer_->render(samplesPerFrame_);
      image_=renderer_->readback();
      toneMap();
      ++frames_; ++statsFrames_;
      auto finish=Clock::now();
      if(finish-lastStats_>std::chrono::milliseconds(500)) {
        fps_=statsFrames_/std::chrono::duration<float>(finish-lastStats_).count(); statsFrames_=0; lastStats_=finish;
      }
      nextFrame=now+std::chrono::microseconds(1000000/options_.fpsLimit);
      InvalidateRect(window_,nullptr,FALSE); UpdateWindow(window_);
      if(options_.smoke) smokeStep();
      if(options_.frames && frames_>=options_.frames) {
        if(options_.smoke) verifySmoke();
        if(!options_.screenshot.empty()) capture(options_.screenshot);
        running_=false;
      }
    }
    require(renderer_->validationErrorCount()==0,"Vulkan validation errors occurred");
    std::cout<<"Presented "<<frames_<<" frames; "<<settings_.width<<'x'<<settings_.height<<"; Vulkan errors: "<<renderer_->validationErrorCount()<<'\n';
    return 0;
  }
private:
  Options options_;
  HWND window_{},quality_{},samples_{},pauseButton_{},autoButton_{};
  HFONT font_{},titleFont_{};
  bool running_=true,minimized_=false,paused_=false,autoOrbit_=false,cameraDirty_=false,resizePending_=false;
  int activeQuality_=0,pendingQuality_=-1,dragButton_=0;
  POINT lastMouse_{};
  std::unique_ptr<caustra::Renderer> renderer_;
  caustra::example::OrbitCamera orbit_;
  caustra::RenderSettings settings_;
  caustra::Image image_;
  std::vector<uint32_t> dib_;
  BackBuffer backBuffer_;
  uint32_t samplesPerFrame_=1,frames_=0,statsFrames_=0;
  float fps_=0;
  Clock::time_point lastFrame_,lastStats_,resizeTime_;
  std::string notice_;
  uint64_t initialHash_=0;
  uint32_t idleSamples_=0;
  int smokeDlssQuality_=0;
  unsigned smokeActions_=0,smokeDlssTransitions_=0;

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
  LRESULT message(UINT msg,WPARAM wp,LPARAM lp) {
    switch(msg) {
      case WM_CLOSE: DestroyWindow(window_); return 0;
      case WM_DESTROY: running_=false; PostQuitMessage(0); return 0;
      case WM_ERASEBKGND: return 1;
      case WM_GETMINMAXINFO: {
        auto* m=reinterpret_cast<MINMAXINFO*>(lp); m->ptMinTrackSize={896,400}; return 0;
      }
      case WM_SIZE:
        minimized_=wp==SIZE_MINIMIZED;
        if(!minimized_) {resizePending_=true;resizeTime_=Clock::now();layoutControls();}
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
      case 'D': pendingQuality_=activeQuality_?0:2; return true;
      case 'A': toggleAuto(); return true;
      case 'S': save(); return true;
      default: return false;
    }
  }
  void togglePause() {paused_=!paused_;SetWindowTextW(pauseButton_,paused_?L"Resume":L"Pause");InvalidateRect(window_,nullptr,FALSE);}
  void toggleAuto() {autoOrbit_=!autoOrbit_;SetWindowTextW(autoButton_,autoOrbit_?L"Auto: On":L"Auto: Off");}
  void resetCamera() {
    orbit_.reset(); autoOrbit_=false;SetWindowTextW(autoButton_,L"Auto: Off");
    if(renderer_) renderer_->setCamera(orbit_.camera());
    cameraDirty_=false; notice_.clear();
  }
  void createControls() {
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
  }
  void resizeRenderer() {
    if(!renderer_) return;
    auto r=viewport(); double w=std::max<LONG>(32,r.right),h=std::max<LONG>(32,r.bottom-r.top);
    // Bound CPU tone mapping and GDI presentation cost on large/high-DPI windows.
    const double scale=std::min(1.0,std::sqrt(1000000.0/(w*h)));
    auto next=settings_; next.width=std::max(32u,uint32_t(w*scale));next.height=std::max(32u,uint32_t(h*scale));
    if(next.width!=settings_.width || next.height!=settings_.height || !image_.width) {renderer_->setSettings(next);settings_=next;}
    resizePending_=false;
  }
  void toneMap() {
    static const auto lut=[] {
      std::array<uint8_t,8192> values{};
      for(size_t i=0;i<values.size();++i) {double x=double(i)/(values.size()-1),s=x<=0.0031308?12.92*x:1.055*std::pow(x,1/2.4)-0.055;values[i]=uint8_t(std::lround(s*255));}
      return values;
    }();
    auto channel=[&](float v) {v=std::max(0.0f,v);float x=v/(1+v);return uint32_t(lut[std::min(size_t(8191),size_t(x*8191+0.5f))]);};
    dib_.resize(image_.pixels.size());
    for(size_t i=0;i<dib_.size();++i) {auto p=image_.pixels[i]; require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),"Nonfinite viewer pixel");dib_[i]=(channel(p.x)<<16)|(channel(p.y)<<8)|channel(p.z);}
  }
  void paint() {
    PAINTSTRUCT ps{}; HDC windowDc=BeginPaint(window_,&ps); RECT client{};GetClientRect(window_,&client);
    HDC dc=backBuffer_.prepare(windowDc,client.right,client.bottom);
    if(!dc) {EndPaint(window_,&ps);return;}
    auto fill=[&](RECT r,COLORREF color) {auto brush=CreateSolidBrush(color);FillRect(dc,&r,brush);DeleteObject(brush);};
    fill(client,RGB(18,21,27)); auto v=viewport();
    if(!dib_.empty()) {
      BITMAPINFO info{};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=image_.width;info.bmiHeader.biHeight=-LONG(image_.height);info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;info.bmiHeader.biCompression=BI_RGB;
      SetStretchBltMode(dc,COLORONCOLOR); StretchDIBits(dc,0,v.top,v.right,v.bottom-v.top,0,0,image_.width,image_.height,dib_.data(),&info,DIB_RGB_COLORS,SRCCOPY);
    }
    fill({0,0,client.right,toolbarHeight},RGB(22,27,35));fill({0,client.bottom-footerHeight,client.right,client.bottom},RGB(22,27,35));
    SetBkMode(dc,TRANSPARENT);SetTextColor(dc,RGB(95,218,200));auto old=SelectObject(dc,titleFont_);TextOutW(dc,18,12,L"CAUSTRA",7);
    SelectObject(dc,font_);SetTextColor(dc,RGB(151,163,183));TextOutW(dc,20,42,L"PATH TRACE / LIVE",17);
    std::wostringstream stats;
    stats<<std::fixed<<std::setprecision(1)<<fps_<<L" FPS  |  "<<image_.width<<L" x "<<image_.height<<L"  |  "<<(renderer_?renderer_->sampleCount():0)<<L" spp"<<(paused_?L"  PAUSED":L"");
    auto text=stats.str(); RECT badge{16,toolbarHeight+14,client.right-16,toolbarHeight+45};
    auto badgeBrush=CreateSolidBrush(RGB(22,27,35)); RECT background=badge;background.right=std::min<LONG>(client.right-16,385);FillRect(dc,&background,badgeBrush);DeleteObject(badgeBrush);
    SetTextColor(dc,RGB(235,241,247));DrawTextW(dc,text.c_str(),-1,&badge,DT_LEFT|DT_VCENTER|DT_SINGLELINE);
    RECT status{16,client.bottom-footerHeight+8,client.right-12,client.bottom-6};
    auto hint=notice_.empty()?std::wstring(L"Left drag: orbit   |   Right drag: pan   |   Wheel: zoom   |   Space: pause   |   R: reset   |   S: save"):wide(notice_);
    SetTextColor(dc,notice_.empty()?RGB(151,163,183):RGB(255,198,108));DrawTextW(dc,hint.c_str(),-1,&status,DT_SINGLELINE|DT_END_ELLIPSIS);
    SelectObject(dc,old);
    // This is the only write to the visible client: publish a complete frame.
    const auto& dirty=ps.rcPaint;
    BitBlt(windowDc,dirty.left,dirty.top,dirty.right-dirty.left,dirty.bottom-dirty.top,dc,dirty.left,dirty.top,SRCCOPY);
    EndPaint(window_,&ps);
  }
  void save() {
    if(image_.pixels.empty()) return;
    try {
      std::filesystem::create_directories("renders");std::ostringstream stem;stem<<"renders/viewer_"<<std::setfill('0')<<std::setw(6)<<frames_;
      image_.savePpm(stem.str()+".ppm");image_.savePfm(stem.str()+".pfm");notice_="Saved "+stem.str()+".ppm / .pfm";
    } catch(const std::exception& e) {notice_=e.what();}
    InvalidateRect(window_,nullptr,FALSE);
  }
  uint64_t imageHash() const {
    uint64_t hash=1469598103934665603ULL;for(auto p:dib_) {hash^=p;hash*=1099511628211ULL;}return hash;
  }
  void verifyDlssTransition(int mode) {
    const auto settings=renderer_->dlssSettings();
    const auto quality=mode<2?caustra::DlssQuality::Native:mode==2?caustra::DlssQuality::Quality:mode==3?caustra::DlssQuality::Balanced:caustra::DlssQuality::Performance;
    require(activeQuality_==mode && SendMessageW(quality_,CB_GETCURSEL,0,0)==mode,"DLSS window selection did not switch");
    require(settings.enabled==(mode!=0) && settings.quality==quality && settings.preset==caustra::DlssPreset::E,"DLSS renderer configuration did not switch");
    require(renderer_->sampleCount()==samplesPerFrame_ && image_.samples==samplesPerFrame_,"DLSS switch did not reset accumulated samples");
    require(image_.width==settings_.width && image_.height==settings_.height && image_.pixels.size()==size_t(image_.width)*image_.height,"DLSS switch returned incorrect image dimensions");
    double energy=0;
    for(const auto p:image_.pixels) {
      require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&std::isfinite(p.w),"DLSS switch returned nonfinite pixels");
      energy+=p.x+p.y+p.z;
    }
    require(energy>0,"DLSS switch returned a black image");
    ++smokeDlssTransitions_;
  }
  void smokeStep() {
    if(frames_==1) initialHash_=imageHash();
    if(frames_==3) {
      require(renderer_->sampleCount()==samplesPerFrame_,"Orbit must reset stale raw accumulation");
      require(initialHash_!=imageHash(),"Same-sized viewport did not change after orbit");
    }
    if(frames_==16) idleSamples_=renderer_->sampleCount();
    if(frames_==17) require(renderer_->sampleCount()==idleSamples_+samplesPerFrame_,"Idle viewport failed to accumulate");
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
    const int x=viewport().right/2,y=toolbarHeight+(viewport().bottom-toolbarHeight)/2;
    auto drag=[&](UINT down,UINT up,WPARAM mask,int dx,int dy) {
      const auto before=orbit_.camera();
      SendMessageW(window_,down,mask,MAKELPARAM(x,y));SendMessageW(window_,WM_MOUSEMOVE,mask,MAKELPARAM(x+dx,y+dy));SendMessageW(window_,up,0,MAKELPARAM(x+dx,y+dy));
      const auto after=orbit_.camera();
      require(before.position.x!=after.position.x || before.position.y!=after.position.y || before.position.z!=after.position.z,"Mouse drag did not change camera");
      ++smokeActions_;
    };
    if(frames_==2) drag(WM_LBUTTONDOWN,WM_LBUTTONUP,MK_LBUTTON,95,24);
    if(frames_==4) {SendMessageW(window_,WM_MOUSEWHEEL,MAKEWPARAM(0,WHEEL_DELTA),0);++smokeActions_;}
    if(frames_==6) drag(WM_RBUTTONDOWN,WM_RBUTTONUP,MK_RBUTTON,24,-12);
    if(frames_==8) {
      RECT r{};GetWindowRect(window_,&r);SetWindowPos(window_,nullptr,0,0,r.right-r.left+64,r.bottom-r.top+32,SWP_NOMOVE|SWP_NOZORDER|SWP_NOACTIVATE);
      resizeRenderer();++smokeActions_;
    }
    if(frames_==10) {SendMessageW(window_,WM_COMMAND,idPause,0);require(paused_,"Pause input failed");SendMessageW(window_,WM_COMMAND,idPause,0);require(!paused_,"Resume input failed");++smokeActions_;}
    if(frames_==12) {SendMessageW(window_,WM_COMMAND,idReset,0);++smokeActions_;}
    if(frames_==14) drag(WM_LBUTTONDOWN,WM_LBUTTONUP,MK_LBUTTON,-40,15);
  }
  void verifySmoke() {
    require(smokeActions_==7,"Viewer input smoke sequence incomplete");
    if(smokeDlssQuality_) require(smokeDlssTransitions_==4,"DLSS window off-on smoke sequence incomplete");
    require(initialHash_!=imageHash(),"Viewport did not change after orbit/pan/zoom");
    double energy=0;for(auto p:image_.pixels) energy+=p.x+p.y+p.z;require(energy>0,"Black viewer image");
    require(renderer_->validationErrorCount()==0,"Viewer Vulkan validation errors");
    verifyRepaint();
    if(smokeDlssQuality_) std::cout<<"DLSS window smoke passed: two off-on cycles via combo selection and D shortcut.\n";
    std::cout<<"Viewer smoke passed: window, orbit, pan, zoom, resize, pause/resume, reset, stable cached repaint, finite nonblack output.\n";
  }
  void verifyRepaint() {
    // An exposed/paused window must repaint the cached frame without changing
    // pixels or leaking GDI objects. Keep sampling below the statistics badge.
    const auto area=viewport();
    HDC dc=GetDC(window_);
    auto hash=[&] {
      uint64_t value=1469598103934665603ULL;
      for(int y=1;y<8;++y) for(int x=1;x<8;++x) {
        value^=GetPixel(dc,x*area.right/8,area.top+50+y*std::max(1L,area.bottom-area.top-50)/8);
        value*=1099511628211ULL;
      }
      return value;
    };
    const auto expected=hash();
    const auto objects=GetGuiResources(GetCurrentProcess(),GR_GDIOBJECTS);
    bool stable=true;
    for(int i=0;i<32;++i) {
      RedrawWindow(window_,nullptr,nullptr,RDW_INVALIDATE|RDW_UPDATENOW|RDW_NOERASE|RDW_NOCHILDREN);
      stable=stable && hash()==expected;
    }
    const auto objectsAfter=GetGuiResources(GetCurrentProcess(),GR_GDIOBJECTS);
    ReleaseDC(window_,dc);
    require(stable,"Cached window repaint changed image pixels");
    require(objectsAfter==objects,"Cached repaint leaked GDI objects");
  }
  void capture(const std::filesystem::path& path) {
    require(path.extension()==".bmp","--screenshot expects .bmp");
    if(!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    RECT r{};GetClientRect(window_,&r);HDC src=GetDC(window_),memory=CreateCompatibleDC(src);
    HBITMAP bitmap=CreateCompatibleBitmap(src,r.right,r.bottom);auto old=SelectObject(memory,bitmap);
    BitBlt(memory,0,0,r.right,r.bottom,src,0,0,SRCCOPY);SelectObject(memory,old);
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
