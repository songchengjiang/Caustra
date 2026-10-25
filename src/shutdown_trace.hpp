#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace caustra::detail {

inline bool shutdownTraceEnabled() noexcept {
  static const bool enabled=[] {
#ifdef _WIN32
    size_t length=0;
    if(getenv_s(&length,nullptr,0,"CAUSTRA_TRACE_SHUTDOWN")!=0 || length!=2) return false;
    char value[2]{};
    return getenv_s(&length,value,sizeof(value),"CAUSTRA_TRACE_SHUTDOWN")==0 && value[0]=='1';
#else
    const char* value=std::getenv("CAUSTRA_TRACE_SHUTDOWN");
    return value && value[0]=='1' && value[1]=='\0';
#endif
  }();
  return enabled;
}

// Opt-in, immediate diagnostics: the last begin without an end identifies a
// blocking resource release. No allocation, rendering state or lifetime change.
class ShutdownTrace {
public:
  explicit ShutdownTrace(const char* phase) noexcept : phase_(phase),active_(shutdownTraceEnabled()) {
    if(active_) {
      start_=Clock::now();
      std::fprintf(stderr,"[shutdown] %s begin\n",phase_);
      std::fflush(stderr);
    }
  }
  ~ShutdownTrace() { finish(); }
  ShutdownTrace(const ShutdownTrace&)=delete;
  ShutdownTrace& operator=(const ShutdownTrace&)=delete;
  void finish() noexcept {
    if(active_) {
      const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start_).count();
      std::fprintf(stderr,"[shutdown] %s end %.3f ms\n",phase_,elapsed);
      std::fflush(stderr);
      active_=false;
    }
  }
private:
  using Clock=std::chrono::steady_clock;
  const char* phase_;
  bool active_;
  Clock::time_point start_{};
};

} // namespace caustra::detail
