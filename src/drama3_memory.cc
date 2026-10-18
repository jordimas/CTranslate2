#include "drama3_memory.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "env.h"

#if defined(__linux__)
#include <chrono>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <unistd.h>

struct sd_bus;
struct sd_bus_error;
struct sd_bus_message;
#endif

namespace ctranslate2 {
  uint64_t parse_drama3_memory_budget(const std::string& value, const char* name) {
    if (value.empty()) return 0;
    try {
      size_t parsed = 0;
      const long double gb = std::stold(value, &parsed);
      const long double bytes = gb * 1000000000.L;
      if (parsed == value.size() && std::isfinite(bytes) && gb >= 0
          && (gb == 0 || bytes >= 1)
          && bytes <= static_cast<long double>(std::numeric_limits<int64_t>::max()))
        return static_cast<uint64_t>(bytes);
    } catch (const std::exception&) {
    }
    throw std::invalid_argument(std::string(name) + " must be 0 (disabled) or a positive "
                                "number of decimal GB");
  }

  uint64_t drama3_memory_budget() {
    return parse_drama3_memory_budget(read_string_from_env("CT2_DRAMA3_MAX_MEMORY_GB"));
  }

  uint64_t drama3_gpu_expert_cache_budget() {
    return parse_drama3_memory_budget(read_string_from_env("CT2_DRAMA3_GPU_EXPERT_CACHE_GB"),
                                     "CT2_DRAMA3_GPU_EXPERT_CACHE_GB");
  }

#if defined(__linux__)
  namespace {
    std::filesystem::path current_cgroup() {
      std::ifstream file("/proc/self/cgroup");
      std::string line;
      while (std::getline(file, line)) {
        if (line.compare(0, 3, "0::") == 0)
          return std::filesystem::path("/sys/fs/cgroup")
            / std::filesystem::path(line.substr(3)).relative_path();
      }
      throw std::runtime_error("CT2_DRAMA3_MAX_MEMORY_GB requires Linux cgroup v2");
    }

    uint64_t effective_limit(std::filesystem::path path, const char* name) {
      uint64_t limit = std::numeric_limits<uint64_t>::max();
      const std::filesystem::path root("/sys/fs/cgroup");
      while (path.string().compare(0, root.string().size(), root.string()) == 0) {
        std::ifstream file(path / name);
        std::string value;
        if (file >> value && value != "max")
          limit = std::min(limit, static_cast<uint64_t>(std::stoull(value)));
        if (path == root) break;
        path = path.parent_path();
      }
      return limit;
    }

    bool budget_enforced(uint64_t bytes) {
      const auto path = current_cgroup();
      return effective_limit(path, "memory.max") <= bytes
        && effective_limit(path, "memory.swap.max") == 0;
    }
  }
#endif

  void enforce_drama3_memory_budget(uint64_t bytes) {
    if (!bytes) return;
#if defined(__linux__)
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    if (budget_enforced(bytes)) return;

    // Move this process only to a new scope. A virtual-address limit would
    // incorrectly count the whole mapped file against the resident budget.
    const std::string pid = std::to_string(getpid());
    static unsigned generation = 0;
    const auto limit = std::min(bytes, effective_limit(current_cgroup(), "memory.max"));
    const auto unit = "ct2-drama3-memory-" + pid + "-" + std::to_string(generation++) + ".scope";
    // Dynamically load the stable sd-bus API: ordinary CT2 use and externally
    // capped containers do not require systemd libraries or development headers.
    void* library = dlopen("libsystemd.so.0", RTLD_NOW | RTLD_LOCAL);
    int result = -1;
    if (library) {
      const auto create = reinterpret_cast<int (*)(sd_bus**)>(dlsym(library, "sd_bus_new"));
      const auto address = reinterpret_cast<int (*)(sd_bus*, const char*)>(
        dlsym(library, "sd_bus_set_address"));
      const auto start = reinterpret_cast<int (*)(sd_bus*)>(dlsym(library, "sd_bus_start"));
      const auto release = reinterpret_cast<sd_bus* (*)(sd_bus*)>(dlsym(library, "sd_bus_unref"));
      const auto timeout = reinterpret_cast<int (*)(sd_bus*, uint64_t)>(
        dlsym(library, "sd_bus_set_method_call_timeout"));
      const auto call = reinterpret_cast<int (*)(sd_bus*, const char*, const char*, const char*,
        const char*, sd_bus_error*, sd_bus_message**, const char*, ...)>(
          dlsym(library, "sd_bus_call_method"));
      sd_bus* bus = nullptr;
      const auto endpoint = "unix:path=" + read_string_from_env("XDG_RUNTIME_DIR",
        "/run/user/" + std::to_string(geteuid())) + "/systemd/private";
      if (create && address && start && release && timeout && call && create(&bus) >= 0) {
        if (address(bus, endpoint.c_str()) >= 0 && start(bus) >= 0 && timeout(bus, 5000000) >= 0)
          result = call(bus, "org.freedesktop.systemd1", "/org/freedesktop/systemd1",
                        "org.freedesktop.systemd1.Manager", "StartTransientUnit", nullptr, nullptr,
                        "ssa(sv)a(sa(sv))", unit.c_str(), "fail", 3,
                        "PIDs", "au", 1, static_cast<uint32_t>(getpid()),
                        "MemoryMax", "t", limit, "MemorySwapMax", "t", uint64_t(0), 0);
        release(bus);
      }
      dlclose(library);
    }
    if (result >= 0) {
      for (int i = 0; i < 100; ++i) {
        if (budget_enforced(bytes)) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    throw std::runtime_error("Unable to enforce CT2_DRAMA3_MAX_MEMORY_GB (systemd error "
                             + std::to_string(result) + "). Use Linux cgroup v2 "
                             "with a running systemd user manager and libsystemd.so.0, or launch in a "
                             "scope/container with memory.max <= the requested budget and "
                             "memory.swap.max=0. No model weights were loaded.");
#else
    throw std::invalid_argument("CT2_DRAMA3_MAX_MEMORY_GB currently requires Linux cgroup v2");
#endif
  }
}
