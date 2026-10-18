#pragma once

#include <cstdint>
#include <string>

namespace ctranslate2 {
  // Zero disables the process budget. Values use decimal GB, not GiB.
  uint64_t parse_drama3_memory_budget(const std::string& value,
                                    const char* name = "CT2_DRAMA3_MAX_MEMORY_GB");
  uint64_t drama3_memory_budget();
  uint64_t drama3_gpu_expert_cache_budget();
  void enforce_drama3_memory_budget(uint64_t bytes);
}
