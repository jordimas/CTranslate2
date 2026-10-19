#pragma once

#include "ctranslate2/types.h"
#include "ctranslate2/devices.h"
#include <cstdint>

namespace ctranslate2 {
  namespace layers {

    // Execution belongs to a replica, independently of checkpoint storage.
    struct DramaExecutionOptions {
      Device device = Device::CPU;
      int device_index = 0;
      ComputeType compute_type = ComputeType::FLOAT32;
      uint64_t expert_cache_bytes = 0;
      bool resident_backbone = true;
      bool waveform = true;
      bool grouped_moe = false;

      bool resident_gpu() const {
        return device == Device::CUDA && resident_backbone;
      }
    };

  }
}
