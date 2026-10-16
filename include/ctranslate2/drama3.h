#pragma once

#include <memory>
#include <cstdint>
#include <string>
#include "storage_view.h"

namespace ctranslate2 {
  // Wall-clock seconds from steady_clock. Total excludes waiting for the
  // request mutex and Python output copying; stage sums exclude bookkeeping.
  struct Drama3Timings {
    double prefill_seconds = 0;
    double backbone_decode_seconds = 0;
    double tower_decode_seconds = 0;
    double codec_seconds = 0;
    double total_native_seconds = 0;
  };

  struct Drama3Result {
    StorageView audio;
    StorageView codes;
    int sample_rate = 44100;
    std::string finish_reason;
    Drama3Timings timings;
  };

  // Single resident CPU model; sequential requests, independent per-call state.
  // Seeded sampling is the default; greedy decoding is available for diagnosis.
  class Drama3 {
  public:
    explicit Drama3(const std::string& model_path, size_t intra_threads = 8,
                    const std::string& compute_type = "int8_float32");
    ~Drama3();
    Drama3Result generate(const StorageView& prompt, dim_t max_frames = 512,
                          dim_t prefill_chunk_size = 16, float temperature = .8f,
                          dim_t top_k = 30, float top_p = .8f, uint64_t seed = 0,
                          bool greedy = false);
  private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
  };
}
