#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Decode E4M3FN bytes stored as INT8 using F32 multipliers per 128x128 block.
    // CPU or CUDA/HIP; output is F32 and must not alias either input.
    class DequantizeFP8 : public Op {
    public:
      void operator()(const StorageView& input, const StorageView& scale_inv,
                      StorageView& output) const;
    private:
      void compute_gpu(const StorageView& input, const StorageView& scale_inv,
                       StorageView& output) const;
    };

  }
}
