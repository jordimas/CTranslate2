#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Filters each channel using only current and past inputs, optionally applying SiLU.
    // CPU float32 cached depthwise cross-correlation. Input/output [B,T,C],
    // weights [C,K] ordered oldest-to-current, optional bias [C], mutable
    // history [B,C,K-1] ordered oldest-to-newest. History stores raw inputs.
    // Arbitrary chunking produces the same results and final history.
    // K>=1; T=0 leaves history unchanged. No mutable tensor may alias another
    // argument; checks include retained mutable allocations.
    class CausalDepthwiseConv1D : public Op {
    public:
      explicit CausalDepthwiseConv1D(bool apply_silu = true);

      void operator()(const StorageView& input,
                      const StorageView& weight,
                      const StorageView* bias,
                      StorageView& history,
                      StorageView& output) const;

    private:
      bool _apply_silu;
    };

  }
}
