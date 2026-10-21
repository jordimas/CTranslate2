#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    // Adds a learned periodic component to codec activations using a sine-squared function.
    // CPU or CUDA/HIP float32 Snake activation for contiguous [batch, channels, time].
    // Alpha is [channels] or [1, channels, 1], storing actual coefficients
    // (not logarithms). Matches DAC: x + sin(alpha*x)^2 / (alpha + 1e-9).
    // Inputs must be finite; alpha + 1e-9 must be nonzero. Zero alpha is valid.
    // Output must not alias either input, including its retained allocation.
    class Snake : public Op {
    public:
      void operator()(const StorageView& x, const StorageView& alpha, StorageView& y) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& x, const StorageView& alpha, StorageView& y) const;
    };

  }
}
