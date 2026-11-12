#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace ops {

    // Structural eligibility shared by the CPU and CUDA/HIP fused weight-only
    // int8 kernels. Checks: no transpose on A, trans on B, no packing, beta==0,
    // dtypes (fp32 input/scale/bias/residual, int8 weight), device of weight/
    // scale/bias/residual matches `expected_device`, k/n > 0 with matching K,
    // m ∈ [1,8], and bias/residual sizes. Does not look at `input.device()`,
    // output, or any device-specific flags (alpha, output dtype, size caps,
    // weight_compensation); callers add those.
    inline bool weight_only_int8_shape_ok(bool trans_a, bool trans_b,
                                          bool a_is_packed, bool b_is_packed,
                                          float beta,
                                          Device expected_device,
                                          const StorageView& input,
                                          const StorageView& weight,
                                          const StorageView& scale,
                                          const StorageView* bias,
                                          const StorageView* residual,
                                          dim_t& m, dim_t& k, dim_t& n) {
      if (trans_a || !trans_b || a_is_packed || b_is_packed || beta != 0.f
          || input.rank() < 1
          || input.dtype() != DataType::FLOAT32
          || weight.dtype() != DataType::INT8 || weight.rank() != 2
          || scale.dtype() != DataType::FLOAT32
          || weight.device() != expected_device
          || scale.device() != expected_device
          || (bias && (bias->dtype() != DataType::FLOAT32
                       || bias->device() != expected_device))
          || (residual && (residual->dtype() != DataType::FLOAT32
                           || residual->device() != expected_device)))
        return false;
      k = input.dim(-1);
      n = weight.dim(0);
      if (k <= 0 || n <= 0 || weight.dim(1) != k)
        return false;
      m = input.size() / k;
      if (m < 1 || m > 8 || scale.size() != n
          || (bias && bias->size() != n)
          || (residual && residual->size() != m * n))
        return false;
      return true;
    }

  }
}
