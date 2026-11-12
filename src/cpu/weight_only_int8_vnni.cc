#include "cpu/kernels.h"

#define TARGET_ISA CpuIsa::AVX2
#include "cpu/vec_avx.h"
#include "cpu/weight_only_int8.h"

namespace ctranslate2 {
  namespace cpu {

    void weight_only_int8_dot_vnni(const int8_t* weight, const float* scale,
                                  const int32_t* compensation, const uint8_t* input,
                                  float* output, dim_t m, dim_t k, dim_t n,
                                  const float* inv_scale, const float* bias, const float* residual,
                                  const ops::ActivationType* activation) {
      weight_only_int8_dot(weight, scale, compensation, input, output,
                             m, k, n, inv_scale, bias, residual, activation);
    }

  }
}
