#include "ctranslate2/ops/dequantize_fp8.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      __global__ void decode_fp8(const uint8_t* input, const float* scales, float* output,
                                 int32_t* invalid, dim_t size, dim_t cols, dim_t block_cols) {
        for (dim_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
             i += dim_t(blockDim.x) * gridDim.x) {
          const auto code = input[i];
          const int exponent = (code & 127) >> 3, mantissa = code & 7;
          const float magnitude = exponent == 0 ? ldexpf(float(mantissa), -9)
                                                : ldexpf(1.f + float(mantissa) / 8, exponent - 7);
          const float scale = scales[(i / cols / 128) * block_cols + (i % cols) / 128];
          const float value = ((code & 128) ? -magnitude : magnitude) * scale;
          if ((code & 127) == 127 || !(scale > 0) || !isfinite(scale) || !isfinite(value))
            atomicExch(invalid, 1);
          output[i] = value;
        }
      }

    }

    template <Device D>
    void DequantizeFP8::compute(const StorageView& input, const StorageView& scale_inv,
                                StorageView& output) const {
      output.resize_as(input);
      StorageView invalid({1}, int32_t(0), Device::CUDA);
      const dim_t threads = 256;
      const dim_t blocks = std::min((input.size() + threads - 1) / threads, dim_t(65535));
      decode_fp8<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          reinterpret_cast<const uint8_t*>(input.data<int8_t>()), scale_inv.data<float>(),
          output.data<float>(), invalid.data<int32_t>(), input.size(), input.dim(1),
          scale_inv.dim(1));
      CUDA_CHECK(cudaGetLastError());
      auto status = invalid.to(Device::CPU);
      synchronize_stream(Device::CUDA);
      if (status.data<int32_t>()[0])
        throw std::invalid_argument(
            "DequantizeFP8 requires positive finite scales and finite decoded weights");
    }

    template void DequantizeFP8::compute<Device::CUDA>(const StorageView&,
                                                       const StorageView&,
                                                       StorageView&) const;

  }
}
