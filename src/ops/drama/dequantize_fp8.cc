#include "ops/drama/dequantize_fp8.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    void DequantizeFP8::operator()(const StorageView& input, const StorageView& scale_inv,
                                   StorageView& output) const {
      PROFILE("DequantizeFP8");
      if (scale_inv.device() != input.device() || output.device() != input.device()
          || input.device_index() != scale_inv.device_index()
          || input.device_index() != output.device_index() || input.dtype() != DataType::INT8
          || scale_inv.dtype() != DataType::FLOAT32 || output.dtype() != DataType::FLOAT32)
        throw std::invalid_argument(
            "DequantizeFP8 requires INT8 bytes and F32 scales/output on the same device");
      if (input.rank() != 2 || input.dim(0) <= 0 || input.dim(1) <= 0 || scale_inv.rank() != 2
          || scale_inv.shape()
                 != Shape{input.dim(0) / 128 + (input.dim(0) % 128 != 0),
                          input.dim(1) / 128 + (input.dim(1) % 128 != 0)})
        throw std::invalid_argument("DequantizeFP8 requires a matrix and 128x128 block scales");
      for (const auto* source : {&input, &scale_inv}) {
        const auto a = reinterpret_cast<uintptr_t>(source->buffer());
        const auto b = reinterpret_cast<uintptr_t>(output.buffer());
        if (source == &output
            || (a && b && output.reserved_memory()
                && (a <= b ? b - a < static_cast<uintptr_t>(source->size() * source->item_size())
                           : a - b < static_cast<uintptr_t>(output.reserved_memory()))))
          throw std::invalid_argument("DequantizeFP8 output must not alias inputs");
      }
#ifdef CT2_WITH_CUDA
      if (input.device() == Device::CUDA) {
        const ScopedDeviceSetter device_setter(input.device(), input.device_index());
        compute_gpu(input, scale_inv, output);
        return;
      }
#endif
      const auto* scales = scale_inv.data<float>();
      for (dim_t i = 0; i < scale_inv.size(); ++i)
        if (!(scales[i] > 0) || !std::isfinite(scales[i]))
          throw std::invalid_argument("DequantizeFP8 scales must be positive and finite");
      static const auto table = []() {
        std::array<float, 256> values{};
        for (int i = 0; i < 256; ++i) {
          const int exponent = (i & 127) >> 3, mantissa = i & 7;
          const float magnitude = exponent == 0
                                      ? std::ldexp(float(mantissa), -9)
                                      : std::ldexp(1.f + float(mantissa) / 8, exponent - 7);
          values[i] = (i & 128) ? -magnitude : magnitude;
        }
        return values;
      }();
      output.resize_as(input);
      const auto* bytes = reinterpret_cast<const uint8_t*>(input.data<int8_t>());
      auto* decoded = output.data<float>();
      const dim_t rows = input.dim(0), cols = input.dim(1), block_cols = scale_inv.dim(1);
      std::atomic<bool> invalid{false};
      cpu::parallel_for(0, rows, 1, [&](dim_t begin, dim_t end) {
        for (dim_t r = begin; r < end; ++r)
          for (dim_t c = 0; c < cols; ++c) {
            const uint8_t code = bytes[r * cols + c];
            const float value = table[code] * scales[(r / 128) * block_cols + c / 128];
            if ((code & 127) == 127 || !std::isfinite(value))
              invalid.store(true, std::memory_order_relaxed);
            decoded[r * cols + c] = value;
          }
      });
      if (invalid.load())
        throw std::invalid_argument(
            "DequantizeFP8 contains NaN codes or nonfinite decoded weights");
    }

  }
}
