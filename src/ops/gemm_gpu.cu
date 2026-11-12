#include "ctranslate2/ops/gemm.h"
#include "ctranslate2/ops/quantize.h"

#include <type_traits>

#include "cuda/helpers.h"
#include "ops/weight_only_int8.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    __device__ __forceinline__ T apply_activation(T value, int activation) {
      switch (activation) {
      case int(ActivationType::ReLU):         return cuda::relu_func<T>()(value);
      case int(ActivationType::GELU):         return cuda::gelu_func<T>()(value);
      case int(ActivationType::GELUTanh):     return cuda::gelu_tanh_func<T>()(value);
      case int(ActivationType::GELUSigmoid):  return cuda::gelu_sigmoid_func<T>()(value);
      case int(ActivationType::Sigmoid):      return cuda::sigmoid_func<T>()(value);
      case int(ActivationType::Swish):        return cuda::swish_func<T>()(value);
      case int(ActivationType::Tanh):         return cuda::tanh_func<T>()(value);
      default:                                return value;
      }
    }

    // Each group of 32 threads computes one output channel and reuses its
    // weights across the small batch. No INT32 output tensor is materialized.
    template <int M, typename T>
    __global__ void weight_only_int8_kernel(const int8_t* input,
                                           const int8_t* weight,
                                           const float* input_scale,
                                           const float* weight_scale,
                                           const T* bias,
                                           const T* residual,
                                           T* output,
                                           int k, int n, int activation) {
      const int channel = blockIdx.x * 4 + threadIdx.x / 32;
      const int lane = threadIdx.x % 32;
      if (channel >= n)
        return;
      int sums[M] = {};
      if (k % 4 == 0) {
        for (int i = lane * 4; i < k; i += 128) {
          const char4 w = *reinterpret_cast<const char4*>(weight + channel * k + i);
#pragma unroll
          for (int row = 0; row < M; ++row) {
            const char4 a = *reinterpret_cast<const char4*>(input + row * k + i);
#ifdef CT2_USE_HIP
            sums[row] = amd_mixed_dot(a, w, sums[row], false);
#elif __CUDA_ARCH__ >= 610
            sums[row] = __dp4a(*reinterpret_cast<const int*>(&a), *reinterpret_cast<const int*>(&w), sums[row]);
#else
            sums[row] += int(a.x) * w.x + int(a.y) * w.y + int(a.z) * w.z + int(a.w) * w.w;
#endif
          }
        }
      } else {
        for (int i = lane; i < k; i += 32) {
          const int w = weight[channel * k + i];
#pragma unroll
          for (int row = 0; row < M; ++row)
            sums[row] += w * int(input[row * k + i]);
        }
      }
#pragma unroll
      for (int row = 0; row < M; ++row) {
#pragma unroll
        for (int offset = 16; offset; offset /= 2)
#ifdef CT2_USE_HIP
          sums[row] += __shfl_down(sums[row], offset, 32);
#else
          sums[row] += __shfl_down_sync(0xffffffff, sums[row], offset, 32);
#endif
        if (lane == 0) {
          // Preserve the fallback path's rounding at each output-type boundary.
          T value = __fdividef(float(sums[row]), input_scale[row] * weight_scale[channel]);
          const auto add = cuda::plus<T>();
          if (bias)
            value = add(value, bias[channel]);
          value = apply_activation(value, activation);
          // Quantized Dense adds the residual after applying its activation.
          if (residual)
            value = add(value, residual[row * n + channel]);
          output[row * n + channel] = value;
        }
      }
    }

    template<>
    bool Gemm::try_weight_only_int8<Device::CUDA>(const StorageView& input,
                                                const StorageView& weight,
                                                const StorageView& scale,
                                                StorageView& output,
                                                const ActivationType* activation,
                                                const StorageView* bias,
                                                const StorageView* residual,
                                                const StorageView*) const {
      dim_t m, k, n;
      if (_alpha != 1.f
          || output.dtype() != input.dtype() || output.device() != Device::CUDA
          || !weight_only_int8_shape_ok(_trans_a, _trans_b, _a_is_packed, _b_is_packed,
                                        _beta, Device::CUDA, input, weight, scale,
                                        bias, residual, m, k, n))
        return false;
      // Large projections lose the launch-overhead advantage at wider batches.
      if ((m > 2 && n > 8192)
          || std::max({input.size(), weight.size(), m * n}) > std::numeric_limits<int>::max() - 3)
        return false;

      resize_weight_only_int8_output(input, n, output);
      StorageView quantized(DataType::INT8, Device::CUDA);
      StorageView input_scale(DataType::FLOAT32, Device::CUDA);
      Quantize()(input, quantized, input_scale);
      auto launch = [&](auto* output_data) {
        using T = std::remove_pointer_t<decltype(output_data)>;
#define LAUNCH(M) case M: weight_only_int8_kernel<M><<< (n + 3) / 4, 128, 0, cuda::get_cuda_stream() >>>( \
        quantized.data<int8_t>(), weight.data<int8_t>(), input_scale.data<float>(), scale.data<float>(), \
        bias ? cuda::device_cast(bias->data<T>()) : nullptr, \
        residual ? cuda::device_cast(residual->data<T>()) : nullptr, \
        cuda::device_cast(output_data), k, n, activation ? int(*activation) : -1); break
        switch (m) {
          LAUNCH(1); LAUNCH(2); LAUNCH(3); LAUNCH(4);
          LAUNCH(5); LAUNCH(6); LAUNCH(7); LAUNCH(8);
        }
#undef LAUNCH
      };
      if (input.dtype() == DataType::FLOAT16)
        launch(output.data<float16_t>());
      else
        launch(output.data<float>());
      CUDA_CHECK(cudaGetLastError());
      return true;
    }

  }
}
