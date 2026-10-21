#include "ctranslate2/ops/snake.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      __global__ void validate(const float* x, const float* alpha, dim_t size, dim_t channels,
                               dim_t time, int32_t* invalid) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < channels && (!isfinite(alpha[i]) || alpha[i] + 1e-9f == 0.f))
          atomicExch(invalid, 1);
        if (i < size && (!isfinite(x[i]) || !isfinite(x[i] * alpha[(i / time) % channels])))
          atomicExch(invalid, 1);
      }

      __global__ void snake(const float* x, const float* alpha, float* out, dim_t n, dim_t time,
                            dim_t channels) {
        const dim_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
          const float a = alpha[(i / time) % channels], s = sinf(a * x[i]);
          out[i] = x[i] + (1.f / (a + 1e-9f)) * (s * s);
        }
      }

    }

    template <Device D, typename T>
    void Snake::compute(const StorageView& x, const StorageView& alpha, StorageView& out) const {
      StorageView invalid({1}, int32_t(0), Device::CUDA);
      const dim_t count = std::max(x.size(), alpha.size());
      validate<<<(count + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          x.data<T>(), alpha.data<T>(), x.size(), x.dim(1), x.dim(2),
          invalid.data<int32_t>());
      CUDA_CHECK(cudaGetLastError());
      auto status = invalid.to(Device::CPU);
      synchronize_stream(Device::CUDA);
      if (status.data<int32_t>()[0])
        throw std::invalid_argument("Snake requires finite input and alpha * input, and finite "
                                    "alpha with nonzero alpha + 1e-9");
      out.resize_as(x);
      if (x.empty())
        return;
      snake<<<(x.size() + 255) / 256, 256, 0, cuda::get_cuda_stream()>>>(
          x.data<T>(), alpha.data<T>(), out.data<T>(), x.size(), x.dim(2), x.dim(1));
      CUDA_CHECK(cudaGetLastError());
    }

    template void Snake::compute<Device::CUDA, float>(const StorageView&,
                                                     const StorageView&,
                                                     StorageView&) const;

  }
}
