#include "ctranslate2/ops/moe_topk.h"
#include "cuda/helpers.h"

#ifdef CT2_USE_HIP
#  include <hipcub/block/block_reduce.hpp>
#  define cub hipcub
#else
#  include <cub/block/block_reduce.cuh>
#endif

namespace ctranslate2 {
  namespace ops {
    namespace {

      struct RoutingCandidate {
        float value;
        int32_t index;
      };

      struct RoutingMaximum {
        __device__ RoutingCandidate operator()(RoutingCandidate a, RoutingCandidate b) const {
          return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
        }
      };

      constexpr int routing_threads = 256;

      __global__ void route(const float* logits, int32_t* indices, float* weights, dim_t experts,
                            dim_t k, int32_t* invalid) {
        // Parallel selection preserves the lowest-index tie ordering. Normalization
        // retains the original sequential double-precision arithmetic.
        using Reduction = cub::BlockReduce<RoutingCandidate, routing_threads>;
        __shared__ typename Reduction::TempStorage temporary;
        const dim_t row = blockIdx.x;
        const float* values = logits + row * experts;
        int32_t* ids = indices + row * k;
        for (dim_t i = 0; i < k; ++i) {
          RoutingCandidate best{-INFINITY, INT32_MAX};
          for (dim_t e = threadIdx.x; e < experts; e += blockDim.x) {
            bool selected = false;
            for (dim_t j = 0; j < i; ++j)
              if (ids[j] == e)
                selected = true;
            if (!selected) {
              const float value = values[e];
              if (!isfinite(value))
                atomicExch(invalid, 1);
              best = RoutingMaximum()(best, {isfinite(value) ? value : 0.f, int32_t(e)});
            }
          }
          best = Reduction(temporary).Reduce(best, RoutingMaximum());
          if (threadIdx.x == 0)
            ids[i] = best.index;
          // Publish the selected ID and allow reuse of reduction shared memory.
          __syncthreads();
        }
        if (threadIdx.x)
          return;
        const double maximum = values[ids[0]];
        double sum = 0;
        for (dim_t i = 0; i < k; ++i)
          sum += exp(double(values[ids[i]]) - maximum);
        for (dim_t i = 0; i < k; ++i)
          weights[row * k + i] = float(exp(double(values[ids[i]]) - maximum) / sum);
      }

    }

    void MoeTopK::compute_gpu(const StorageView& logits, StorageView& indices,
                              StorageView& weights) const {
      indices.resize({logits.dim(0), _k});
      weights.resize({logits.dim(0), _k});
      if (logits.empty())
        return;
      StorageView invalid({1}, int32_t(0), Device::CUDA);
      route<<<logits.dim(0), routing_threads, 0, cuda::get_cuda_stream()>>>(
          logits.data<float>(), indices.data<int32_t>(), weights.data<float>(), logits.dim(1), _k,
          invalid.data<int32_t>());
      CUDA_CHECK(cudaGetLastError());
      auto status = invalid.to(Device::CPU);
      synchronize_stream(Device::CUDA);
      if (status.data<int32_t>()[0])
        throw std::invalid_argument("MoeTopK logits must be finite");
    }

  }
}
