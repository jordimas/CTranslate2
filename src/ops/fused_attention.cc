#include "ctranslate2/ops/fused_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "ctranslate2/primitives.h"
#include "cpu/parallel.h"
#include "device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // Attention tensors are [batch, heads, time, depth].
    constexpr dim_t fused_attention_rank = 4;

    // Decoding steps (fewer queries) stay on the batched GEMV path.
    constexpr dim_t fused_attention_min_queries = 32;

    // A block of queries is processed against chunks of keys, so that the block of scores
    // (64 x 64 floats = 16 KB) stays in the L2 cache. These are the tile sizes llama.cpp uses
    // for its CPU flash attention. An op-level sweep on Whisper, long-sequence and
    // head-dim-128 shapes showed that key chunks of 512 were about 25-30% faster here, since
    // every tile goes through two library GEMM calls whose overhead larger chunks amortize.
    constexpr dim_t fused_attention_block_queries = 64;
    constexpr dim_t fused_attention_block_keys = 64;

    // Tiled attention with an online softmax (flash-attention style) for FP32 on CPU.
    // The full [batch, heads, queries, keys] score tensor is never materialized: scores are
    // computed for a block of queries and a chunk of keys at a time, which keeps them in the
    // L2 cache instead of writing then re-reading hundreds of MB per layer.
    static bool cpu_fused_attention_enabled() {
      static const bool enabled = []() {
        const char* v = std::getenv("CT2_CPU_FUSED_ATTENTION");
        return !(v && v[0] == '0');
      }();
      return enabled;
    }

    template<>
    bool FusedAttention::compute<Device::CPU>(const StorageView& queries,
                                               const StorageView& keys,
                                               const StorageView& values,
                                               const StorageView* values_lengths,
                                               float queries_scale,
                                               bool interleaved_queries,
                                               StorageView& output) const {

      if (interleaved_queries
          || !cpu_fused_attention_enabled()
          || queries.dtype() != DataType::FLOAT32
          || keys.dtype() != DataType::FLOAT32
          || values.dtype() != DataType::FLOAT32
          || queries.rank() != fused_attention_rank || keys.rank() != fused_attention_rank
          || values.rank() != fused_attention_rank
          || queries.dim(0) != keys.dim(0) || queries.dim(0) != values.dim(0)
          || queries.dim(1) != keys.dim(1) || queries.dim(1) != values.dim(1)
          || keys.dim(2) != values.dim(2)
          || queries.dim(3) != keys.dim(3))
        return false;

      const dim_t batch = queries.dim(0);
      const dim_t heads = queries.dim(1);
      const dim_t tq = queries.dim(2);
      const dim_t depth = queries.dim(3);
      const dim_t tk = keys.dim(2);
      const dim_t depth_v = values.dim(3);

      if (tq < fused_attention_min_queries || tk == 0 || depth == 0 || depth_v == 0)
        return false;
      if (values_lengths && values_lengths->size() != batch * heads * tq)
        return false;

      output.resize({batch, heads, tq, depth_v});

      const float* q_data = queries.data<float>();
      const float* k_data = keys.data<float>();
      const float* v_data = values.data<float>();
      const int32_t* lengths = values_lengths ? values_lengths->data<int32_t>() : nullptr;
      float* out_data = output.data<float>();

      using cpu_primitives = primitives<Device::CPU>;
      const dim_t num_query_blocks = ceil_divide(tq, fused_attention_block_queries);

      cpu::parallel_for(0, batch * heads * num_query_blocks, 1, [&](dim_t begin, dim_t end) {
        std::vector<float> scores(fused_attention_block_queries * fused_attention_block_keys);
        std::vector<float> row_max(fused_attention_block_queries);
        std::vector<float> row_sum(fused_attention_block_queries);

        for (dim_t task = begin; task < end; ++task) {
          const dim_t bh = task / num_query_blocks;
          const dim_t q0 = (task % num_query_blocks) * fused_attention_block_queries;
          const dim_t nq = std::min(fused_attention_block_queries, tq - q0);

          const float* q = q_data + (bh * tq + q0) * depth;
          const float* k = k_data + bh * tk * depth;
          const float* v = v_data + bh * tk * depth_v;
          float* out = out_data + (bh * tq + q0) * depth_v;  // Also the running accumulator.
          const int32_t* row_lengths = lengths ? lengths + bh * tq + q0 : nullptr;
          const auto length_of = [&](dim_t r) -> dim_t {
            return row_lengths ? std::min<dim_t>(row_lengths[r], tk) : tk;
          };

          // Keys beyond the longest row of this block do not contribute (e.g. causal masks).
          dim_t max_length = 0;
          for (dim_t r = 0; r < nq; ++r)
            max_length = std::max(max_length, length_of(r));

          std::fill(out, out + nq * depth_v, 0.f);
          for (dim_t r = 0; r < nq; ++r) {
            row_max[r] = -std::numeric_limits<float>::infinity();
            row_sum[r] = 0.f;
          }

          for (dim_t k0 = 0; k0 < max_length; k0 += fused_attention_block_keys) {
            const dim_t nk = std::min(fused_attention_block_keys, max_length - k0);

            cpu_primitives::gemm(/*a_is_packed=*/false, /*b_is_packed=*/false,
                                 /*transpose_a=*/false, /*transpose_b=*/true,
                                 nq, nk, depth,
                                 queries_scale,
                                 q, depth,
                                 k + k0 * depth, depth,
                                 0.f,
                                 scores.data(), fused_attention_block_keys);

            for (dim_t r = 0; r < nq; ++r) {
              float* s = scores.data() + r * fused_attention_block_keys;
              const dim_t valid = std::max<dim_t>(0, std::min(nk, length_of(r) - k0));

              if (valid == 0) {
                std::fill(s, s + nk, 0.f);
                continue;
              }

              // Online softmax: exp(-inf - new_max) is 0, which zeroes the (empty)
              // accumulator on the first chunk.
              const float new_max = std::max(row_max[r], cpu_primitives::max(s, valid));
              const float correction = std::exp(row_max[r] - new_max);
              cpu_primitives::add(-new_max, s, s, valid);
              cpu_primitives::exp(s, s, valid);
              std::fill(s + valid, s + nk, 0.f);

              row_sum[r] = row_sum[r] * correction + cpu_primitives::sum(s, valid);
              if (correction != 1.f)
                cpu_primitives::mul(correction, out + r * depth_v, depth_v);
              row_max[r] = new_max;
            }

            cpu_primitives::gemm(/*a_is_packed=*/false, /*b_is_packed=*/false,
                                 /*transpose_a=*/false, /*transpose_b=*/false,
                                 nq, depth_v, nk,
                                 1.f,
                                 scores.data(), fused_attention_block_keys,
                                 v + k0 * depth_v, depth_v,
                                 1.f,
                                 out, depth_v);
          }

          for (dim_t r = 0; r < nq; ++r)
            cpu_primitives::mul(row_sum[r] > 0.f ? 1.f / row_sum[r] : 0.f,
                                out + r * depth_v, depth_v);
        }
      });

      return true;
    }

    template<>
    bool FusedAttention::compute<Device::CUDA>(const StorageView&,
                                                const StorageView&,
                                                const StorageView&,
                                                const StorageView*,
                                                float,
                                                bool,
                                                StorageView&) const {
      return false;
    }

    template<>
    bool FusedAttention::supports_interleaved_queries<Device::CPU>(DataType) const {
      return false;
    }

    template<>
    bool FusedAttention::supports_interleaved_queries<Device::CUDA>(DataType) const {
      return false;
    }

    bool FusedAttention::operator()(const StorageView& queries,
                                    const StorageView& keys,
                                    const StorageView& values,
                                    const StorageView* values_lengths,
                                    float queries_scale,
                                    bool interleaved_queries,
                                    StorageView& output) const {
      bool supported = false;
      DEVICE_DISPATCH(queries.device(),
                      supported = compute<D>(queries,
                                             keys,
                                             values,
                                             values_lengths,
                                             queries_scale,
                                             interleaved_queries,
                                             output));
      return supported;
    }

    bool FusedAttention::supports_interleaved_queries(Device device, DataType dtype) const {
      bool supported = false;
      DEVICE_DISPATCH(device, supported = supports_interleaved_queries<D>(dtype));
      return supported;
    }

  }
}
