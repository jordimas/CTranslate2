#include "ctranslate2/ops/fused_attention.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "ctranslate2/primitives.h"
#include "cpu/backend.h"
#include "cpu/cpu_isa.h"
#include "cpu/kernels.h"
#include "cpu/parallel.h"
#include "device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // Attention tensors are [batch, heads, time, depth].
    constexpr dim_t fused_attention_rank = 4;

    // Decoding steps (fewer queries) stay on the batched GEMV path.
    constexpr dim_t fused_attention_min_queries = 32;

    // The scores of a block of queries against all the keys are kept in cache. Every block goes
    // through two GEMMs that pack the keys and the values again, so large blocks are faster than
    // small ones: up to 256 queries, fewer when the keys are many so that the scores stay within
    // fused_attention_scores_bytes. With so many keys that the block would have fewer than 64
    // queries (more than 8192 keys), the GEMMs get too thin and the regular path is faster.
    constexpr dim_t fused_attention_max_block_queries = 256;
    constexpr dim_t fused_attention_min_block_queries = 64;
    constexpr dim_t fused_attention_scores_bytes = 2 * 1024 * 1024;

    // oneDNN can round GEMMs differently when their row count changes. Fuse whole
    // heads instead of query blocks on this backend, preserving the regular GEMM
    // dimensions. Bound the score scratch allocation per worker for long sequences.
    constexpr dim_t fused_attention_dnnl_scores_bytes = 64 * 1024 * 1024;

    // Block-wise attention for FP32 on CPU. The full [batch, heads, queries, keys] score
    // tensor is never materialized: it is computed, normalized and multiplied by the values one
    // block of queries at a time, which avoids writing then re-reading hundreds of MB per layer.
    // The operations are those of the regular path (scaled QK^T, the same softmax kernel, then
    // multiplication by the values), only applied to a block of queries at a time. An online
    // softmax (rescaling partial results, as in flash attention) would round differently, and
    // with int8 weights those differences flip quantized activations and change transcripts.
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
      if (tk > fused_attention_scores_bytes / dim_t(sizeof(float)) / fused_attention_min_block_queries)
        return false;

      static const bool use_dnnl = cpu::get_gemm_backend(ComputeType::FLOAT32)
        == cpu::GemmBackend::DNNL;
      dim_t block_queries;
      if (use_dnnl) {
        if (tq > fused_attention_dnnl_scores_bytes / dim_t(sizeof(float)) / tk)
          return false;
        block_queries = tq;
      } else {
        block_queries = std::min(
          fused_attention_max_block_queries,
          fused_attention_scores_bytes / (tk * dim_t(sizeof (float))) / 8 * 8);
        if (block_queries < fused_attention_min_block_queries)
          return false;
      }

      output.resize({batch, heads, tq, depth_v});

      const float* q_data = queries.data<float>();
      const float* k_data = keys.data<float>();
      const float* v_data = values.data<float>();
      const int32_t* lengths = values_lengths ? values_lengths->data<int32_t>() : nullptr;
      float* out_data = output.data<float>();

      using cpu_primitives = primitives<Device::CPU>;
      const dim_t num_query_blocks = ceil_divide(tq, block_queries);

      cpu::parallel_for(0, batch * heads * num_query_blocks, 1, [&](dim_t begin, dim_t end) {
        std::vector<float> scores(block_queries * tk);
        std::vector<int32_t> block_lengths(lengths ? block_queries : 0);

        for (dim_t task = begin; task < end; ++task) {
          const dim_t bh = task / num_query_blocks;
          const dim_t q0 = (task % num_query_blocks) * block_queries;
          const dim_t nq = std::min(block_queries, tq - q0);

          const float* q = q_data + (bh * tq + q0) * depth;
          const float* k = k_data + bh * tk * depth;
          const float* v = v_data + bh * tk * depth_v;
          float* out = out_data + (bh * tq + q0) * depth_v;

          // Keys beyond the longest row of this block do not contribute (e.g. causal masks).
          dim_t max_length = tk;
          if (lengths) {
            max_length = 0;
            for (dim_t r = 0; r < nq; ++r) {
              const dim_t length = std::max<dim_t>(0, std::min<dim_t>(lengths[bh * tq + q0 + r], tk));
              block_lengths[r] = static_cast<int32_t>(length);
              max_length = std::max(max_length, length);
            }
          }

          if (max_length == 0) {
            std::fill(out, out + nq * depth_v, 0.f);
            continue;
          }

          // Truncating masked keys would also change oneDNN's GEMM dimensions.
          if (use_dnnl)
            max_length = tk;

          cpu_primitives::gemm(/*a_is_packed=*/false, /*b_is_packed=*/false,
                               /*transpose_a=*/false, /*transpose_b=*/true,
                               nq, max_length, depth,
                               queries_scale,
                               q, depth,
                               k, depth,
                               0.f,
                               scores.data(), max_length);

          CPU_ISA_DISPATCH((cpu::softmax<ISA>(scores.data(),
                                              lengths ? block_lengths.data() : nullptr,
                                              scores.data(),
                                              nq,
                                              max_length,
                                              /*log=*/false)));

          cpu_primitives::gemm(/*a_is_packed=*/false, /*b_is_packed=*/false,
                               /*transpose_a=*/false, /*transpose_b=*/false,
                               nq, depth_v, max_length,
                               1.f,
                               scores.data(), max_length,
                               v, depth_v,
                               0.f,
                               out, depth_v);
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
