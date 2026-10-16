#pragma once

#include "common.h"

namespace ctranslate2 {
  namespace layers {

    struct DramaAttentionConfig {
      dim_t num_heads;
      dim_t num_kv_heads;
      dim_t head_dim;
      dim_t rotary_dim;
      float rope_base = 10000.f;
      float norm_epsilon = 1e-6f;
      bool rotary_interleaved = false;
      bool zero_centered_qk_norm = true;
    };

    // Caller allocates CPU float32 [capacity, kv_heads, head_dim] keys/values.
    // Only [0,length) is initialized/read; reset length=0 to start a sequence.
    struct DramaAttentionCache {
      StorageView keys;
      StorageView values;
      dim_t length = 0;
    };

    // Batch-one causal attention with CPU float32 [time, hidden] activations.
    // Model must outlive this layer and keep unpacked matrices unchanged.
    // q_proj rows are per-head [query_d | gate_d]; k/v/o use ordinary Dense
    // layouts. Tower's fused wqkv must be split/reordered during conversion.
    // q_norm/weight and k_norm/weight each have shape [head_dim].
    class DramaAttention : public Layer {
    public:
      DramaAttention(const models::Model& model, const std::string& scope,
                     DramaAttentionConfig config);
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const StorageView& input, DramaAttentionCache& cache,
                      StorageView& output) const;

    private:
      DramaAttentionConfig _config;
      Dense _query;
      Dense _key;
      Dense _value;
      Dense _output;
      const StorageView& _query_norm;
      const StorageView& _key_norm;
      dim_t _hidden_size;
      std::vector<const StorageView*> _parameters;
    };

  }
}
