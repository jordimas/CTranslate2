#pragma once

#include "ctranslate2/layers/attention_layer.h"
#include "ctranslate2/padder.h"
#include "ctranslate2/layers/transformer.h"

namespace ctranslate2 {
  namespace layers {

    StorageView make_relative_positions(dim_t queries_length,
                                        dim_t keys_length,
                                        dim_t max_position);

    StorageView make_asymmetric_relative_positions(dim_t queries_length,
                                                   dim_t keys_length,
                                                   dim_t left_max_position,
                                                   dim_t right_max_position);

    // Configures attention heads, rotary positions, and query/key normalization.
    struct GatedSelfAttentionConfig {
      dim_t num_heads;
      dim_t num_kv_heads;
      dim_t head_dim;
      dim_t rotary_dim;
      float rope_base = 10000.f;
      float norm_epsilon = 1e-6f;
      bool rotary_interleaved = false;
      bool zero_centered_qk_norm = true;
    };

    // Retains keys and values from earlier tokens for causal decoding.
    // Caller allocates CPU float32 [capacity, kv_heads, head_dim] keys/values.
    // Only [0,length) is initialized/read; reset length=0 to start a sequence.
    struct GatedSelfAttentionCache {
      StorageView keys;
      StorageView values;
      dim_t length = 0;
    };

    // Applies causal self-attention with a learned sigmoid gate on each output feature.
    // Batch-one causal attention with CPU float32 [time, hidden] activations.
    // Model must outlive this layer and keep unpacked matrices unchanged.
    // q_proj rows are per-head [query_d | gate_d]; k/v/o use ordinary Dense
    // layouts. Fused checkpoints must be split/reordered during conversion.
    // q_norm/weight and k_norm/weight each have shape [head_dim].
    class GatedSelfAttention : public Layer {
    public:
      GatedSelfAttention(const models::Model& model, const std::string& scope,
                         GatedSelfAttentionConfig config);
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const StorageView& input, GatedSelfAttentionCache& cache,
                      StorageView& output) const;

    private:
      void forward_gpu(const StorageView&, GatedSelfAttentionCache&, StorageView&) const;
      StorageView _gpu_query_norm, _gpu_key_norm;
      GatedSelfAttentionConfig _config;
      Dense _query;
      Dense _key;
      Dense _value;
      Dense _output;
      const StorageView& _query_norm;
      const StorageView& _key_norm;
      dim_t _hidden_size;
      mutable RotaryEmbeddings _rotary;
      std::vector<const StorageView*> _parameters;
    };

    class RotaryEmbeddings;
    class Alibi;

    class MultiHeadAttention : public AttentionLayer
    {
    public:
      MultiHeadAttention(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention,
                         bool pre_norm = true,
                         bool is_decoder = false,
                         Alibi* alibi = nullptr);
      DataType output_type() const override;
      dim_t output_size() const override;
      virtual void operator()(const StorageView& queries,
                      const StorageView& values,
                      const StorageView* values_lengths,
                      StorageView& output,
                      StorageView* cached_keys = nullptr,
                      StorageView* cached_values = nullptr,
                      StorageView* attention = nullptr,
                      const Padder* queries_padder = nullptr,
                      const Padder* values_padder = nullptr,
                      bool return_normalized_attention = true,
                      StorageView* position_bias = nullptr,
                      dim_t offset = 0) const override;

      virtual bool has_positional_embeddings() const override {
            return _relative_position_keys || _relative_attention_bias || _rotary_embeddings || _alibi;
      }

      bool has_merged_encoder_attention() const {
        return bool(_memory_kv);
      }

      void forward_merged(const StorageView& queries,
                          const StorageView* memory,
                          const StorageView* memory_lengths_mask,
                          const StorageView* self_lengths_mask,
                          StorageView& output,
                          StorageView* cached_self_keys,
                          StorageView* cached_self_values,
                          StorageView* cached_memory_keys,
                          StorageView* cached_memory_values,
                          const Padder* queries_padder,
                          const Padder* memory_padder,
                          dim_t offset) const;

    protected:
      void process_cross_attention(const StorageView& queries,
                                const StorageView& values,
                                StorageView& fused_proj,
                                StorageView& queries_proj,
                                StorageView& keys_proj,
                                StorageView& values_proj,
                                StorageView* cached_keys,
                                StorageView* cached_values,
                                const Padder* queries_padder,
                                const Padder* values_padder,
                                dim_t& beam_size) const;

    private:
      static void split_heads(StorageView& x,
                               dim_t num_heads,
                               const Padder* padder = nullptr,
                               dim_t beam_size = 1);

      static void combine_heads(StorageView& x,
                                 dim_t num_heads,
                                 const Padder* padder = nullptr,
                                 dim_t beam_size = 1);

      void apply_k_norm(StorageView& keys_proj) const;
      void apply_v_norm(StorageView& values_proj) const;

      void apply_qk_norm(StorageView& queries_proj,
                          StorageView& keys_proj) const;

      const StorageView* _relative_attention_bias;
      const StorageView* _relative_position_keys;
      const StorageView* _relative_asymmetric_position_keys;
      const StorageView* _relative_position_values;
      dim_t _maximum_relative_position;
      dim_t _relative_left_max_position;
      dim_t _relative_right_max_position;
      const bool _merge_time_and_head_dims;
      const dim_t _cache_time_dim;
      std::unique_ptr<const LayerNorm> _q_norm;  // Query normalization
      std::unique_ptr<const LayerNorm> _k_norm;  // Key normalization
      std::unique_ptr<const LayerNorm> _v_norm;  // Value normalization (no learnable scale)
      std::unique_ptr<const Dense> _memory_kv;  // Fused K+V projection of encoder memory (merged attention)
    };
  }
}
