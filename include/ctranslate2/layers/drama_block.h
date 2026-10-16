#pragma once

#include "drama_attention.h"
#include "gated_delta.h"
#include "moe.h"

namespace ctranslate2 {
  namespace layers {

    enum class DramaMixerKind { LinearAttention, FullAttention };

    struct DramaBlockConfig {
      DramaMixerKind kind = DramaMixerKind::LinearAttention;
      float norm_epsilon = 1e-6f;
      dim_t num_experts = 0;
      dim_t top_k = 0;
      dim_t key_heads = 0, value_heads = 0, key_dim = 0, value_dim = 0;
      DramaAttentionConfig attention{0, 0, 0, 0};
    };

    struct DramaBlockState {
      StorageView recurrent;
      StorageView history;
      DramaAttentionCache attention;
    };

    // CPU float32 batch-one [time,hidden] backbone block. Model variables must
    // outlive this layer and remain unchanged; weight prepacking is disabled.
    // Each call advances caller-owned sequence state. On runtime failure after
    // mixer execution, discard/reset that state before retrying the sequence.
    class DramaBlock : public Layer {
    public:
      DramaBlock(const models::Model& model, const std::string& scope,
                 DramaBlockConfig config);
      DataType output_type() const override;
      dim_t output_size() const override;
      void initialize_state(DramaBlockState& state, dim_t max_tokens = 0) const;
      void operator()(const StorageView& input, DramaBlockState& state,
                      StorageView& output) const;

    private:
      DramaBlockConfig _config;
      const StorageView& _input_norm;
      const StorageView& _post_norm;
      SparseMoE _moe;
      std::unique_ptr<GatedDeltaNet> _linear;
      std::unique_ptr<DramaAttention> _attention;
    };

  }
}
