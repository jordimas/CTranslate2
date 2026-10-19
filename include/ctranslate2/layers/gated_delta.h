#pragma once

#include "common.h"

namespace ctranslate2 {
  namespace layers {

    // Mixes token information through recurrent linear attention and output gating.
    // Batch-one CPU float32 GatedDeltaNet. Dense handles supported quantized
    // projection weights; raw /weight variables and model lifetime are required.
    // State/history belong to the caller; no layer-internal sequence cache.
    class GatedDeltaNet : public Layer {
    public:
      GatedDeltaNet(const models::Model& model, const std::string& scope, dim_t key_heads,
                    dim_t value_heads, dim_t key_dim, dim_t value_dim, float rms_epsilon);
      DataType output_type() const override;
      dim_t output_size() const override;

      // Allocates zero state [1,Hv,K,V] and history [1,2*Hk*K+Hv*V,kernel-1].
      void initialize_state(StorageView& state, StorageView& history) const;
      // Input/output [time,hidden]. T=0 preserves caches. Inputs, parameters,
      // state, history and output must not overlap mutable allocations.
      void operator()(const StorageView& input, StorageView& state, StorageView& history,
                      StorageView& output) const;

    private:
      void forward_gpu(const StorageView&, StorageView&, StorageView&, StorageView&) const;
      bool _resident_gpu = false;
      StorageView _gpu_conv, _gpu_conv_bias, _gpu_a_log, _gpu_dt_bias, _gpu_norm;
      Dense _qkv, _z, _a, _b, _out;
      const StorageView& _a_log;
      const StorageView& _dt_bias;
      const StorageView& _norm_weight;
      const StorageView* _conv_bias;
      StorageView _conv_weight;
      std::vector<const StorageView*> _parameters;
      dim_t _key_heads, _value_heads, _key_dim, _value_dim;
      dim_t _hidden = 0, _key_width = 0, _value_width = 0;
      dim_t _conv_channels = 0, _kernel = 0;
      float _rms_epsilon;
    };

  }
}
