#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    // Updates recurrent attention memory and reads it for each query.
    // CPU or CUDA/HIP float32 recurrent gated-delta update. Inputs are contiguous:
    //   query, key: [batch, time, key_heads, key_dim]
    //   value: [batch, time, value_heads, value_dim]
    //   log_decay, beta: [batch, time, value_heads]
    //   state (updated in place): [batch, value_heads, key_dim, value_dim]
    //   output: [batch, time, value_heads, value_dim]
    // value_heads must be a multiple of key_heads (consecutive head groups).
    // Query/key normalization and gate projections are the caller's responsibility.
    // At each step, S <- exp(g) S; S <- S + beta k (v - k^T S)^T;
    // output = S^T q / sqrt(key_dim). No input may alias state or output.
    class GatedDeltaNet : public Op {
    public:
      void operator()(const StorageView& query, const StorageView& key, const StorageView& value,
                      const StorageView& log_decay, const StorageView& beta, StorageView& state,
                      StorageView& output) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& query, const StorageView& key, const StorageView& value,
                   const StorageView& log_decay, const StorageView& beta, StorageView& state,
                   StorageView& output) const;

      void validate_input(const StorageView& query, const StorageView& key,
                          const StorageView& value, const StorageView& log_decay,
                          const StorageView& beta, const StorageView& state,
                          const StorageView& output) const;
    };

  }
}
