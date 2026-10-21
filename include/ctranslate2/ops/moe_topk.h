#pragma once

#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    // Selects the highest-scoring experts and normalizes their routing weights.
    // CPU or CUDA/HIP float32 router logits [rows, experts] -> int32 expert_indices
    // and float32 routing_weights [rows, k]. Experts are sorted by descending
    // logit, breaking ties by ascending expert id. Weights are softmax over
    // selected logits: equivalent to full softmax, top-k, then renormalization.
    // This primitive applies no routing bias, grouping, or external scaling.
    // All logits must be finite; inputs/outputs must not alias, including
    // retained output allocations. Empty row dimensions are supported.
    class MoeTopK : public Op {
    public:
      explicit MoeTopK(dim_t k);

      void operator()(const StorageView& router_logits, StorageView& expert_indices,
                      StorageView& routing_weights) const;

    private:
      void compute_gpu(const StorageView& router_logits, StorageView& expert_indices,
                       StorageView& routing_weights) const;

      dim_t _k;
    };

  }
}
