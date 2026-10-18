#pragma once

#include "common.h"
#include "ctranslate2/ops/moe_topk.h"

namespace ctranslate2 {
  namespace layers {

    // Routes each token through selected experts and combines their outputs.
    // CPU float32 activation baseline: [rows, hidden] -> [rows, hidden].
    // Computes only selected routed experts, plus a sigmoid-gated shared expert.
    // Model variables must outlive this layer and remain unchanged. Requires
    // unpacked /weight matrices: model loading must disable weight prepacking.
    // Dense handles INT8 matmul; all weights are resident, not offloaded.
    // No residual connection or normalization is applied.
    class SparseMoE : public Layer {
    public:
      SparseMoE(const models::Model& model, const std::string& scope,
                dim_t num_experts, dim_t k);
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const StorageView& input, StorageView& output) const;

    private:
      // Applies a gated feed-forward network for one routed or shared expert.
      struct Expert {
        Expert(const models::Model& model, const std::string& scope);
        void operator()(const StorageView& input, StorageView& output) const;
        Dense gate;
        Dense up;
        Dense down;
        int gpu_device_index;
      };

      void forward_grouped_gpu(const StorageView& input, StorageView& output) const;
      void forward_resident_gpu(const StorageView& input, StorageView& output) const;
      mutable StorageView _gpu_experts;
      bool _resident_gpu = false;
      bool _grouped_gpu = false;
      Dense _router;
      Dense _shared_gate;
      Expert _shared;
      std::vector<std::unique_ptr<Expert>> _experts;
      ops::MoeTopK _topk;
      dim_t _input_size;
      dim_t _k;
      std::vector<const StorageView*> _parameters;
    };

  }
}
