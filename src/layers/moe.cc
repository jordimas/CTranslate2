#include "ctranslate2/layers/moe.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ctranslate2 {
  namespace layers {
    namespace {

      bool overlaps_output(const StorageView& input, const StorageView& output) {
        if (input.empty() || !output.buffer() || output.reserved_memory() == 0)
          return false;
        const auto a = reinterpret_cast<uintptr_t>(input.buffer());
        const auto b = reinterpret_cast<uintptr_t>(output.buffer());
        return a <= b ? b - a < static_cast<uintptr_t>(input.size() * input.item_size())
                      : a - b < static_cast<uintptr_t>(output.reserved_memory());
      }

    }

    SparseMoE::Expert::Expert(const models::Model& model, const std::string& scope)
      : gate(model, scope + "/gate_proj")
      , up(model, scope + "/up_proj")
      , down(model, scope + "/down_proj") {
    }

    void SparseMoE::Expert::operator()(const StorageView& input, StorageView& output) const {
      StorageView gated, values, activated;
      gate(input, gated);
      up(input, values);
      ops::get_activation_op(ops::ActivationType::Swish)(gated, activated);
      ops::Mul()(activated, values, activated);
      down(activated, output);
    }

    SparseMoE::SparseMoE(const models::Model& model, const std::string& scope,
                         dim_t num_experts, dim_t k)
      : _router(model, scope + "/router")
      , _shared_gate(model, scope + "/shared_expert_gate")
      , _shared(model, scope + "/shared_expert")
      , _topk(k)
      , _input_size(0)
      , _k(k) {
      if (model.device() != Device::CPU || _router.output_type() != DataType::FLOAT32
          || num_experts <= 0 || k > num_experts)
        throw std::invalid_argument("SparseMoE requires CPU float32 activations and 0 < k <= num_experts");

      // Requires unpacked /weight variables; the model must disable packing.
      // Dense still handles quantized matrices and their scale metadata.
      auto matrix = [&](const std::string& name) -> const StorageView& {
        const auto& weight = model.get_variable(name + "/weight");
        if (weight.rank() != 2 || weight.device() != Device::CPU)
          throw std::invalid_argument("SparseMoE weights must be rank-2 matrices");
        if (model.get_variable_if_exists(name + "/weight_packed"))
          throw std::invalid_argument("SparseMoE requires model weight prepacking to be disabled");
        const auto* bias = model.get_variable_if_exists(name + "/bias");
        if (bias && bias->shape() != Shape{weight.dim(0)})
          throw std::invalid_argument("SparseMoE projection bias dimension is incompatible");
        _parameters.push_back(&weight);
        for (const auto* suffix : {"/bias", "/weight_scale", "/weight_zero", "/weight_packed", "/weight_compensation"}) {
          if (const auto* value = model.get_variable_if_exists(name + suffix))
            _parameters.push_back(value);
        }
        return weight;
      };
      const auto& router = matrix(scope + "/router");
      _input_size = router.dim(1);
      if (_input_size <= 0 || router.dim(0) != num_experts)
        throw std::invalid_argument("SparseMoE router dimensions must be [num_experts, hidden]");
      const auto& shared_gate = matrix(scope + "/shared_expert_gate");
      if (shared_gate.shape() != Shape{1, _input_size})
        throw std::invalid_argument("SparseMoE shared gate dimensions must be [1, hidden]");
      auto validate_expert = [&](const std::string& name) {
        const auto& gate = matrix(name + "/gate_proj");
        const auto& up = matrix(name + "/up_proj");
        const auto& down = matrix(name + "/down_proj");
        if (gate.dim(0) <= 0 || gate.dim(1) != _input_size || up.shape() != gate.shape()
            || down.shape() != Shape{_input_size, gate.dim(0)})
          throw std::invalid_argument("SparseMoE expert projection dimensions are incompatible");
      };
      validate_expert(scope + "/shared_expert");
      for (dim_t e = 0; e < num_experts; ++e) {
        const auto name = scope + "/experts_" + std::to_string(e);
        validate_expert(name);
        _experts.emplace_back(std::make_unique<Expert>(model, name));
      }
    }

    DataType SparseMoE::output_type() const {
      return DataType::FLOAT32;
    }

    dim_t SparseMoE::output_size() const {
      return _input_size;
    }

    void SparseMoE::operator()(const StorageView& input, StorageView& output) const {
      if (input.device() != Device::CPU || output.device() != Device::CPU
          || input.dtype() != DataType::FLOAT32 || output.dtype() != DataType::FLOAT32
          || input.rank() != 2 || input.dim(1) != _input_size)
        throw std::invalid_argument("SparseMoE expects CPU float32 [rows, hidden] input/output");
      if (&input == &output || overlaps_output(input, output))
        throw std::invalid_argument("SparseMoE output must not alias input");
      for (const auto* parameter : _parameters) {
        if (overlaps_output(*parameter, output))
          throw std::invalid_argument("SparseMoE output must not alias model parameters");
      }
      for (dim_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input.data<float>()[i]))
          throw std::invalid_argument("SparseMoE input must be finite");
      }
      const dim_t rows = input.dim(0);
      if (rows == 0) {
        output.resize_as(input);
        return;
      }
      StorageView logits, indices(DataType::INT32), weights;
      _router(input, logits);
      _topk(logits, indices, weights);
      StorageView result(input.shape(), 0.f);
      StorageView row({1, _input_size}, DataType::FLOAT32);
      StorageView expert_output, shared_output, shared_gate;
      for (dim_t r = 0; r < rows; ++r) {
        for (dim_t d = 0; d < _input_size; ++d)
          row.data<float>()[d] = input.data<float>()[r * _input_size + d];
        auto* destination = result.data<float>() + r * _input_size;
        for (dim_t i = 0; i < _k; ++i) {
          const auto expert = indices.data<int32_t>()[r * _k + i];
          (*_experts[expert])(row, expert_output);
          const float probability = weights.data<float>()[r * _k + i];
          for (dim_t d = 0; d < _input_size; ++d)
            destination[d] += probability * expert_output.data<float>()[d];
        }
        _shared(row, shared_output);
        _shared_gate(row, shared_gate);
        const float gate = 1.f / (1.f + std::exp(-shared_gate.data<float>()[0]));
        for (dim_t d = 0; d < _input_size; ++d)
          destination[d] += gate * shared_output.data<float>()[d];
      }
      output = std::move(result);
    }

  }
}
