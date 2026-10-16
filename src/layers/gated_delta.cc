#include "ctranslate2/layers/gated_delta.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "ctranslate2/ops/causal_depthwise_conv1d.h"
#include "ctranslate2/ops/gated_delta_net.h"

namespace ctranslate2 {
  namespace layers {
    namespace {

      bool overlaps(const StorageView& a, const StorageView& b) {
        const auto an = std::max(a.reserved_memory(), a.size() * a.item_size());
        const auto bn = std::max(b.reserved_memory(), b.size() * b.item_size());
        if (!a.buffer() || !b.buffer() || !an || !bn)
          return false;
        const auto ap = reinterpret_cast<uintptr_t>(a.buffer());
        const auto bp = reinterpret_cast<uintptr_t>(b.buffer());
        return ap <= bp ? bp - ap < static_cast<uintptr_t>(an)
                        : ap - bp < static_cast<uintptr_t>(bn);
      }

      dim_t product(dim_t a, dim_t b) {
        if (a <= 0 || b <= 0 || b > std::numeric_limits<dim_t>::max() / a)
          throw std::invalid_argument("GatedDeltaNet dimensions must be positive and bounded");
        return a * b;
      }

      void require_float(const StorageView& tensor) {
        if (tensor.device() != Device::CPU || tensor.dtype() != DataType::FLOAT32)
          throw std::invalid_argument("GatedDeltaNet requires CPU float32 tensors");
      }

      void require_finite(const StorageView& tensor) {
        for (dim_t i = 0; i < tensor.size(); ++i) {
          if (!std::isfinite(tensor.data<float>()[i]))
            throw std::invalid_argument("GatedDeltaNet floating tensors must be finite");
        }
      }

      float sigmoid(float x) {
        if (x >= 0)
          return 1.f / (1.f + std::exp(-x));
        const auto e = std::exp(x);
        return e / (1.f + e);
      }

      float softplus(float x) {
        return std::max(x, 0.f) + std::log1p(std::exp(-std::abs(x)));
      }

    }

    GatedDeltaNet::GatedDeltaNet(const models::Model& model, const std::string& scope,
                                dim_t key_heads, dim_t value_heads,
                                dim_t key_dim, dim_t value_dim, float rms_epsilon)
      : _qkv(model, scope + "/in_proj_qkv")
      , _z(model, scope + "/in_proj_z")
      , _a(model, scope + "/in_proj_a")
      , _b(model, scope + "/in_proj_b")
      , _out(model, scope + "/out_proj")
      , _a_log(model.get_variable(scope + "/A_log"))
      , _dt_bias(model.get_variable(scope + "/dt_bias"))
      , _norm_weight(model.get_variable(scope + "/norm/weight"))
      , _conv_bias(model.get_variable_if_exists(scope + "/conv1d/bias"))
      , _key_heads(key_heads), _value_heads(value_heads)
      , _key_dim(key_dim), _value_dim(value_dim), _rms_epsilon(rms_epsilon) {
      if (model.device() != Device::CPU || _qkv.output_type() != DataType::FLOAT32
          || key_heads <= 0 || value_heads <= 0 || value_heads % key_heads
          || !std::isfinite(rms_epsilon) || rms_epsilon <= 0)
        throw std::invalid_argument("GatedDeltaNet requires CPU float32 and valid dimensions/epsilon");
      _key_width = product(key_heads, key_dim);
      _value_width = product(value_heads, value_dim);
      if (_key_width > (std::numeric_limits<dim_t>::max() - _value_width) / 2)
        throw std::invalid_argument("GatedDeltaNet projection dimensions overflow");
      _conv_channels = 2 * _key_width + _value_width;
      product(product(value_heads, key_dim), value_dim);
      auto matrix = [&](const std::string& name) -> const StorageView& {
        const auto& weight = model.get_variable(scope + name + "/weight");
        if (weight.rank() != 2 || weight.device() != Device::CPU
            || model.get_variable_if_exists(scope + name + "/weight_packed")
            || model.get_variable_if_exists(scope + name + "/bias"))
          throw std::invalid_argument("GatedDeltaNet requires unpacked bias-free projection matrices");
        _parameters.push_back(&weight);
        for (const auto* suffix : {"/weight_scale", "/weight_zero", "/weight_compensation"}) {
          if (const auto* parameter = model.get_variable_if_exists(scope + name + suffix))
            _parameters.push_back(parameter);
        }
        return weight;
      };
      const auto& qkv = matrix("/in_proj_qkv");
      _hidden = qkv.dim(1);
      if (_hidden <= 0 || qkv.dim(0) != _conv_channels
          || matrix("/in_proj_z").shape() != Shape{_value_width, _hidden}
          || matrix("/in_proj_a").shape() != Shape{value_heads, _hidden}
          || matrix("/in_proj_b").shape() != Shape{value_heads, _hidden}
          || matrix("/out_proj").shape() != Shape{_hidden, _value_width})
        throw std::invalid_argument("GatedDeltaNet projection dimensions are incompatible");
      const auto& conv = model.get_variable(scope + "/conv1d/weight");
      require_float(conv);
      if (conv.rank() != 3 || conv.dim(0) != _conv_channels || conv.dim(1) != 1 || conv.dim(2) < 1)
        throw std::invalid_argument("GatedDeltaNet conv1d weight must be [channels,1,kernel]");
      _kernel = conv.dim(2);
      _conv_weight = conv;
      _conv_weight.reshape({_conv_channels, _kernel});
      _parameters.insert(_parameters.end(), {&conv, &_a_log, &_dt_bias, &_norm_weight});
      if (_conv_bias)
        _parameters.push_back(_conv_bias);
      for (const auto* parameter : {&conv, &_a_log, &_dt_bias, &_norm_weight, _conv_bias}) {
        if (parameter) {
          require_float(*parameter);
          require_finite(*parameter);
        }
      }
      if (_a_log.shape() != Shape{value_heads} || _dt_bias.shape() != Shape{value_heads}
          || _norm_weight.shape() != Shape{value_dim}
          || (_conv_bias && _conv_bias->shape() != Shape{_conv_channels}))
        throw std::invalid_argument("GatedDeltaNet gate/norm/conv bias dimensions are incompatible");
    }

    DataType GatedDeltaNet::output_type() const { return DataType::FLOAT32; }
    dim_t GatedDeltaNet::output_size() const { return _hidden; }

    void GatedDeltaNet::initialize_state(StorageView& state, StorageView& history) const {
      if (&state == &history)
        throw std::invalid_argument("GatedDeltaNet state and history must be distinct");
      state = StorageView({1, _value_heads, _key_dim, _value_dim}, 0.f);
      history = StorageView({1, _conv_channels, _kernel - 1}, 0.f);
    }

    void GatedDeltaNet::operator()(const StorageView& input, StorageView& state,
                                   StorageView& history, StorageView& output) const {
      for (const auto* tensor : {&input, static_cast<const StorageView*>(&state),
                                 static_cast<const StorageView*>(&history),
                                 static_cast<const StorageView*>(&output)})
        require_float(*tensor);
      if (input.rank() != 2 || input.dim(0) < 0 || input.dim(1) != _hidden
          || state.shape() != Shape{1, _value_heads, _key_dim, _value_dim}
          || history.shape() != Shape{1, _conv_channels, _kernel - 1})
        throw std::invalid_argument("GatedDeltaNet input/cache shapes are incompatible");
      const StorageView* mutable_tensors[] = {&state, &history, &output};
      for (int i = 0; i < 3; ++i) {
        const auto* tensor = mutable_tensors[i];
        if (tensor == &input || overlaps(*tensor, input))
          throw std::invalid_argument("GatedDeltaNet mutable tensors must not alias input");
        for (int j = 0; j < i; ++j) {
          if (tensor == mutable_tensors[j] || overlaps(*tensor, *mutable_tensors[j]))
            throw std::invalid_argument("GatedDeltaNet mutable tensors must not alias");
        }
        for (const auto* parameter : _parameters) {
          if (tensor == parameter || overlaps(*tensor, *parameter))
            throw std::invalid_argument("GatedDeltaNet mutable tensors must not alias parameters");
        }
      }
      require_finite(input);
      require_finite(state);
      require_finite(history);
      const auto steps = input.dim(0);
      if (!steps) {
        output.resize({0, _hidden});
        return;
      }
      // Commit caller caches only after all projections and numerical checks
      // succeed. These small cache copies also preserve external-view storage.
      StorageView next_state(state), next_history(history);
      StorageView projected, mixed, z, a, beta;
      _qkv(input, projected);
      projected.reshape({1, steps, _conv_channels});
      ops::CausalDepthwiseConv1D()(projected, _conv_weight, _conv_bias, next_history, mixed);
      _z(input, z);
      _a(input, a);
      _b(input, beta);
      require_finite(z);
      require_finite(a);
      require_finite(beta);
      StorageView query({1, steps, _key_heads, _key_dim});
      StorageView key(query.shape());
      StorageView value({1, steps, _value_heads, _value_dim});
      for (dim_t t = 0; t < steps; ++t) {
        const auto* row = mixed.data<float>() + t * _conv_channels;
        std::copy(row, row + _key_width, query.data<float>() + t * _key_width);
        std::copy(row + _key_width, row + 2 * _key_width, key.data<float>() + t * _key_width);
        std::copy(row + 2 * _key_width, row + _conv_channels, value.data<float>() + t * _value_width);
      }
      for (auto* tensor : {&query, &key}) {
        for (dim_t row = 0; row < steps * _key_heads; ++row) {
          auto* values = tensor->data<float>() + row * _key_dim;
          float sum = 1e-6f;
          for (dim_t i = 0; i < _key_dim; ++i)
            sum += values[i] * values[i];
          const auto inverse = 1.f / std::sqrt(sum);
          for (dim_t i = 0; i < _key_dim; ++i)
            values[i] *= inverse;
        }
      }
      for (dim_t i = 0; i < steps * _value_heads; ++i) {
        const auto head = i % _value_heads;
        beta.data<float>()[i] = sigmoid(beta.data<float>()[i]);
        a.data<float>()[i] = -std::exp(_a_log.data<float>()[head])
          * softplus(a.data<float>()[i] + _dt_bias.data<float>()[head]);
      }
      a.reshape({1, steps, _value_heads});
      beta.reshape(a.shape());
      require_finite(query);
      require_finite(key);
      require_finite(value);
      require_finite(a);
      StorageView core;
      ops::GatedDeltaNet()(query, key, value, a, beta, next_state, core);
      for (dim_t row = 0; row < steps * _value_heads; ++row) {
        auto* values = core.data<float>() + row * _value_dim;
        float mean = 0;
        for (dim_t i = 0; i < _value_dim; ++i)
          mean += values[i] * values[i];
        const auto inverse = 1.f / std::sqrt(mean / _value_dim + _rms_epsilon);
        for (dim_t i = 0; i < _value_dim; ++i) {
          const auto gate = z.data<float>()[row * _value_dim + i];
          values[i] *= inverse * _norm_weight.data<float>()[i] * gate * sigmoid(gate);
        }
      }
      core.reshape({steps, _value_width});
      StorageView result;
      _out(core, result);
      require_finite(result);
      require_finite(next_state);
      output.copy_from(result);
      state.copy_from(next_state);
      history.copy_from(next_history);
    }

  }
}
