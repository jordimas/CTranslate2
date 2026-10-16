#include "ctranslate2/layers/drama_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ctranslate2 {
  namespace layers {
    namespace {

      bool overlaps(const StorageView& a, const StorageView& b) {
        if (!a.buffer() || !b.buffer() || !a.reserved_memory() || !b.reserved_memory())
          return false;
        const auto x = reinterpret_cast<uintptr_t>(a.buffer());
        const auto y = reinterpret_cast<uintptr_t>(b.buffer());
        return x <= y ? y - x < static_cast<uintptr_t>(a.reserved_memory())
                      : x - y < static_cast<uintptr_t>(b.reserved_memory());
      }

      void require_float_cpu(const StorageView& tensor) {
        if (tensor.device() != Device::CPU || tensor.dtype() != DataType::FLOAT32)
          throw std::invalid_argument("DramaAttention requires CPU float32 activations and cache");
      }

    }

    DramaAttention::DramaAttention(const models::Model& model, const std::string& scope,
                                   DramaAttentionConfig config)
      : _config(config)
      , _query(model, scope + "/q_proj")
      , _key(model, scope + "/k_proj")
      , _value(model, scope + "/v_proj")
      , _output(model, scope + "/o_proj")
      , _query_norm(model.get_variable(scope + "/q_norm/weight"))
      , _key_norm(model.get_variable(scope + "/k_norm/weight"))
      , _hidden_size(_output.output_size()) {
      const auto& cfg = _config;
      if (model.device() != Device::CPU || _query.output_type() != DataType::FLOAT32
          || cfg.num_heads <= 0 || cfg.num_kv_heads <= 0 || cfg.head_dim <= 0
          || cfg.num_heads % cfg.num_kv_heads || cfg.rotary_dim < 0
          || cfg.rotary_dim > cfg.head_dim || cfg.rotary_dim % 2
          || !std::isfinite(cfg.rope_base) || cfg.rope_base <= 0
          || !std::isfinite(cfg.norm_epsilon) || cfg.norm_epsilon <= 0 || _hidden_size <= 0)
        throw std::invalid_argument("DramaAttention configuration is incompatible");
      const dim_t widths[] = {cfg.num_heads * cfg.head_dim * 2,
                              cfg.num_kv_heads * cfg.head_dim,
                              cfg.num_kv_heads * cfg.head_dim, _hidden_size};
      size_t i = 0;
      for (const auto* name : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        const std::string path = scope + "/" + name;
        const auto& weight = model.get_variable(path + "/weight");
        const dim_t in_width = i == 3 ? cfg.num_heads * cfg.head_dim : _hidden_size;
        if (weight.device() != Device::CPU || weight.shape() != Shape{widths[i], in_width}
            || model.get_variable_if_exists(path + "/weight_packed"))
          throw std::invalid_argument("DramaAttention requires compatible unpacked projection matrices");
        _parameters.push_back(&weight);
        for (const auto* suffix : {"/bias", "/weight_scale", "/weight_compensation", "/weight_zero"}) {
          if (const auto* parameter = model.get_variable_if_exists(path + suffix)) {
            if (std::string(suffix) == "/bias" && parameter->shape() != Shape{widths[i]})
              throw std::invalid_argument("DramaAttention bias dimensions are incompatible");
            _parameters.push_back(parameter);
          }
        }
        ++i;
      }
      for (const auto* norm : {&_query_norm, &_key_norm}) {
        require_float_cpu(*norm);
        if (norm->shape() != Shape{cfg.head_dim})
          throw std::invalid_argument("DramaAttention QK norm weights must have shape [head_dim]");
        for (dim_t d = 0; d < cfg.head_dim; ++d) {
          if (!std::isfinite(norm->data<float>()[d]))
            throw std::invalid_argument("DramaAttention QK norm weights must be finite");
        }
        _parameters.push_back(norm);
      }
    }

    DataType DramaAttention::output_type() const { return DataType::FLOAT32; }
    dim_t DramaAttention::output_size() const { return _hidden_size; }

    void DramaAttention::operator()(const StorageView& input, DramaAttentionCache& cache,
                                    StorageView& output) const {
      for (const auto* tensor : {&input, static_cast<const StorageView*>(&cache.keys),
                                static_cast<const StorageView*>(&cache.values),
                                static_cast<const StorageView*>(&output)})
        require_float_cpu(*tensor);
      const auto& cfg = _config;
      if (input.rank() != 2 || input.dim(1) != _hidden_size
          || cache.keys.rank() != 3 || cache.keys.dim(1) != cfg.num_kv_heads
          || cache.keys.dim(2) != cfg.head_dim || cache.values.shape() != cache.keys.shape()
          || cache.length < 0 || cache.length > cache.keys.dim(0)
          || input.dim(0) > cache.keys.dim(0) - cache.length)
        throw std::invalid_argument("DramaAttention input/cache shape or capacity is incompatible");
      for (const auto* a : {&cache.keys, &cache.values, &output}) {
        if (a == &input || overlaps(input, *a))
          throw std::invalid_argument("DramaAttention input must not alias cache/output");
        for (const auto* parameter : _parameters) {
          if (overlaps(*parameter, *a))
            throw std::invalid_argument("DramaAttention cache/output must not alias model parameters");
        }
      }
      if (&cache.keys == &output || &cache.values == &output
          || overlaps(cache.keys, cache.values) || overlaps(cache.keys, output) || overlaps(cache.values, output))
        throw std::invalid_argument("DramaAttention cache/output tensors must not alias");
      const dim_t steps = input.dim(0);
      const dim_t kv_width = cfg.num_kv_heads * cfg.head_dim;
      for (const auto* tensor : {&input, static_cast<const StorageView*>(&cache.keys),
                                static_cast<const StorageView*>(&cache.values)}) {
        const dim_t count = tensor == &input ? input.size() : cache.length * kv_width;
        for (dim_t i = 0; i < count; ++i) {
          if (!std::isfinite(tensor->data<float>()[i]))
            throw std::invalid_argument("DramaAttention input and initialized cache prefix must be finite");
        }
      }
      if (steps == 0) {
        output.resize_as(input);
        return;
      }
      StorageView queries, keys, values;
      _query(input, queries);
      _key(input, keys);
      _value(input, values);
      const dim_t q_width = cfg.num_heads * cfg.head_dim;
      StorageView normalized_queries({steps, q_width}, DataType::FLOAT32);
      auto normalize_rotate = [&](float* destination, const float* source,
                                  const StorageView& gamma, dim_t position) {
        float square_sum = 0;
        for (dim_t d = 0; d < cfg.head_dim; ++d)
          square_sum += source[d] * source[d];
        const float scale = 1.f / std::sqrt(square_sum / cfg.head_dim + cfg.norm_epsilon);
        for (dim_t d = 0; d < cfg.head_dim; ++d)
          destination[d] = source[d] * scale
            * (gamma.data<float>()[d] + (cfg.zero_centered_qk_norm ? 1.f : 0.f));
        for (dim_t pair = 0; pair < cfg.rotary_dim / 2; ++pair) {
          const dim_t first = cfg.rotary_interleaved ? 2 * pair : pair;
          const dim_t second = cfg.rotary_interleaved ? 2 * pair + 1 : pair + cfg.rotary_dim / 2;
          const float angle = static_cast<float>(position)
            * std::pow(cfg.rope_base, -2.f * pair / cfg.rotary_dim);
          const float cosine = std::cos(angle), sine = std::sin(angle);
          const float a = destination[first], b = destination[second];
          destination[first] = a * cosine - b * sine;
          destination[second] = b * cosine + a * sine;
        }
      };
      for (dim_t t = 0; t < steps; ++t) {
        for (dim_t h = 0; h < cfg.num_heads; ++h)
          normalize_rotate(normalized_queries.data<float>() + t * q_width + h * cfg.head_dim,
                           queries.data<float>() + (t * cfg.num_heads + h) * 2 * cfg.head_dim,
                           _query_norm, cache.length + t);
        for (dim_t h = 0; h < cfg.num_kv_heads; ++h) {
          auto* key = keys.data<float>() + t * kv_width + h * cfg.head_dim;
          normalize_rotate(key, key, _key_norm, cache.length + t);
        }
      }
      StorageView attended({steps, q_width}, 0.f);
      const dim_t groups = cfg.num_heads / cfg.num_kv_heads;
      std::vector<double> scores(cache.length + steps);
      for (dim_t t = 0; t < steps; ++t) {
        const dim_t length = cache.length + t + 1;
        for (dim_t h = 0; h < cfg.num_heads; ++h) {
          const auto* query = normalized_queries.data<float>() + t * q_width + h * cfg.head_dim;
          double maximum = -std::numeric_limits<double>::infinity();
          for (dim_t s = 0; s < length; ++s) {
            const auto* key = s < cache.length ? cache.keys.data<float>() + s * kv_width
                                              : keys.data<float>() + (s - cache.length) * kv_width;
            key += (h / groups) * cfg.head_dim;
            double score = 0;
            for (dim_t d = 0; d < cfg.head_dim; ++d)
              score += static_cast<double>(query[d]) * key[d];
            scores[s] = score / std::sqrt(static_cast<double>(cfg.head_dim));
            maximum = std::max(maximum, scores[s]);
          }
          double total = 0;
          for (dim_t s = 0; s < length; ++s)
            total += scores[s] = std::exp(scores[s] - maximum);
          for (dim_t d = 0; d < cfg.head_dim; ++d) {
            double value = 0;
            for (dim_t s = 0; s < length; ++s) {
              const auto* source = s < cache.length ? cache.values.data<float>() + s * kv_width
                                                   : values.data<float>() + (s - cache.length) * kv_width;
              value += (scores[s] / total) * source[(h / groups) * cfg.head_dim + d];
            }
            const float gate = queries.data<float>()[(t * cfg.num_heads + h) * 2 * cfg.head_dim + cfg.head_dim + d];
            attended.data<float>()[t * q_width + h * cfg.head_dim + d]
              = static_cast<float>(value) / (1.f + std::exp(-gate));
          }
        }
      }
      StorageView result;
      _output(attended, result);
      for (const auto* tensor : {&result, &keys, &values}) {
        for (dim_t i = 0; i < tensor->size(); ++i) {
          if (!std::isfinite(tensor->data<float>()[i]))
            throw std::runtime_error("DramaAttention produced non-finite values; cache was not updated");
        }
      }
      std::copy_n(keys.data<float>(), keys.size(), cache.keys.data<float>() + cache.length * kv_width);
      std::copy_n(values.data<float>(), values.size(), cache.values.data<float>() + cache.length * kv_width);
      cache.length += steps;
      output = std::move(result);
    }

  }
}
