#include "ctranslate2/layers/drama_tower.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ctranslate2 {
  namespace layers {

    int32_t sample_drama_token(const std::vector<float>& logits,
                               const DramaSamplingOptions& options,
                               std::mt19937_64& rng) {
      if (logits.empty() || logits.size() > size_t(std::numeric_limits<int32_t>::max())
          || !std::isfinite(options.temperature) || options.temperature <= 0
          || !std::isfinite(options.top_p) || options.top_p < 0 || options.top_p > 1
          || options.top_k <= 0)
        throw std::invalid_argument("Drama sampling requires finite positive temperature/top_k and top_p in [0,1]");
      for (const auto value : logits) {
        if (!std::isfinite(value))
          throw std::invalid_argument("Drama sampling logits must be finite");
      }
      std::vector<int32_t> order(logits.size());
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
      });
      const double temperature = std::max(options.temperature, 1e-5f);
      const double maximum = logits[order[0]];
      std::vector<double> probabilities(logits.size());
      double full_sum = 0;
      for (size_t rank = 0; rank < order.size(); ++rank)
        full_sum += probabilities[rank] = std::exp((double(logits[order[rank]]) - maximum) / temperature);
      double cumulative = 0, kept_sum = 0;
      size_t kept = 0;
      for (size_t rank = 0; rank < order.size(); ++rank) {
        cumulative += probabilities[rank] / full_sum;
        if (rank && (cumulative > options.top_p || rank >= size_t(options.top_k)))
          break;
        kept_sum += probabilities[rank];
        ++kept;
      }
      // Categorical-equivalent to PyTorch's exponential-race draw, without
      // promising identical random sequences between the two implementations.
      const double threshold = std::generate_canonical<double, 53>(rng) * kept_sum;
      double sum = 0;
      for (size_t rank = 0; rank < kept; ++rank) {
        sum += probabilities[rank];
        if (threshold < sum)
          return order[rank];
      }
      return order[kept - 1];
    }
    namespace {

      void require_cpu_float(const StorageView& tensor) {
        if (tensor.device() != Device::CPU || tensor.dtype() != DataType::FLOAT32)
          throw std::invalid_argument("DramaTower requires CPU float32 activations");
      }

      bool overlaps(const StorageView& a, const StorageView& b) {
        if (!a.buffer() || !b.buffer() || !a.reserved_memory() || !b.reserved_memory())
          return false;
        const auto x = reinterpret_cast<uintptr_t>(a.buffer()), y = reinterpret_cast<uintptr_t>(b.buffer());
        return x <= y ? y - x < static_cast<uintptr_t>(a.reserved_memory())
                      : x - y < static_cast<uintptr_t>(b.reserved_memory());
      }

      void norm(const StorageView& input, const StorageView& gamma, float epsilon, StorageView& output) {
        output.resize_as(input);
        const dim_t dim = input.dim(1);
        for (dim_t t = 0; t < input.dim(0); ++t) {
          float square = 0;
          for (dim_t d = 0; d < dim; ++d) {
            const float x = input.data<float>()[t * dim + d];
            square += x * x;
          }
          const float scale = 1.f / std::sqrt(square / dim + epsilon);
          for (dim_t d = 0; d < dim; ++d)
            output.data<float>()[t * dim + d] = input.data<float>()[t * dim + d]
              * scale * (1.f + gamma.data<float>()[d]);
        }
      }

      void validate_embedding(const models::Model& model, const std::string& scope,
                              const StorageView& weights) {
        if (weights.device() != Device::CPU
            || (weights.dtype() != DataType::FLOAT32 && weights.dtype() != DataType::INT8
                && weights.dtype() != DataType::INT16))
          throw std::invalid_argument("DramaTower embedding type is unsupported");
        if (weights.dtype() != DataType::FLOAT32) {
          const auto* scale = model.get_variable_if_exists(scope + "/weight_scale");
          if (!scale || scale->device() != Device::CPU || scale->dtype() != DataType::FLOAT32
              || (!scale->is_scalar() && scale->shape() != Shape{weights.dim(0)}))
            throw std::invalid_argument("DramaTower quantized embeddings need float32 per-row or scalar scales");
          for (dim_t i = 0; i < scale->size(); ++i) {
            if (!std::isfinite(scale->data<float>()[i]) || scale->data<float>()[i] <= 0)
              throw std::invalid_argument("DramaTower embedding scales must be finite and positive");
          }
        }
      }

    }

    struct DramaTower::TowerLayer {
      const StorageView& embedding;
      Embeddings embedding_lookup;
      const StorageView& enorm;
      const StorageView& hnorm;
      const StorageView& attention_norm;
      const StorageView& ffn_norm;
      const StorageView& final_norm;
      Dense eh_proj, gate, up, down;
      DramaAttention attention;
      float epsilon;
      dim_t dim;

      TowerLayer(const models::Model& model, const std::string& scope,
                 const DramaTowerConfig& cfg, bool shared)
        : embedding(model.get_variable(scope + (shared ? "/semantic_embedding/weight" : "/codebook_embedding/weight")))
        , embedding_lookup(model, scope + (shared ? "/semantic_embedding" : "/codebook_embedding"))
        , enorm(model.get_variable(scope + "/enorm/weight"))
        , hnorm(model.get_variable(scope + "/hnorm/weight"))
        , attention_norm(model.get_variable(scope + "/transformer_layer/attention_norm/weight"))
        , ffn_norm(model.get_variable(scope + "/transformer_layer/ffn_norm/weight"))
        , final_norm(model.get_variable(scope + "/final_layernorm/weight"))
        , eh_proj(model, scope + "/eh_proj")
        , gate(model, scope + "/transformer_layer/feed_forward/gate_proj")
        , up(model, scope + "/transformer_layer/feed_forward/up_proj")
        , down(model, scope + "/transformer_layer/feed_forward/down_proj")
        , attention(model, scope + "/transformer_layer/attention", cfg.attention)
        , epsilon(cfg.attention.norm_epsilon)
        , dim(attention.output_size()) {
        if (embedding.shape() != Shape{cfg.vocab_size, dim})
          throw std::invalid_argument("DramaTower layer embedding dimensions are incompatible");
        validate_embedding(model, scope + (shared ? "/semantic_embedding" : "/codebook_embedding"), embedding);
        for (const auto* gamma : {&enorm, &hnorm, &attention_norm, &ffn_norm, &final_norm}) {
          require_cpu_float(*gamma);
          if (gamma->shape() != Shape{dim})
            throw std::invalid_argument("DramaTower norm dimensions are incompatible");
          for (dim_t i = 0; i < gamma->size(); ++i) {
            if (!std::isfinite(gamma->data<float>()[i]))
              throw std::invalid_argument("DramaTower norm coefficients must be finite");
          }
        }
        auto weight = [&](const std::string& path, dim_t rows, dim_t cols) {
          const auto& value = model.get_variable(path + "/weight");
          if (value.device() != Device::CPU || value.shape() != Shape{rows, cols}
              || model.get_variable_if_exists(path + "/weight_packed"))
            throw std::invalid_argument("DramaTower requires compatible unpacked projection matrices");
        };
        if (gate.output_size() <= 0 || gate.output_type() != DataType::FLOAT32)
          throw std::invalid_argument("DramaTower requires float32 activation projections");
        weight(scope + "/eh_proj", dim, 2 * dim);
        weight(scope + "/transformer_layer/feed_forward/gate_proj", gate.output_size(), dim);
        weight(scope + "/transformer_layer/feed_forward/up_proj", gate.output_size(), dim);
        weight(scope + "/transformer_layer/feed_forward/down_proj", dim, gate.output_size());
      }

      void operator()(const StorageView& hidden, const StorageView& next_frames,
                      dim_t code_column, int32_t semantic_begin, int32_t semantic_end,
                      const StorageView* shared_input, DramaAttentionCache& cache,
                      StorageView& output) const {
        const dim_t rows = hidden.dim(0), columns = next_frames.dim(1);
        StorageView e(hidden.shape(), 0.f);
        StorageView ids({rows}, DataType::INT32), selected;
        for (dim_t t = 0; t < rows; ++t)
          ids.data<int32_t>()[t] = next_frames.data<int32_t>()[t * columns + code_column];
        if (rows)
          embedding_lookup(ids, selected);
        for (dim_t t = 0; t < rows; ++t) {
          const int32_t main = next_frames.data<int32_t>()[t * columns];
          const bool semantic = main >= semantic_begin && main <= semantic_end;
          for (dim_t d = 0; d < dim; ++d)
            e.data<float>()[t * dim + d] = (semantic ? selected.data<float>()[t * dim + d] : 0.f)
              + (shared_input ? shared_input->data<float>()[t * dim + d] : 0.f);
        }
        StorageView en, hn;
        norm(e, enorm, epsilon, en);
        norm(hidden, hnorm, epsilon, hn);
        StorageView joined({rows, 2 * dim}, DataType::FLOAT32);
        for (dim_t t = 0; t < rows; ++t)
          for (dim_t d = 0; d < dim; ++d) {
            joined.data<float>()[t * 2 * dim + d] = en.data<float>()[t * dim + d];
            joined.data<float>()[t * 2 * dim + dim + d] = hn.data<float>()[t * dim + d];
          }
        StorageView mixed;
        if (rows == 0) {
          output.resize_as(hidden);
          return;
        }
        eh_proj(joined, mixed);
        StorageView normalized, attended;
        norm(mixed, attention_norm, epsilon, normalized);
        attention(normalized, cache, attended);
        for (dim_t i = 0; i < mixed.size(); ++i)
          mixed.data<float>()[i] += attended.data<float>()[i];
        norm(mixed, ffn_norm, epsilon, normalized);
        StorageView gates, ups, activated, projected;
        gate(normalized, gates);
        up(normalized, ups);
        ops::get_activation_op(ops::ActivationType::Swish)(gates, activated);
        for (dim_t i = 0; i < activated.size(); ++i)
          activated.data<float>()[i] *= ups.data<float>()[i];
        down(activated, projected);
        for (dim_t i = 0; i < mixed.size(); ++i)
          mixed.data<float>()[i] += projected.data<float>()[i];
        norm(mixed, final_norm, epsilon, output);
      }
    };

    DramaTower::DramaTower(const models::Model& model, const std::string& scope,
                           DramaTowerConfig config)
      : _config(config)
      , _embeddings(model.get_variable(scope + "/codebook_embeddings/weight"))
      , _embedding_lookup(model, scope + "/codebook_embeddings") {
      const auto& cfg = _config;
      if (cfg.num_codebooks < 2 || cfg.vocab_size <= 0 || cfg.shared_layers < 1
          || cfg.vq_backbone_codebooks < 0 || cfg.vq_backbone_codebooks >= cfg.num_codebooks
          || cfg.semantic_begin < 0 || cfg.semantic_end < cfg.semantic_begin || cfg.eos_id < 0
          || int64_t(cfg.semantic_end) - cfg.semantic_begin + 1 > cfg.vocab_size
          || is_semantic(cfg.eos_id) || _embeddings.rank() != 2
          || _embeddings.dim(0) != cfg.num_codebooks * cfg.vocab_size || _embeddings.dim(1) <= 0
          || !cfg.attention.rotary_interleaved || !cfg.attention.zero_centered_qk_norm)
        throw std::invalid_argument("DramaTower configuration or codebook embedding dimensions are incompatible");
      validate_embedding(model, scope + "/codebook_embeddings", _embeddings);
      for (dim_t i = 0; i < cfg.shared_layers; ++i) {
        auto layer = std::make_unique<TowerLayer>(model, scope + "/shared_layers_" + std::to_string(i), cfg, true);
        if (layer->dim != output_size())
          throw std::invalid_argument("DramaTower shared hidden dimension is incompatible");
        _shared.emplace_back(std::move(layer));
      }
      for (dim_t i = 0; i < cfg.num_codebooks - 1; ++i) {
        auto layer = std::make_unique<TowerLayer>(model, scope + "/residual_layers_" + std::to_string(i), cfg, false);
        if (layer->dim != output_size())
          throw std::invalid_argument("DramaTower residual hidden dimension is incompatible");
        _residual.emplace_back(std::move(layer));
        const auto head_scope = scope + "/heads_" + std::to_string(i);
        const auto& weight = model.get_variable(head_scope + "/weight");
        if (weight.shape() != Shape{cfg.vocab_size, output_size()}
            || model.get_variable_if_exists(head_scope + "/weight_packed"))
          throw std::invalid_argument("DramaTower requires compatible unpacked head matrices");
        _heads.emplace_back(std::make_unique<Dense>(model, head_scope));
      }
    }

    DramaTower::~DramaTower() = default;
    DataType DramaTower::output_type() const { return DataType::FLOAT32; }
    dim_t DramaTower::output_size() const { return _embeddings.dim(1); }
    bool DramaTower::is_semantic(int32_t token) const {
      return token >= _config.semantic_begin && token <= _config.semantic_end;
    }

    void DramaTower::initialize_state(DramaTowerState& state, dim_t max_tokens) const {
      if (max_tokens <= 0)
        throw std::invalid_argument("DramaTower requires positive cache capacity");
      DramaTowerState fresh;
      fresh.capacity = max_tokens;
      auto allocate = [&](size_t count, std::vector<DramaAttentionCache>& caches) {
        for (size_t i = 0; i < count; ++i) {
          const Shape shape{max_tokens, _config.attention.num_kv_heads, _config.attention.head_dim};
          caches.push_back({StorageView(shape, DataType::FLOAT32), StorageView(shape, DataType::FLOAT32), 0});
        }
      };
      allocate(_shared.size(), fresh.shared);
      allocate(_residual.size(), fresh.residual);
      state = std::move(fresh);
    }

    void DramaTower::validate_frames(const StorageView& frames, dim_t rows) const {
      if (frames.device() != Device::CPU || frames.dtype() != DataType::INT32
          || frames.shape() != Shape{rows, _config.num_codebooks + 1})
        throw std::invalid_argument("DramaTower frames must be CPU int32 [time,1+num_codebooks]");
      for (dim_t t = 0; t < rows; ++t)
        for (dim_t c = 1; c <= _config.num_codebooks; ++c) {
          const auto code = frames.data<int32_t>()[t * (_config.num_codebooks + 1) + c];
          if (code < 0 || code >= _config.vocab_size)
            throw std::invalid_argument("DramaTower frame code is out of range");
        }
    }

    void DramaTower::validate_hidden(const StorageView& hidden) const {
      require_cpu_float(hidden);
      if (hidden.rank() != 2 || hidden.dim(1) != output_size())
        throw std::invalid_argument("DramaTower hidden must be [time,dim]");
      for (dim_t i = 0; i < hidden.size(); ++i) {
        if (!std::isfinite(hidden.data<float>()[i]))
          throw std::invalid_argument("DramaTower hidden values must be finite");
      }
    }

    void DramaTower::validate_state(const DramaTowerState& state, dim_t rows) const {
      if (state.shared.size() != _shared.size() || state.residual.size() != _residual.size()
          || state.capacity <= 0 || state.length < 0 || state.length > state.capacity
          || rows > state.capacity - state.length)
        throw std::invalid_argument("DramaTower state/context limit is incompatible");
      for (const auto* caches : {&state.shared, &state.residual})
        for (const auto& cache : *caches) {
          require_cpu_float(cache.keys);
          require_cpu_float(cache.values);
          if (cache.length != state.length || cache.keys.shape() != Shape{state.capacity, _config.attention.num_kv_heads, _config.attention.head_dim}
              || cache.values.shape() != cache.keys.shape())
            throw std::invalid_argument("DramaTower temporal cache position/dimensions are incompatible");
        }
    }

    void DramaTower::validate_state_aliases(const DramaTowerState& state,
                                           const std::vector<const StorageView*>& tensors) const {
      std::vector<const StorageView*> allocations;
      for (const auto* caches : {&state.shared, &state.residual})
        for (const auto& cache : *caches)
          for (const auto* tensor : {&cache.keys, &cache.values}) {
            for (const auto* input : tensors) {
              if (input == tensor || overlaps(*input, *tensor))
                throw std::invalid_argument("DramaTower temporal caches must not alias inputs/outputs");
            }
            for (const auto* prior : allocations) {
              if (overlaps(*prior, *tensor))
                throw std::invalid_argument("DramaTower temporal caches must not alias each other");
            }
            allocations.push_back(tensor);
          }
    }

    StorageView DramaTower::embedding_sum(const StorageView& frames, dim_t first, dim_t end) const {
      StorageView result({frames.dim(0), output_size()}, 0.f);
      const dim_t count = end - first;
      if (!count || !frames.dim(0))
        return result;
      StorageView ids({frames.dim(0) * count}, DataType::INT32), selected;
      for (dim_t t = 0; t < frames.dim(0); ++t)
        for (dim_t c = first; c < end; ++c) {
          const dim_t code = frames.data<int32_t>()[t * (_config.num_codebooks + 1) + 1 + c];
          ids.data<int32_t>()[t * count + c - first] = static_cast<int32_t>(c * _config.vocab_size + code);
        }
      _embedding_lookup(ids, selected);
      for (dim_t t = 0; t < frames.dim(0); ++t)
        for (dim_t c = first; c < end; ++c)
          for (dim_t d = 0; d < output_size(); ++d)
            result.data<float>()[t * output_size() + d] += selected.data<float>()[((t * count) + c - first) * output_size() + d];
      return result;
    }

    void DramaTower::embed_backbone(const StorageView& frames, const StorageView& text_embeddings,
                                    StorageView& output) const {
      validate_hidden(text_embeddings);
      validate_frames(frames, text_embeddings.dim(0));
      require_cpu_float(output);
      if (&text_embeddings == &output || overlaps(text_embeddings, output)
          || overlaps(frames, output) || overlaps(_embeddings, output))
        throw std::invalid_argument("DramaTower embedding output must not alias inputs/weights");
      auto extra = embedding_sum(frames, 1, 1 + _config.vq_backbone_codebooks);
      const float divisor = std::sqrt(float(1 + _config.vq_backbone_codebooks));
      for (dim_t t = 0; t < text_embeddings.dim(0); ++t) {
        const bool semantic = is_semantic(frames.data<int32_t>()[t * (_config.num_codebooks + 1)]);
        for (dim_t d = 0; d < output_size(); ++d) {
          const dim_t i = t * output_size() + d;
          extra.data<float>()[i] = semantic ? (text_embeddings.data<float>()[i] + extra.data<float>()[i]) / divisor
                                           : text_embeddings.data<float>()[i];
        }
      }
      output = std::move(extra);
    }

    void DramaTower::run_shared(const StorageView& hidden, const StorageView& frames,
                                const StorageView& next_frames, DramaTowerState& state,
                                StorageView& output) const {
      auto unseen = embedding_sum(frames, 1 + _config.vq_backbone_codebooks, _config.num_codebooks);
      for (dim_t t = 0; t < frames.dim(0); ++t)
        if (!is_semantic(frames.data<int32_t>()[t * (_config.num_codebooks + 1)]))
          for (dim_t d = 0; d < output_size(); ++d)
            unseen.data<float>()[t * output_size() + d] = 0;
      StorageView current(hidden);
      for (size_t i = 0; i < _shared.size(); ++i) {
        StorageView next;
        (*_shared[i])(current, next_frames, 1, _config.semantic_begin, _config.semantic_end,
                     i == 0 ? &unseen : nullptr, state.shared[i], next);
        current = std::move(next);
      }
      output = std::move(current);
    }

    void DramaTower::prefill(const StorageView& hidden, const StorageView& frames,
                             const StorageView& next_frames, DramaTowerState& state) const {
      validate_hidden(hidden);
      validate_frames(frames, hidden.dim(0));
      validate_frames(next_frames, hidden.dim(0));
      validate_state(state, hidden.dim(0));
      validate_state_aliases(state, {&hidden, &frames, &next_frames, &_embeddings});
      StorageView current;
      run_shared(hidden, frames, next_frames, state, current);
      for (size_t i = 0; i < _residual.size(); ++i) {
        StorageView next;
        (*_residual[i])(current, next_frames, i + 1, _config.semantic_begin, _config.semantic_end,
                       nullptr, state.residual[i], next);
        current = std::move(next);
      }
      state.length += hidden.dim(0);
    }

    void DramaTower::decode_greedy(const StorageView& logits, const StorageView& hidden,
                                   const StorageView& previous_frame, DramaTowerState& state,
                                   StorageView& next_frame) const {
      decode_impl(logits, hidden, previous_frame, state, next_frame, nullptr, nullptr, nullptr);
    }

    void DramaTower::decode_sample(const StorageView& logits, const StorageView& hidden,
                                   const StorageView& previous_frame, DramaTowerState& state,
                                   const std::vector<int32_t>& history_main_tokens,
                                   const DramaSamplingOptions& options, std::mt19937_64& rng,
                                   StorageView& next_frame) const {
      decode_impl(logits, hidden, previous_frame, state, next_frame, &history_main_tokens, &options, &rng);
    }

    void DramaTower::decode_impl(const StorageView& logits, const StorageView& hidden,
                                 const StorageView& previous_frame, DramaTowerState& state,
                                 StorageView& next_frame,
                                 const std::vector<int32_t>* history_main_tokens,
                                 const DramaSamplingOptions* options, std::mt19937_64* rng) const {
      validate_hidden(hidden);
      validate_frames(previous_frame, 1);
      validate_state(state, 1);
      validate_state_aliases(state, {&logits, &hidden, &previous_frame, &next_frame, &_embeddings});
      require_cpu_float(logits);
      if (hidden.dim(0) != 1 || logits.rank() != 2 || logits.dim(0) != 1
          || logits.dim(1) <= std::max(_config.semantic_end, _config.eos_id)
          || next_frame.device() != Device::CPU || next_frame.dtype() != DataType::INT32)
        throw std::invalid_argument("DramaTower greedy decode expects one hidden/logit row and int32 frame output");
      for (const auto* input : {&logits, &hidden, &previous_frame, &_embeddings}) {
        if (input == &next_frame || overlaps(*input, next_frame))
          throw std::invalid_argument("DramaTower frame output must not alias inputs/weights");
      }
      int32_t main = _config.eos_id;
      float best = logits.data<float>()[main];
      for (int64_t candidate = _config.semantic_begin; candidate <= _config.semantic_end; ++candidate) {
        const float value = logits.data<float>()[candidate];
        if (!std::isfinite(value) || !std::isfinite(best))
          throw std::invalid_argument("DramaTower candidate logits must be finite");
        if (value > best || (value == best && candidate < main)) {
          main = static_cast<int32_t>(candidate);
          best = value;
        }
      }
      if (options) {
        std::vector<int32_t> candidates{_config.eos_id};
        for (int64_t candidate = _config.semantic_begin; candidate <= _config.semantic_end; ++candidate)
          candidates.push_back(static_cast<int32_t>(candidate));
        std::sort(candidates.begin(), candidates.end());
        std::vector<float> candidate_logits;
        for (const auto candidate : candidates)
          candidate_logits.push_back(logits.data<float>()[candidate]);
        main = candidates[sample_drama_token(candidate_logits, *options, *rng)];
        DramaSamplingOptions retry;
        retry.temperature = 1.f;
        retry.top_k = 30;
        retry.top_p = .95f;
        // The official graph draws the retry unconditionally. Only semantic
        // repeats in the last16 main tokens use it; EOS is never resampled.
        const auto replacement = candidates[sample_drama_token(candidate_logits, retry, *rng)];
        const size_t first = history_main_tokens->size() > 16 ? history_main_tokens->size() - 16 : 0;
        if (is_semantic(main)
            && std::find(history_main_tokens->begin() + first, history_main_tokens->end(), main) != history_main_tokens->end())
          main = replacement;
      }
      StorageView frame({1, _config.num_codebooks + 1}, int32_t(0));
      frame.data<int32_t>()[0] = main;
      frame.data<int32_t>()[1] = static_cast<int32_t>(std::max<int64_t>(0, std::min<int64_t>(_config.vocab_size - 1, int64_t(main) - _config.semantic_begin)));
      StorageView current;
      run_shared(hidden, previous_frame, frame, state, current);
      for (size_t i = 0; i < _residual.size(); ++i) {
        StorageView next, code_logits;
        (*_residual[i])(current, frame, i + 1, _config.semantic_begin, _config.semantic_end,
                       nullptr, state.residual[i], next);
        _heads[i]->operator()(next, code_logits);
        int32_t chosen = 0;
        for (dim_t code = 0; code < _config.vocab_size; ++code) {
          const float value = code_logits.data<float>()[code];
          if (!std::isfinite(value))
            throw std::runtime_error("DramaTower code logits are non-finite; reset state before retry");
          if (value > code_logits.data<float>()[chosen])
            chosen = static_cast<int32_t>(code);
        }
        if (options) {
          const std::vector<float> values(code_logits.data<float>(), code_logits.data<float>() + _config.vocab_size);
          chosen = sample_drama_token(values, *options, *rng);
        }
        frame.data<int32_t>()[i + 2] = chosen;
        current = std::move(next);
      }
      ++state.length;
      next_frame = std::move(frame);
    }

  }
}
