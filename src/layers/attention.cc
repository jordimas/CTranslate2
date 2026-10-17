#include "ctranslate2/layers/attention.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/rms_norm.h"
#include "ctranslate2/utils.h"


#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <stdexcept>

#include "dispatch.h"
#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace layers {

    StorageView make_relative_positions(dim_t queries_length,
                                        dim_t keys_length,
                                        dim_t max_position) {
      StorageView positions({queries_length, keys_length}, DataType::INT32);
      auto* positions_data = positions.data<int32_t>();

      const dim_t offset = keys_length - queries_length;

      for (dim_t i = 0; i < queries_length; ++i) {
        auto* row = positions_data + i * keys_length;
        for (dim_t j = 0; j < keys_length; ++j) {
          row[j] = std::min(std::max(j - (i + offset), -max_position), max_position) + max_position;
        }
      }

      return positions;
    }

    StorageView make_asymmetric_relative_positions(dim_t queries_length,
                                                  dim_t keys_length,
                                                  dim_t left_max_position,
                                                  dim_t right_max_position) {
      StorageView positions({queries_length, keys_length}, DataType::INT32);
      auto* positions_data = positions.data<int32_t>();

      for (dim_t i = 0; i < queries_length; ++i) {
        auto* row = positions_data + i * keys_length;
        for (dim_t j = 0; j < keys_length; ++j) {
          row[j] = std::max(std::min(j - i, right_max_position), -left_max_position) + left_max_position;
        }
      }

      return positions;
    }

    static StorageView get_relative_position_bucket(bool bidirectional,
                                                    dim_t query_length,
                                                    dim_t key_length,
                                                    dim_t num_buckets,
                                                    dim_t max_distance,
                                                    dim_t query_offset = 0) {
      StorageView relative_buckets({query_length, key_length}, DataType::INT32);
      int32_t* relative_buckets_data = relative_buckets.data<int32_t>();

      if (bidirectional)
        num_buckets /= 2;

      const dim_t max_exact = num_buckets / 2;
      const float log_max_distance_over_max_exact = std::log(float(max_distance) / float(max_exact));

      cpu::parallel_for(0, query_length * key_length, 8, [&](const dim_t begin, const dim_t end) {
        for (dim_t flat_index = begin; flat_index < end; ++flat_index) {
          const dim_t i = flat_index / key_length;
          const dim_t j = flat_index % key_length;

          int32_t relative_position = j - (i + query_offset);
          int32_t relative_bucket = 0;

          if (bidirectional) {
            if (relative_position > 0)
              relative_bucket += num_buckets;
            else
              relative_position = std::abs(relative_position);
          } else {
            relative_position = -std::min(relative_position, 0);
          }

          const bool is_small = relative_position < max_exact;

          if (!is_small) {
            relative_position = std::min(
              int32_t(float(max_exact)
                      + std::log(float(relative_position) / float(max_exact))
                      / log_max_distance_over_max_exact
                      * float(num_buckets - max_exact)),
              int32_t(num_buckets - 1));
          }

          relative_bucket += relative_position;
          relative_buckets_data[flat_index] = relative_bucket;
        }
      });

      return relative_buckets;
    }

    static StorageView compute_relative_bias(const StorageView& relative_attention_bias,
                                             dim_t query_length,
                                             dim_t key_length,
                                             dim_t max_distance,
                                             bool is_decoder,
                                             dim_t query_offset = 0) {
      const Device device = relative_attention_bias.device();
      const DataType dtype = relative_attention_bias.dtype();
      const dim_t num_buckets = relative_attention_bias.dim(0);

      StorageView relative_attention_bucket = get_relative_position_bucket(!is_decoder,
                                                                           query_length,
                                                                           key_length,
                                                                           num_buckets,
                                                                           max_distance,
                                                                           query_offset);

      StorageView values(dtype, device);
      ops::Gather()(relative_attention_bias, relative_attention_bucket.to(device), values);

      StorageView values_t(dtype, device);
      ops::Transpose({2, 0, 1})(values, values_t);

      return values_t;
    }

    static void matmul_with_relative_representations(const ops::MatMul& matmul_op,
                                                     const StorageView& a,
                                                     const StorageView& b,
                                                     StorageView& c) {
      const Device device = a.device();
      const DataType dtype = a.dtype();
      const dim_t batch = a.dim(0);
      const dim_t head = a.dim(1);
      const dim_t time = a.dim(2);

      StorageView a_t(dtype, device);
      ops::Transpose({2, 0, 1, 3})(a, a_t);
      a_t.reshape({time, batch * head, -1});

      StorageView c_t(dtype, device);
      matmul_op(a_t, b, c_t);
      c_t.reshape({time, batch, head, -1});
      ops::Transpose({1, 2, 0, 3})(c_t, c);
    }

    static void add_relative_representations(const StorageView& queries,
                                             const StorageView& relative_positions,
                                             const StorageView& relative_values,
                                             const ops::MatMul& matmul_op,
                                             StorageView& dot) {
      const Device device = queries.device();
      const DataType dtype = queries.dtype();

      StorageView relative_representations(dtype, device);
      ops::Gather()(relative_values, relative_positions, relative_representations);

      StorageView dot_relative(dtype, device);
      matmul_with_relative_representations(matmul_op,
                                           queries,
                                           relative_representations,
                                           dot_relative);
      ops::Add()(dot_relative, dot, dot);
    }

    static const ops::Transpose transpose_op({0, 2, 1, 3});

    static inline void save_attention(StorageView& attention, StorageView weights, dim_t beam_size) {
      if (beam_size == 1)
        attention = std::move(weights);
      else {
        transpose_op(weights, attention);
        attention.reshape({-1, weights.dim(1), 1, weights.dim(-1)});
      }
    }

    static void dot_product_attention(const StorageView& queries,
                                      const StorageView& keys,
                                      const StorageView& values,
                                      const StorageView* values_lengths,
                                      const StorageView* relative_position_keys,
                                      const StorageView* relative_asymmetric_position_keys,
                                      const StorageView* relative_position_values,
                                      const StorageView* relative_attention_bias,
                                      dim_t relative_left_max_position,
                                      dim_t relative_right_max_position,
                                      dim_t maximum_relative_position,
                                      StorageView& output,
                                      StorageView* attention = nullptr,
                                      bool return_normalized_attention = true,
                                      float queries_scale = 1,
                                      bool is_decoder = false,
                                      bool with_cache = false,
                                      dim_t beam_size = 1,
                                      Alibi* alibi = nullptr,
                                      StorageView* position_bias = nullptr) {
      PROFILE("dot_product_attention");

      std::unique_ptr<const StorageView> relative_positions;
      if (relative_position_keys || relative_position_values || relative_asymmetric_position_keys) {
        const dim_t query_length = queries.dim(2);
        const dim_t key_length = keys.dim(2);
        if (relative_asymmetric_position_keys)
          relative_positions = std::make_unique<StorageView>(
            make_asymmetric_relative_positions(query_length,
                                    key_length,
                                    relative_left_max_position,
                                    relative_right_max_position).to(queries.device()));
        else relative_positions = std::make_unique<StorageView>(
               make_relative_positions(query_length,
                                       key_length,
                                       maximum_relative_position).to(queries.device()));
      }

      const ops::MatMul keys_matmul(/*trans_a=*/false, /*trans_b=*/true, queries_scale);
      keys_matmul(queries, keys, output);
      if (relative_position_keys)
        add_relative_representations(queries,
                                     *relative_positions,
                                     *relative_position_keys,
                                     keys_matmul,
                                     output);

      if (relative_asymmetric_position_keys)
        add_relative_representations(queries,
                                     *relative_positions,
                                     *relative_asymmetric_position_keys,
                                     keys_matmul,
                                     output);
      if (relative_attention_bias) {
        StorageView local_position_bias(output.dtype(), output.device());

        if (!position_bias)
          position_bias = &local_position_bias;

        if (position_bias->empty()) {
          const dim_t query_length = queries.dim(2);
          const dim_t key_length = keys.dim(2);
          *position_bias = compute_relative_bias(*relative_attention_bias,
                                                 query_length,
                                                 key_length,
                                                 maximum_relative_position,
                                                 is_decoder,
                                                 with_cache ? key_length - 1 : 0);
        }
        StorageView* position_bias_per_gpu = position_bias;
        StorageView position_bias_tmp(position_bias->dtype(), position_bias->device());
        if (ScopedMPISetter::getCurRank() != 0) {
          const dim_t num_head_per_gpu = SAFE_DIVIDE(position_bias->dim(0), ScopedMPISetter::getNRanks());
          ops::Slide slide_ops(0, num_head_per_gpu * ScopedMPISetter::getCurRank(),
                               num_head_per_gpu, true);
          slide_ops(*position_bias, position_bias_tmp);
          position_bias_per_gpu = &position_bias_tmp;
        }

        DEVICE_AND_TYPE_DISPATCH(output.device(), output.dtype(),
                                 primitives<D>::add_batch_broadcast(position_bias_per_gpu->data<T>(),
                                                                    output.data<T>(),
                                                                    position_bias_per_gpu->size(),
                                                                    output.size()));
      }

      if (alibi)
        alibi->apply(output, queries_scale);

      StorageView attn(values.dtype(), values.device());
      if (attention && !return_normalized_attention) {
        ops::SoftMax()(output, values_lengths, attn);
        save_attention(*attention, std::move(output), beam_size);
      } else {
        attn = std::move(output);
        ops::SoftMax()(attn, values_lengths, attn);
      }

      const ops::MatMul values_matmul;
      values_matmul(attn, values, output);
      if (relative_position_values)
        add_relative_representations(attn,
                                     *relative_positions,
                                     *relative_position_values,
                                     values_matmul,
                                     output);

      if (attention && return_normalized_attention)
        save_attention(*attention, std::move(attn), beam_size);
    }



    static void replicate_heads(StorageView& x, dim_t repeats) {
      x.expand_dims(2);
      ops::Tile(2, repeats)(x);
      x.reshape({x.dim(0), x.dim(1) * x.dim(2), x.dim(3), x.dim(4)});
    }

    MultiHeadAttention::MultiHeadAttention(const models::Model& model,
                                           const std::string& scope,
                                           dim_t num_heads,
                                           bool self_attention,
                                           bool pre_norm,
                                           bool is_decoder,
                                           Alibi* alibi)
      : AttentionLayer(model, scope, num_heads, self_attention, pre_norm, is_decoder, alibi, false)
      , _relative_attention_bias(model.get_variable_if_exists(scope + "/relative_attention_bias"))
      , _relative_position_keys(model.get_variable_if_exists(scope + "/relative_position_keys"))
      , _relative_asymmetric_position_keys(model.get_variable_if_exists(scope + "/relative_asymmetric_position_keys"))
      , _relative_position_values(model.get_variable_if_exists(scope + "/relative_position_values"))
      , _merge_time_and_head_dims(_multi_query
                                  && !_relative_attention_bias
                                  && !_relative_position_keys
                                  && !_relative_position_values)
      ,_cache_time_dim(_merge_time_and_head_dims ? 1 : 2)
      , _q_norm(build_optional_layer<LayerNorm>(model, scope + "/q_norm"))
      , _k_norm(build_optional_layer<LayerNorm>(model, scope + "/k_norm"))
      , _v_norm(build_optional_layer<LayerNorm>(model, scope + "/v_norm"))
      , _memory_kv((model.get_variable_if_exists(scope + "/memory_kv/weight")
                    || model.get_variable_if_exists(scope + "/memory_kv/weight_packed"))
                     ? std::make_unique<const Dense>(model, scope + "/memory_kv")
                     : nullptr)
    {
      if (_relative_position_keys)
        _maximum_relative_position = (_relative_position_keys->dim(0) - 1) / 2;
      else if (_relative_asymmetric_position_keys) {
        _relative_left_max_position = model.get_attribute<int32_t>(
          scope + "/relative_left_max_position");
        _relative_right_max_position = model.get_attribute<int32_t>(
          scope + "/relative_right_max_position");
      }
      else if (_relative_attention_bias)
        _maximum_relative_position = model.get_attribute<int32_t>(
          scope + "/relative_attention_max_distance");
      else
        _maximum_relative_position = 0;
    }

    DataType MultiHeadAttention::output_type() const {
      return _linear.back().output_type();
    }

    dim_t MultiHeadAttention::output_size() const {
      return _d_model;
    }

    void MultiHeadAttention::apply_k_norm(StorageView& keys_proj) const {
      if (_k_norm) {
        StorageView keys_normed(keys_proj.dtype(), keys_proj.device());
        (*_k_norm)(keys_proj, keys_normed);
        keys_proj = std::move(keys_normed);
      }
    }

    void MultiHeadAttention::apply_qk_norm(StorageView& queries_proj, StorageView& keys_proj) const {
      if (_q_norm) {
        StorageView queries_normed(queries_proj.dtype(), queries_proj.device());
        (*_q_norm)(queries_proj, queries_normed);
        queries_proj = std::move(queries_normed);
      }

      apply_k_norm(keys_proj);
    }

    void MultiHeadAttention::apply_v_norm(StorageView& values_proj) const {
      if (_v_norm) {
        StorageView values_normed(values_proj.dtype(), values_proj.device());
        (*_v_norm)(values_proj, values_normed);
        values_proj = std::move(values_normed);
      }
    }

    void MultiHeadAttention::process_cross_attention(
        const StorageView& queries,
        const StorageView& values,
        StorageView& fused_proj,
        StorageView& queries_proj,
        StorageView& keys_proj,
        StorageView& values_proj,
        StorageView* cached_keys,
        StorageView* cached_values,
        const Padder* queries_padder,
        const Padder* values_padder,
        dim_t& beam_size) const {

      queries_proj = std::move(fused_proj);

      if (cached_keys == nullptr || cached_keys->empty()) {
        _linear[1](values, fused_proj);

        if (_num_heads_kv == 1) { // MQA (Multi-Query Attention)
          if (values_padder)
            values_padder->add_padding(fused_proj);
          ops::Split(2, {_d_head, _d_head})(fused_proj, keys_proj, values_proj);

          apply_k_norm(keys_proj);
          apply_v_norm(values_proj);
          keys_proj.expand_dims(1);
          values_proj.expand_dims(1);
          replicate_heads(keys_proj, _num_heads);
          replicate_heads(values_proj, _num_heads);

        } else if (_num_heads_kv < _num_heads) { // GQA (Grouped-Query Attention)
          if (values_padder)
            values_padder->add_padding(fused_proj);

          const ops::Split split_op(2, {_num_heads_kv * _d_head, _num_heads_kv * _d_head});
          split_op(fused_proj, keys_proj, values_proj);

          split_heads(keys_proj, _num_heads_kv);
          split_heads(values_proj, _num_heads_kv);

          apply_k_norm(keys_proj);
          apply_v_norm(values_proj);

          replicate_heads(keys_proj, _num_heads / _num_heads_kv);
          replicate_heads(values_proj, _num_heads / _num_heads_kv);
        } else { //Standard Multi-Head Attention (MHA)
          split_heads(fused_proj, 2 * _num_heads, values_padder);
          ops::Split(1)(fused_proj, keys_proj, values_proj);

          apply_k_norm(keys_proj);
          apply_v_norm(values_proj);
        }

        if (cached_keys != nullptr) {
          *cached_keys = std::move(keys_proj);
          *cached_values = std::move(values_proj);
        }
      }

      if (_q_norm) {
        StorageView queries_normed(queries_proj.dtype(), queries_proj.device());
        (*_q_norm)(queries_proj, queries_normed);
        queries_proj = std::move(queries_normed);
      }

      if (queries_proj.dim(1) == 1 && cached_keys)
        beam_size = queries_proj.dim(0) / cached_keys->dim(0);

      split_heads(queries_proj, _num_heads, queries_padder, beam_size);
    }

    void MultiHeadAttention::operator()(const StorageView& queries,
                                        const StorageView& values,
                                        const StorageView* values_lengths,
                                        StorageView& output,
                                        StorageView* cached_keys,
                                        StorageView* cached_values,
                                        StorageView* attention,
                                        const Padder* queries_padder,
                                        const Padder* values_padder,
                                        bool return_normalized_attention,
                                        StorageView* position_bias,
                                        dim_t offset) const {
      PROFILE("MultiHeadAttention");
      const Device device = queries.device();
      const DataType dtype = queries.dtype();
      StorageView fused_proj(dtype, device);
      StorageView queries_proj(dtype, device);
      StorageView keys_proj(dtype, device);
      StorageView values_proj(dtype, device);

      const StorageView* q = &queries;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(queries, queries_proj);
        q = &queries_proj;
      }

      _linear[0](*q, fused_proj);

      dim_t beam_size = 1;

      bool prefilling = (_sliding_window > 0 && values_lengths);

      if (!_self_attention) {

        process_cross_attention(queries, values, fused_proj, queries_proj, keys_proj,
                                values_proj, cached_keys, cached_values,
                                queries_padder, values_padder, beam_size);
      } else {

        if (_num_heads_kv < _num_heads) {// MQA or GQA: queries stay in merged time/head format
          if (queries_padder)
            queries_padder->add_padding(fused_proj);

          const ops::Split split_op(2, {_num_heads * _d_head, _num_heads_kv * _d_head, _num_heads_kv * _d_head});
          split_op(fused_proj, queries_proj, keys_proj, values_proj);

          if (_merge_time_and_head_dims) {
            queries_proj.reshape({queries_proj.dim(0), -1, _d_head});
            // k_norm/v_norm operate per-head (gamma size = _d_head), so temporarily
            // split keys/values into heads, apply norms, then restore merged layout.
            if (_k_norm || _q_norm || _v_norm) {
              split_heads(keys_proj, _num_heads_kv);
              split_heads(values_proj, _num_heads_kv);
              apply_qk_norm(queries_proj, keys_proj);
              apply_v_norm(values_proj);
              // Restore flat [batch, time, num_kv_heads*head_dim] layout.
              combine_heads(keys_proj, _num_heads_kv);
              combine_heads(values_proj, _num_heads_kv);
            }
          } else {
            split_heads(queries_proj, _num_heads);
            split_heads(keys_proj, _num_heads_kv);
            split_heads(values_proj, _num_heads_kv);

            apply_qk_norm(queries_proj, keys_proj);
            apply_v_norm(values_proj);

            replicate_heads(keys_proj, _num_heads / _num_heads_kv);
            replicate_heads(values_proj, _num_heads / _num_heads_kv);
          }

        } else {
          split_heads(fused_proj, 3 * _num_heads, queries_padder);
          ops::Split(1)(fused_proj, queries_proj, keys_proj, values_proj);

          apply_qk_norm(queries_proj, keys_proj);
          apply_v_norm(values_proj);
        }

        if (_rotary_embeddings) {
          if (_merge_time_and_head_dims) {
            queries_proj.reshape({queries_proj.dim(0), -1, _d_model});
            split_heads(queries_proj, _num_heads);
          }

          _rotary_embeddings->apply(queries_proj, offset);
          _rotary_embeddings->apply(keys_proj, offset);

          if (_merge_time_and_head_dims) {
            combine_heads(queries_proj, _num_heads);
            queries_proj.reshape({queries_proj.dim(0), -1, _d_head});
          }
        }

        if (cached_keys != nullptr) {
          if (cached_keys->empty()) {
            *cached_keys = std::move(keys_proj);
            *cached_values = std::move(values_proj);
          } else {
            const ops::Concat concat_op(_cache_time_dim);
            StorageView& tmp = fused_proj;  // Reuse storage.
            tmp = std::move(*cached_keys);
            concat_op({&tmp, &keys_proj}, *cached_keys);
            tmp = std::move(*cached_values);
            concat_op({&tmp, &values_proj}, *cached_values);

            if (!prefilling && _sliding_window > 0 && cached_keys->shape()[2] > _sliding_window) {
              // only for generation
              const ops::Slide slide_op(2, 1, cached_keys->shape()[2] - 1);
              slide_op(*cached_keys, tmp);
              *cached_keys = std::move(tmp);
              slide_op(*cached_values, tmp);
              *cached_values = std::move(tmp);
            }
          }
        }
      }

      if (cached_keys) {
        keys_proj.shallow_copy(*cached_keys);
        values_proj.shallow_copy(*cached_values);
      }

      StorageView& context = fused_proj;  // Reuse storage.
      dot_product_attention(queries_proj,
                            keys_proj,
                            values_proj,
                            values_lengths,
                            _relative_position_keys,
                            _relative_asymmetric_position_keys,
                            _relative_position_values,
                            _relative_attention_bias,
                            _relative_left_max_position,
                            _relative_right_max_position,
                            _maximum_relative_position,
                            context,
                            attention,
                            return_normalized_attention,
                            _queries_scale,
                            _is_decoder,
                            bool(cached_keys),
                            beam_size,
                            _alibi,
                            position_bias);

      if (prefilling && cached_keys && cached_keys->shape()[2] > _sliding_window) {
        // set only last sliding_window tokens to cached_keys and cached_values after computing attention
        const ops::Slide slide_op(2, cached_keys->shape()[2] - _sliding_window, _sliding_window);
        StorageView tmp(dtype, device);
        slide_op(*cached_keys, tmp);
        *cached_keys = std::move(tmp);
        slide_op(*cached_values, tmp);
        *cached_values = std::move(tmp);
      }

      if (_merge_time_and_head_dims) {
        context.reshape(queries.shape());
        if (queries_padder)
          queries_padder->remove_padding(context);
      } else {
        combine_heads(context, _num_heads, queries_padder, beam_size);
      }
      _linear.back()(context, output, _layer_norm ? &queries : nullptr);

      if (_tensor_parallel) {
        Shape shape = output.shape();
        StorageView tmp(std::move(shape), output.dtype(), output.device());
        ops::ReduceAll ops_reduce_all(ops::ReduceAll::RED_OP::SUM);
        ops_reduce_all(output, tmp);
        output = std::move(tmp);
      }
      if (_layer_norm && !_pre_norm)
        (*_layer_norm)(output, output);
    }

    // Expand a 4-D tensor [batch, heads, time, d_head] along the batch dimension
    // by `repeats`, producing [batch*repeats, heads, time, d_head].
    static void replicate_batches(StorageView& x, dim_t repeats) {
      x.expand_dims(1);                         // [batch, 1, heads, time, d_head]
      ops::Tile(1, repeats)(x);                 // [batch, repeats, heads, time, d_head]
      x.reshape({x.dim(0) * x.dim(1), x.dim(2), x.dim(3), x.dim(4)});
    }

    void MultiHeadAttention::forward_merged(const StorageView& queries,
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
                                            dim_t offset) const {
      PROFILE("MultiHeadAttentionMerged");
      const Device device = queries.device();

      const DataType dtype = queries.dtype();

      StorageView fused_proj(dtype, device);
      StorageView queries_proj(dtype, device);
      StorageView keys_proj(dtype, device);
      StorageView values_proj(dtype, device);

      const StorageView* q = &queries;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(queries, queries_proj);
        q = &queries_proj;
      }

      _linear[0](*q, fused_proj);

      if (_num_heads_kv < _num_heads) {
        if (queries_padder)
          queries_padder->add_padding(fused_proj);
        const ops::Split split_op(2, {_num_heads * _d_head,
                                      _num_heads_kv * _d_head,
                                      _num_heads_kv * _d_head});
        split_op(fused_proj, queries_proj, keys_proj, values_proj);
        split_heads(queries_proj, _num_heads);
        split_heads(keys_proj, _num_heads_kv);
        split_heads(values_proj, _num_heads_kv);
        apply_qk_norm(queries_proj, keys_proj);
        apply_v_norm(values_proj);
        replicate_heads(keys_proj, _num_heads / _num_heads_kv);
        replicate_heads(values_proj, _num_heads / _num_heads_kv);
      } else {
        split_heads(fused_proj, 3 * _num_heads, queries_padder);
        ops::Split(1)(fused_proj, queries_proj, keys_proj, values_proj);
        apply_qk_norm(queries_proj, keys_proj);
        apply_v_norm(values_proj);
      }

      if (_rotary_embeddings) {
        _rotary_embeddings->apply(queries_proj, offset);
        _rotary_embeddings->apply(keys_proj, offset);
      }

      // Update self-attention KV cache.
      if (cached_self_keys) {
        if (cached_self_keys->empty()) {
          *cached_self_keys = std::move(keys_proj);
          *cached_self_values = std::move(values_proj);
        } else {
          const ops::Concat concat_op(2);
          StorageView tmp(dtype, device);
          tmp = std::move(*cached_self_keys);
          concat_op({&tmp, &keys_proj}, *cached_self_keys);
          tmp = std::move(*cached_self_values);
          concat_op({&tmp, &values_proj}, *cached_self_values);
        }
        keys_proj.shallow_copy(*cached_self_keys);
        values_proj.shallow_copy(*cached_self_values);
      }

      // Compute beam_size: queries batch = batch * beam_size; memory cache at batch granularity.
      // During the first decoding step memory is not yet cached; use the live memory batch.
      const bool memory_already_cached = cached_memory_keys && !cached_memory_keys->empty();
      const dim_t batch_size_no_beam = memory_already_cached
                                          ? cached_memory_keys->dim(0)
                                          : (memory ? memory->dim(0) : queries.dim(0));
      const dim_t beam_size = queries.dim(0) / batch_size_no_beam;

      // Project encoder memory through memory_kv (no RoPE; k_norm only).
      // Memory K/V is stored at batch granularity (without beam expansion).
      StorageView memory_keys(dtype, device);
      StorageView memory_values(dtype, device);
      if (memory_already_cached) {
        memory_keys.shallow_copy(*cached_memory_keys);
        memory_values.shallow_copy(*cached_memory_values);
      } else {
        StorageView mem_kv(dtype, device);
        (*_memory_kv)(*memory, mem_kv);
        if (_num_heads_kv < _num_heads) {
          if (memory_padder)
            memory_padder->add_padding(mem_kv);
          const ops::Split split_op(2, {_num_heads_kv * _d_head, _num_heads_kv * _d_head});
          split_op(mem_kv, memory_keys, memory_values);
          split_heads(memory_keys, _num_heads_kv);
          split_heads(memory_values, _num_heads_kv);
          apply_k_norm(memory_keys);
          replicate_heads(memory_keys, _num_heads / _num_heads_kv);
          replicate_heads(memory_values, _num_heads / _num_heads_kv);
        } else {
          split_heads(mem_kv, 2 * _num_heads, memory_padder);
          ops::Split(1)(mem_kv, memory_keys, memory_values);
          apply_k_norm(memory_keys);
        }
        if (cached_memory_keys) {
          *cached_memory_keys = memory_keys;
          *cached_memory_values = memory_values;
        }
      }

      // Expand memory K/V from [batch, heads, src_len, d] to [batch*beam, heads, src_len, d].
      if (beam_size > 1) {
        replicate_batches(memory_keys, beam_size);
        replicate_batches(memory_values, beam_size);
      }

      // Concatenate: self K/V (decoder steps) || memory K/V (src_len) along time dim.
      // This matches HF: key_states = cat([self_keys, cross_keys], dim=2)
      const ops::Concat time_concat(2);
      StorageView merged_keys(dtype, device);
      StorageView merged_values(dtype, device);
      time_concat({&keys_proj, &memory_keys}, merged_keys);
      time_concat({&values_proj, &memory_values}, merged_values);

      // Build merged lengths mask.
      // During prefix processing (training-style), both self_lengths_mask and
      // memory_lengths_mask are provided; we add them to get the merged valid count.
      // During single-step generation, self_lengths_mask is null; we create a
      // full-coverage mask that allows attending to all merged keys.
      const dim_t total_keys = merged_keys.dim(2);
      const dim_t q_len = queries_proj.dim(2);

      std::unique_ptr<StorageView> merged_lengths;
      if (self_lengths_mask && memory_lengths_mask) {
        // Normalise memory_lengths_mask shape to match self_lengths_mask if needed
        // (beam search may produce different leading dims).
        std::unique_ptr<StorageView> norm_mem_lengths;
        const StorageView* mem_len_ptr = memory_lengths_mask;
        if (memory_lengths_mask->shape() != self_lengths_mask->shape()) {
          norm_mem_lengths = std::make_unique<StorageView>(*memory_lengths_mask);
          norm_mem_lengths->reshape(self_lengths_mask->shape());
          mem_len_ptr = norm_mem_lengths.get();
        }
        merged_lengths = std::make_unique<StorageView>(self_lengths_mask->dtype(), device);
        merged_lengths->resize_as(*self_lengths_mask);
        ops::Add()(*self_lengths_mask, *mem_len_ptr, *merged_lengths);
      } else if (self_lengths_mask) {
        merged_lengths = std::make_unique<StorageView>(*self_lengths_mask);
      } else {
        // Single-step generation: no causal mask needed (only 1 query), and all
        // merged keys are valid. Create a constant mask = total_keys for each query.
        const dim_t batch_queries = queries_proj.dim(0) * _num_heads * q_len;
        merged_lengths = std::make_unique<StorageView>(
          Shape{batch_queries}, int32_t(total_keys), device);
      }

      StorageView context(dtype, device);
      const ops::MatMul keys_matmul(false, true, _queries_scale);
      keys_matmul(queries_proj, merged_keys, context);
      StorageView attn(dtype, device);
      ops::SoftMax()(context, merged_lengths.get(), attn);
      const ops::MatMul values_matmul;
      values_matmul(attn, merged_values, context);

      combine_heads(context, _num_heads, queries_padder, /*beam_size=*/1);
      _linear.back()(context, output, _layer_norm ? &queries : nullptr);

      if (_layer_norm && !_pre_norm)
        (*_layer_norm)(output, output);
    }

    void MultiHeadAttention::split_heads(StorageView& x,
                     dim_t num_heads,
                     const Padder* padder,
                     dim_t beam_size) {
      if (padder)
        padder->add_padding(x);

      if (beam_size > 1)
        x.reshape({x.dim(0) / beam_size, beam_size, x.dim(2)});

      // x has shape [batch_size, time, depth]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(1);
      const dim_t head_dim = x.dim(2) / num_heads;

      if (time == 1) {
        x.reshape({batch_size, num_heads, 1, head_dim});
      } else {
        x.reshape({batch_size, time, num_heads, head_dim});
        StorageView y(x.device(), x.dtype());
        transpose_op(x, y);
        x = std::move(y);
      }
    }

    void MultiHeadAttention::combine_heads(StorageView& x,
                                         dim_t num_heads,
                                         const Padder* padder,
                                         dim_t beam_size) {
      // x has shape [batch_size, num_heads, time, head_dim]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(2);
      const dim_t depth = x.dim(3) * num_heads;

      if (time > 1) {
        StorageView y(x.device(), x.dtype());
        transpose_op(x, y);
        x = std::move(y);
      }

      x.reshape({batch_size, time, depth});

      if (beam_size > 1)
        x.reshape({batch_size * beam_size, 1, depth});

      if (padder)
        padder->remove_padding(x);
    }

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
          throw std::invalid_argument("GatedSelfAttention requires CPU float32 activations and cache");
      }

    }

    GatedSelfAttention::GatedSelfAttention(const models::Model& model, const std::string& scope,
                                         GatedSelfAttentionConfig config)
      : _config(config)
      , _query(model, scope + "/q_proj")
      , _key(model, scope + "/k_proj")
      , _value(model, scope + "/v_proj")
      , _output(model, scope + "/o_proj")
      , _query_norm(model.get_variable(scope + "/q_norm/weight"))
      , _key_norm(model.get_variable(scope + "/k_norm/weight"))
      , _hidden_size(_output.output_size())
      , _rotary(config.rotary_dim, config.rotary_interleaved, RotaryScalingType::None,
                1.f, config.rope_base, 64) {
      const auto& cfg = _config;
      if (model.device() != Device::CPU || _query.output_type() != DataType::FLOAT32
          || cfg.num_heads <= 0 || cfg.num_kv_heads <= 0 || cfg.head_dim <= 0
          || cfg.num_heads % cfg.num_kv_heads || cfg.rotary_dim < 0
          || cfg.rotary_dim > cfg.head_dim || cfg.rotary_dim % 2
          || !std::isfinite(cfg.rope_base) || cfg.rope_base <= 0
          || !std::isfinite(cfg.norm_epsilon) || cfg.norm_epsilon <= 0 || _hidden_size <= 0)
        throw std::invalid_argument("GatedSelfAttention configuration is incompatible");
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
          throw std::invalid_argument("GatedSelfAttention requires compatible unpacked projection matrices");
        _parameters.push_back(&weight);
        for (const auto* suffix : {"/bias", "/weight_scale", "/weight_compensation", "/weight_zero"}) {
          if (const auto* parameter = model.get_variable_if_exists(path + suffix)) {
            if (std::string(suffix) == "/bias" && parameter->shape() != Shape{widths[i]})
              throw std::invalid_argument("GatedSelfAttention bias dimensions are incompatible");
            _parameters.push_back(parameter);
          }
        }
        ++i;
      }
      for (const auto* norm : {&_query_norm, &_key_norm}) {
        require_float_cpu(*norm);
        if (norm->shape() != Shape{cfg.head_dim})
          throw std::invalid_argument("GatedSelfAttention QK norm weights must have shape [head_dim]");
        for (dim_t d = 0; d < cfg.head_dim; ++d) {
          if (!std::isfinite(norm->data<float>()[d]))
            throw std::invalid_argument("GatedSelfAttention QK norm weights must be finite");
        }
        _parameters.push_back(norm);
      }
    }

    DataType GatedSelfAttention::output_type() const { return DataType::FLOAT32; }
    dim_t GatedSelfAttention::output_size() const { return _hidden_size; }

    void GatedSelfAttention::operator()(const StorageView& input, GatedSelfAttentionCache& cache,
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
        throw std::invalid_argument("GatedSelfAttention input/cache shape or capacity is incompatible");
      for (const auto* a : {&cache.keys, &cache.values, &output}) {
        if (a == &input || overlaps(input, *a))
          throw std::invalid_argument("GatedSelfAttention input must not alias cache/output");
        for (const auto* parameter : _parameters) {
          if (overlaps(*parameter, *a))
            throw std::invalid_argument("GatedSelfAttention cache/output must not alias model parameters");
        }
      }
      if (&cache.keys == &output || &cache.values == &output
          || overlaps(cache.keys, cache.values) || overlaps(cache.keys, output) || overlaps(cache.values, output))
        throw std::invalid_argument("GatedSelfAttention cache/output tensors must not alias");
      const dim_t steps = input.dim(0);
      const dim_t kv_width = cfg.num_kv_heads * cfg.head_dim;
      for (const auto* tensor : {&input, static_cast<const StorageView*>(&cache.keys),
                                static_cast<const StorageView*>(&cache.values)}) {
        const dim_t count = tensor == &input ? input.size() : cache.length * kv_width;
        for (dim_t i = 0; i < count; ++i) {
          if (!std::isfinite(tensor->data<float>()[i]))
            throw std::invalid_argument("GatedSelfAttention input and initialized cache prefix must be finite");
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
      for (const auto* tensor : {&queries, &keys, &values})
        for (dim_t i = 0; i < tensor->size(); ++i)
          if (!std::isfinite(tensor->data<float>()[i]))
            throw std::runtime_error("GatedSelfAttention projection overflow; cache was not updated");
      const dim_t q_width = cfg.num_heads * cfg.head_dim;
      StorageView normalized_queries({steps, q_width}, DataType::FLOAT32);
      for (dim_t t = 0; t < steps; ++t) {
        for (dim_t h = 0; h < cfg.num_heads; ++h)
          std::copy_n(queries.data<float>() + (t * cfg.num_heads + h) * 2 * cfg.head_dim,
                      cfg.head_dim, normalized_queries.data<float>() + t * q_width + h * cfg.head_dim);
      }
      normalized_queries.reshape({1, steps, cfg.num_heads, cfg.head_dim});
      keys.reshape({1, steps, cfg.num_kv_heads, cfg.head_dim});
      const ops::RMSNorm norm(cfg.norm_epsilon, cfg.zero_centered_qk_norm);
      norm(_query_norm, normalized_queries, normalized_queries);
      norm(_key_norm, keys, keys);
      if (cfg.rotary_dim) {
        StorageView transposed_queries, transposed_keys;
        transpose_op(normalized_queries, transposed_queries);
        transpose_op(keys, transposed_keys);
        _rotary.apply(transposed_queries, cache.length);
        _rotary.apply(transposed_keys, cache.length);
        transpose_op(transposed_queries, normalized_queries);
        transpose_op(transposed_keys, keys);
      }
      keys.reshape({steps, kv_width});
      // Use the same batched MatMul/SoftMax attention implementation as MHA.
      // Pack grouped query heads as additional query rows, avoiding KV replication.
      const dim_t groups = cfg.num_heads / cfg.num_kv_heads;
      const dim_t length = cache.length + steps;
      StorageView packed_queries({1, cfg.num_kv_heads, groups * steps, cfg.head_dim});
      StorageView packed_keys({1, cfg.num_kv_heads, length, cfg.head_dim});
      StorageView packed_values(packed_keys.shape());
      StorageView lengths({cfg.num_kv_heads * groups * steps}, DataType::INT32);
      for (dim_t h = 0; h < cfg.num_kv_heads; ++h) {
        for (dim_t s = 0; s < length; ++s) {
          const auto* key = s < cache.length ? cache.keys.data<float>() + s * kv_width
                                            : keys.data<float>() + (s - cache.length) * kv_width;
          const auto* value = s < cache.length ? cache.values.data<float>() + s * kv_width
                                              : values.data<float>() + (s - cache.length) * kv_width;
          const dim_t offset = (h * length + s) * cfg.head_dim;
          std::copy_n(key + h * cfg.head_dim, cfg.head_dim, packed_keys.data<float>() + offset);
          std::copy_n(value + h * cfg.head_dim, cfg.head_dim, packed_values.data<float>() + offset);
        }
        for (dim_t g = 0; g < groups; ++g) {
          for (dim_t t = 0; t < steps; ++t) {
            const dim_t row = (h * groups + g) * steps + t;
            std::copy_n(normalized_queries.data<float>() + t * q_width + (h * groups + g) * cfg.head_dim,
                        cfg.head_dim, packed_queries.data<float>() + row * cfg.head_dim);
            lengths.data<int32_t>()[row] = cache.length + t + 1;
          }
        }
      }
      StorageView context;
      dot_product_attention(packed_queries, packed_keys, packed_values, &lengths,
                            nullptr, nullptr, nullptr, nullptr, 0, 0, 0, context,
                            nullptr, true, 1.f / std::sqrt(static_cast<float>(cfg.head_dim)));
      StorageView attended({steps, q_width});
      for (dim_t h = 0; h < cfg.num_heads; ++h) {
        for (dim_t t = 0; t < steps; ++t) {
          for (dim_t d = 0; d < cfg.head_dim; ++d) {
            const float gate = queries.data<float>()[(t * cfg.num_heads + h) * 2 * cfg.head_dim + cfg.head_dim + d];
            attended.data<float>()[t * q_width + h * cfg.head_dim + d]
              = context.data<float>()[(h * steps + t) * cfg.head_dim + d] / (1.f + std::exp(-gate));
          }
        }
      }
      StorageView result;
      _output(attended, result);
      for (const auto* tensor : {&result, &keys, &values}) {
        for (dim_t i = 0; i < tensor->size(); ++i) {
          if (!std::isfinite(tensor->data<float>()[i]))
            throw std::runtime_error("GatedSelfAttention produced non-finite values; cache was not updated");
        }
      }
      std::copy_n(keys.data<float>(), keys.size(), cache.keys.data<float>() + cache.length * kv_width);
      std::copy_n(values.data<float>(), values.size(), cache.values.data<float>() + cache.length * kv_width);
      cache.length += steps;
      output = std::move(result);
    }

  }
}
