#include "ctranslate2/models/drama3.h"
#include <regex>
#include <stdexcept>
#include "ctranslate2/layers/drama_backbone.h"
#include "ctranslate2/layers/drama_tower.h"
#include "ctranslate2/layers/fish_dac_decoder.h"
#include "ctranslate2/utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
namespace ctranslate2 { namespace models {
  namespace {
    std::string quantization_policy(const nlohmann::json& config) {
      const auto policy = config.value("quantization_policy", std::string("all_matrices_int8"));
      if (policy != "all_matrices_int8" && policy != "experts_int8_sensitive_float32")
        throw std::invalid_argument("Unknown Drama3 quantization policy: " + policy);
      return policy;
    }
  }
  bool Drama3Model::is_quantizable(const std::string& name) const {
    // Model::set_compute_type runs before initialize. Enforce this policy here
    // so loading INT8 experts cannot silently re-quantize preserved F32 matrices.
    const auto policy = quantization_policy(config);
    if (name.compare(0,6,"codec/") == 0 || name.size() < 7
        || name.compare(name.size()-7,7,"/weight") != 0) return false;
    const auto* variable = get_variable_if_exists(name);
    if (!variable || variable->rank() != 2) return false;
    if (policy == "experts_int8_sensitive_float32") {
      static const std::regex routed("backbone/layers_[0-9]+/mlp/experts_[0-9]+/(gate_proj|up_proj|down_proj)/weight");
      return std::regex_match(name, routed);
    }
    return true;
  }
  bool Drama3Model::is_linear_weight(const std::string& name) const {
    return is_quantizable(name) && name.find("embedding") == std::string::npos;
  }
  void Drama3Model::initialize(ModelReader&) {
    quantization_policy(config);
    if (device() != Device::CPU)
      throw std::invalid_argument("Drama3 currently supports CPU execution only");
    if (effective_compute_type() != ComputeType::FLOAT32
        && effective_compute_type() != ComputeType::INT8_FLOAT32)
      throw std::invalid_argument("Drama3 requires float32 or int8_float32 compute");
    if (!config.is_object() || config.value("model_type",std::string()) != "drama3"
        || config.value("ct2_layout_version",0) != 1
        || !config.contains("source_config") || !config["source_config"].is_object())
      throw std::invalid_argument("Drama3 requires verified layout-v1 source configuration");
  }
  std::unique_ptr<Model> Drama3Model::clone() const {
    return std::make_unique<Drama3Model>(*this);
  }

  namespace {
    layers::DramaAttentionConfig attention_config(const nlohmann::json& cfg, bool tower) {
      layers::DramaAttentionConfig result;
      result.num_heads = cfg.at(tower ? "n_head" : "num_attention_heads").get<dim_t>();
      result.num_kv_heads = cfg.at(tower ? "n_local_heads" : "num_key_value_heads").get<dim_t>();
      result.head_dim = cfg.at("head_dim").get<dim_t>();
      const auto& rope = tower ? cfg : cfg.at("rope_parameters");
      const float fraction = rope.at(tower ? "rotary_percent" : "partial_rotary_factor").get<float>();
      result.rotary_dim = static_cast<dim_t>(result.head_dim * fraction);
      result.rope_base = rope.at(tower ? "rope_base" : "rope_theta").get<float>();
      result.norm_epsilon = cfg.at(tower ? "norm_eps" : "rms_norm_eps").get<float>();
      result.rotary_interleaved = cfg.at("rotary_interleaved").get<bool>();
      result.zero_centered_qk_norm = true;
      if (!cfg.value("layernorm_zero_centered_gamma", false)
          || !cfg.value(tower ? "attention_output_gate" : "attn_output_gate", false))
        throw std::invalid_argument("Unsupported Drama3 attention normalization/gate configuration");
      return result;
    }

    template<typename T>
    StorageView rows(const StorageView& source, dim_t first, dim_t count) {
      const dim_t columns = source.dim(1);
      return StorageView({count, columns}, std::vector<T>(source.data<T>() + first*columns,
                                                        source.data<T>() + (first+count)*columns));
    }
  }

  struct Drama3::Impl {
    std::shared_ptr<const Model> model;
    std::unique_ptr<layers::Embeddings> embeddings;
    std::unique_ptr<layers::DramaBackbone> backbone;
    std::unique_ptr<layers::DramaTower> tower;
    std::unique_ptr<layers::FishDACDecoder> codec;
    size_t threads;
    dim_t max_context;
    dim_t vocabulary;
    int32_t eos;
    std::mutex mutex;

    Impl(const std::string& path, size_t count, const std::string& compute)
      : threads(count) {
      if (!count || (compute != "int8_float32" && compute != "float32"))
        throw std::invalid_argument("Drama3 requires positive intra_threads and int8_float32 or float32");
      set_num_threads(threads);
      model = Model::load(path, Device::CPU, 0, str_to_compute_type(compute));
      if (!dynamic_cast<const Drama3Model*>(model.get()))
        throw std::invalid_argument("Expected a converted Drama3Spec model");
      const auto& source = model->config.at("source_config");
      const auto& text = source.at("text_config");
      const auto& audio = source.at("audio_decoder_config");
      layers::DramaBackboneConfig bc;
      bc.norm_epsilon = text.at("rms_norm_eps").get<float>();
      bc.max_context = max_context = text.at("max_position_embeddings").get<dim_t>();
      vocabulary = text.at("vocab_size").get<dim_t>();
      if (!text.value("norm_topk_prob", true))
        throw std::invalid_argument("Drama3 requires normalized top-k routing");
      for (const auto& type : text.at("layer_types")) {
        layers::DramaBlockConfig block;
        if (type == "linear_attention") block.kind = layers::DramaMixerKind::LinearAttention;
        else if (type == "full_attention") block.kind = layers::DramaMixerKind::FullAttention;
        else throw std::invalid_argument("Unsupported Drama3 mixer type");
        block.norm_epsilon = bc.norm_epsilon;
        block.num_experts = text.at("num_experts").get<dim_t>();
        block.top_k = text.at("num_experts_per_tok").get<dim_t>();
        block.key_heads = text.at("linear_num_key_heads").get<dim_t>();
        block.value_heads = text.at("linear_num_value_heads").get<dim_t>();
        block.key_dim = text.at("linear_key_head_dim").get<dim_t>();
        block.value_dim = text.at("linear_value_head_dim").get<dim_t>();
        block.attention = attention_config(text, false);
        bc.layers.push_back(block);
      }
      if (bc.layers.size() != text.at("num_hidden_layers").get<size_t>())
        throw std::invalid_argument("Drama3 layer count mismatch");
      layers::DramaTowerConfig tc;
      tc.attention = attention_config(audio, true);
      tc.num_codebooks = audio.at("num_codebooks").get<dim_t>();
      tc.vocab_size = audio.at("vocab_size").get<dim_t>();
      tc.shared_layers = audio.at("shared_enrichment_layers").get<dim_t>();
      tc.vq_backbone_codebooks = audio.at("vq_backbone_codebooks").get<dim_t>();
      tc.semantic_begin = source.at("semantic_start_token_id").get<int32_t>();
      tc.semantic_end = source.at("semantic_end_token_id").get<int32_t>();
      tc.eos_id = eos = source.at("eos_token_id").get<int32_t>();
      if (tc.num_codebooks != 10 || tc.semantic_end-tc.semantic_begin+1 != 4096)
        throw std::invalid_argument("Drama3 codec requires ten codebooks and 4096 semantic codes");
      bc.lm_candidates.push_back(tc.eos_id);
      for (int32_t id = tc.semantic_begin; id <= tc.semantic_end; ++id)
        bc.lm_candidates.push_back(id);
      embeddings = std::make_unique<layers::Embeddings>(*model,"backbone/embeddings");
      backbone = std::make_unique<layers::DramaBackbone>(*model,"backbone",std::move(bc));
      tower = std::make_unique<layers::DramaTower>(*model,"tower",tc);
      codec = std::make_unique<layers::FishDACDecoder>(*model);
    }

    StorageView embed(const StorageView& frames) const {
      std::vector<int32_t> ids(frames.dim(0));
      for (dim_t i = 0; i < frames.dim(0); ++i) ids[i] = frames.data<int32_t>()[i*11];
      StorageView text, result;
      (*embeddings)(StorageView({frames.dim(0)}, ids), text);
      tower->embed_backbone(frames, text, result);
      return result;
    }
  };

  Drama3::Drama3(const std::string& path, size_t threads, const std::string& compute)
    : _impl(std::make_unique<Impl>(path,threads,compute)) {}
  Drama3::~Drama3() = default;

  Drama3Result Drama3::generate(const StorageView& prompt, dim_t max_frames,
                               dim_t prefill_chunk_size, float temperature,
                               dim_t top_k, float top_p, uint64_t seed, bool greedy) {
    std::lock_guard<std::mutex> lock(_impl->mutex);
    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();
    const auto seconds_since = [](const Clock::time_point& start) {
      return std::chrono::duration<double>(Clock::now() - start).count();
    };
    Drama3Result result;
    set_num_threads(_impl->threads);
    if (prompt.device() != Device::CPU || prompt.dtype() != DataType::INT32
        || prompt.rank() != 2 || prompt.dim(1) != 11 || prompt.dim(0) <= 0
        || prompt.dim(0) > 8192 || max_frames <= 0 || max_frames > 8192
        || prefill_chunk_size <= 0 || max_frames > _impl->max_context-prompt.dim(0)
        || !std::isfinite(temperature) || temperature <= 0 || top_k <= 0
        || !std::isfinite(top_p) || top_p <= 0 || top_p > 1)
      throw std::invalid_argument("Invalid Drama3 prompt, frame budget or prefill chunk size");
    for (dim_t t = 0; t < prompt.dim(0); ++t)
      if (prompt.data<int32_t>()[t*11] < 0 || prompt.data<int32_t>()[t*11] >= _impl->vocabulary)
        throw std::invalid_argument("Drama3 prompt token outside vocabulary");
    layers::DramaBackboneState backbone_state;
    layers::DramaTowerState tower_state;
    const dim_t capacity = prompt.dim(0)+max_frames;
    _impl->backbone->initialize_state(backbone_state,capacity);
    _impl->tower->initialize_state(tower_state,capacity);
    StorageView hidden, logits;
    const auto prefill_start = Clock::now();
    for (dim_t start = 0; start < prompt.dim(0); start += prefill_chunk_size) {
      const dim_t count = std::min(prefill_chunk_size,prompt.dim(0)-start);
      auto frames = rows<int32_t>(prompt,start,count);
      auto embedded = _impl->embed(frames);
      (*_impl->backbone)(embedded,backbone_state,hidden,logits);
      const dim_t paired = std::min(count,prompt.dim(0)-1-start);
      if (paired > 0)
        _impl->tower->prefill(rows<float>(hidden,0,paired),rows<int32_t>(prompt,start,paired),
                              rows<int32_t>(prompt,start+1,paired),tower_state);
    }
    hidden = rows<float>(hidden,hidden.dim(0)-1,1);
    logits = rows<float>(logits,logits.dim(0)-1,1);
    result.timings.prefill_seconds = seconds_since(prefill_start);
    auto previous = rows<int32_t>(prompt,prompt.dim(0)-1,1);
    std::vector<int32_t> codes;
    std::vector<int32_t> history;
    for (dim_t t = 0; t < prompt.dim(0); ++t) history.push_back(prompt.data<int32_t>()[t*11]);
    layers::DramaSamplingOptions sampling;
    sampling.temperature = temperature;
    sampling.top_k = top_k;
    sampling.top_p = top_p;
    sampling.seed = seed;
    std::mt19937_64 rng(seed);
    result.finish_reason = "length";
    for (dim_t step = 0; step < max_frames; ++step) {
      StorageView next(DataType::INT32);
      const auto tower_start = Clock::now();
      if (greedy) _impl->tower->decode_greedy(logits,hidden,previous,tower_state,next);
      else _impl->tower->decode_sample(logits,hidden,previous,tower_state,history,sampling,rng,next);
      result.timings.tower_decode_seconds += seconds_since(tower_start);
      if (next.data<int32_t>()[0] == _impl->eos) {
        result.finish_reason = "eos";
        break;
      }
      codes.insert(codes.end(),next.data<int32_t>()+1,next.data<int32_t>()+11);
      history.push_back(next.data<int32_t>()[0]);
      if (step+1 < max_frames) {
        const auto backbone_start = Clock::now();
        auto embedded = _impl->embed(next);
        (*_impl->backbone)(embedded,backbone_state,hidden,logits);
        result.timings.backbone_decode_seconds += seconds_since(backbone_start);
      }
      previous = std::move(next);
    }
    const dim_t frames = codes.size()/10;
    result.codes = StorageView({frames,10},codes);
    if (frames > 0) {
      const auto codec_start = Clock::now();
      StorageView codec_codes({1,10,frames},DataType::INT32);
      for (dim_t t = 0; t < frames; ++t)
        // Official DAC clamps residual lookup indices to its 1024-entry books.
        // Preserve raw tower codes for feedback and in the returned diagnostic.
        for (dim_t b = 0; b < 10; ++b)
          codec_codes.data<int32_t>()[b*frames+t] = std::min(codes[t*10+b],b == 0 ? 4095 : 1023);
      (*_impl->codec)(codec_codes,result.audio);
      result.audio.reshape({result.audio.size()});
      result.timings.codec_seconds = seconds_since(codec_start);
    } else result.audio = StorageView({0},DataType::FLOAT32);
    result.timings.total_native_seconds = seconds_since(total_start);
    return result;
  }
}}
