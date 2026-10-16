#include <cmath>
#include <limits>
#include <set>

#include "test_utils.h"
#include "ctranslate2/layers/drama_tower.h"

namespace {

  layers::DramaTowerConfig config() {
    layers::DramaTowerConfig cfg;
    cfg.attention = {1, 1, 2, 2, 100.f, 1e-5f, true, true};
    cfg.num_codebooks = 3;
    cfg.vocab_size = 3;
    cfg.shared_layers = 1;
    cfg.vq_backbone_codebooks = 1;
    cfg.semantic_begin = 5;
    cfg.semantic_end = 7;
    cfg.eos_id = 1;
    return cfg;
  }

  class TinyTowerModel : public models::Model {
  public:
    explicit TinyTowerModel(bool quantized_embeddings = false) {
      put("codebook_embeddings/weight", {9, 2}, {0, 0, 1, 0, 0, 1,
                                                .2f, .4f, .5f, -.1f, -.3f, .8f,
                                                .1f, -.5f, -.4f, .2f, .6f, .3f});
      add_layer("shared_layers_0", true);
      for (dim_t i = 0; i < 2; ++i) {
        add_layer("residual_layers_" + std::to_string(i), false);
        put("heads_" + std::to_string(i) + "/weight", {3, 2}, {-1, 0, 0, 0, 1, 0});
      }
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
      if (quantized_embeddings) {
        const auto variables = get_variables();
        for (const auto& pair : variables) {
          if (pair.first.find("embedding/weight") == std::string::npos
              && pair.first.find("embeddings/weight") == std::string::npos)
            continue;
          StorageView quantized(pair.second.shape(), DataType::INT8);
          for (dim_t i = 0; i < pair.second.size(); ++i)
            quantized.data<int8_t>()[i] = static_cast<int8_t>(std::round(10.f * pair.second.data<float>()[i]));
          remove_variable(pair.first);
          register_variable(pair.first, std::move(quantized));
          register_variable(pair.first + "_scale", StorageView({pair.second.dim(0)}, 10.f));
        }
      }
    }
  protected:
    bool is_packable(const std::string&) const override { return false; }
    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<TinyTowerModel>(*this);
    }
  private:
    void put(const std::string& name, Shape shape, std::vector<float> values) {
      register_variable("tower/" + name, StorageView(std::move(shape), values));
    }
    void add_layer(const std::string& name, bool shared) {
      put(name + (shared ? "/semantic_embedding/weight" : "/codebook_embedding/weight"),
          {3, 2}, {-1, .2f, .2f, 1, 1, -.4f});
      for (const auto* norm : {"enorm", "hnorm", "final_layernorm", "transformer_layer/attention_norm", "transformer_layer/ffn_norm"})
        put(name + "/" + norm + "/weight", {2}, {0, 0});
      put(name + "/eh_proj/weight", {2, 4}, {.5f, 0, 1, 0, 0, .5f, 0, 1});
      const auto attn = name + "/transformer_layer/attention/";
      put(attn + "q_proj/weight", {4, 2}, {1, 0, 0, 1, 0, 0, 0, 0});
      put(attn + "k_proj/weight", {2, 2}, {1, 0, 0, 1});
      put(attn + "v_proj/weight", {2, 2}, {.1f, 0, 0, .1f});
      put(attn + "o_proj/weight", {2, 2}, {1, 0, 0, 1});
      put(attn + "q_norm/weight", {2}, {0, 0});
      put(attn + "k_norm/weight", {2}, {0, 0});
      const auto ffn = name + "/transformer_layer/feed_forward/";
      put(ffn + "gate_proj/weight", {1, 2}, {0, 0});
      put(ffn + "up_proj/weight", {1, 2}, {0, 0});
      put(ffn + "down_proj/weight", {2, 1}, {0, 0});
    }
  };

  void compare_caches(const layers::DramaTowerState& a, const layers::DramaTowerState& b) {
    EXPECT_EQ(a.length, b.length);
    for (size_t i = 0; i < a.shared.size(); ++i) {
      expect_storage_eq(a.shared[i].keys, b.shared[i].keys, 1e-6);
      expect_storage_eq(a.shared[i].values, b.shared[i].values, 1e-6);
    }
    for (size_t i = 0; i < a.residual.size(); ++i) {
      expect_storage_eq(a.residual[i].keys, b.residual[i].keys, 1e-6);
      expect_storage_eq(a.residual[i].values, b.residual[i].values, 1e-6);
    }
  }

}

TEST(DramaTowerTest, IndependentBackboneEmbeddingFusion) {
  TinyTowerModel model;
  layers::DramaTower tower(model, "tower", config());
  StorageView frames({2, 4}, std::vector<int32_t>{6, 1, 1, 2, 2, 0, 2, 1});
  StorageView text({2, 2}, std::vector<float>{1, 2, -1, 3}), output;
  tower.embed_backbone(frames, text, output);
  EXPECT_NEAR(output.data<float>()[0], 1.5 / std::sqrt(2.), 1e-6);
  EXPECT_NEAR(output.data<float>()[1], 1.9 / std::sqrt(2.), 1e-6);
  EXPECT_FLOAT_EQ(output.data<float>()[2], -1);
  EXPECT_FLOAT_EQ(output.data<float>()[3], 3);
}

TEST(DramaTowerTest, TeacherPrefillUsesNextCodeAndPreviousUnseenCodes) {
  TinyTowerModel model;
  layers::DramaTower tower(model, "tower", config());
  layers::DramaTowerState state;
  tower.initialize_state(state, 1);
  StorageView hidden({1, 2}, std::vector<float>{1, 2});
  StorageView frames({1, 4}, std::vector<int32_t>{6, 1, 1, 2});
  StorageView next({1, 4}, std::vector<int32_t>{5, 0, 2, 1});
  tower.prefill(hidden, frames, next, state);
  // Shared embedding comes from next code0=0: [-1,.2], while unseen code2=2
  // comes from current frame: [.6,.3]. Independently derive attention's V cache.
  const double ex = -.4, ey = .5;
  const double e_rms = std::sqrt((ex * ex + ey * ey) / 2 + 1e-5);
  const double h_rms = std::sqrt(2.5 + 1e-5);
  const double mx = .5 * ex / e_rms + 1 / h_rms;
  const double my = .5 * ey / e_rms + 2 / h_rms;
  const double m_rms = std::sqrt((mx * mx + my * my) / 2 + 1e-5);
  EXPECT_NEAR(state.shared[0].values.data<float>()[0], .1 * mx / m_rms, 1e-6);
  EXPECT_NEAR(state.shared[0].values.data<float>()[1], .1 * my / m_rms, 1e-6);
  EXPECT_EQ(state.length, 1);
}

TEST(DramaTowerTest, GreedyAndShiftedTeacherPrefillHaveMatchingTemporalCaches) {
  TinyTowerModel model;
  layers::DramaTower tower(model, "tower", config());
  layers::DramaTowerState generated, teacher;
  tower.initialize_state(generated, 3);
  tower.initialize_state(teacher, 3);
  StorageView hidden({3, 2}, std::vector<float>{1, 2, -.5f, 1, 2, -.25f});
  StorageView logits({1, 9}, std::vector<float>{0, -2, 0, 0, 0, 0, 3, 0, 100});
  StorageView previous({1, 4}, std::vector<int32_t>{6, 1, 1, 2});
  StorageView frames({3, 4}, DataType::INT32), next_frames({3, 4}, DataType::INT32);
  for (dim_t t = 0; t < 3; ++t) {
    for (dim_t c = 0; c < 4; ++c)
      frames.data<int32_t>()[t * 4 + c] = previous.data<int32_t>()[c];
    StorageView row({1, 2}, hidden.data<float>() + t * 2), next(DataType::INT32);
    tower.decode_greedy(logits, row, previous, generated, next);
    EXPECT_EQ(next.data<int32_t>()[0], 6); // Higher out-of-band token8 is ignored.
    EXPECT_EQ(next.data<int32_t>()[1], 1);
    for (dim_t c = 0; c < 4; ++c)
      next_frames.data<int32_t>()[t * 4 + c] = next.data<int32_t>()[c];
    previous = std::move(next);
  }
  tower.prefill(hidden, frames, next_frames, teacher);
  compare_caches(generated, teacher);
  tower.initialize_state(teacher, 3);
  tower.prefill(hidden, frames, next_frames, teacher);
  compare_caches(generated, teacher);
}

TEST(DramaTowerTest, EosAndInputStateGuards) {
  TinyTowerModel model;
  layers::DramaTower tower(model, "tower", config());
  layers::DramaTowerState state;
  tower.initialize_state(state, 1);
  StorageView hidden({1, 2}, 1.f), logits({1, 9}, 0.f);
  logits.data<float>()[1] = 10;
  StorageView previous({1, 4}, int32_t(0)), output(DataType::INT32);
  tower.decode_greedy(logits, hidden, previous, state, output);
  EXPECT_EQ(output.data<int32_t>()[0], 1);
  EXPECT_EQ(output.data<int32_t>()[1], 0);
  EXPECT_EQ(state.length, 1);
  EXPECT_THROW(tower.decode_greedy(logits, hidden, previous, state, output), std::invalid_argument);
  tower.initialize_state(state, 1);
  previous.data<int32_t>()[2] = 3;
  EXPECT_THROW(tower.decode_greedy(logits, hidden, previous, state, output), std::invalid_argument);
  previous.data<int32_t>()[2] = 0;
  state.residual[0].length = 1;
  EXPECT_THROW(tower.decode_greedy(logits, hidden, previous, state, output), std::invalid_argument);
}

TEST(DramaTowerTest, Int8EmbeddingTablesUsePerRowScales) {
  TinyTowerModel floating, quantized(true);
  ASSERT_EQ(quantized.get_variable("tower/codebook_embeddings/weight").dtype(), DataType::INT8);
  ASSERT_EQ(quantized.get_variable("tower/shared_layers_0/semantic_embedding/weight").dtype(), DataType::INT8);
  ASSERT_EQ(quantized.get_variable("tower/residual_layers_0/codebook_embedding/weight").dtype(), DataType::INT8);
  layers::DramaTower a(floating, "tower", config()), b(quantized, "tower", config());
  StorageView current({1, 4}, std::vector<int32_t>{6, 1, 1, 2});
  StorageView next({1, 4}, std::vector<int32_t>{5, 0, 2, 1});
  StorageView hidden({1, 2}, std::vector<float>{1, 2}), af, bf;
  a.embed_backbone(current, hidden, af);
  b.embed_backbone(current, hidden, bf);
  expect_storage_eq(af, bf, 1e-6);
  layers::DramaTowerState ac, bc;
  a.initialize_state(ac, 1);
  b.initialize_state(bc, 1);
  a.prefill(hidden, current, next, ac);
  b.prefill(hidden, current, next, bc);
  compare_caches(ac, bc);
}

TEST(DramaTowerTest, SamplingDropsCrossingTokenAndKeepsHighest) {
  const std::vector<float> logits{std::log(.6f), std::log(.3f), std::log(.1f)};
  layers::DramaSamplingOptions options;
  options.temperature = 1;
  options.top_k = 3;
  options.top_p = .7f;
  std::mt19937_64 rng(42);
  // The .3 token crosses .7 and is dropped, rather than kept as in some other
  // nucleus samplers. Only token0 remains.
  for (int i = 0; i < 100; ++i)
    EXPECT_EQ(layers::sample_drama_token(logits, options, rng), 0);
  options.top_p = 0;
  EXPECT_EQ(layers::sample_drama_token(logits, options, rng), 0);
  options.top_p = 1;
  options.top_k = 1;
  EXPECT_EQ(layers::sample_drama_token({2, 2, 2}, options, rng), 0);
}

TEST(DramaTowerTest, SamplingDistributionAndSeedReproducibility) {
  layers::DramaSamplingOptions options;
  options.temperature = 1;
  options.top_k = 3;
  options.top_p = .95f;
  const std::vector<float> logits{std::log(.6f), std::log(.3f), std::log(.1f)};
  std::mt19937_64 a(1234), b(1234);
  int count0 = 0;
  for (int i = 0; i < 4000; ++i) {
    const auto token = layers::sample_drama_token(logits, options, a);
    EXPECT_EQ(token, layers::sample_drama_token(logits, options, b));
    EXPECT_LT(token, 2);
    count0 += token == 0;
  }
  EXPECT_NEAR(count0 / 4000., 2. / 3, .03);
}

TEST(DramaTowerTest, SampledEosIsNeverResampledAndCodeIdsStayInRange) {
  TinyTowerModel model;
  layers::DramaTower tower(model, "tower", config());
  layers::DramaSamplingOptions options;
  options.top_k = 1;
  options.top_p = 1;
  options.temperature = 1;
  StorageView hidden({1, 2}, 1.f), logits({1, 9}, 0.f);
  logits.data<float>()[1] = .1f;
  logits.data<float>()[8] = 100; // Must remain outside main-token candidates.
  StorageView previous({1, 4}, int32_t(0));
  for (uint64_t seed = 0; seed < 16; ++seed) {
    layers::DramaTowerState state;
    tower.initialize_state(state, 1);
    StorageView next(DataType::INT32);
    std::mt19937_64 rng(seed);
    tower.decode_sample(logits, hidden, previous, state, {1, 1, 1}, options, rng, next);
    EXPECT_EQ(next.data<int32_t>()[0], 1);
    for (dim_t c = 1; c < 4; ++c) {
      EXPECT_GE(next.data<int32_t>()[c], 0);
      EXPECT_LT(next.data<int32_t>()[c], 3);
    }
  }
}

TEST(DramaTowerTest, RepetitionResamplesSemanticOnceWithinLastSixteenTokens) {
  TinyTowerModel model;
  layers::DramaTower tower(model, "tower", config());
  layers::DramaSamplingOptions options;
  options.temperature = 1;
  options.top_k = 1;
  options.top_p = 1;
  layers::DramaSamplingOptions retry;
  retry.temperature = 1;
  retry.top_k = 30;
  retry.top_p = .95f;
  StorageView hidden({1, 2}, 1.f), logits({1, 9}, 0.f);
  logits.data<float>()[6] = .1f;
  logits.data<float>()[8] = 100;
  StorageView previous({1, 4}, int32_t(0));
  const std::vector<int32_t> candidate_ids{1, 5, 6, 7};
  const std::vector<float> candidate_logits{0, 0, .1f, 0};
  for (uint64_t seed = 0; seed < 16; ++seed) {
    std::mt19937_64 reference_rng(seed);
    EXPECT_EQ(candidate_ids[layers::sample_drama_token(candidate_logits, options, reference_rng)], 6);
    const auto expected_retry = candidate_ids[layers::sample_drama_token(candidate_logits, retry, reference_rng)];
    layers::DramaTowerState state;
    tower.initialize_state(state, 1);
    StorageView next(DataType::INT32);
    std::mt19937_64 rng(seed);
    tower.decode_sample(logits, hidden, previous, state, {6}, options, rng, next);
    EXPECT_EQ(next.data<int32_t>()[0], expected_retry);
    tower.initialize_state(state, 1);
    std::vector<int32_t> old_history(17, 7);
    old_history[0] = 6; // Outside the 16-token repetition window.
    rng.seed(seed);
    tower.decode_sample(logits, hidden, previous, state, old_history, options, rng, next);
    EXPECT_EQ(next.data<int32_t>()[0], 6);
  }
}

TEST(DramaTowerTest, ResidualCodesAreSampledAndRequestSeedsReproduceFrames) {
  TinyTowerModel model;
  layers::DramaTower tower(model, "tower", config());
  layers::DramaSamplingOptions options;
  options.temperature = 1;
  options.top_k = 3;
  options.top_p = 1;
  StorageView hidden({1, 2}, 1.f), logits({1, 9}, -50.f);
  logits.data<float>()[1] = 10;
  StorageView previous({1, 4}, int32_t(0));
  std::set<int32_t> residual_codes;
  for (uint64_t seed = 0; seed < 24; ++seed) {
    layers::DramaTowerState a, b;
    tower.initialize_state(a, 1);
    tower.initialize_state(b, 1);
    StorageView af(DataType::INT32), bf(DataType::INT32);
    std::mt19937_64 ar(seed), br(seed);
    tower.decode_sample(logits, hidden, previous, a, {}, options, ar, af);
    tower.decode_sample(logits, hidden, previous, b, {}, options, br, bf);
    expect_storage_eq(af, bf);
    residual_codes.insert(af.data<int32_t>()[2]);
    for (dim_t c = 1; c < 4; ++c) {
      EXPECT_GE(af.data<int32_t>()[c], 0);
      EXPECT_LT(af.data<int32_t>()[c], 3);
    }
  }
  EXPECT_GT(residual_codes.size(), 1);
}

TEST(DramaTowerTest, SamplingRejectsInvalidOptionsAndNonfiniteLogits) {
  std::mt19937_64 rng(0);
  layers::DramaSamplingOptions options;
  options.temperature = 0;
  EXPECT_THROW(layers::sample_drama_token({0, 1}, options, rng), std::invalid_argument);
  options.temperature = 1;
  options.top_p = -1;
  EXPECT_THROW(layers::sample_drama_token({0, 1}, options, rng), std::invalid_argument);
  options.top_p = 1;
  options.top_k = 0;
  EXPECT_THROW(layers::sample_drama_token({0, 1}, options, rng), std::invalid_argument);
  options.top_k = 1;
  EXPECT_THROW(layers::sample_drama_token({}, options, rng), std::invalid_argument);
  EXPECT_THROW(layers::sample_drama_token({std::numeric_limits<float>::infinity()}, options, rng), std::invalid_argument);
}
