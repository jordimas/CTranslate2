#include <cmath>

#include "drama_test_utils.h"
#include "ctranslate2/layers/drama_backbone.h"

namespace {

  layers::DramaBackboneConfig config(bool hybrid = true) {
    return {{drama_test::block_config(false, 1), drama_test::block_config(hybrid, 1)}, 1e-5f, 8};
  }

  class TinyBackboneModel : public models::Model {
  public:
    explicit TinyBackboneModel(bool hybrid = true) {
      for (dim_t i = 0; i < 2; ++i) {
        const auto path = "backbone/layers_" + std::to_string(i) + "/";
        put(path + "input_layernorm/weight", {2}, {.1f, -.1f});
        put(path + "post_attention_layernorm/weight", {2}, {0, .2f});
        put(path + "mlp/router/weight", {1, 2}, {.1f, -.2f});
        put(path + "mlp/shared_expert_gate/weight", {1, 2}, {0, 0});
        for (const auto* expert : {"experts_0", "shared_expert"}) {
          const auto scope = path + "mlp/" + expert + "/";
          put(scope + "gate_proj/weight", {1, 2}, {.3f, -.2f});
          put(scope + "up_proj/weight", {1, 2}, {-.2f, .1f});
          put(scope + "down_proj/weight", {2, 1}, {.5f, -.1f});
        }
        drama_test::add_mixer([&](const std::string& name, Shape shape, std::vector<float> values) {
          put(path + name, std::move(shape), std::move(values));
        }, i == 1 && hybrid);
      }
      put("backbone/norm/weight", {2}, {.2f, -.1f});
      put("backbone/lm_head/weight", {3, 2}, {1, 0, 0, 1, .5f, -.25f});
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }
    void install_int8_lm_head() {
      StorageView weight({3, 2}, std::vector<int8_t>{4, 0, 0, 4, 2, -1});
      auto compensation = ops::Gemm::compensate_u8_input(weight, true, 2, 3, 1.f);
      remove_variable("backbone/lm_head/weight");
      register_variable("backbone/lm_head/weight", std::move(weight));
      register_variable("backbone/lm_head/weight_scale", StorageView({3}, 4.f));
      register_variable("backbone/lm_head/weight_compensation", std::move(compensation));
      set_compute_type(ComputeType::INT8_FLOAT32, Device::CPU, 0, false);
    }
  protected:
    bool is_packable(const std::string&) const override { return false; }
    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<TinyBackboneModel>(*this);
    }
  private:
    void put(std::string name, Shape shape, std::vector<float> values) {
      register_variable(std::move(name), StorageView(std::move(shape), values));
    }
  };

}

TEST(DramaBackboneTest, HybridCompositionAndIndependentFinalLogits) {
  TinyBackboneModel model;
  const auto cfg = config();
  layers::DramaBackbone backbone(model, "backbone", cfg);
  layers::DramaBackboneState state;
  backbone.initialize_state(state, 3);
  const auto embedded = drama_test::inputs();
  StorageView hidden, logits;
  backbone(embedded, state, hidden, logits);
  EXPECT_EQ(backbone.output_size(), 2);
  EXPECT_EQ(backbone.vocabulary_size(), 3);
  EXPECT_EQ(logits.shape(), (Shape{3, 3}));
  EXPECT_EQ(state.length, 3);
  StorageView composed(embedded);
  for (size_t i = 0; i < 2; ++i) {
    layers::DramaBlock layer(model, "backbone/layers_" + std::to_string(i), cfg.layers[i]);
    layers::DramaBlockState local;
    layer.initialize_state(local, 3);
    StorageView output;
    layer(composed, local, output);
    composed = std::move(output);
  }
  for (dim_t t = 0; t < 3; ++t) {
    const double a = composed.data<float>()[t * 2], b = composed.data<float>()[t * 2 + 1];
    const double rms = std::sqrt((a * a + b * b) / 2 + 1e-5);
    const double x = 1.2 * a / rms, y = .9 * b / rms;
    EXPECT_NEAR(hidden.data<float>()[t * 2], x, 2e-6);
    EXPECT_NEAR(hidden.data<float>()[t * 2 + 1], y, 2e-6);
    EXPECT_NEAR(logits.data<float>()[t * 3], x, 2e-6);
    EXPECT_NEAR(logits.data<float>()[t * 3 + 1], y, 2e-6);
    EXPECT_NEAR(logits.data<float>()[t * 3 + 2], .5 * x - .25 * y, 2e-6);
  }
}

TEST(DramaBackboneTest, HybridChunkParityAndReset) {
  TinyBackboneModel model;
  layers::DramaBackbone backbone(model, "backbone", config());
  layers::DramaBackboneState whole, chunked;
  backbone.initialize_state(whole);
  backbone.initialize_state(chunked);
  auto embedded = drama_test::inputs();
  StorageView full_hidden, full_logits;
  backbone(embedded, whole, full_hidden, full_logits);
  dim_t offset = 0;
  for (const dim_t size : {2, 1}) {
    StorageView chunk({size, 2}, embedded.data<float>() + offset * 2), hidden, logits;
    backbone(chunk, chunked, hidden, logits);
    for (dim_t t = 0; t < size; ++t) {
      for (dim_t d = 0; d < 2; ++d)
        EXPECT_NEAR(hidden.data<float>()[t * 2 + d], full_hidden.data<float>()[(offset + t) * 2 + d], 1e-6);
      for (dim_t v = 0; v < 3; ++v)
        EXPECT_NEAR(logits.data<float>()[t * 3 + v], full_logits.data<float>()[(offset + t) * 3 + v], 1e-6);
    }
    offset += size;
  }
  EXPECT_EQ(chunked.length, whole.length);
  backbone.initialize_state(chunked);
  StorageView reset_hidden, reset_logits;
  backbone(embedded, chunked, reset_hidden, reset_logits);
  expect_storage_eq(reset_hidden, full_hidden);
  expect_storage_eq(reset_logits, full_logits);
}

TEST(DramaBackboneTest, ContextLimitAlsoAppliesToGdnOnlyBackbones) {
  TinyBackboneModel model(false);
  layers::DramaBackbone backbone(model, "backbone", config(false));
  layers::DramaBackboneState state;
  backbone.initialize_state(state, 3);
  auto embedded = drama_test::inputs();
  StorageView hidden, logits;
  backbone(embedded, state, hidden, logits);
  StorageView row({1, 2}, embedded.data<float>());
  EXPECT_THROW(backbone(row, state, hidden, logits), std::invalid_argument);
  EXPECT_EQ(state.length, 3);
  EXPECT_THROW(backbone.initialize_state(state, 9), std::invalid_argument);
}

TEST(DramaBackboneTest, InputStateAndAliasGuards) {
  TinyBackboneModel model;
  layers::DramaBackbone backbone(model, "backbone", config());
  layers::DramaBackboneState state;
  auto embedded = drama_test::inputs();
  StorageView hidden, logits;
  EXPECT_THROW(backbone(embedded, state, hidden, logits), std::invalid_argument);
  backbone.initialize_state(state);
  EXPECT_THROW(backbone(embedded, state, hidden, hidden), std::invalid_argument);
  EXPECT_THROW(backbone(embedded, state, embedded, logits), std::invalid_argument);
  state.layers[1].attention.length = 1;
  EXPECT_THROW(backbone(embedded, state, hidden, logits), std::invalid_argument);
}

TEST(DramaBackboneTest, CandidateHeadMatchesFullProjectionAtRequestedIds) {
  TinyBackboneModel model;
  auto selected_config = config();
  selected_config.lm_candidates = {2, 0}; // Deliberately preserve arbitrary order.
  layers::DramaBackbone full(model, "backbone", config()), selected(model, "backbone", selected_config);
  EXPECT_EQ(selected.vocabulary_size(), full.vocabulary_size());
  layers::DramaBackboneState fs, ss;
  full.initialize_state(fs);
  selected.initialize_state(ss);
  const auto input = drama_test::inputs();
  StorageView fh, fl, sh, sl;
  full(input, fs, fh, fl);
  selected(input, ss, sh, sl);
  expect_storage_eq(fh, sh);
  EXPECT_EQ(sl.shape(), fl.shape());
  for (dim_t t = 0; t < input.dim(0); ++t) {
    EXPECT_NEAR(sl.data<float>()[t * 3], fl.data<float>()[t * 3], 1e-6);
    EXPECT_NEAR(sl.data<float>()[t * 3 + 2], fl.data<float>()[t * 3 + 2], 1e-6);
    EXPECT_EQ(sl.data<float>()[t * 3 + 1], 0.f);
  }
}

TEST(DramaBackboneTest, CandidateHeadGathersInt8ScalesAndCompensation) {
  if (!mayiuse_int8(Device::CPU))
    GTEST_SKIP() << "CPU INT8 GEMM unavailable";
  TinyBackboneModel model;
  model.install_int8_lm_head();
  auto cfg = config();
  cfg.lm_candidates = {2, 0};
  layers::DramaBackbone full(model, "backbone", config()), selected(model, "backbone", cfg);
  layers::DramaBackboneState fs, ss;
  full.initialize_state(fs);
  selected.initialize_state(ss);
  StorageView fh, fl, sh, sl;
  const auto input = drama_test::inputs();
  full(input, fs, fh, fl);
  selected(input, ss, sh, sl);
  for (dim_t t = 0; t < input.dim(0); ++t) {
    EXPECT_NEAR(sl.data<float>()[t * 3], fl.data<float>()[t * 3], 1e-6);
    EXPECT_NEAR(sl.data<float>()[t * 3 + 2], fl.data<float>()[t * 3 + 2], 1e-6);
    EXPECT_EQ(sl.data<float>()[t * 3 + 1], 0.f);
  }
}

TEST(DramaBackboneTest, CandidateHeadValidatesUniqueInRangeIds) {
  TinyBackboneModel model;
  for (const std::vector<int32_t> ids : {std::vector<int32_t>{0, 0}, {-1}, {3}}) {
    auto cfg = config();
    cfg.lm_candidates = ids;
    EXPECT_THROW((layers::DramaBackbone(model, "backbone", cfg)), std::invalid_argument);
  }
}
