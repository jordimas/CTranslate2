#include <algorithm>
#include <cmath>
#include <limits>

#include "test_utils.h"
#include "ctranslate2/layers/drama_attention.h"

namespace {

  class TinyAttentionModel : public models::Model {
  public:
    explicit TinyAttentionModel(float gamma = 0.f, float gate_scale = 0.f) {
      StorageView q({16, 4}, 0.f), k({4, 4}, 0.f), v({4, 4}, 0.f), o({4, 8}, 0.f);
      for (dim_t d = 0; d < 4; ++d) {
        q.data<float>()[d * 4 + d] = 1;
        q.data<float>()[(8 + d) * 4 + d] = -1;
        q.data<float>()[(4 + d) * 4 + d] = gate_scale;
        q.data<float>()[(12 + d) * 4 + d] = gate_scale * 2;
        k.data<float>()[d * 4 + d] = 1;
        v.data<float>()[d * 4 + d] = 1;
        o.data<float>()[d * 8 + d] = .25f;
        o.data<float>()[d * 8 + 4 + d] = .75f;
      }
      register_variable("attn/q_proj/weight", std::move(q));
      register_variable("attn/k_proj/weight", std::move(k));
      register_variable("attn/v_proj/weight", std::move(v));
      register_variable("attn/o_proj/weight", std::move(o));
      register_variable("attn/q_norm/weight", StorageView({4}, gamma));
      register_variable("attn/k_norm/weight", StorageView({4}, gamma));
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0);
    }
  protected:
    bool is_packable(const std::string&) const override { return false; }
    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<TinyAttentionModel>(*this);
    }
  };

  layers::DramaAttentionConfig config(bool interleaved = false, dim_t rotary = 4) {
    return {2, 1, 4, rotary, 100.f, 1e-5f, interleaved, true};
  }

  layers::DramaAttentionCache cache(dim_t capacity) {
    return {StorageView({capacity, 1, 4}, DataType::FLOAT32),
            StorageView({capacity, 1, 4}, DataType::FLOAT32), 0};
  }

  StorageView inputs() {
    return StorageView({3, 4}, std::vector<float>{1, 2, -1, .5f, -.5f, 1, 2, -1, 2, -.25f, .5f, 1});
  }

  // Independent double-precision causal attention over dense tiny arrays.
  std::vector<double> reference(const StorageView& x, bool interleaved, dim_t rotary) {
    std::vector<std::vector<double>> vectors(3, std::vector<double>(4));
    for (dim_t t = 0; t < 3; ++t) {
      double square = 0;
      for (dim_t d = 0; d < 4; ++d)
        square += std::pow(double(x.data<float>()[t * 4 + d]), 2);
      for (dim_t d = 0; d < 4; ++d)
        vectors[t][d] = x.data<float>()[t * 4 + d] / std::sqrt(square / 4 + 1e-5);
      const auto normalized = vectors[t];
      for (dim_t p = 0; p < rotary / 2; ++p) {
        const dim_t a = interleaved ? 2 * p : p;
        const dim_t b = interleaved ? 2 * p + 1 : p + rotary / 2;
        const double angle = t / std::pow(100., 2. * p / rotary);
        vectors[t][a] = normalized[a] * std::cos(angle) - normalized[b] * std::sin(angle);
        vectors[t][b] = normalized[a] * std::sin(angle) + normalized[b] * std::cos(angle);
      }
    }
    std::vector<double> output(12, 0);
    for (dim_t t = 0; t < 3; ++t) {
      for (dim_t h = 0; h < 2; ++h) {
        std::vector<double> scores(t + 1);
        double sum = 0;
        for (dim_t s = 0; s <= t; ++s) {
          double dot = 0;
          for (dim_t d = 0; d < 4; ++d)
            dot += vectors[t][d] * vectors[s][d];
          scores[s] = std::exp((h ? -1 : 1) * dot / 2);
          sum += scores[s];
        }
        for (dim_t d = 0; d < 4; ++d)
          for (dim_t s = 0; s <= t; ++s)
            output[t * 4 + d] += .5 * (h ? .75 : .25) * (scores[s] / sum) * x.data<float>()[s * 4 + d];
      }
    }
    return output;
  }

}

TEST(DramaAttentionTest, IndependentGroupedAttentionAndBothRopeLayouts) {
  TinyAttentionModel model;
  const auto x = inputs();
  for (const bool interleaved : {false, true}) {
    for (const dim_t rotary : {0, 2, 4}) {
      layers::DramaAttention layer(model, "attn", config(interleaved, rotary));
      auto state = cache(3);
      StorageView output;
      layer(x, state, output);
      const auto expected = reference(x, interleaved, rotary);
      for (dim_t i = 0; i < output.size(); ++i)
        EXPECT_NEAR(output.data<float>()[i], expected[i], 2e-6);
      EXPECT_EQ(state.length, 3);
    }
  }
}

TEST(DramaAttentionTest, CachedStepsMatchPrefill) {
  TinyAttentionModel model;
  layers::DramaAttention layer(model, "attn", config());
  const auto x = inputs();
  auto full_cache = cache(3), step_cache = cache(3);
  StorageView full;
  layer(x, full_cache, full);
  for (dim_t t = 0; t < 3; ++t) {
    StorageView row({1, 4}, const_cast<float*>(x.data<float>()) + 4 * t), output;
    layer(row, step_cache, output);
    for (dim_t d = 0; d < 4; ++d)
      EXPECT_FLOAT_EQ(output.data<float>()[d], full.data<float>()[t * 4 + d]);
  }
  expect_storage_eq(step_cache.keys, full_cache.keys);
  expect_storage_eq(step_cache.values, full_cache.values);
}

TEST(DramaAttentionTest, PerHeadQueryGatePacking) {
  TinyAttentionModel model(0.f, 1.f);
  layers::DramaAttention layer(model, "attn", config());
  auto state = cache(1);
  StorageView input({1, 4}, std::vector<float>{1, -1, .5f, -2}), output;
  layer(input, state, output);
  for (dim_t d = 0; d < 4; ++d) {
    const double x = input.data<float>()[d];
    const double expected = x * (.25 / (1 + std::exp(-x)) + .75 / (1 + std::exp(-2 * x)));
    EXPECT_NEAR(output.data<float>()[d], expected, 1e-6);
  }
}

TEST(DramaAttentionTest, DirectAndZeroCenteredGammaAgree) {
  TinyAttentionModel zero_centered(0), direct(1);
  auto cfg = config();
  layers::DramaAttention a(zero_centered, "attn", cfg);
  cfg.zero_centered_qk_norm = false;
  layers::DramaAttention b(direct, "attn", cfg);
  auto ac = cache(3), bc = cache(3);
  StorageView ay, by;
  const auto x = inputs();
  a(x, ac, ay);
  b(x, bc, by);
  expect_storage_eq(ay, by);
}

TEST(DramaAttentionTest, CapacityAndShapeGuardsDoNotUpdateCache) {
  TinyAttentionModel model;
  layers::DramaAttention layer(model, "attn", config());
  auto state = cache(2);
  const auto x = inputs();
  StorageView y;
  EXPECT_THROW(layer(x, state, y), std::invalid_argument);
  EXPECT_EQ(state.length, 0);
  StorageView wrong({1, 3}, 1.f);
  EXPECT_THROW(layer(wrong, state, y), std::invalid_argument);
  state.length = -1;
  EXPECT_THROW(layer(x, state, y), std::invalid_argument);
  auto cfg = config();
  cfg.rotary_dim = 3;
  EXPECT_THROW((layers::DramaAttention(model, "attn", cfg)), std::invalid_argument);
}

TEST(DramaAttentionTest, EmptyAndAliasingAndNonfiniteGuards) {
  TinyAttentionModel model;
  layers::DramaAttention layer(model, "attn", config());
  auto state = cache(3);
  StorageView empty({0, 4}, DataType::FLOAT32), output;
  layer(empty, state, output);
  EXPECT_EQ(output.shape(), empty.shape());
  EXPECT_EQ(state.length, 0);
  auto x = inputs();
  EXPECT_THROW(layer(x, state, x), std::invalid_argument);
  StorageView alias({3, 4}, state.keys.data<float>());
  EXPECT_THROW(layer(alias, state, output), std::invalid_argument);
  x.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(layer(x, state, output), std::invalid_argument);
  EXPECT_EQ(state.length, 0);
}
