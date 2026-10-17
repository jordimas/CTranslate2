#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>

#include "test_utils.h"
#include "ctranslate2/layers/attention.h"
#include "ctranslate2/models/model_factory.h"

namespace {
  class AttentionParityFixture : public models::Model {
  protected:
    bool is_quantizable(const std::string&) const override { return false; }
    bool is_packable(const std::string&) const override { return false; }
    std::unique_ptr<models::Model> clone() const override {
      return std::make_unique<AttentionParityFixture>(*this);
    }
  };

  StorageView read(const std::string& path, Shape shape) {
    StorageView result(std::move(shape), DataType::FLOAT32);
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
      throw std::runtime_error("cannot read attention oracle fixture " + path);
    const auto bytes = result.size() * sizeof(float);
    stream.read(reinterpret_cast<char*>(result.data<float>()), bytes);
    if (stream.gcount() != std::streamsize(bytes) || stream.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("attention oracle fixture has incorrect byte count");
    return result;
  }

  void compare(const StorageView& actual, const StorageView& expected,
               const std::string& name, float maximum_limit, double rms_limit) {
    ASSERT_EQ(actual.shape(), expected.shape());
    float maximum = 0;
    double squared = 0;
    for (dim_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual.data<float>()[i]));
      const float error = std::abs(actual.data<float>()[i] - expected.data<float>()[i]);
      maximum = std::max(maximum, error);
      squared += double(error) * error;
    }
    const double rms = std::sqrt(squared / std::max(dim_t(1), actual.size()));
    ::testing::Test::RecordProperty(name + "_max_absolute_error", std::to_string(maximum));
    ::testing::Test::RecordProperty(name + "_rmse", std::to_string(rms));
    EXPECT_LT(maximum, maximum_limit) << name;
    EXPECT_LT(rms, rms_limit) << name;
  }
}

TEST(Drama3AttentionParityTest, ActualBackboneAndFusedTowerWeightsMatchIndependentOracle) {
  const char* path = std::getenv("CT2_DRAMA3_ATTENTION_FIXTURE");
  if (!path)
    GTEST_SKIP() << "Set CT2_DRAMA3_ATTENTION_FIXTURE to an independent attention fixture";
  models::register_model<AttentionParityFixture>("Drama3AttentionParityFixture");
  for (const auto* kind : {"backbone", "tower"}) {
    const std::string directory = std::string(path) + "/" + kind;
    std::ifstream stream(directory + "/manifest.json");
    ASSERT_TRUE(stream.good());
    const auto manifest = nlohmann::json::parse(stream);
    const auto model = models::Model::load(directory, Device::CPU, 0, ComputeType::FLOAT32);
    const dim_t hidden = manifest.at("hidden"), kv_heads = manifest.at("kv_heads"), dim = manifest.at("head_dim");
    layers::GatedSelfAttentionConfig cfg{manifest.at("heads"), kv_heads, dim, manifest.at("rotary_dim"),
                                     manifest.at("rope_base"), manifest.at("norm_epsilon"), manifest.at("interleaved"), true};
    layers::GatedSelfAttention layer(*model, "attn", cfg);
    for (const auto& fixture : manifest.at("cases")) {
      const dim_t steps = fixture.at("steps");
      const std::string label = std::string(kind) + "_t" + std::to_string(steps);
      SCOPED_TRACE(label);
      auto get = [&](const char* name, Shape shape) {
        return read(directory + "/" + fixture.at(name).get<std::string>(), std::move(shape));
      };
      auto input = get("input", {steps, hidden});
      auto make_cache = [&]() {
        return layers::GatedSelfAttentionCache{StorageView({steps, kv_heads, dim}, DataType::FLOAT32),
                                            StorageView({steps, kv_heads, dim}, DataType::FLOAT32), 0};
      };
      auto cache = make_cache(), incremental = make_cache();
      StorageView output;
      layer(input, cache, output);
      compare(output, get("expected", {steps, hidden}), label + "_output", 2e-4f, 2e-5);
      compare(cache.keys, get("keys", {steps, kv_heads, dim}), label + "_keys", 5e-5f, 5e-6);
      compare(cache.values, get("values", {steps, kv_heads, dim}), label + "_values", 5e-5f, 5e-6);
      for (dim_t t = 0; t < steps; ++t) {
        StorageView row({1, hidden}, input.data<float>() + t * hidden), one;
        layer(row, incremental, one);
        StorageView full_row({1, hidden}, output.data<float>() + t * hidden);
        compare(one, full_row, label + "_step" + std::to_string(t), 2e-4f, 2e-5);
      }
      compare(incremental.keys, cache.keys, label + "_step_keys", 5e-5f, 5e-6);
      compare(incremental.values, cache.values, label + "_step_values", 5e-5f, 5e-6);
    }
  }
}
