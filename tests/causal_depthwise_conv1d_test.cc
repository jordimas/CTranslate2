#include <cmath>
#include <limits>

#include "test_utils.h"
#include "ctranslate2/ops/causal_depthwise_conv1d.h"

TEST(CausalDepthwiseConv1DTest, AnalyticalNonzeroHistory) {
  StorageView x({1, 3, 1}, std::vector<float>{3, 4, 5});
  StorageView weight({1, 3}, std::vector<float>{1, 10, 100});
  StorageView history({1, 1, 2}, std::vector<float>{1, 2});
  StorageView y;
  ops::CausalDepthwiseConv1D(false)(x, weight, nullptr, history, y);
  expect_storage_eq(y, StorageView({1, 3, 1}, std::vector<float>{321, 432, 543}));
  expect_storage_eq(history, StorageView({1, 1, 2}, std::vector<float>{4, 5}));
}

TEST(CausalDepthwiseConv1DTest, BatchChannelBiasAndSilu) {
  StorageView x({2, 2, 2}, std::vector<float>{1, 2, 3, 4, -1, -2, -3, -4});
  StorageView weight({2, 2}, std::vector<float>{2, 3, -1, .5f});
  StorageView bias({2}, std::vector<float>{1, -2});
  StorageView history({2, 2, 1}, std::vector<float>{4, 5, -4, -5});
  StorageView y;
  ops::CausalDepthwiseConv1D()(x, weight, &bias, history, y);
  const std::vector<double> preactivation{12, -6, 12, -2, -10, 2, -10, -2};
  for (dim_t i = 0; i < y.size(); ++i)
    EXPECT_NEAR(y.data<float>()[i], preactivation[i] / (1 + std::exp(-preactivation[i])), 1e-6);
  expect_storage_eq(history, StorageView({2, 2, 1}, std::vector<float>{3, 4, -3, -4}));
}

TEST(CausalDepthwiseConv1DTest, ArbitraryChunksMatchPrefill) {
  StorageView x({2, 9, 3}, DataType::FLOAT32);
  for (dim_t i = 0; i < x.size(); ++i)
    x.data<float>()[i] = float(i % 13 - 6) / 7;
  StorageView weight({3, 4}, std::vector<float>{1, 2, 3, 4, -.1f, .2f, -.3f, .4f, 2, -1, .5f, .1f});
  StorageView initial({2, 3, 3}, .25f);
  StorageView full_state(initial), chunk_state(initial), full;
  ops::CausalDepthwiseConv1D op;
  op(x, weight, nullptr, full_state, full);
  dim_t offset = 0;
  for (const dim_t size : {2, 1, 5, 1}) {
    StorageView chunk({2, size, 3}, DataType::FLOAT32);
    for (dim_t b = 0; b < 2; ++b)
      for (dim_t t = 0; t < size; ++t)
        for (dim_t c = 0; c < 3; ++c)
          chunk.data<float>()[(b * size + t) * 3 + c] =
              x.data<float>()[(b * 9 + offset + t) * 3 + c];
    StorageView y;
    op(chunk, weight, nullptr, chunk_state, y);
    for (dim_t b = 0; b < 2; ++b)
      for (dim_t t = 0; t < size; ++t)
        for (dim_t c = 0; c < 3; ++c)
          EXPECT_FLOAT_EQ(y.data<float>()[(b * size + t) * 3 + c],
                          full.data<float>()[(b * 9 + offset + t) * 3 + c]);
    offset += size;
  }
  expect_storage_eq(chunk_state, full_state);
}

TEST(CausalDepthwiseConv1DTest, KernelOneAndEmptyTime) {
  StorageView x({1, 2, 2}, std::vector<float>{1, 2, 3, 4});
  StorageView weight({2, 1}, std::vector<float>{2, 3});
  StorageView state({1, 2, 0}, DataType::FLOAT32), y;
  ops::CausalDepthwiseConv1D(false)(x, weight, nullptr, state, y);
  expect_storage_eq(y, StorageView({1, 2, 2}, std::vector<float>{2, 6, 6, 12}));
  StorageView empty({1, 0, 2}, DataType::FLOAT32);
  StorageView wider_weight({2, 3}, 1.f), wider_state({1, 2, 2}, 7.f);
  ops::CausalDepthwiseConv1D()(empty, wider_weight, nullptr, wider_state, y);
  EXPECT_EQ(y.shape(), empty.shape());
  expect_storage_eq(wider_state, StorageView({1, 2, 2}, 7.f));
}

TEST(CausalDepthwiseConv1DTest, Guards) {
  StorageView x({1, 2, 2}, 1.f), weight({2, 3}, 1.f), state({1, 2, 2}, 0.f), y;
  StorageView bad_weight({3, 2}, 1.f), bad_state({1, 2, 3}, 0.f), bad_bias({1}, 0.f);
  ops::CausalDepthwiseConv1D op;
  EXPECT_THROW(op(x, bad_weight, nullptr, state, y), std::invalid_argument);
  EXPECT_THROW(op(x, weight, nullptr, bad_state, y), std::invalid_argument);
  EXPECT_THROW(op(x, weight, &bad_bias, state, y), std::invalid_argument);
  StorageView integers({1, 2, 2}, int32_t(1));
  EXPECT_THROW(op(integers, weight, nullptr, state, y), std::invalid_argument);
  x.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(op(x, weight, nullptr, state, y), std::invalid_argument);
  expect_storage_eq(state, StorageView({1, 2, 2}, 0.f));
}

TEST(CausalDepthwiseConv1DTest, AliasingRetainedAllocations) {
  StorageView x({1, 2, 2}, 1.f), weight({2, 3}, 1.f), state({1, 2, 2}, 0.f), y;
  ops::CausalDepthwiseConv1D op;
  EXPECT_THROW(op(x, weight, nullptr, state, x), std::invalid_argument);
  EXPECT_THROW(op(x, weight, nullptr, state, state), std::invalid_argument);
  StorageView input_alias({1, 2, 2}, state.data<float>());
  EXPECT_THROW(op(input_alias, weight, nullptr, state, y), std::invalid_argument);
  StorageView buffer({12}, 0.f);
  StorageView input_view({1, 2, 2}, buffer.data<float>() + 4);
  buffer.clear();
  EXPECT_THROW(op(input_view, weight, nullptr, state, buffer), std::invalid_argument);
}
