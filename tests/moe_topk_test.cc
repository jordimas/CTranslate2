#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include "test_utils.h"
#include "ctranslate2/ops/moe_topk.h"
#include "ctranslate2/ops/grouped_moe.h"

#ifdef CT2_WITH_CUDA
TEST(MoeTopKGpuTest, ParallelRoutingMatchesCpuAcrossWidthsAndTies) {
  if (!get_device_count(Device::CUDA))
    GTEST_SKIP() << "No CUDA/HIP device";
  const ScopedDeviceSetter setter(Device::CUDA, 0);
  for (dim_t width : {1, 17, 256, 513}) {
    for (dim_t k : {dim_t(1), std::min(dim_t(8), width), width}) {
      SCOPED_TRACE("width=" + std::to_string(width) + ", k=" + std::to_string(k));
      StorageView logits({3, width});
      for (dim_t e = 0; e < width; ++e) {
        logits.data<float>()[e] = 0.f;
        logits.data<float>()[width + e] = float((e * 37) % 19 - 9);
        logits.data<float>()[2 * width + e] = 10000.f + float(e % 13);
      }
      StorageView expected_ids(DataType::INT32), expected_weights;
      ops::MoeTopK{k}(logits, expected_ids, expected_weights);
      auto uploaded = logits.to(Device::CUDA);
      StorageView ids(DataType::INT32, Device::CUDA), weights(DataType::FLOAT32, Device::CUDA);
      ops::moe_topk_gpu(uploaded, k, ids, weights);
      auto actual_ids = ids.to(Device::CPU);
      auto actual_weights = weights.to(Device::CPU);
      synchronize_stream(Device::CUDA);
      expect_storage_eq(actual_ids, expected_ids);
      expect_storage_eq(actual_weights, expected_weights, 1e-7f);
    }
  }
}
#endif

TEST(MoeTopKTest, AnalyticSelectedSoftmax) {
  StorageView logits({2, 4}, std::vector<float>{0, 1, 2, 3, 1, 3, 2, 0});
  StorageView indices(DataType::INT32), weights;
  ops::MoeTopK(2)(logits, indices, weights);
  const std::vector<int32_t> expected{3, 2, 1, 2};
  EXPECT_EQ(indices.shape(), (Shape{2, 2}));
  EXPECT_EQ(weights.shape(), indices.shape());
  for (dim_t i = 0; i < 4; ++i) {
    EXPECT_EQ(indices.data<int32_t>()[i], expected[i]);
    EXPECT_NEAR(weights.data<float>()[i], i % 2 ? 1 / (1 + std::exp(1.)) : 1 / (1 + std::exp(-1.)),
                1e-7);
  }
}

TEST(MoeTopKTest, DeterministicTiesAndSingleExpert) {
  StorageView logits({1, 5}, std::vector<float>{2, 3, 3, 3, 1});
  StorageView indices(DataType::INT32), weights;
  ops::MoeTopK(2)(logits, indices, weights);
  EXPECT_EQ(indices.data<int32_t>()[0], 1);
  EXPECT_EQ(indices.data<int32_t>()[1], 2);
  EXPECT_EQ(weights.data<float>()[0], .5f);
  EXPECT_EQ(weights.data<float>()[1], .5f);
  ops::MoeTopK(1)(logits, indices, weights);
  EXPECT_EQ(indices.data<int32_t>()[0], 1);
  EXPECT_EQ(weights.data<float>()[0], 1.f);
}

TEST(MoeTopKTest, LargeFiniteLogits) {
  const float largest = std::numeric_limits<float>::max();
  StorageView logits({2, 3},
                     std::vector<float>{largest, largest, -largest, -10000, -10001, -10002});
  StorageView indices(DataType::INT32), weights;
  ops::MoeTopK(3)(logits, indices, weights);
  EXPECT_EQ(weights.data<float>()[0], .5f);
  EXPECT_EQ(weights.data<float>()[1], .5f);
  EXPECT_EQ(weights.data<float>()[2], 0.f);
  EXPECT_NEAR(weights.data<float>()[3], 1. / (1 + std::exp(-1.) + std::exp(-2.)), 1e-7);
}

TEST(MoeTopKTest, IndependentFullSoftmaxAndRowPermutation) {
  StorageView logits({13, 17}, DataType::FLOAT32);
  for (dim_t i = 0; i < logits.size(); ++i)
    logits.data<float>()[i] = float((i * 19) % 61 - 30) / 7;
  StorageView indices(DataType::INT32), weights;
  ops::MoeTopK(8)(logits, indices, weights);
  for (dim_t row = 0; row < 13; ++row) {
    std::vector<double> full(17);
    double total = 0;
    for (dim_t e = 0; e < 17; ++e)
      total += full[e] = std::exp(double(logits.data<float>()[row * 17 + e]));
    for (auto& p : full)
      p /= total;
    auto sorted = full;
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    double selected_total = 0;
    for (dim_t i = 0; i < 8; ++i)
      selected_total += sorted[i];
    std::set<int32_t> seen;
    double row_sum = 0;
    for (dim_t i = 0; i < 8; ++i) {
      const auto e = indices.data<int32_t>()[row * 8 + i];
      ASSERT_GE(e, 0);
      ASSERT_LT(e, 17);
      EXPECT_TRUE(seen.insert(e).second);
      EXPECT_NEAR(full[e], sorted[i], 1e-12);
      EXPECT_NEAR(weights.data<float>()[row * 8 + i], full[e] / selected_total, 1e-7);
      row_sum += weights.data<float>()[row * 8 + i];
    }
    EXPECT_NEAR(row_sum, 1, 1e-7);
  }
}

TEST(MoeTopKTest, EmptyRows) {
  StorageView logits({0, 3}, DataType::FLOAT32);
  StorageView indices(DataType::INT32), weights;
  ops::MoeTopK(2)(logits, indices, weights);
  EXPECT_EQ(indices.shape(), (Shape{0, 2}));
  EXPECT_EQ(weights.shape(), indices.shape());
}

TEST(MoeTopKTest, Validation) {
  EXPECT_THROW(ops::MoeTopK(0), std::invalid_argument);
  EXPECT_THROW(ops::MoeTopK(-1), std::invalid_argument);
  StorageView logits({1, 3}, 1.f);
  StorageView indices(DataType::INT32), weights;
  EXPECT_THROW(ops::MoeTopK(4)(logits, indices, weights), std::invalid_argument);
  StorageView rank1({3}, 1.f);
  EXPECT_THROW(ops::MoeTopK(2)(rank1, indices, weights), std::invalid_argument);
  StorageView wrong_indices;
  EXPECT_THROW(ops::MoeTopK(2)(logits, wrong_indices, weights), std::invalid_argument);
  logits.data<float>()[0] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(ops::MoeTopK(2)(logits, indices, weights), std::invalid_argument);
  logits.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(ops::MoeTopK(2)(logits, indices, weights), std::invalid_argument);
}

TEST(MoeTopKTest, RetainedCapacityAliasing) {
  StorageView weights({12}, 0.f);
  StorageView logits({1, 3}, weights.data<float>() + 5);
  weights.clear();
  StorageView indices(DataType::INT32);
  EXPECT_THROW(ops::MoeTopK(2)(logits, indices, weights), std::invalid_argument);
  StorageView independent_logits({1, 3}, 1.f);
  StorageView overlapping_indices({2}, reinterpret_cast<int32_t*>(weights.data<float>() + 2));
  EXPECT_THROW(ops::MoeTopK(2)(independent_logits, overlapping_indices, weights),
               std::invalid_argument);
}
