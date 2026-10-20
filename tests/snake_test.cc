#include <cmath>
#include <limits>

#include "test_utils.h"
#include "ops/drama/snake.h"

TEST(SnakeTest, AnalyticalAndChannelBroadcast) {
  const float pi = std::acos(-1.f);
  StorageView x({2, 2, 2}, std::vector<float>{0, pi / 2, 0, pi / 4, -pi / 2, pi, -pi / 4, pi / 2});
  StorageView alpha({2}, std::vector<float>{1, 2});
  StorageView y;
  ops::Snake()(x, alpha, y);
  const std::vector<float> expected{0,  pi / 2 + 1,    0,     pi / 4 + .5f, -pi / 2 + 1,
                                    pi, -pi / 4 + .5f, pi / 2};
  ASSERT_EQ(y.shape(), x.shape());
  for (dim_t i = 0; i < y.size(); ++i)
    EXPECT_NEAR(y.data<float>()[i], expected[i], 1e-6);
}

TEST(SnakeTest, IndependentDoublePrecisionEquation) {
  StorageView x({2, 3, 11}, DataType::FLOAT32);
  for (dim_t i = 0; i < x.size(); ++i)
    x.data<float>()[i] = float(i - 30) / 9;
  StorageView alpha({1, 3, 1}, std::vector<float>{.25f, -1.5f, 3.f});
  StorageView y;
  ops::Snake()(x, alpha, y);
  for (dim_t i = 0; i < x.size(); ++i) {
    const double a = alpha.data<float>()[(i / 11) % 3];
    const double value = x.data<float>()[i];
    const double expected = value + std::pow(std::sin(a * value), 2) / (a + 1e-9);
    EXPECT_NEAR(y.data<float>()[i], expected, 2e-6);
  }
}

TEST(SnakeTest, ZeroAndTinyAlpha) {
  StorageView x({1, 2, 2}, std::vector<float>{-2, 3, 1, -1});
  StorageView alpha({2}, std::vector<float>{0, 1e-9f});
  StorageView y;
  ops::Snake()(x, alpha, y);
  expect_storage_eq(y, x);
}

TEST(SnakeTest, ParallelChannelBoundaries) {
  StorageView x({2, 3, 10001}, DataType::FLOAT32);
  StorageView alpha({3}, std::vector<float>{.125f, 1.f, 2.5f});
  for (dim_t i = 0; i < x.size(); ++i)
    x.data<float>()[i] = float(i % 73 - 36) / 13;
  StorageView y;
  ops::Snake()(x, alpha, y);
  for (dim_t i = 0; i < x.size(); ++i) {
    const double a = alpha.data<float>()[(i / 10001) % 3];
    const double value = x.data<float>()[i];
    EXPECT_NEAR(y.data<float>()[i], value + std::pow(std::sin(a * value), 2) / (a + 1e-9), 2e-6);
  }
}

TEST(SnakeTest, EmptyTime) {
  StorageView x({1, 2, 0}, DataType::FLOAT32);
  StorageView alpha({2}, std::vector<float>{1, 2});
  StorageView y;
  ops::Snake()(x, alpha, y);
  EXPECT_EQ(y.shape(), x.shape());
  EXPECT_TRUE(y.empty());
}

TEST(SnakeTest, InvalidShapeAndDtype) {
  StorageView x({1, 2, 3}, 1.f);
  StorageView alpha({3}, 1.f);
  StorageView y;
  EXPECT_THROW(ops::Snake()(x, alpha, y), std::invalid_argument);
  StorageView wrong_rank({2, 3}, 1.f);
  EXPECT_THROW(ops::Snake()(wrong_rank, alpha, y), std::invalid_argument);
  StorageView valid_alpha({2}, 1.f);
  StorageView integers({1, 2, 3}, int32_t(1));
  EXPECT_THROW(ops::Snake()(integers, valid_alpha, y), std::invalid_argument);
}

TEST(SnakeTest, InvalidValuesLeaveOutputUntouched) {
  StorageView x({1, 1, 1}, 1.f);
  StorageView alpha({1}, -1e-9f);
  StorageView y({1}, 42.f);
  EXPECT_THROW(ops::Snake()(x, alpha, y), std::invalid_argument);
  EXPECT_EQ(y.data<float>()[0], 42);
  alpha.data<float>()[0] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(ops::Snake()(x, alpha, y), std::invalid_argument);
  alpha.data<float>()[0] = 1;
  x.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(ops::Snake()(x, alpha, y), std::invalid_argument);
}

TEST(SnakeTest, RejectsAliasingIncludingClearedOutput) {
  StorageView x({1, 1, 3}, 1.f);
  StorageView alpha({1}, 1.f);
  EXPECT_THROW(ops::Snake()(x, alpha, x), std::invalid_argument);
  EXPECT_THROW(ops::Snake()(x, alpha, alpha), std::invalid_argument);
  StorageView y({8}, 0.f);
  StorageView view({1, 1, 3}, y.data<float>() + 2);
  y.clear();
  EXPECT_THROW(ops::Snake()(view, alpha, y), std::invalid_argument);
}
