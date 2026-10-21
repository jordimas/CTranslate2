#include <algorithm>
#include <limits>
#include <memory>

#include "test_utils.h"
#include "ctranslate2/ops/causal_depthwise_conv1d.h"
#include "ctranslate2/ops/conv_transpose1d.h"
#include "ctranslate2/ops/gated_delta_net.h"

namespace {

void fill_pattern(StorageView& tensor, float scale = .03125f) {
  for (dim_t i = 0; i < tensor.size(); ++i)
    tensor.data<float>()[i] = float(i % 17 - 8) * scale;
}

class PublicOperatorDeviceTest : public ::testing::TestWithParam<Device> {
protected:
  void SetUp() override {
    device = GetParam();
    if (device == Device::CUDA && !get_device_count(device))
      GTEST_SKIP() << "No CUDA/HIP device";
    setter.reset(new ScopedDeviceSetter(device, 0));
  }

  Device device = Device::CPU;
  std::unique_ptr<ScopedDeviceSetter> setter;
};

TEST_P(PublicOperatorDeviceTest, CausalConvolutionMatchesCpuAndPreservesHistory) {
  for (dim_t kernel : {1, 4, 7}) {
    for (dim_t steps : {0, 1, 9, 65}) {
      for (bool silu : {false, true}) {
        SCOPED_TRACE("kernel=" + std::to_string(kernel) + ", steps=" + std::to_string(steps)
                     + ", silu=" + std::to_string(silu));
        StorageView x({2, steps, 5}), w({5, kernel}), bias({5}), initial({2, 5, kernel - 1});
        for (auto* tensor : {&x, &w, &bias, &initial})
          fill_pattern(*tensor);
        ops::CausalDepthwiseConv1D op(silu);
        StorageView expected, expected_state(initial);
        const auto* cpu_bias = silu ? &bias : nullptr;
        op(x, w, cpu_bias, expected_state, expected);
        auto dx = x.to(device), dw = w.to(device), db = bias.to(device), state = initial.to(device);
        StorageView actual(DataType::FLOAT32, device);
        op(dx, dw, silu ? &db : nullptr, state, actual);
        expect_storage_eq(actual, expected, 2e-6f);
        expect_storage_eq(state, expected_state);
      }
    }
  }
}

TEST_P(PublicOperatorDeviceTest, CausalConvolutionArbitraryChunksMatchPrefill) {
  StorageView input({2, 9, 3}), weight({3, 4}), initial({2, 3, 3});
  for (auto* tensor : {&input, &weight, &initial})
    fill_pattern(*tensor);
  ops::CausalDepthwiseConv1D op;
  StorageView expected, expected_state(initial);
  op(input, weight, nullptr, expected_state, expected);
  auto state = initial.to(device), dw = weight.to(device);
  dim_t offset = 0;
  for (dim_t count : {0, 2, 1, 5, 1}) {
    StorageView chunk({2, count, 3}), expected_chunk(chunk.shape());
    for (dim_t n = 0; n < 2; ++n) {
      if (count) {
        std::copy_n(input.data<float>() + (n * 9 + offset) * 3, count * 3,
                    chunk.data<float>() + n * count * 3);
        std::copy_n(expected.data<float>() + (n * 9 + offset) * 3, count * 3,
                    expected_chunk.data<float>() + n * count * 3);
      }
    }
    auto dx = chunk.to(device);
    StorageView actual(DataType::FLOAT32, device);
    op(dx, dw, nullptr, state, actual);
    expect_storage_eq(actual, expected_chunk, 2e-6f);
    offset += count;
  }
  expect_storage_eq(state, expected_state);
}

TEST_P(PublicOperatorDeviceTest, CausalValidationDoesNotMutateHistoryOrOutput) {
  StorageView x({1, 2, 2}, 1.f), w({2, 3}, .5f), bias({2}, .25f), initial({1, 2, 2}, 2.f);
  auto dx = x.to(device), dw = w.to(device), db = bias.to(device), state = initial.to(device);
  StorageView actual({1}, 42.f, device);
  ops::CausalDepthwiseConv1D op;
  EXPECT_THROW(op(dx, dw, &db, state, state), std::invalid_argument);
  EXPECT_THROW(op(dx, dw, &db, state, dx), std::invalid_argument);
  StorageView retained({16}, 3.f, device);
  StorageView alias({1, 2, 2}, retained.data<float>() + 2, device);
  retained.clear();
  EXPECT_THROW(op(alias, dw, &db, state, retained), std::invalid_argument);
  StorageView wrong_weight({2, 2}, 1.f, device), wrong_bias({3}, 1.f, device);
  StorageView wrong_type(x.shape(), DataType::INT32, device);
  EXPECT_THROW(op(dx, wrong_weight, &db, state, actual), std::invalid_argument);
  EXPECT_THROW(op(dx, dw, &wrong_bias, state, actual), std::invalid_argument);
  EXPECT_THROW(op(wrong_type, dw, &db, state, actual), std::invalid_argument);
  for (int bad_tensor = 0; device == Device::CPU && bad_tensor < 4; ++bad_tensor) {
    StorageView hx(x), hw(w), hb(bias), hs(initial);
    StorageView* tensors[] = {&hx, &hw, &hb, &hs};
    tensors[bad_tensor]->data<float>()[0] = std::numeric_limits<float>::infinity();
    dx = hx.to(device);
    dw = hw.to(device);
    db = hb.to(device);
    state = hs.to(device);
    EXPECT_THROW(op(dx, dw, &db, state, actual), std::invalid_argument);
    expect_storage_eq(state, hs);
    expect_storage_eq(actual, StorageView({1}, 42.f));
  }
  if (device == Device::CUDA) {
    EXPECT_THROW(op(x, dw, &db, state, actual), std::invalid_argument);
  }
}

TEST_P(PublicOperatorDeviceTest, GatedDeltaMatchesCpuAcrossBatchesHeadsAndKeySpecializations) {
  for (dim_t kd : {3, 64, 128}) {
    for (dim_t vd : {2, 257, 1025}) {
      for (dim_t steps : {0, 1, 5}) {
        SCOPED_TRACE("kd=" + std::to_string(kd) + ", vd=" + std::to_string(vd)
                     + ", steps=" + std::to_string(steps));
        StorageView q({2, steps, 2, kd}), k(q.shape()), v({2, steps, 4, vd});
        StorageView g({2, steps, 4}, -.2f), beta({2, steps, 4}, .3f), initial({2, 4, kd, vd});
        for (auto* tensor : {&q, &k, &v, &initial})
          fill_pattern(*tensor, .0078125f);
        StorageView expected, expected_state(initial);
        ops::GatedDeltaNet op;
        op(q, k, v, g, beta, expected_state, expected);
        auto dq = q.to(device), dk = k.to(device), dv = v.to(device);
        auto dg = g.to(device), db = beta.to(device), state = initial.to(device);
        StorageView actual(DataType::FLOAT32, device);
        op(dq, dk, dv, dg, db, state, actual);
        expect_storage_eq(actual, expected, 2e-5f);
        expect_storage_eq(state, expected_state, 2e-5f);
      }
    }
  }
}

TEST_P(PublicOperatorDeviceTest, GatedDeltaSingleStepsMatchPrefill) {
  const dim_t steps = 5, kd = 64, vd = 3;
  StorageView q({2, steps, 2, kd}), k(q.shape()), v({2, steps, 4, vd});
  StorageView g({2, steps, 4}, -.3f), beta({2, steps, 4}, .25f), initial({2, 4, kd, vd});
  for (auto* tensor : {&q, &k, &v, &initial})
    fill_pattern(*tensor, .015625f);
  StorageView expected, expected_state(initial);
  ops::GatedDeltaNet op;
  op(q, k, v, g, beta, expected_state, expected);
  auto dq = q.to(device), dk = k.to(device), dv = v.to(device);
  auto dg = g.to(device), db = beta.to(device), state = initial.to(device);
  for (dim_t n = 0; n < 2; ++n) {
    StorageView s({1, 4, kd, vd}, state.data<float>() + n * 4 * kd * vd, device);
    for (dim_t t = 0; t < steps; ++t) {
      const dim_t row = n * steps + t;
      StorageView qt({1, 1, 2, kd}, dq.data<float>() + row * 2 * kd, device);
      StorageView kt({1, 1, 2, kd}, dk.data<float>() + row * 2 * kd, device);
      StorageView vt({1, 1, 4, vd}, dv.data<float>() + row * 4 * vd, device);
      StorageView gt({1, 1, 4}, dg.data<float>() + row * 4, device);
      StorageView bt({1, 1, 4}, db.data<float>() + row * 4, device);
      StorageView actual(DataType::FLOAT32, device);
      op(qt, kt, vt, gt, bt, s, actual);
      StorageView reference({1, 1, 4, vd}, expected.data<float>() + row * 4 * vd);
      expect_storage_eq(actual, reference, 2e-5f);
    }
  }
  expect_storage_eq(state, expected_state, 2e-5f);
}

TEST_P(PublicOperatorDeviceTest, GatedDeltaValidationPreservesState) {
  StorageView q({1, 1, 1, 2}, 1.f, device), k(q), v(q);
  StorageView g({1, 1, 1}, 0.f, device), beta({1, 1, 1}, .5f, device);
  StorageView state({1, 1, 2, 2}, 2.f, device), actual({1}, 42.f, device);
  ops::GatedDeltaNet op;
  StorageView bad_key({1, 1, 1, 3}, 1.f, device), bad_gate({1}, 0.f, device);
  EXPECT_THROW(op(q, bad_key, v, g, beta, state, actual), std::invalid_argument);
  EXPECT_THROW(op(q, k, v, bad_gate, beta, state, actual), std::invalid_argument);
  EXPECT_THROW(op(q, k, v, g, beta, state, q), std::invalid_argument);
  EXPECT_THROW(op(q, k, v, g, beta, state, state), std::invalid_argument);
  StorageView alias({1, 1, 1, 2}, state.data<float>() + 1, device);
  EXPECT_THROW(op(alias, k, v, g, beta, state, actual), std::invalid_argument);
  StorageView wrong_type(q.shape(), DataType::INT32, device);
  EXPECT_THROW(op(wrong_type, k, v, g, beta, state, actual), std::invalid_argument);
  if (device == Device::CUDA) {
    auto host = q.to(Device::CPU);
    EXPECT_THROW(op(host, k, v, g, beta, state, actual), std::invalid_argument);
  }
  expect_storage_eq(state, StorageView({1, 1, 2, 2}, 2.f));
  expect_storage_eq(actual, StorageView({1}, 42.f));
}

TEST_P(PublicOperatorDeviceTest, TransposedConvolutionMatchesScalarAcrossParameters) {
  // A scatter reference, independent of the GPU's GEMM/gather implementation.
  for (dim_t groups : {1, 2}) {
    for (dim_t dilation : {1, 2}) {
      for (dim_t output_padding : {0, 1}) {
        for (bool use_bias : {false, true}) {
          StorageView x({2, 4, 7}), w({4, 3, 3}), bias({3 * groups});
          for (auto* tensor : {&x, &w, &bias})
            fill_pattern(*tensor);
          const dim_t length = 12 - 2 + 2 * dilation + output_padding + 1;
          StorageView expected({2, 3 * groups, length}, 0.f);
          for (dim_t n = 0; n < 2; ++n)
            for (dim_t oc = 0; oc < 3 * groups; ++oc) {
              auto* out = expected.data<float>() + (n * 3 * groups + oc) * length;
              std::fill_n(out, length, use_bias ? bias.data<float>()[oc] : 0.f);
              const dim_t first_ic = (oc / 3) * (4 / groups);
              for (dim_t ic = first_ic; ic < first_ic + 4 / groups; ++ic)
                for (dim_t t = 0; t < 7; ++t)
                  for (dim_t tap = 0; tap < 3; ++tap) {
                    const dim_t pos = t * 2 - 1 + tap * dilation;
                    if (pos >= 0 && pos < length)
                      out[pos] += x.data<float>()[(n * 4 + ic) * 7 + t]
                                  * w.data<float>()[(ic * 3 + oc % 3) * 3 + tap];
                  }
            }
          auto dx = x.to(device), dw = w.to(device), db = bias.to(device);
          StorageView actual(DataType::FLOAT32, device);
          ops::ConvTranspose1D op(2, 1, output_padding, dilation, groups);
          if (use_bias)
            op(dx, dw, db, actual);
          else
            op(dx, dw, actual);
          expect_storage_eq(actual, expected, 1e-6f);
        }
      }
    }
  }
}

TEST_P(PublicOperatorDeviceTest, TransposedConvolutionBoundaryCases) {
  StorageView x({1, 2, 1}, std::vector<float>{2, 3}, device);
  StorageView w({2, 2, 1}, std::vector<float>{1, 10, 100, 1000}, device);
  StorageView bias({2}, std::vector<float>{-1, 2}, device);
  StorageView actual(DataType::FLOAT32, device);
  ops::ConvTranspose1D()(x, w, bias, actual);
  expect_storage_eq(actual, StorageView({1, 2, 1}, std::vector<float>{301, 3022}));
  StorageView single({1, 1, 1}, 2.f, device), filter({1, 1, 2}, 3.f, device);
  ops::ConvTranspose1D(1, 0, 1, 2)(single, filter, actual);
  expect_storage_eq(actual, StorageView({1, 1, 4}, std::vector<float>{6, 0, 6, 0}));
  // Fish's right-side crop belongs to its layer, not the public transpose op.
  StorageView fish_input({1, 1, 2}, 1.f, device), fish_filter({1, 1, 6}, 1.f, device);
  ops::ConvTranspose1D(3)(fish_input, fish_filter, actual);
  expect_storage_eq(actual, StorageView({1, 1, 9}, std::vector<float>{1, 1, 1, 2, 2, 2, 1, 1, 1}));
}

TEST_P(PublicOperatorDeviceTest, TransposedConvolutionTimeTilesPreserveOverlaps) {
  // Forces a partial final tile and overlapping contributions across tiles.
  StorageView x({1, 8, 4096}), w({8, 256, 5}), bias({256});
  for (auto* tensor : {&x, &w, &bias})
    fill_pattern(*tensor, .00390625f);
  ops::ConvTranspose1D op(2, 1, 1, 2);
  StorageView expected;
  op(x, w, bias, expected);
  auto dx = x.to(device), dw = w.to(device), db = bias.to(device);
  StorageView actual(DataType::FLOAT32, device);
  op(dx, dw, db, actual);
  expect_storage_eq(actual, expected, 1e-6f);
}

TEST_P(PublicOperatorDeviceTest, TransposedConvolutionValidationPreservesOutput) {
  StorageView x({1, 2, 2}, 1.f, device), w({2, 1, 2}, 1.f, device);
  StorageView bias({1}, .5f, device), actual({1}, 42.f, device);
  ops::ConvTranspose1D op;
  StorageView bad_weight({1, 1, 2}, 1.f, device), bad_bias({2}, 0.f, device);
  StorageView wrong_type(x.shape(), DataType::INT32, device);
  EXPECT_THROW(op(x, bad_weight, actual), std::invalid_argument);
  EXPECT_THROW(op(x, w, bad_bias, actual), std::invalid_argument);
  EXPECT_THROW(op(wrong_type, w, actual), std::invalid_argument);
  EXPECT_THROW(op(x, w, x), std::invalid_argument);
  EXPECT_THROW(op(x, w, bias, bias), std::invalid_argument);
  StorageView retained({16}, 3.f, device);
  StorageView alias({1, 2, 2}, retained.data<float>() + 2, device);
  retained.clear();
  EXPECT_THROW(op(alias, w, retained), std::invalid_argument);
  if (device == Device::CUDA) {
    auto host = x.to(Device::CPU);
    EXPECT_THROW(op(host, w, actual), std::invalid_argument);
  }
  expect_storage_eq(actual, StorageView({1}, 42.f));
}

INSTANTIATE_TEST_SUITE_P(CPU, PublicOperatorDeviceTest, ::testing::Values(Device::CPU));
INSTANTIATE_TEST_SUITE_P(GPU, PublicOperatorDeviceTest, ::testing::Values(Device::CUDA));

}
