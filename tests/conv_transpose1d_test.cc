#include <limits>
#include <stdexcept>

#include "test_utils.h"
#include "ctranslate2/ops/conv_transpose1d.h"

TEST(ConvTranspose1DTest, ScattersOverlappingKernelWithoutReversal) {
  StorageView x({1, 1, 3}, std::vector<float>{1, 2, 3});
  StorageView w({1, 1, 3}, std::vector<float>{1, 10, 100});
  StorageView y;
  ops::ConvTranspose1D()(x, w, y);
  expect_storage_eq(y, StorageView({1, 1, 5}, std::vector<float>{1, 12, 123, 230, 300}));
}

TEST(ConvTranspose1DTest, SumsInputChannelsAndAddsBiasOnceForEachBatch) {
  StorageView x({2, 2, 1}, std::vector<float>{2, 3, 4, 5});
  StorageView w({2, 2, 1}, std::vector<float>{1, 10, 100, 1000});
  StorageView b({2}, std::vector<float>{-1, 2});
  StorageView y;
  ops::ConvTranspose1D()(x, w, b, y);
  expect_storage_eq(y, StorageView({2, 2, 1}, std::vector<float>{301, 3022, 503, 5042}));
}

TEST(ConvTranspose1DTest, GroupsDoNotMixChannels) {
  StorageView x({1, 2, 2}, std::vector<float>{1, 2, 3, 4});
  StorageView w({2, 2, 1}, std::vector<float>{1, 10, 100, 1000});
  StorageView y;
  ops::ConvTranspose1D(1, 0, 0, 1, 2)(x, w, y);
  expect_storage_eq(y,
                    StorageView({1, 4, 2}, std::vector<float>{1, 2, 10, 20, 300, 400, 3000, 4000}));
}

TEST(ConvTranspose1DTest, DilationPaddingAndOutputPadding) {
  StorageView x({1, 1, 2}, std::vector<float>{2, 3});
  StorageView w({1, 1, 2}, std::vector<float>{5, 7});
  StorageView b({1}, 1.f), y;
  ops::ConvTranspose1D(2, 1, 1, 2)(x, w, b, y);
  // Uncropped contributions at positions 0,2,4 are 10,29,21.
  expect_storage_eq(y, StorageView({1, 1, 4}, std::vector<float>{1, 30, 1, 22}));
}

TEST(ConvTranspose1DTest, OutputPaddingCanBeSmallerThanDilationOnly) {
  StorageView x({1, 1, 1}, 2.f), w({1, 1, 2}, 3.f), y;
  ops::ConvTranspose1D(1, 0, 1, 2)(x, w, y);
  expect_storage_eq(y, StorageView({1, 1, 4}, std::vector<float>{6, 0, 6, 0}));
}

TEST(ConvTranspose1DTest, RejectsInvalidParameters) {
  EXPECT_THROW(ops::ConvTranspose1D(0), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, -1), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 0, -1), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 0, 1), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 0, 0, 0), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 0, 0, 1, 0), std::invalid_argument);
}

TEST(ConvTranspose1DTest, RejectsInvalidShapesTypesAndLengths) {
  StorageView x({1, 2, 2}, 1.f), w({2, 1, 2}, 1.f), y;
  StorageView rank2({2, 2}, 1.f), channels({1, 1, 2}, 1.f);
  StorageView bias({2}, 1.f), bad_type(x.shape(), DataType::INT32);
  StorageView bad_output(DataType::INT32), empty({1, 2, 0});
  ops::ConvTranspose1D op;
  EXPECT_THROW(op(rank2, w, y), std::invalid_argument);
  EXPECT_THROW(op(x, rank2, y), std::invalid_argument);
  EXPECT_THROW(op(x, channels, y), std::invalid_argument);
  EXPECT_THROW(op(x, w, bias, y), std::invalid_argument);
  EXPECT_THROW(op(bad_type, w, y), std::invalid_argument);
  EXPECT_THROW(op(x, w, bad_output), std::invalid_argument);
  EXPECT_THROW(op(empty, w, y), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 0, 0, 1, 3)(x, w, y), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D(1, 10)(x, w, y), std::invalid_argument);
  EXPECT_THROW(ops::ConvTranspose1D{std::numeric_limits<dim_t>::max()}(x, w, y),
               std::invalid_argument);
}

TEST(ConvTranspose1DTest, RejectsDirectAndPartialOutputAliasing) {
  StorageView x({1, 1, 3}, 1.f), w({1, 1, 2}, 1.f), b({1}, 0.f);
  ops::ConvTranspose1D op;
  EXPECT_THROW(op(x, w, x), std::invalid_argument);
  EXPECT_THROW(op(x, w, w), std::invalid_argument);
  EXPECT_THROW(op(x, w, b, b), std::invalid_argument);
  StorageView partial({1}, x.data<float>() + 1);
  EXPECT_THROW(op(x, w, partial), std::invalid_argument);
}

TEST(ConvTranspose1DTest, RejectsAliasingInRetainedOutputCapacity) {
  StorageView y({8}, 1.f);
  StorageView x({1, 1, 2}, y.data<float>() + 4), w({1, 1, 2}, 1.f);
  y.resize({0});
  EXPECT_THROW(ops::ConvTranspose1D()(x, w, y), std::invalid_argument);
}

namespace {

  void compare_transpose_with_scalar(dim_t batch, dim_t cin, dim_t cout_group, dim_t time,
                                     dim_t kernel, dim_t stride, dim_t padding,
                                     dim_t output_padding, dim_t dilation, dim_t groups) {
    StorageView x({batch, cin, time}), w({cin, cout_group, kernel});
    const dim_t cout = cout_group * groups;
    StorageView bias(Shape{cout});
    for (dim_t i = 0; i < x.size(); ++i)
      x.data<float>()[i] = (i % 17 - 8) * .03125f;
    for (dim_t i = 0; i < w.size(); ++i)
      w.data<float>()[i] = (i % 13 - 6) * .015625f;
    for (dim_t i = 0; i < cout; ++i)
      bias.data<float>()[i] = i * .00390625f;
    const dim_t length =
        (time - 1) * stride - 2 * padding + (kernel - 1) * dilation + output_padding + 1;
    StorageView expected({batch, cout, length});
    for (dim_t n = 0; n < batch; ++n)
      for (dim_t oc = 0; oc < cout; ++oc) {
        auto* row = expected.data<float>() + (n * cout + oc) * length;
        std::fill_n(row, length, bias.data<float>()[oc]);
        const dim_t group = oc / cout_group;
        for (dim_t ic = group * (cin / groups); ic < (group + 1) * (cin / groups); ++ic)
          for (dim_t t = 0; t < time; ++t)
            for (dim_t k = 0; k < kernel; ++k) {
              const dim_t pos = t * stride - padding + k * dilation;
              if (pos >= 0 && pos < length)
                row[pos] += x.data<float>()[(n * cin + ic) * time + t]
                            * w.data<float>()[(ic * cout_group + oc % cout_group) * kernel + k];
            }
      }
    StorageView actual;
    ops::ConvTranspose1D op(stride, padding, output_padding, dilation, groups);
    op(x, w, bias, actual);
    expect_storage_eq(actual, expected);
  }

}

TEST(ConvTranspose1DTest, GemmSizedInputsMatchScalarAcrossGroupsAndBoundaryParameters) {
  for (dim_t groups : {1, 2})
    for (dim_t dilation : {1, 2})
      for (dim_t output_padding : {0, 1})
        compare_transpose_with_scalar(2, 16, 5, 7, 3, 2, 1, output_padding, dilation, groups);
}

TEST(ConvTranspose1DTest, GemmTimeTilesPreserveOverlapsAndBiasAcrossTileBoundary) {
  // Contributions alone exceed the 16 MiB workspace bound, forcing multiple
  // input tiles whose overlapping output regions must accumulate, not reset.
  compare_transpose_with_scalar(1, 8, 256, 4096, 5, 2, 1, 1, 2, 1);
}
