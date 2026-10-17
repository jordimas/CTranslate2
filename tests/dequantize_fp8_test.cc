#include "test_utils.h"
#include "ctranslate2/ops/dequantize_fp8.h"

#include <cmath>
#include <cstring>
#include <limits>

TEST(DequantizeFP8Test, EveryFiniteEncodingAndSignedBytes) {
  StorageView input({1, 254}, DataType::INT8), scale({1, 2}, 2.f), output;
  auto* bytes = reinterpret_cast<uint8_t*>(input.data<int8_t>());
  dim_t column = 0;
  for (unsigned code = 0; code < 256; ++code)
    if ((code & 127) != 127) bytes[column++] = code;
  ops::DequantizeFP8()(input, scale, output);
  column = 0;
  for (unsigned code = 0; code < 256; ++code) {
    if ((code & 127) == 127) continue;
    const unsigned exponent = (code & 127) / 8, mantissa = code % 8;
    const double magnitude = exponent == 0 ? double(mantissa) / 512
      : (1. + double(mantissa) / 8) * std::pow(2., int(exponent) - 7);
    const float expected = float((code & 128 ? -magnitude : magnitude) * 2);
    EXPECT_EQ(output.data<float>()[column], expected);
    EXPECT_EQ(std::signbit(output.data<float>()[column]), std::signbit(expected));
    ++column;
  }
}

TEST(DequantizeFP8Test, EdgeBlocksAndFloat32ScaleMultiplication) {
  StorageView input({129, 130}, int8_t(0x38));  // E4M3FN 1.0.
  StorageView scale({2, 2}, std::vector<float>{.1f, .3f, 1.7f, 9.1f}), output;
  ops::DequantizeFP8()(input, scale, output);
  EXPECT_EQ(output.shape(), input.shape());
  for (dim_t r = 0; r < 129; ++r)
    for (dim_t c = 0; c < 130; ++c)
      EXPECT_EQ(output.data<float>()[r * 130 + c], scale.data<float>()[(r / 128) * 2 + c / 128]);
}

TEST(DequantizeFP8Test, RejectsInvalidScalesAndEncodings) {
  StorageView input({1, 1}, int8_t(0x38)), scale({1, 1}, 1.f), output;
  for (float bad : {0.f, -1.f, std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()}) {
    scale.data<float>()[0] = bad;
    EXPECT_THROW(ops::DequantizeFP8()(input, scale, output), std::invalid_argument);
  }
  scale.data<float>()[0] = 1.f;
  for (uint8_t bad : {uint8_t(0x7f), uint8_t(0xff)}) {
    std::memcpy(input.data<int8_t>(), &bad, 1);
    EXPECT_THROW(ops::DequantizeFP8()(input, scale, output), std::invalid_argument);
  }
  input.data<int8_t>()[0] = 0x7e;  // Largest finite FP8 times finite F32 can overflow.
  scale.data<float>()[0] = std::numeric_limits<float>::max();
  EXPECT_THROW(ops::DequantizeFP8()(input, scale, output), std::invalid_argument);
}

TEST(DequantizeFP8Test, RejectsMalformedShapesTypesAndAliasing) {
  StorageView input({1, 1}, int8_t(0x38)), scale({1, 1}, 1.f), output;
  for (const Shape& shape : {Shape{1}, Shape{0, 1}, Shape{1, 1, 1}}) {
    StorageView bad(shape, DataType::INT8);
    EXPECT_THROW(ops::DequantizeFP8()(bad, scale, output), std::invalid_argument);
  }
  StorageView bad_scale({2, 1}, 1.f), bad_rank({1}, 1.f), bad_type({1, 1}, int8_t(1));
  EXPECT_THROW(ops::DequantizeFP8()(input, bad_scale, output), std::invalid_argument);
  EXPECT_THROW(ops::DequantizeFP8()(input, bad_rank, output), std::invalid_argument);
  EXPECT_THROW(ops::DequantizeFP8()(scale, scale, output), std::invalid_argument);
  EXPECT_THROW(ops::DequantizeFP8()(input, bad_type, output), std::invalid_argument);
  EXPECT_THROW(ops::DequantizeFP8()(input, scale, bad_type), std::invalid_argument);
  EXPECT_THROW(ops::DequantizeFP8()(input, scale, scale), std::invalid_argument);
  StorageView byte_alias({1, 1}, reinterpret_cast<float*>(input.data<int8_t>()));
  EXPECT_THROW(ops::DequantizeFP8()(input, scale, byte_alias), std::invalid_argument);
  StorageView alias({1, 1}, scale.data<float>());
  EXPECT_THROW(ops::DequantizeFP8()(input, scale, alias), std::invalid_argument);
}
