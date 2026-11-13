#pragma once

#include "cpu/kernels.h"
#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace cpu {

    template <typename Function>
    static void with_activation(const ops::ActivationType* activation, const Function& func);

#if defined(__AVX2__) && !defined(__AVX512F__)
#if !defined(__AVXVNNI__)
    static __m256i load_weight_s16(const int8_t* p) {
      return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    }

    static __m256i load_input_u16(const uint8_t* p) {
      return _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    }

#endif

    static int32_t reduce_int32(__m256i sum) {
      auto x = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
      x = _mm_hadd_epi32(x, x);
      return _mm_cvtsi128_si32(_mm_hadd_epi32(x, x));
    }

    template<int M>
    static void weight_only_int8_dot_rows(const int8_t* weight, const float* scale,
                                         const int32_t* compensation, const uint8_t* input,
                                         float* output, dim_t k, dim_t n,
                                         const float* inv_scale, const float* bias, const float* residual,
                                         const ops::ActivationType* activation) {
      using VecType = Vec<float, CpuIsa::AVX2>;
      constexpr dim_t width = VecType::width;
      // Buffer output channels so activations use full SIMD vectors even for M=1.
      parallel_for(0, (n + width - 1) / width, 1, [&](dim_t begin, dim_t end) {
#if defined(__AVXVNNI__)
        constexpr int chains = 1;
#else
        // Widen byte products to avoid INT16 saturation; limit registers for M=5..8.
        constexpr int chains = M <= 4 ? 2 : 1;
#endif
        for (dim_t tile = begin; tile < end; ++tile) {
          const dim_t first = tile * width;
          const dim_t count = std::min(width, n - first);
          float values[M][VecType::width] = {};
          for (dim_t lane = 0; lane < count; ++lane) {
            const dim_t c = first + lane;
            __m256i acc[chains * M];
            for (int i = 0; i < chains * M; ++i)
              acc[i] = _mm256_setzero_si256();
            const int8_t* row = weight + c * k;
            dim_t j = 0;
#if defined(__AVXVNNI__)
            // vpdpbusd fuses u8*s8 widen+madd+add; one accumulator per row.
            for (; j + 32 <= k; j += 32) {
              const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + j));
              for (int r = 0; r < M; ++r) {
                const __m256i x = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(input + r * k + j));
                acc[r] = _mm256_dpbusd_avx_epi32(acc[r], x, w);
              }
            }
#else
            for (; j + 32 <= k; j += 32) {
              const auto w0 = load_weight_s16(row + j);
              const auto w1 = load_weight_s16(row + j + 16);
              for (int r = 0; r < M; ++r) {
                const uint8_t* x = input + r * k + j;
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(load_input_u16(x), w0));
                acc[(chains - 1) * M + r] = _mm256_add_epi32(acc[(chains - 1) * M + r],
                    _mm256_madd_epi16(load_input_u16(x + 16), w1));
              }
            }
            if (j + 16 <= k) {
              const auto w = load_weight_s16(row + j);
              for (int r = 0; r < M; ++r)
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(load_input_u16(input + r * k + j), w));
              j += 16;
            }
#endif
            for (int r = 0; r < M; ++r) {
              auto total = acc[r];
              if constexpr (chains == 2)
                total = _mm256_add_epi32(total, acc[M + r]);
              int32_t sum = reduce_int32(total) + compensation[c];
              for (dim_t t = j; t < k; ++t)
                sum += int32_t(input[r * k + t]) * row[t];
              values[r][lane] = float(sum) * inv_scale[r] / scale[c];
            }
          }
          for (int r = 0; r < M; ++r) {
            auto v = VecType::load(values[r]);
            if (bias)
              v = VecType::add(v, VecType::load(bias + first, count));
            if (residual)
              v = VecType::add(v, VecType::load(residual + r * n + first, count));
            with_activation(activation, [&](const auto& epilogue) {
              VecType::store(epilogue(v), output + r * n + first, count);
            });
          }
        }
      });
    }
    static void weight_only_int8_dot(const int8_t* weight, const float* scale,
                                      const int32_t* compensation, const uint8_t* input,
                                      float* output, dim_t m, dim_t k, dim_t n,
                                      const float* inv_scale, const float* bias, const float* residual,
                                      const ops::ActivationType* activation) {
      // Fixed M keeps accumulators in registers, especially for beam-size-five decoding.
#define CT2_WEIGHT_ONLY_CASE(M) \
      case M: weight_only_int8_dot_rows<M>(weight, scale, compensation, input, output, \
                                           k, n, inv_scale, bias, residual, activation); break
      switch (m) {
        CT2_WEIGHT_ONLY_CASE(1); CT2_WEIGHT_ONLY_CASE(2);
        CT2_WEIGHT_ONLY_CASE(3); CT2_WEIGHT_ONLY_CASE(4);
        CT2_WEIGHT_ONLY_CASE(5); CT2_WEIGHT_ONLY_CASE(6);
        CT2_WEIGHT_ONLY_CASE(7); CT2_WEIGHT_ONLY_CASE(8);
      }
#undef CT2_WEIGHT_ONLY_CASE
    }
#endif

  }
}
