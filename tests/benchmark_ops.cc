#include "benchmark_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#if defined(__x86_64__)
#  include <immintrin.h>
#endif

#include "ctranslate2/ops/ops.h"

using namespace ctranslate2;

void benchmark_gather(Device device) {
  StorageView data({512, 512}, DataType::FLOAT32, device);
  std::vector<int32_t> input_v(250);
  std::iota(input_v.begin(), input_v.end(), 0);
  StorageView input({static_cast<dim_t>(input_v.size())}, input_v, device);
  StorageView output(device);
  const ops::Gather gather_op;
  BENCHMARK(gather_op(data, input, output), 100000);
}

void benchmark_transpose(Device device) {
  StorageView x({64, 48, 8, 64}, DataType::FLOAT32, device);
  StorageView y(device);
  const ops::Transpose transpose_op({0, 2, 1, 3});
  BENCHMARK(transpose_op(x, y), 1000);
}

void benchmark_split(Device device) {
  StorageView x({64, 512*3}, DataType::FLOAT32, device);
  StorageView a(device);
  StorageView b(device);
  StorageView c(device);
  const ops::Split split_op(-1);
  BENCHMARK(split_op(x, a, b, c), 10000);
}

void benchmark_layer_norm(Device device) {
  std::vector<float> gamma_ = rand_vector(512);
  std::vector<float> beta_ = rand_vector(512);
  std::vector<float> x_ = rand_vector(100 * 512);

  StorageView gamma({512}, gamma_, device);
  StorageView beta({512}, beta_, device);
  StorageView x({100, 512}, x_, device);
  StorageView y(x.device());
  const ops::LayerNorm layer_norm_op{};
  BENCHMARK(layer_norm_op(beta, gamma, x, y), 10000);
}

void benchmark_softmax(Device device) {
  std::vector<float> x_ = rand_vector(100 * 512);
  StorageView x({100, 512}, x_, device);
  StorageView y(x.device());
  const ops::SoftMax softmax_op{};
  BENCHMARK(softmax_op(x, y), 10000);
}

void benchmark_masked_softmax(Device device) {
  const dim_t batch_size = 32;
  const dim_t num_heads = 8;
  const dim_t max_source = 24;
  const dim_t max_target = 36;
  const dim_t rows = batch_size * num_heads * max_source;
  StorageView lengths(
      {rows},
      std::vector<int32_t>(rows, max_target - 5),
      device);
  StorageView x({batch_size, num_heads, max_source, max_target},
      rand_vector(batch_size * num_heads * max_source * max_target),
      device);
  StorageView y(x.device());
  const ops::SoftMax softmax_op{};
  BENCHMARK(softmax_op(x, lengths, y), 10000);
}

void benchmark_topk(Device device) {
  const size_t k = 4;
  const size_t batch_size = 8;
  const size_t vocab_size = 32000;
  std::vector<float> x = rand_vector(batch_size * k * vocab_size);
  StorageView input({batch_size, k * vocab_size}, x, device);
  StorageView values(input.dtype(), device);
  StorageView indices(DataType::INT32,  device);
  const ops::TopK op(k);
  BENCHMARK(op(input, values, indices), 2000);
}

void benchmark_gemm(Device device, DataType dtype) {
  DataType output_dtype = dtype != DataType::FLOAT32 ? DataType::INT32 : dtype;
  StorageView a({32 * 32, 512}, dtype, device);
  StorageView b({2048, 512}, dtype, device);
  StorageView c(output_dtype, device);
  const ops::Gemm gemm_op(1, 0, false, true);
  BENCHMARK(gemm_op(a, b, c), 1000);
}

void benchmark_quantize(Device device, DataType dtype) {
  StorageView x({32, 512}, rand_vector(32 * 512), device);
  StorageView y(dtype, device);
  StorageView scale(DataType::FLOAT32, device);
  const ops::Quantize quantize_op;
  BENCHMARK(quantize_op(x, y, scale), 10000);
}

void benchmark_dequantize(Device device) {
  StorageView x({64, 8192}, DataType::INT32, device);
  StorageView input_scale({32}, DataType::FLOAT32, device);
  StorageView weight_scale({8192}, DataType::FLOAT32, device);
  StorageView bias({8192}, DataType::FLOAT32, device);
  StorageView y(device);
  const ops::ActivationType activation_type = ops::ActivationType::ReLU;
  const ops::Dequantize dequantize_op(&activation_type);
  BENCHMARK(dequantize_op(x, input_scale, weight_scale, false, true, y, &bias), 10000);
}

void benchmark_conv1d(Device device) {
  StorageView x({1, 768, 3000}, DataType::FLOAT32, device);
  StorageView weight({768, 768, 3}, DataType::FLOAT32, device);
  StorageView bias({768}, DataType::FLOAT32, device);
  StorageView y(device);
  const ops::Conv1D conv_op{2, 1};
  BENCHMARK(conv_op(x, weight, bias, y), 100);
}

void benchmark_median_filter(Device device) {
  const dim_t width = 5;
  std::vector<float> x_ = rand_vector(100 * 512);
  StorageView x({100, 512}, x_, device);
  StorageView y(device);
  const ops::MedianFilter median_filter_op(width);
  BENCHMARK(median_filter_op(x, y), 10000);
}

void benchmark_snake(Device device) {
  StorageView x({1, 128, 1024}, 0.1f, device);
  StorageView alpha({128}, 1.f, device);
  StorageView y(device);
  const ops::Snake snake_op;
  BENCHMARK(snake_op(x, alpha, y), 1000);
}

void benchmark_moe_topk(Device device) {
  StorageView logits({32, 128}, rand_vector(32 * 128), device);
  StorageView indices(DataType::INT32, device);
  StorageView weights(device);
  const ops::MoeTopK moe_topk_op(8);
  BENCHMARK(moe_topk_op(logits, indices, weights), 1000);
}

void benchmark_dequantize_fp8(Device device) {
  StorageView x({1024, 1024}, int8_t(0x38), device);
  StorageView scale({8, 8}, 1.f, device);
  StorageView y(device);
  const ops::DequantizeFP8 dequantize_op;
  BENCHMARK(dequantize_op(x, scale, y), 100);
}

void benchmark_causal_depthwise_conv1d(Device device) {
  StorageView x({1, 1, 2048}, 0.1f, device);
  StorageView weight({2048, 4}, 0.1f, device);
  StorageView bias({2048}, 0.f, device);
  StorageView history({1, 2048, 3}, 0.f, device);
  StorageView y(device);
  const ops::CausalDepthwiseConv1D conv_op;
  BENCHMARK(conv_op(x, weight, &bias, history, y), 1000);
}

void benchmark_conv_transpose1d(Device device) {
  StorageView x({1, 128, 128}, 0.1f, device);
  StorageView weight({128, 64, 4}, 0.01f, device);
  StorageView bias({64}, 0.f, device);
  StorageView y(device);
  const ops::ConvTranspose1D conv_op(2);
  BENCHMARK(conv_op(x, weight, bias, y), 100);
}

#if defined(__x86_64__)
// Fused s8xs8 GEMV via VNNI vpdpbusd. Quantize fp32 input to u8 (shift by +128),
// dot with s8 weight using vpdpbusd, apply +128 compensation using precomputed
// weight row sums, dequantize by (act_scale * weight_scale[n]).
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")))
static void fused_vnni_gemv(const float* __restrict__ input,
                            const int8_t* __restrict__ weight,
                            const float* __restrict__ weight_scale,
                            const int32_t* __restrict__ weight_row_sum,
                            uint8_t* __restrict__ qinput_scratch,
                            float* __restrict__ output,
                            int k, int n, float alpha) {
    // 1) amax(input)
    __m512 vamax = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= k; i += 16) {
        __m512 v = _mm512_loadu_ps(input + i);
        vamax = _mm512_max_ps(vamax, _mm512_abs_ps(v));
    }
    float amax = _mm512_reduce_max_ps(vamax);
    for (; i < k; ++i) amax = std::max(amax, std::fabs(input[i]));
    const float act_scale = (amax > 0.f) ? (127.f / amax) : 1.f;

    // 2) quantize input to u8 shifted by +128
    __m512 vscale = _mm512_set1_ps(act_scale);
    __m512i v128 = _mm512_set1_epi32(128);
    for (i = 0; i + 16 <= k; i += 16) {
        __m512 v = _mm512_mul_ps(_mm512_loadu_ps(input + i), vscale);
        __m512i qi = _mm512_cvtps_epi32(v);
        qi = _mm512_add_epi32(qi, v128);
        // saturate to [0, 255] and pack to u8
        qi = _mm512_min_epi32(_mm512_max_epi32(qi, _mm512_setzero_si512()),
                              _mm512_set1_epi32(255));
        __m128i u8 = _mm512_cvtusepi32_epi8(qi);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(qinput_scratch + i), u8);
    }
    for (; i < k; ++i) {
        int q = static_cast<int>(std::nearbyintf(input[i] * act_scale)) + 128;
        qinput_scratch[i] = static_cast<uint8_t>(std::clamp(q, 0, 255));
    }

    // 3) VNNI dot over N (parallel)
    const float inv_act = 1.f / act_scale;
    #pragma omp parallel for schedule(static)
    for (int ni = 0; ni < n; ++ni) {
        const int8_t* wrow = weight + static_cast<size_t>(ni) * k;
        __m512i acc = _mm512_setzero_si512();
        int kk = 0;
        for (; kk + 64 <= k; kk += 64) {
            __m512i x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(qinput_scratch + kk));
            __m512i w = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(wrow + kk));
            acc = _mm512_dpbusd_epi32(acc, x, w);
        }
        int32_t sum = _mm512_reduce_add_epi32(acc);
        for (; kk < k; ++kk) {
            sum += static_cast<int32_t>(qinput_scratch[kk]) * static_cast<int32_t>(wrow[kk]);
        }
        // compensation: u8 = s8 + 128  =>  real_dot = sum - 128 * weight_row_sum[n]
        sum -= 128 * weight_row_sum[ni];
        output[ni] = alpha * static_cast<float>(sum) * inv_act / weight_scale[ni];
    }
}
#endif  // __x86_64__

void benchmark_dense_int8_m1(Device device) {
  if (device != Device::CPU) {
    std::cerr << "dense_int8_m1 is CPU-only" << std::endl;
    return;
  }
  struct Shape { dim_t k; dim_t n; const char* label; };
  std::vector<Shape> shapes = {
    {1024, 1024, "K=1024 N=1024 (attn)"},
    {1024, 3072, "K=1024 N=3072 (QKV)"},
    {1024, 4096, "K=1024 N=4096 (FFN up)"},
    {4096, 1024, "K=4096 N=1024 (FFN down)"},
  };

  for (const auto& sh : shapes) {
    const dim_t k = sh.k;
    const dim_t n = sh.n;
    std::cerr << "\n### " << sh.label << " ###" << std::endl;

    // M=1 fp32 input
    StorageView input({1, k}, rand_vector(k), device);
    // Random int8 weight [N, K] (trans_b=true layout)
    std::vector<float> weight_src = rand_vector(n * k);
    for (auto& v : weight_src) v = static_cast<int8_t>(static_cast<int>(v) % 127);
    StorageView weight_fp({n, k}, weight_src, device);
    StorageView weight(DataType::INT8, device);
    StorageView weight_scale(DataType::FLOAT32, device);
    ops::Quantize quantize_w;
    quantize_w(weight_fp, weight, weight_scale);

    StorageView output(device);
    const ops::ActivationType* no_act = nullptr;

    // Fast path (weight-only int8 directly).
    ops::Gemm gemm_op(1.f, 0.f, /*trans_a=*/false, /*trans_b=*/true);
    BENCHMARK(gemm_op.try_weight_only_int8(input, weight, weight_scale, output,
                                            no_act, nullptr, nullptr),
              2000);

    // Fallback path: quantize input, int8 GEMM, dequantize.
    StorageView qinput(DataType::INT8, device);
    StorageView qinput_scale(DataType::FLOAT32, device);
    StorageView qoutput(DataType::INT32, device);
    StorageView dq_output(device);
    ops::Quantize quantize_a;
    ops::Dequantize dequantize_op;
    auto fallback = [&]() {
      quantize_a(input, qinput, qinput_scale);
      gemm_op(qinput, weight, qoutput);
      dequantize_op(qoutput, qinput_scale, weight_scale,
                    /*trans_a=*/false, /*trans_b=*/true, dq_output);
    };
    BENCHMARK(fallback(), 2000);

#if defined(__x86_64__)
    // Fused VNNI (option 2): activation quantization + u8xs8 vpdpbusd + dequant.
    std::vector<int32_t> wrow_sum(n, 0);
    const int8_t* wptr = weight.data<int8_t>();
    const float* wscale_ptr = weight_scale.data<float>();
    for (dim_t ni = 0; ni < n; ++ni) {
      int32_t s = 0;
      for (dim_t ki = 0; ki < k; ++ki) s += wptr[ni * k + ki];
      wrow_sum[ni] = s;
    }
    std::vector<uint8_t> qinput_scratch(((k + 63) / 64) * 64);
    std::vector<float> vnni_output(n);
    const float* in_ptr = input.data<float>();
    BENCHMARK(fused_vnni_gemv(in_ptr, wptr, wscale_ptr, wrow_sum.data(),
                              qinput_scratch.data(), vnni_output.data(),
                              static_cast<int>(k), static_cast<int>(n), 1.f),
              2000);
#endif
  }
}

void benchmark_gated_delta_net(Device device) {
  StorageView query({1, 1, 4, 128}, 0.01f, device);
  StorageView key({1, 1, 4, 128}, 0.01f, device);
  StorageView value({1, 1, 8, 128}, 0.1f, device);
  StorageView log_decay({1, 1, 8}, -0.1f, device);
  StorageView beta({1, 1, 8}, 0.5f, device);
  StorageView state({1, 8, 128, 128}, 0.f, device);
  StorageView y(device);
  const ops::GatedDeltaNet gated_delta_net_op;
  BENCHMARK(gated_delta_net_op(query, key, value, log_decay, beta, state, y), 1000);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " op device [dtype]" << std::endl;
    return 1;
  }

  std::string op = argv[1];
  Device device = std::string(argv[2]) == "cuda" ? Device::CUDA : Device::CPU;
  std::string dtype_str = argc > 3 ? argv[3] : "float32";
  DataType dtype = DataType::FLOAT32;
  if (dtype_str == "int16")
    dtype = DataType::INT16;
  else if (dtype_str == "int8")
    dtype = DataType::INT8;

  if (op == "gather")
    benchmark_gather(device);
  else if (op == "transpose")
    benchmark_transpose(device);
  else if (op == "split")
    benchmark_split(device);
  else if (op == "layer_norm")
    benchmark_layer_norm(device);
  else if (op == "softmax")
    benchmark_softmax(device);
  else if (op == "masked_softmax")
    benchmark_masked_softmax(device);
  else if (op == "topk")
    benchmark_topk(device);
  else if (op == "gemm")
    benchmark_gemm(device, dtype);
  else if (op == "quantize")
    benchmark_quantize(device, dtype);
  else if (op == "dequantize")
    benchmark_dequantize(device);
  else if (op == "conv1d")
    benchmark_conv1d(device);
  else if (op == "median_filter")
    benchmark_median_filter(device);

  else if (op == "snake")
    benchmark_snake(device);
  else if (op == "moe_topk")
    benchmark_moe_topk(device);
  else if (op == "dequantize_fp8")
    benchmark_dequantize_fp8(device);
  else if (op == "causal_depthwise_conv1d")
    benchmark_causal_depthwise_conv1d(device);
  else if (op == "conv_transpose1d")
    benchmark_conv_transpose1d(device);
  else if (op == "gated_delta_net")
    benchmark_gated_delta_net(device);
  else if (op == "dense_int8_m1")
    benchmark_dense_int8_m1(device);

  return 0;
}
