#pragma once
#include "ctranslate2/types.h"
#include <cstdint>
#include <string>
namespace ctranslate2 { namespace experimental {
bool int8_pack_enabled();
bool pack_int8_weight_name(const std::string& name);
dim_t pack_int8_weights(const int8_t* src, bool transpose, dim_t k, dim_t n,
                        float alpha, int8_t* dst);
void packed_int8_gemm(bool transpose_a, bool transpose_b, dim_t m, dim_t n, dim_t k,
                      float alpha, const int8_t* a, dim_t lda, const int8_t* b, dim_t ldb,
                      float beta, int32_t* c, dim_t ldc, const int32_t* compensation);
}}
