#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "metal_sgemm.h"
#include <stdexcept>
#include <memory>
#include <mutex>
#include "ctranslate2/types.h"  // Add this line

void metal_sgemm(bool a_is_packed, bool b_is_packed,
                 bool transpose_a, bool transpose_b,
                 dim_t m, dim_t n, dim_t k,
                 float alpha,
                 const float* a, dim_t lda,
                 const float* b, dim_t ldb,
                 float beta,
                 float* c, dim_t ldc,
                 const float* a_shift_compensation) {

    return;
}