#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "metal_sgemm.h"
#include <stdexcept>
#include <memory>
#include <mutex>
#include "ctranslate2/types.h"  // Add this line

/*void metal_sgemm(bool a_is_packed, bool b_is_packed,
                 bool transpose_a, bool transpose_b,
                 dim_t m, dim_t n, dim_t k,
                 float alpha,
                 const float* a, dim_t lda,
                 const float* b, dim_t ldb,
                 float beta,
                 float* c, dim_t ldc,
                 const float* a_shift_compensation) {

    return;
}*/



namespace {
    struct MetalContext {
        id<MTLDevice> device;
        id<MTLCommandQueue> queue;
        
        static MetalContext& instance() {
            static MetalContext ctx;
            return ctx;
        }
        
    private:
        MetalContext() : device(MTLCreateSystemDefaultDevice()), 
                         queue([device newCommandQueue]) {}
    };
}

void metal_sgemm(bool a_is_packed, bool b_is_packed, bool transpose_a, 
                     bool transpose_b, dim_t m, dim_t n, dim_t k,
                     float alpha, const float* a, dim_t lda,
                     const float* b, dim_t ldb, float beta,
                     float* c, dim_t ldc, const float* a_shift_compensation) {
    
    @autoreleasepool {
        MetalContext& ctx = MetalContext::instance();
        
        dim_t a_rows = transpose_a ? k : m;
        dim_t a_cols = transpose_a ? m : k;
        dim_t b_rows = transpose_b ? n : k;
        dim_t b_cols = transpose_b ? k : n;
        
        id<MTLBuffer> buf_a = [ctx.device newBufferWithBytes:a 
                                                      length:a_rows * lda * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b = [ctx.device newBufferWithBytes:b
                                                      length:b_rows * ldb * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_c = [ctx.device newBufferWithBytes:c
                                                      length:m * ldc * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        
        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a
                                                  descriptor:[MPSMatrixDescriptor 
                                                      matrixDescriptorWithRows:a_rows
                                                                       columns:a_cols
                                                                      rowBytes:lda * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32]];
        
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b
                                                  descriptor:[MPSMatrixDescriptor
                                                      matrixDescriptorWithRows:b_rows
                                                                       columns:b_cols
                                                                      rowBytes:ldb * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32]];
        
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c
                                                  descriptor:[MPSMatrixDescriptor
                                                      matrixDescriptorWithRows:m
                                                                       columns:n
                                                                      rowBytes:ldc * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32]];
        
        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        
        [[[MPSMatrixMultiplication alloc]
            initWithDevice:ctx.device
            transposeLeft:transpose_a
            transposeRight:transpose_b
            resultRows:m
            resultColumns:n
            interiorColumns:k
            alpha:alpha
            beta:beta] encodeToCommandBuffer:cmd
                                  leftMatrix:mat_a
                                 rightMatrix:mat_b
                                resultMatrix:mat_c];
        
        [cmd commit];
        [cmd waitUntilCompleted];
        
        memcpy(c, [buf_c contents], m * ldc * sizeof(float));
    }
}