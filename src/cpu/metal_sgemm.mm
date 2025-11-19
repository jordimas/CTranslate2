#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "metal_sgemm.h"
#include <stdexcept>
#include <memory>
#include <mutex>
#include <vector>
#include <condition_variable>
#include "ctranslate2/types.h"

namespace {
    // Buffer pool for a single matrix
    struct BufferSet {
        id<MTLBuffer> buf_a;
        id<MTLBuffer> buf_b;
        id<MTLBuffer> buf_c;
        MPSMatrix* mat_a;
        MPSMatrix* mat_b;
        MPSMatrix* mat_c;
        bool in_use;
        
        BufferSet() : buf_a(nil), buf_b(nil), buf_c(nil),
                      mat_a(nil), mat_b(nil), mat_c(nil), in_use(false) {}
    };
    
    struct MetalContext {
        id<MTLDevice> device;
        id<MTLCommandQueue> queue;
        
        // Double buffering: 2 sets of buffers
        static constexpr size_t NUM_BUFFER_SETS = 2;
        std::vector<BufferSet> buffer_sets;
        size_t current_set_index;
        
        std::mutex mtx;
        std::condition_variable cv;
        
        static MetalContext& instance() {
            static MetalContext ctx;
            return ctx;
        }
        
        // Get next available buffer set (blocks if all are in use)
        size_t acquire_buffer_set() {
            std::unique_lock<std::mutex> lock(mtx);
            
            // Wait until a buffer set is available
            cv.wait(lock, [this] {
                for (size_t i = 0; i < NUM_BUFFER_SETS; ++i) {
                    if (!buffer_sets[i].in_use) return true;
                }
                return false;
            });
            
            // Find and mark an available set
            for (size_t i = 0; i < NUM_BUFFER_SETS; ++i) {
                if (!buffer_sets[i].in_use) {
                    buffer_sets[i].in_use = true;
                    return i;
                }
            }
            
            return 0; // Should never reach here
        }
        
        void release_buffer_set(size_t index) {
            std::lock_guard<std::mutex> lock(mtx);
            buffer_sets[index].in_use = false;
            cv.notify_one();
        }
        
        // Allocate or resize buffers in a set
        void ensure_buffers(size_t set_index, size_t a_size, size_t b_size, size_t c_size,
                           dim_t a_rows, dim_t a_cols, dim_t b_rows, dim_t b_cols,
                           dim_t m, dim_t n, dim_t lda, dim_t ldb, dim_t ldc) {
            BufferSet& set = buffer_sets[set_index];
            
            // Check if we need to reallocate (size changed)
            bool need_realloc = false;
            if (set.buf_a == nil || [set.buf_a length] < a_size) need_realloc = true;
            if (set.buf_b == nil || [set.buf_b length] < b_size) need_realloc = true;
            if (set.buf_c == nil || [set.buf_c length] < c_size) need_realloc = true;
            
            if (need_realloc) {
                set.buf_a = [device newBufferWithLength:a_size
                                                options:MTLResourceStorageModeShared];
                set.buf_b = [device newBufferWithLength:b_size
                                                options:MTLResourceStorageModeShared];
                set.buf_c = [device newBufferWithLength:c_size
                                                options:MTLResourceStorageModeShared];
            }
            
            // ALWAYS recreate matrix descriptors with current dimensions
            // Even if buffers weren't reallocated, dimensions may have changed
            set.mat_a = [[MPSMatrix alloc] initWithBuffer:set.buf_a
                                               descriptor:[MPSMatrixDescriptor 
                                                   matrixDescriptorWithRows:a_rows
                                                                    columns:a_cols
                                                                   rowBytes:lda * sizeof(float)
                                                                   dataType:MPSDataTypeFloat32]];
            
            set.mat_b = [[MPSMatrix alloc] initWithBuffer:set.buf_b
                                               descriptor:[MPSMatrixDescriptor
                                                   matrixDescriptorWithRows:b_rows
                                                                    columns:b_cols
                                                                   rowBytes:ldb * sizeof(float)
                                                                   dataType:MPSDataTypeFloat32]];
            
            set.mat_c = [[MPSMatrix alloc] initWithBuffer:set.buf_c
                                               descriptor:[MPSMatrixDescriptor
                                                   matrixDescriptorWithRows:m
                                                                    columns:n
                                                                   rowBytes:ldc * sizeof(float)
                                                                   dataType:MPSDataTypeFloat32]];
        }
        
    private:
        MetalContext() : device(MTLCreateSystemDefaultDevice()), 
                         queue([device newCommandQueue]),
                         current_set_index(0) {
            buffer_sets.resize(NUM_BUFFER_SETS);
        }
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
        
        size_t a_size = a_rows * lda * sizeof(float);
        size_t b_size = b_rows * ldb * sizeof(float);
        size_t c_size = m * ldc * sizeof(float);
        
        // Acquire an available buffer set (blocks if all busy)
        size_t set_index = ctx.acquire_buffer_set();
        
        // Ensure buffers are allocated and properly sized
        ctx.ensure_buffers(set_index, a_size, b_size, c_size,
                          a_rows, a_cols, b_rows, b_cols,
                          m, n, lda, ldb, ldc);
        
        BufferSet& set = ctx.buffer_sets[set_index];
        
        // Copy input data to GPU buffers
        memcpy([set.buf_a contents], a, a_size);
        memcpy([set.buf_b contents], b, b_size);
        
        // Create command buffer and encode operation
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
                                  leftMatrix:set.mat_a
                                 rightMatrix:set.mat_b
                                resultMatrix:set.mat_c];
        
        // Add completion handler to copy results and release buffer set
        [cmd addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            // Copy result back to output
            memcpy(c, [set.buf_c contents], c_size);
            
            // Release this buffer set for reuse
            ctx.release_buffer_set(set_index);
        }];
        
        [cmd commit];
        
        // Wait for THIS specific command to complete
        // While waiting, other buffer sets can be prepared on CPU
        [cmd waitUntilCompleted];
    }
}