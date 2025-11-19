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
        bool in_use;
        
        // Track current dimensions
        dim_t curr_a_rows, curr_a_cols, curr_lda;
        dim_t curr_b_rows, curr_b_cols, curr_ldb;
        dim_t curr_m, curr_n, curr_ldc;
        
        BufferSet() : buf_a(nil), buf_b(nil), buf_c(nil), in_use(false),
                      curr_a_rows(0), curr_a_cols(0), curr_lda(0),
                      curr_b_rows(0), curr_b_cols(0), curr_ldb(0),
                      curr_m(0), curr_n(0), curr_ldc(0) {}
        
        ~BufferSet() {
            buf_a = nil;
            buf_b = nil;
            buf_c = nil;
        }
    };
    
    struct MetalContext {
        id<MTLDevice> device;
        id<MTLCommandQueue> queue;
        
        // Double buffering: 2 sets of buffers
        static constexpr size_t NUM_BUFFER_SETS = 2;
        std::vector<BufferSet> buffer_sets;
        
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
            if (index < buffer_sets.size()) {
                buffer_sets[index].in_use = false;
                cv.notify_one();
            }
        }
        
        // Allocate or resize buffers in a set
        void ensure_buffers(size_t set_index, size_t a_size, size_t b_size, size_t c_size) {
            BufferSet& set = buffer_sets[set_index];
            
            // Check if we need to reallocate buffers (size changed)
            if (set.buf_a == nil || [set.buf_a length] < a_size) {
                set.buf_a = [device newBufferWithLength:a_size
                                                options:MTLResourceStorageModeShared];
            }
            if (set.buf_b == nil || [set.buf_b length] < b_size) {
                set.buf_b = [device newBufferWithLength:b_size
                                                options:MTLResourceStorageModeShared];
            }
            if (set.buf_c == nil || [set.buf_c length] < c_size) {
                set.buf_c = [device newBufferWithLength:c_size
                                                options:MTLResourceStorageModeShared];
            }
        }
        
    private:
        MetalContext() : device(MTLCreateSystemDefaultDevice()), 
                         queue([device newCommandQueue]) {
            if (!device) {
                throw std::runtime_error("Failed to create Metal device");
            }
            if (!queue) {
                throw std::runtime_error("Failed to create Metal command queue");
            }
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
        ctx.ensure_buffers(set_index, a_size, b_size, c_size);
        
        BufferSet& set = ctx.buffer_sets[set_index];
        
        // Copy input data to GPU buffers
        memcpy([set.buf_a contents], a, a_size);
        memcpy([set.buf_b contents], b, b_size);
        
        // Create matrix descriptors (these are lightweight and autoreleased)
        MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor 
            matrixDescriptorWithRows:a_rows
                             columns:a_cols
                            rowBytes:lda * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        
        MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor
            matrixDescriptorWithRows:b_rows
                             columns:b_cols
                            rowBytes:ldb * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        
        MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor
            matrixDescriptorWithRows:m
                             columns:n
                            rowBytes:ldc * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        
        // Create matrices using the descriptors
        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:set.buf_a
                                                  descriptor:desc_a];
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:set.buf_b
                                                  descriptor:desc_b];
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:set.buf_c
                                                  descriptor:desc_c];
        
        // Create command buffer and encode operation
        id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
        
        MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:ctx.device
            transposeLeft:transpose_a
            transposeRight:transpose_b
            resultRows:m
            resultColumns:n
            interiorColumns:k
            alpha:alpha
            beta:beta];
        
        [matmul encodeToCommandBuffer:cmd
                           leftMatrix:mat_a
                          rightMatrix:mat_b
                          resultMatrix:mat_c];
        
        [cmd commit];
        
        // Wait for completion synchronously
        [cmd waitUntilCompleted];
        
        // Copy result back after completion
        memcpy(c, [set.buf_c contents], c_size);
        
        // Release this buffer set for reuse
        ctx.release_buffer_set(set_index);
        
        // Clean up matrix objects (buffers are retained in BufferSet)
        mat_a = nil;
        mat_b = nil;
        mat_c = nil;
        matmul = nil;
    }
}