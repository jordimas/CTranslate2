#include "ctranslate2/ops/conv1d.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    // Unified arithmetic operations
    template <typename T>
    struct ArithOps {
      __device__ static T zero() { return T(0); }
      __device__ static T add(T a, T b) { return a + b; }
      __device__ static T mul(T a, T b) { return a * b; }
    };

    template <>
    struct ArithOps<__half> {
      __device__ static __half zero() { return __float2half(0.0f); }
      __device__ static __half add(__half a, __half b) { return __hadd(a, b); }
      __device__ static __half mul(__half a, __half b) { return __hmul(a, b); }
    };

    template <>
    struct ArithOps<__nv_bfloat16> {
      __device__ static __nv_bfloat16 zero() { return __float2bfloat16(0.0f); }
      __device__ static __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }
      __device__ static __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
    };

    // Optimized im2col - parallel over all batches with coalesced memory access
    template <typename T>
    __global__ void im2col_1d_kernel_optimized(
        const T* __restrict__ input,
        T* __restrict__ col_buffer,
        int batch_size,
        int in_channels,
        int input_length,
        int kernel_size,
        int stride,
        int padding,
        int dilation,
        int output_length,
        int in_channels_per_group,
        int groups) {
      
      // Parallelize over ALL elements: batch * in_channels * kernel_size * output_length
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = batch_size * in_channels * kernel_size * output_length;
      
      if (idx < total) {
        // Unpack indices - innermost is output_length for coalesced writes
        int out_pos = idx % output_length;
        int k = (idx / output_length) % kernel_size;
        int ic = (idx / (output_length * kernel_size)) % in_channels;
        int b = idx / (output_length * kernel_size * in_channels);
        
        int w_in = out_pos * stride - padding + k * dilation;
        
        int col_idx = b * in_channels * kernel_size * output_length + 
                      (ic * kernel_size + k) * output_length + out_pos;
        
        if (w_in >= 0 && w_in < input_length) {
          int input_idx = (b * in_channels + ic) * input_length + w_in;
          col_buffer[col_idx] = input[input_idx];
        } else {
          col_buffer[col_idx] = ArithOps<T>::zero();
        }
      }
    }

    // Broadcast bias to output before GEMM
    template <typename T>
    __global__ void broadcast_bias_for_gemm(
        T* __restrict__ output,
        const T* __restrict__ bias,
        int batch_size,
        int out_channels,
        int output_length) {
      
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = batch_size * out_channels * output_length;
      
      if (idx < total) {
        int oc = (idx / output_length) % out_channels;
        output[idx] = bias[oc];
      }
    }

    // Single GEMM wrappers (keep original logic)
    template <typename T>
    void gemm_cuda(
        cublasHandle_t handle,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        const T* a, int lda,
        const T* b, int ldb,
        T* c, int ldc);

    template <>
    void gemm_cuda<float>(
        cublasHandle_t handle,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        const float* a, int lda,
        const float* b, int ldb,
        float* c, int ldc) {
      
      float alpha = 1.0f;
      float beta = 0.0f;
      
      CUBLAS_CHECK(cublasSgemm(
          handle,
          trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
          trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
          n, m, k,
          &alpha,
          b, ldb,
          a, lda,
          &beta,
          c, ldc));
    }

    template <>
    void gemm_cuda<__half>(
        cublasHandle_t handle,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        const __half* a, int lda,
        const __half* b, int ldb,
        __half* c, int ldc) {
      
      float alpha = 1.0f;
      float beta = 0.0f;
      
      CUBLAS_CHECK(cublasGemmEx(
          handle,
          trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
          trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
          n, m, k,
          &alpha,
          b, CUDA_R_16F, ldb,
          a, CUDA_R_16F, lda,
          &beta,
          c, CUDA_R_16F, ldc,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    template <>
    void gemm_cuda<__nv_bfloat16>(
        cublasHandle_t handle,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        const __nv_bfloat16* a, int lda,
        const __nv_bfloat16* b, int ldb,
        __nv_bfloat16* c, int ldc) {
      
      float alpha = 1.0f;
      float beta = 0.0f;
      
      CUBLAS_CHECK(cublasGemmEx(
          handle,
          trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
          trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
          n, m, k,
          &alpha,
          b, CUDA_R_16BF, ldb,
          a, CUDA_R_16BF, lda,
          &beta,
          c, CUDA_R_16BF, ldc,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Batched GEMM wrappers
    template <typename T>
    void gemm_batched_cuda(
        cublasHandle_t handle,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        const T* a, int lda, long long stride_a,
        const T* b, int ldb, long long stride_b,
        T* c, int ldc, long long stride_c,
        int batch_count);

    template <>
    void gemm_batched_cuda<float>(
        cublasHandle_t handle,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        const float* a, int lda, long long stride_a,
        const float* b, int ldb, long long stride_b,
        float* c, int ldc, long long stride_c,
        int batch_count) {
      
      float alpha = 1.0f;
      float beta = 0.0f;
      
      CUBLAS_CHECK(cublasSgemmStridedBatched(
          handle,
          trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
          trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
          n, m, k,
          &alpha,
          b, ldb, stride_b,
          a, lda, stride_a,
          &beta,
          c, ldc, stride_c,
          batch_count));
    }

    template <>
    void gemm_batched_cuda<__half>(
        cublasHandle_t handle,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        const __half* a, int lda, long long stride_a,
        const __half* b, int ldb, long long stride_b,
        __half* c, int ldc, long long stride_c,
        int batch_count) {
      
      float alpha = 1.0f;
      float beta = 0.0f;
      
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(
          handle,
          trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
          trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
          n, m, k,
          &alpha,
          b, CUDA_R_16F, ldb, stride_b,
          a, CUDA_R_16F, lda, stride_a,
          &beta,
          c, CUDA_R_16F, ldc, stride_c,
          batch_count,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    template <>
    void gemm_batched_cuda<__nv_bfloat16>(
        cublasHandle_t handle,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        const __nv_bfloat16* a, int lda, long long stride_a,
        const __nv_bfloat16* b, int ldb, long long stride_b,
        __nv_bfloat16* c, int ldc, long long stride_c,
        int batch_count) {
      
      float alpha = 1.0f;
      float beta = 0.0f;
      
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(
          handle,
          trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
          trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
          n, m, k,
          &alpha,
          b, CUDA_R_16BF, ldb, stride_b,
          a, CUDA_R_16BF, lda, stride_a,
          &beta,
          c, CUDA_R_16BF, ldc, stride_c,
          batch_count,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Main implementation - using batched GEMM instead of loop
    template <typename CudaT, typename HostT>
    void conv1d_compute_impl(
        const StorageView& input,
        const StorageView& weight,
        const StorageView* bias,
        StorageView& output,
        int stride,
        int padding,
        int dilation,
        int groups) {
      
      const int batch_size = input.dim(0);
      const int in_channels = input.dim(1);
      const int input_length = input.dim(2);
      const int output_length = output.dim(2);
      const int out_channels = weight.dim(0);
      const int in_channels_per_group = weight.dim(1);
      const int kernel_size = weight.dim(2);

      const CudaT* input_ptr = reinterpret_cast<const CudaT*>(input.data<HostT>());
      const CudaT* weight_ptr = reinterpret_cast<const CudaT*>(weight.data<HostT>());
      const CudaT* bias_ptr = bias ? reinterpret_cast<const CudaT*>(bias->data<HostT>()) : nullptr;
      CudaT* output_ptr = reinterpret_cast<CudaT*>(output.data<HostT>());

      cublasHandle_t cublas_handle = cuda::get_cublas_handle();
      
      // Enable tensor core optimizations
#if CUDA_VERSION >= 11000
      cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_TENSOR_OP_MATH);
#elif CUDA_VERSION >= 9000
      cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
#endif

      // im2col + GEMM approach
      size_t col_size = batch_size * in_channels_per_group * kernel_size * output_length * groups;
      CudaT* col_buffer = static_cast<CudaT*>(
          get_allocator<Device::CUDA>().allocate(col_size * sizeof(CudaT)));
      
      // Launch im2col with parallelism over ALL batches
      int total = batch_size * in_channels * kernel_size * output_length;
      int threads = 256;
      int blocks = (total + threads - 1) / threads;
      
      im2col_1d_kernel_optimized<CudaT><<<blocks, threads>>>(
          input_ptr, col_buffer,
          batch_size, in_channels, input_length,
          kernel_size, stride, padding, dilation, output_length,
          in_channels_per_group, groups);
      
      // Perform grouped GEMM with optimized bias handling
      int out_channels_per_group = out_channels / groups;
      
      if (groups == 1) {
        // Single group: use batched GEMM with fused bias
        long long stride_col = in_channels * kernel_size * output_length;
        long long stride_output = out_channels * output_length;
        
        // Broadcast bias first if present
        if (bias_ptr) {
          int total = batch_size * out_channels * output_length;
          int bias_threads = 256;
          int bias_blocks = (total + bias_threads - 1) / bias_threads;
          broadcast_bias_for_gemm<CudaT><<<bias_blocks, bias_threads>>>(
              output_ptr, bias_ptr, batch_size, out_channels, output_length);
        }
        
        // Then GEMM with beta=1.0 to accumulate with bias
        float alpha = 1.0f;
        float beta = bias_ptr ? 1.0f : 0.0f;
        
        if (std::is_same<CudaT, float>::value) {
          CUBLAS_CHECK(cublasSgemmStridedBatched(
              cublas_handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              output_length, out_channels_per_group, in_channels_per_group * kernel_size,
              &alpha,
              reinterpret_cast<const float*>(col_buffer), output_length, stride_col,
              reinterpret_cast<const float*>(weight_ptr), in_channels_per_group * kernel_size, 0,
              &beta,
              reinterpret_cast<float*>(output_ptr), output_length, stride_output,
              batch_size));
        } else if (std::is_same<CudaT, __half>::value) {
          CUBLAS_CHECK(cublasGemmStridedBatchedEx(
              cublas_handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              output_length, out_channels_per_group, in_channels_per_group * kernel_size,
              &alpha,
              col_buffer, CUDA_R_16F, output_length, stride_col,
              weight_ptr, CUDA_R_16F, in_channels_per_group * kernel_size, 0,
              &beta,
              output_ptr, CUDA_R_16F, output_length, stride_output,
              batch_size,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else if (std::is_same<CudaT, __nv_bfloat16>::value) {
          CUBLAS_CHECK(cublasGemmStridedBatchedEx(
              cublas_handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              output_length, out_channels_per_group, in_channels_per_group * kernel_size,
              &alpha,
              col_buffer, CUDA_R_16BF, output_length, stride_col,
              weight_ptr, CUDA_R_16BF, in_channels_per_group * kernel_size, 0,
              &beta,
              output_ptr, CUDA_R_16BF, output_length, stride_output,
              batch_size,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        
      } else {
        // Multiple groups: need to handle bias differently
        // First broadcast all biases if present
        if (bias_ptr) {
          int total = batch_size * out_channels * output_length;
          int bias_threads = 256;
          int bias_blocks = (total + bias_threads - 1) / bias_threads;
          broadcast_bias_for_gemm<CudaT><<<bias_blocks, bias_threads>>>(
              output_ptr, bias_ptr, batch_size, out_channels, output_length);
        }
        
        // Then process each group
        for (int g = 0; g < groups; g++) {
          const CudaT* group_weight = weight_ptr + 
              g * out_channels_per_group * in_channels_per_group * kernel_size;
          
          long long stride_col = in_channels * kernel_size * output_length;
          long long stride_output = out_channels * output_length;
          
          CudaT* batch_col = col_buffer + g * in_channels_per_group * kernel_size * output_length;
          CudaT* batch_output = output_ptr + g * out_channels_per_group * output_length;
          
          // GEMM for this group with beta=1.0 if bias present
          float alpha = 1.0f;
          float beta = bias_ptr ? 1.0f : 0.0f;
          
          if (std::is_same<CudaT, float>::value) {
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                output_length, out_channels_per_group, in_channels_per_group * kernel_size,
                &alpha,
                reinterpret_cast<const float*>(batch_col), output_length, stride_col,
                reinterpret_cast<const float*>(group_weight), in_channels_per_group * kernel_size, 0,
                &beta,
                reinterpret_cast<float*>(batch_output), output_length, stride_output,
                batch_size));
          } else if (std::is_same<CudaT, __half>::value) {
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                output_length, out_channels_per_group, in_channels_per_group * kernel_size,
                &alpha,
                batch_col, CUDA_R_16F, output_length, stride_col,
                group_weight, CUDA_R_16F, in_channels_per_group * kernel_size, 0,
                &beta,
                batch_output, CUDA_R_16F, output_length, stride_output,
                batch_size,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
          } else if (std::is_same<CudaT, __nv_bfloat16>::value) {
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                output_length, out_channels_per_group, in_channels_per_group * kernel_size,
                &alpha,
                batch_col, CUDA_R_16BF, output_length, stride_col,
                group_weight, CUDA_R_16BF, in_channels_per_group * kernel_size, 0,
                &beta,
                batch_output, CUDA_R_16BF, output_length, stride_output,
                batch_size,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
          }
        }
      }
      
      get_allocator<Device::CUDA>().free(col_buffer);
      
      CUDA_CHECK(cudaGetLastError());
    }

    // Consolidated compute using macro
    #define CONV1D_COMPUTE_SPECIALIZATION(HostT, CudaT)                \
      template <>                                                      \
      void Conv1D::compute<Device::CUDA, HostT>(                      \
          const StorageView& input,                                    \
          const StorageView& weight,                                   \
          const StorageView* bias,                                     \
          StorageView& output,                                         \
          const StorageView* qscale) const {                          \
                                                                       \
        if (qscale)                                                    \
          throw std::runtime_error(                                    \
              "Quantization is not supported in this Conv1D implementation"); \
                                                                       \
        conv1d_compute_impl<CudaT, HostT>(input, weight, bias, output, \
                                         _stride, _padding, _dilation, _groups); \
      }

    CONV1D_COMPUTE_SPECIALIZATION(float, float)
    CONV1D_COMPUTE_SPECIALIZATION(float16_t, __half)
    CONV1D_COMPUTE_SPECIALIZATION(bfloat16_t, __nv_bfloat16)

    #undef CONV1D_COMPUTE_SPECIALIZATION

  }
}
