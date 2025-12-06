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

    // Optimized im2col with better memory coalescing
    template <typename T>
    __global__ void im2col_1d_kernel_coalesced(
        const T* __restrict__ input,
        T* __restrict__ col_buffer,
        int batch_size,
        int in_channels,
        int input_length,
        int kernel_size,
        int stride,
        int padding,
        int dilation,
        int output_length) {
      
      // Thread processes one element in col_buffer
      // Layout: col_buffer[b][ic*ks + k][out_pos]
      // This ensures consecutive threads write consecutive memory (coalesced)
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int stride_y = gridDim.x * blockDim.x;
      
      int total = batch_size * in_channels * kernel_size * output_length;
      
      for (int i = idx; i < total; i += stride_y) {
        int out_pos = i % output_length;
        int k = (i / output_length) % kernel_size;
        int ic = (i / (output_length * kernel_size)) % in_channels;
        int b = i / (output_length * kernel_size * in_channels);
        
        int w_in = out_pos * stride - padding + k * dilation;
        
        T value;
        if (w_in >= 0 && w_in < input_length) {
          int input_idx = (b * in_channels + ic) * input_length + w_in;
          value = input[input_idx];
        } else {
          value = ArithOps<T>::zero();
        }
        
        col_buffer[i] = value;
      }
    }

    // Optimized direct convolution with shared memory
    template <typename T, int BLOCK_SIZE = 256>
    __global__ void conv1d_direct_shared(
        const T* __restrict__ input,
        const T* __restrict__ weight,
        const T* __restrict__ bias,
        T* __restrict__ output,
        int batch_size,
        int in_channels,
        int out_channels,
        int input_length,
        int output_length,
        int kernel_size,
        int stride,
        int padding,
        int dilation) {
      
      extern __shared__ char shared_mem[];
      T* shared_weight = reinterpret_cast<T*>(shared_mem);
      
      int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
      
      if (out_idx < batch_size * out_channels * output_length) {
        int w = out_idx % output_length;
        int oc = (out_idx / output_length) % out_channels;
        int b = out_idx / (output_length * out_channels);
        
        // Cooperatively load weights for this output channel
        int weight_size = in_channels * kernel_size;
        for (int i = threadIdx.x; i < weight_size; i += blockDim.x) {
          shared_weight[i] = weight[oc * weight_size + i];
        }
        __syncthreads();
        
        T sum = bias ? bias[oc] : ArithOps<T>::zero();
        
        #pragma unroll 4
        for (int ic = 0; ic < in_channels; ic++) {
          #pragma unroll
          for (int k = 0; k < kernel_size; k++) {
            int w_in = w * stride - padding + k * dilation;
            
            if (w_in >= 0 && w_in < input_length) {
              int input_idx = (b * in_channels + ic) * input_length + w_in;
              int weight_idx = ic * kernel_size + k;
              sum = ArithOps<T>::add(sum, ArithOps<T>::mul(input[input_idx], shared_weight[weight_idx]));
            }
          }
        }
        
        output[out_idx] = sum;
      }
    }

    // Broadcast bias initialization kernel
    template <typename T>
    __global__ void broadcast_bias_kernel(
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

    // Batched GEMM wrapper using strided batched GEMM
    template <typename T>
    void gemm_batched_cuda(
        cublasHandle_t handle,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        const T* a, int lda, long long stride_a,
        const T* b, int ldb, long long stride_b,
        T* c, int ldc, long long stride_c,
        int batch_count,
        float alpha_val = 1.0f,
        float beta_val = 0.0f);

    template <>
    void gemm_batched_cuda<float>(
        cublasHandle_t handle,
        bool trans_a, bool trans_b,
        int m, int n, int k,
        const float* a, int lda, long long stride_a,
        const float* b, int ldb, long long stride_b,
        float* c, int ldc, long long stride_c,
        int batch_count,
        float alpha_val,
        float beta_val) {
      
      CUBLAS_CHECK(cublasSgemmStridedBatched(
          handle,
          trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
          trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
          n, m, k,
          &alpha_val,
          b, ldb, stride_b,
          a, lda, stride_a,
          &beta_val,
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
        int batch_count,
        float alpha_val,
        float beta_val) {
      
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(
          handle,
          trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
          trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
          n, m, k,
          &alpha_val,
          b, CUDA_R_16F, ldb, stride_b,
          a, CUDA_R_16F, lda, stride_a,
          &beta_val,
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
        int batch_count,
        float alpha_val,
        float beta_val) {
      
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(
          handle,
          trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
          trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
          n, m, k,
          &alpha_val,
          b, CUDA_R_16BF, ldb, stride_b,
          a, CUDA_R_16BF, lda, stride_a,
          &beta_val,
          c, CUDA_R_16BF, ldc, stride_c,
          batch_count,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Main implementation with all optimizations
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
      // Heuristic: use direct convolution for small kernels and reasonable output sizes
      const bool use_direct = (kernel_size <= 7 && 
                               groups == 1 && 
                               output_length * out_channels <= 16384 &&
                               in_channels <= 512);

      if (use_direct) {
        // Direct convolution with shared memory
        int total_outputs = batch_size * out_channels * output_length;
        int threads = 256;
        int blocks = (total_outputs + threads - 1) / threads;
        
        size_t shared_mem_size = in_channels * kernel_size * sizeof(CudaT);
        
        conv1d_direct_shared<CudaT, 256><<<blocks, threads, shared_mem_size>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, out_channels,
            input_length, output_length, kernel_size,
            stride, padding, dilation);
            
      } else {
        // im2col + Batched GEMM approach
        size_t col_size = batch_size * in_channels * kernel_size * output_length;
        CudaT* col_buffer = static_cast<CudaT*>(
            get_allocator<Device::CUDA>().allocate(col_size * sizeof(CudaT)));
        
        // Optimized im2col with coalesced memory access
        int threads = 256;
        int blocks = std::min(65535, (int)((col_size + threads - 1) / threads));
        
        im2col_1d_kernel_coalesced<CudaT><<<blocks, threads>>>(
            input_ptr, col_buffer,
            batch_size, in_channels, input_length,
            kernel_size, stride, padding, dilation, output_length);
        
        // Initialize output with bias if present (allows using beta=1.0 in GEMM)
        if (bias_ptr) {
          int total = batch_size * out_channels * output_length;
          int bias_threads = 256;
          int bias_blocks = (total + bias_threads - 1) / bias_threads;
          broadcast_bias_kernel<CudaT><<<bias_blocks, bias_threads>>>(
              output_ptr, bias_ptr, batch_size, out_channels, output_length);
        }
        
        // Use strided batched GEMM for all batches and groups at once
        int out_channels_per_group = out_channels / groups;
        int total_batches = batch_size * groups;
        
        // Strides for batched GEMM
        long long stride_col = in_channels_per_group * kernel_size * output_length;
        long long stride_weight = out_channels_per_group * in_channels_per_group * kernel_size;
        long long stride_output = out_channels_per_group * output_length;
        
        // Adjust for group structure
        if (groups > 1) {
          // For groups, we need to handle the interleaving properly
          // Perform one batched GEMM per group
          for (int g = 0; g < groups; g++) {
            const CudaT* group_weight = weight_ptr + g * out_channels_per_group * in_channels_per_group * kernel_size;
            
            gemm_batched_cuda<CudaT>(
                cublas_handle,
                false, false,
                out_channels_per_group,
                output_length,
                in_channels_per_group * kernel_size,
                group_weight, in_channels_per_group * kernel_size, 0, // weight has no stride (shared across batches)
                col_buffer + g * in_channels_per_group * kernel_size * output_length, 
                output_length, 
                in_channels * kernel_size * output_length, // stride between batches
                output_ptr + g * out_channels_per_group * output_length, 
                output_length, 
                out_channels * output_length, // stride between batches
                batch_size,
                1.0f,
                bias_ptr ? 1.0f : 0.0f); // Use beta=1.0 to accumulate with bias
          }
        } else {
          // Single batched GEMM for all batches when groups=1
          gemm_batched_cuda<CudaT>(
              cublas_handle,
              false, false,
              out_channels,
              output_length,
              in_channels * kernel_size,
              weight_ptr, in_channels * kernel_size, 0,
              col_buffer, output_length, stride_col,
              output_ptr, output_length, out_channels * output_length,
              batch_size,
              1.0f,
              bias_ptr ? 1.0f : 0.0f);
        }
        
        get_allocator<Device::CUDA>().free(col_buffer);
      }
      
      CUDA_CHECK(cudaGetLastError());
    }

    // Template specializations
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
