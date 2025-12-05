#include "ctranslate2/ops/conv1d.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    // Helper to get zero value for different types
    template <typename T>
    __device__ __forceinline__ T get_zero() {
      return T(0);
    }

    template <>
    __device__ __forceinline__ __half get_zero<__half>() {
      return __float2half(0.0f);
    }

    template <>
    __device__ __forceinline__ __nv_bfloat16 get_zero<__nv_bfloat16>() {
      return __float2bfloat16(0.0f);
    }

    // Half-precision arithmetic helpers
    template <typename T>
    __device__ __forceinline__ T add(T a, T b) {
      return a + b;
    }

    template <>
    __device__ __forceinline__ __half add<__half>(__half a, __half b) {
      return __hadd(a, b);
    }

    template <>
    __device__ __forceinline__ __nv_bfloat16 add<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) {
      return __hadd(a, b);
    }

    template <typename T>
    __device__ __forceinline__ T mul(T a, T b) {
      return a * b;
    }

    template <>
    __device__ __forceinline__ __half mul<__half>(__half a, __half b) {
      return __hmul(a, b);
    }

    template <>
    __device__ __forceinline__ __nv_bfloat16 mul<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) {
      return __hmul(a, b);
    }

    // Optimized im2col kernel - produces column-major output for cuBLAS
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
      
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = output_length * in_channels * kernel_size;
      
      if (idx < total) {
        int k = idx % kernel_size;
        int ic = (idx / kernel_size) % in_channels;
        int out_pos = idx / (kernel_size * in_channels);
        
        int w_in = out_pos * stride - padding + k * dilation;
        
        for (int b = 0; b < batch_size; b++) {
          int col_idx = b * in_channels * kernel_size * output_length + 
                        (ic * kernel_size + k) * output_length + out_pos;
          
          if (w_in >= 0 && w_in < input_length) {
            int input_idx = (b * in_channels + ic) * input_length + w_in;
            col_buffer[col_idx] = input[input_idx];
          } else {
            col_buffer[col_idx] = get_zero<T>();
          }
        }
      }
    }

    // Direct convolution kernel with proper half-precision arithmetic
    template <typename T>
    __global__ void conv1d_direct_kernel_optimized(
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
        int dilation,
        int groups) {
      
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total_outputs = batch_size * out_channels * output_length;
      
      if (idx < total_outputs) {
        int w = idx % output_length;
        int oc = (idx / output_length) % out_channels;
        int b = idx / (output_length * out_channels);
        
        int in_channels_per_group = in_channels / groups;
        int out_channels_per_group = out_channels / groups;
        int group = oc / out_channels_per_group;
        int ic_start = group * in_channels_per_group;
        int ic_end = ic_start + in_channels_per_group;
        
        T sum = bias ? bias[oc] : get_zero<T>();
        
        for (int ic = ic_start; ic < ic_end; ic++) {
          for (int k = 0; k < kernel_size; k++) {
            int w_in = w * stride - padding + k * dilation;
            
            if (w_in >= 0 && w_in < input_length) {
              int input_idx = (b * in_channels + ic) * input_length + w_in;
              int weight_idx = (oc * in_channels_per_group + (ic - ic_start)) * kernel_size + k;
              sum = add(sum, mul(input[input_idx], weight[weight_idx]));
            }
          }
        }
        
        output[idx] = sum;
      }
    }

    // Bias addition kernel
    template <typename T>
    __global__ void add_bias_kernel(
        T* __restrict__ output,
        const T* __restrict__ bias,
        int batch_size,
        int out_channels,
        int output_length) {
      
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = batch_size * out_channels * output_length;
      
      if (idx < total) {
        int oc = (idx / output_length) % out_channels;
        output[idx] = add(output[idx], bias[oc]);
      }
    }

    // GEMM wrappers for different types
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

    // Type conversion helpers
    template <typename CudaT, typename HostT>
    inline CudaT* to_cuda_type(HostT* ptr) {
      return reinterpret_cast<CudaT*>(ptr);
    }

    template <typename CudaT, typename HostT>
    inline const CudaT* to_cuda_type(const HostT* ptr) {
      return reinterpret_cast<const CudaT*>(ptr);
    }

    // Template implementation for all types
    template <typename CudaT, typename HostT>
    void conv1d_compute_impl(
        const Conv1D& op,
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

      const CudaT* input_ptr = to_cuda_type<const CudaT>(input.data<HostT>());
      const CudaT* weight_ptr = to_cuda_type<const CudaT>(weight.data<HostT>());
      const CudaT* bias_ptr = bias ? to_cuda_type<const CudaT>(bias->data<HostT>()) : nullptr;
      CudaT* output_ptr = to_cuda_type<CudaT>(output.data<HostT>());

      const bool use_direct = (kernel_size <= 5 && groups == 1 && output_length <= 512);

      if (use_direct) {
        int total_outputs = batch_size * out_channels * output_length;
        int threads = 256;
        int blocks = (total_outputs + threads - 1) / threads;
        
        conv1d_direct_kernel_optimized<CudaT><<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, out_channels,
            input_length, output_length, kernel_size,
            stride, padding, dilation, groups);
      } else {
        // im2col + GEMM approach
        size_t col_size = batch_size * in_channels_per_group * kernel_size * output_length * groups;
        CudaT* col_buffer = static_cast<CudaT*>(
            get_allocator<Device::CUDA>().allocate(col_size * sizeof(CudaT)));
        
        int total = output_length * in_channels * kernel_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        im2col_1d_kernel_optimized<CudaT><<<blocks, threads>>>(
            input_ptr, col_buffer,
            batch_size, in_channels, input_length,
            kernel_size, stride, padding, dilation, output_length,
            in_channels_per_group, groups);
        
        cublasHandle_t cublas_handle = cuda::get_cublas_handle();
        
        // Perform grouped GEMM
        for (int b = 0; b < batch_size; b++) {
          for (int g = 0; g < groups; g++) {
            int out_channels_per_group = out_channels / groups;
            
            CudaT* batch_output = output_ptr + 
                (b * out_channels + g * out_channels_per_group) * output_length;
            CudaT* batch_col = col_buffer + 
                b * in_channels * kernel_size * output_length + 
                g * in_channels_per_group * kernel_size * output_length;
            const CudaT* group_weight = weight_ptr + 
                g * out_channels_per_group * in_channels_per_group * kernel_size;
            
            // GEMM: output = weight * col
            // weight: [out_channels_per_group, in_channels_per_group * kernel_size]
            // col: [in_channels_per_group * kernel_size, output_length]
            // output: [out_channels_per_group, output_length]
            gemm_cuda<CudaT>(
                cublas_handle,
                false, false,  // No transpose
                out_channels_per_group,
                output_length,
                in_channels_per_group * kernel_size,
                group_weight, in_channels_per_group * kernel_size,
                batch_col, output_length,
                batch_output, output_length);
          }
        }
        
        if (bias_ptr) {
          int total = batch_size * out_channels * output_length;
          int threads = 256;
          int blocks = (total + threads - 1) / threads;
          add_bias_kernel<CudaT><<<blocks, threads>>>(
              output_ptr, bias_ptr, batch_size, out_channels, output_length);
        }
        
        get_allocator<Device::CUDA>().free(col_buffer);
      }
      
      CUDA_CHECK(cudaGetLastError());
    }

    // Main compute functions
    template <>
    void Conv1D::compute<Device::CUDA, float>(
        const StorageView& input,
        const StorageView& weight,
        const StorageView* bias,
        StorageView& output,
        const StorageView* qscale) const {
      
      if (qscale)
        throw std::runtime_error("Quantization is not supported in this Conv1D implementation");
      
      conv1d_compute_impl<float, float>(*this, input, weight, bias, output,
                                        _stride, _padding, _dilation, _groups);
    }

    template <>
    void Conv1D::compute<Device::CUDA, float16_t>(
        const StorageView& input,
        const StorageView& weight,
        const StorageView* bias,
        StorageView& output,
        const StorageView* qscale) const {
      
      if (qscale)
        throw std::runtime_error("Quantization is not supported in this Conv1D implementation");
      
      conv1d_compute_impl<__half, float16_t>(*this, input, weight, bias, output,
                                             _stride, _padding, _dilation, _groups);
    }

    template <>
    void Conv1D::compute<Device::CUDA, bfloat16_t>(
        const StorageView& input,
        const StorageView& weight,
        const StorageView* bias,
        StorageView& output,
        const StorageView* qscale) const {
      
      if (qscale)
        throw std::runtime_error("Quantization is not supported in this Conv1D implementation");
      
      conv1d_compute_impl<__nv_bfloat16, bfloat16_t>(*this, input, weight, bias, output,
                                                     _stride, _padding, _dilation, _groups);
    }

  }
}
