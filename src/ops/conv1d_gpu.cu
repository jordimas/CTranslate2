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

    // Optimized im2col kernel - works with native CUDA types
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
        int output_length) {
      
      int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
      
      if (out_pos < output_length) {
        for (int b = 0; b < batch_size; b++) {
          for (int c = 0; c < in_channels; c++) {
            for (int k = 0; k < kernel_size; k++) {
              int w_in = out_pos * stride - padding + k * dilation;
              int col_idx = ((b * in_channels + c) * kernel_size + k) * output_length + out_pos;
              
              if (w_in >= 0 && w_in < input_length) {
                int input_idx = (b * in_channels + c) * input_length + w_in;
                col_buffer[col_idx] = input[input_idx];
              } else {
                col_buffer[col_idx] = get_zero<T>();
              }
            }
          }
        }
      }
    }

    // Direct convolution kernel - uses native CUDA types
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
              sum = sum + input[input_idx] * weight[weight_idx];
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
        int out_channels,
        int output_length) {
      
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = out_channels * output_length;
      
      if (idx < total) {
        int oc = idx / output_length;
        output[idx] = output[idx] + bias[oc];
      }
    }

    // GEMM wrappers for different types
    template <typename T>
    void gemm_cuda(
        cublasHandle_t handle,
        int m, int n, int k,
        const T* a,
        const T* b,
        T* c);

    template <>
    void gemm_cuda<float>(
        cublasHandle_t handle,
        int m, int n, int k,
        const float* a,
        const float* b,
        float* c) {
      
      float alpha = 1.0f;
      float beta = 0.0f;
      
      CUBLAS_CHECK(cublasSgemm(
          handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          n, m, k,
          &alpha,
          b, n,
          a, k,
          &beta,
          c, n));
    }

    template <>
    void gemm_cuda<__half>(
        cublasHandle_t handle,
        int m, int n, int k,
        const __half* a,
        const __half* b,
        __half* c) {
      
      __half alpha = __float2half(1.0f);
      __half beta = __float2half(0.0f);
      
      CUBLAS_CHECK(cublasHgemm(
          handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          n, m, k,
          &alpha,
          b, n,
          a, k,
          &beta,
          c, n));
    }

    template <>
    void gemm_cuda<__nv_bfloat16>(
        cublasHandle_t handle,
        int m, int n, int k,
        const __nv_bfloat16* a,
        const __nv_bfloat16* b,
        __nv_bfloat16* c) {
      
      float alpha = 1.0f;
      float beta = 0.0f;
      
      CUBLAS_CHECK(cublasGemmEx(
          handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          n, m, k,
          &alpha,
          b, CUDA_R_16BF, n,
          a, CUDA_R_16BF, k,
          &beta,
          c, CUDA_R_16BF, n,
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

    // Main compute function for float
    template <>
    void Conv1D::compute<Device::CUDA, float>(
        const StorageView& input,
        const StorageView& weight,
        const StorageView* bias,
        StorageView& output,
        const StorageView* qscale) const {
      
      if (qscale)
        throw std::runtime_error("Quantization is not supported in this Conv1D implementation");

      const int batch_size = input.dim(0);
      const int in_channels = input.dim(1);
      const int input_length = input.dim(2);
      const int output_length = output.dim(2);
      const int out_channels = weight.dim(0);
      const int kernel_size = weight.dim(2);

      const float* input_ptr = input.data<float>();
      const float* weight_ptr = weight.data<float>();
      const float* bias_ptr = bias ? bias->data<float>() : nullptr;
      float* output_ptr = output.data<float>();

      const bool use_direct = (kernel_size <= 5 && _groups == 1 && output_length <= 512);

      if (use_direct) {
        int total_outputs = batch_size * out_channels * output_length;
        int threads = 256;
        int blocks = (total_outputs + threads - 1) / threads;
        
        conv1d_direct_kernel_optimized<float><<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, out_channels,
            input_length, output_length, kernel_size,
            _stride, _padding, _dilation, _groups);
      } else {
        size_t col_size = batch_size * in_channels * kernel_size * output_length;
        float* col_buffer = static_cast<float*>(
            get_allocator<Device::CUDA>().allocate(col_size * sizeof(float)));
        
        int threads = 256;
        int blocks = (output_length + threads - 1) / threads;
        
        im2col_1d_kernel_optimized<float><<<blocks, threads>>>(
            input_ptr, col_buffer,
            batch_size, in_channels, input_length,
            kernel_size, _stride, _padding, _dilation, output_length);
        
        cublasHandle_t cublas_handle = cuda::get_cublas_handle();
        
        for (int b = 0; b < batch_size; b++) {
          float* batch_output = output_ptr + b * out_channels * output_length;
          float* batch_col = col_buffer + b * in_channels * kernel_size * output_length;
          
          gemm_cuda<float>(
              cublas_handle,
              out_channels,
              output_length,
              in_channels * kernel_size,
              weight_ptr,
              batch_col,
              batch_output);
        }
        
        if (bias_ptr) {
          int total = batch_size * out_channels * output_length;
          int threads = 256;
          int blocks = (total + threads - 1) / threads;
          add_bias_kernel<float><<<blocks, threads>>>(
              output_ptr, bias_ptr, out_channels, output_length);
        }
        
        get_allocator<Device::CUDA>().free(col_buffer);
      }
      
      CUDA_CHECK(cudaGetLastError());
    }

    // Main compute function for float16_t (converts to __half)
    template <>
    void Conv1D::compute<Device::CUDA, float16_t>(
        const StorageView& input,
        const StorageView& weight,
        const StorageView* bias,
        StorageView& output,
        const StorageView* qscale) const {
      
      if (qscale)
        throw std::runtime_error("Quantization is not supported in this Conv1D implementation");

      const int batch_size = input.dim(0);
      const int in_channels = input.dim(1);
      const int input_length = input.dim(2);
      const int output_length = output.dim(2);
      const int out_channels = weight.dim(0);
      const int kernel_size = weight.dim(2);

      const __half* input_ptr = to_cuda_type<const __half>(input.data<float16_t>());
      const __half* weight_ptr = to_cuda_type<const __half>(weight.data<float16_t>());
      const __half* bias_ptr = bias ? to_cuda_type<const __half>(bias->data<float16_t>()) : nullptr;
      __half* output_ptr = to_cuda_type<__half>(output.data<float16_t>());

      const bool use_direct = (kernel_size <= 5 && _groups == 1 && output_length <= 512);

      if (use_direct) {
        int total_outputs = batch_size * out_channels * output_length;
        int threads = 256;
        int blocks = (total_outputs + threads - 1) / threads;
        
        conv1d_direct_kernel_optimized<__half><<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, out_channels,
            input_length, output_length, kernel_size,
            _stride, _padding, _dilation, _groups);
      } else {
        size_t col_size = batch_size * in_channels * kernel_size * output_length;
        __half* col_buffer = static_cast<__half*>(
            get_allocator<Device::CUDA>().allocate(col_size * sizeof(__half)));
        
        int threads = 256;
        int blocks = (output_length + threads - 1) / threads;
        
        im2col_1d_kernel_optimized<__half><<<blocks, threads>>>(
            input_ptr, col_buffer,
            batch_size, in_channels, input_length,
            kernel_size, _stride, _padding, _dilation, output_length);
        
        cublasHandle_t cublas_handle = cuda::get_cublas_handle();
        
        for (int b = 0; b < batch_size; b++) {
          __half* batch_output = output_ptr + b * out_channels * output_length;
          __half* batch_col = col_buffer + b * in_channels * kernel_size * output_length;
          
          gemm_cuda<__half>(
              cublas_handle,
              out_channels,
              output_length,
              in_channels * kernel_size,
              weight_ptr,
              batch_col,
              batch_output);
        }
        
        if (bias_ptr) {
          int total = batch_size * out_channels * output_length;
          int threads = 256;
          int blocks = (total + threads - 1) / threads;
          add_bias_kernel<__half><<<blocks, threads>>>(
              output_ptr, bias_ptr, out_channels, output_length);
        }
        
        get_allocator<Device::CUDA>().free(col_buffer);
      }
      
      CUDA_CHECK(cudaGetLastError());
    }

    // Main compute function for bfloat16_t (converts to __nv_bfloat16)
    template <>
    void Conv1D::compute<Device::CUDA, bfloat16_t>(
        const StorageView& input,
        const StorageView& weight,
        const StorageView* bias,
        StorageView& output,
        const StorageView* qscale) const {
      
      if (qscale)
        throw std::runtime_error("Quantization is not supported in this Conv1D implementation");

      const int batch_size = input.dim(0);
      const int in_channels = input.dim(1);
      const int input_length = input.dim(2);
      const int output_length = output.dim(2);
      const int out_channels = weight.dim(0);
      const int kernel_size = weight.dim(2);

      const __nv_bfloat16* input_ptr = to_cuda_type<const __nv_bfloat16>(input.data<bfloat16_t>());
      const __nv_bfloat16* weight_ptr = to_cuda_type<const __nv_bfloat16>(weight.data<bfloat16_t>());
      const __nv_bfloat16* bias_ptr = bias ? to_cuda_type<const __nv_bfloat16>(bias->data<bfloat16_t>()) : nullptr;
      __nv_bfloat16* output_ptr = to_cuda_type<__nv_bfloat16>(output.data<bfloat16_t>());

      const bool use_direct = (kernel_size <= 5 && _groups == 1 && output_length <= 512);

      if (use_direct) {
        int total_outputs = batch_size * out_channels * output_length;
        int threads = 256;
        int blocks = (total_outputs + threads - 1) / threads;
        
        conv1d_direct_kernel_optimized<__nv_bfloat16><<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, out_channels,
            input_length, output_length, kernel_size,
            _stride, _padding, _dilation, _groups);
      } else {
        size_t col_size = batch_size * in_channels * kernel_size * output_length;
        __nv_bfloat16* col_buffer = static_cast<__nv_bfloat16*>(
            get_allocator<Device::CUDA>().allocate(col_size * sizeof(__nv_bfloat16)));
        
        int threads = 256;
        int blocks = (output_length + threads - 1) / threads;
        
        im2col_1d_kernel_optimized<__nv_bfloat16><<<blocks, threads>>>(
            input_ptr, col_buffer,
            batch_size, in_channels, input_length,
            kernel_size, _stride, _padding, _dilation, output_length);
        
        cublasHandle_t cublas_handle = cuda::get_cublas_handle();
        
        for (int b = 0; b < batch_size; b++) {
          __nv_bfloat16* batch_output = output_ptr + b * out_channels * output_length;
          __nv_bfloat16* batch_col = col_buffer + b * in_channels * kernel_size * output_length;
          
          gemm_cuda<__nv_bfloat16>(
              cublas_handle,
              out_channels,
              output_length,
              in_channels * kernel_size,
              weight_ptr,
              batch_col,
              batch_output);
        }
        
        if (bias_ptr) {
          int total = batch_size * out_channels * output_length;
          int threads = 256;
          int blocks = (total + threads - 1) / threads;
          add_bias_kernel<__nv_bfloat16><<<blocks, threads>>>(
              output_ptr, bias_ptr, out_channels, output_length);
        }
        
        get_allocator<Device::CUDA>().free(col_buffer);
      }
      
      CUDA_CHECK(cudaGetLastError());
    }

  }
}
