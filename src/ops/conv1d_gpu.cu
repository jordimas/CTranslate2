#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/primitives.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    // Direct 1D convolution kernel using circular buffer in registers
    // Optimized for small kernels (typical in SSM/Mamba: 3-4)
    template <typename T, int KERNEL_SIZE>
    __global__ void direct_conv1d_kernel(
        const T* __restrict__ input,
        const T* __restrict__ weight,
        const T* __restrict__ bias,
        T* __restrict__ output,
        int batch_size,
        int in_channels,
        int out_channels,
        int input_length,
        int output_length,
        int stride,
        int padding,
        int dilation,
        int groups) {
      
      const int tid = threadIdx.x;
      const int out_ch_idx = blockIdx.x;  // Output channel
      const int batch_idx = blockIdx.y;   // Batch
      
      // Early exit for invalid threads
      if (out_ch_idx >= out_channels) return;
      
      const int group = out_ch_idx / (out_channels / groups);
      const int in_channels_per_group = in_channels / groups;
      const int in_ch_start = group * in_channels_per_group;
      
      // Circular buffer for input values (stored in registers)
      T x_buffer[KERNEL_SIZE];
      T w_cache[KERNEL_SIZE];
      
      // Process each output position (strided by blockDim.x)
      for (int out_pos = tid; out_pos < output_length; out_pos += blockDim.x) {
        
        T result = bias ? bias[out_ch_idx] : T(0);
        
        // Convolve across input channels in this group
        for (int ic = 0; ic < in_channels_per_group; ic++) {
          int in_ch = in_ch_start + ic;
          
          // Weight layout: [out_channels, in_channels_per_group, kernel_size]
          // Load weights for this (output_ch, input_ch) pair
          #pragma unroll
          for (int k = 0; k < KERNEL_SIZE; k++) {
            int weight_idx = (out_ch_idx * in_channels_per_group + ic) * KERNEL_SIZE + k;
            w_cache[k] = weight[weight_idx];
          }
          
          // Load input values into buffer
          #pragma unroll
          for (int k = 0; k < KERNEL_SIZE; k++) {
            int w_in = out_pos * stride - padding + k * dilation;
            if (w_in >= 0 && w_in < input_length) {
              int input_idx = (batch_idx * in_channels + in_ch) * input_length + w_in;
              x_buffer[k] = input[input_idx];
            } else {
              x_buffer[k] = T(0);
            }
          }
          
          // Compute dot product
          #pragma unroll
          for (int k = 0; k < KERNEL_SIZE; k++) {
            result += x_buffer[k] * w_cache[k];
          }
        }
        
        // Write output
        int out_idx = (batch_idx * out_channels + out_ch_idx) * output_length + out_pos;
        output[out_idx] = result;
      }
    }

    // Specialized kernel for causal convolution (like SSM)
    // Uses rotating circular buffer for temporal locality
    template <typename T, int KERNEL_SIZE>
    __global__ void causal_conv1d_kernel(
        const T* __restrict__ input,
        const T* __restrict__ weight,
        const T* __restrict__ bias,
        T* __restrict__ output,
        int batch_size,
        int channels,
        int seq_length) {
      
      const int ch = blockIdx.x * blockDim.x + threadIdx.x;
      const int batch = blockIdx.y;
      
      if (ch >= channels) return;
      
      // Circular buffer in registers
      T x_buffer[KERNEL_SIZE];
      T w_cache[KERNEL_SIZE];
      
      // Load weights once - weight layout: [out_channels, in_channels_per_group, kernel_size]
      // For causal conv: in_channels == out_channels and groups == 1
      #pragma unroll
      for (int k = 0; k < KERNEL_SIZE; k++) {
        w_cache[k] = weight[ch * KERNEL_SIZE + k];
      }
      
      const T* x_ptr = input + (batch * channels + ch) * seq_length;
      T* y_ptr = output + (batch * channels + ch) * seq_length;
      
      // Initialize circular buffer with first KERNEL_SIZE inputs (with zero padding)
      #pragma unroll
      for (int k = 0; k < KERNEL_SIZE; k++) {
        if (k < seq_length) {
          x_buffer[k] = x_ptr[k];
        } else {
          x_buffer[k] = T(0);
        }
      }
      
      // Process sequence with rotating buffer
      for (int t = 0; t < seq_length; t++) {
        T sum = bias ? bias[ch] : T(0);
        
        // Compute convolution using circular indexing
        #pragma unroll
        for (int k = 0; k < KERNEL_SIZE; k++) {
          sum += x_buffer[(t + k) % KERNEL_SIZE] * w_cache[k];
        }
        
        y_ptr[t] = sum;
        
        // Rotate buffer: replace oldest with next input
        if (t + KERNEL_SIZE < seq_length) {
          x_buffer[t % KERNEL_SIZE] = x_ptr[t + KERNEL_SIZE];
        }
      }
    }

    // Dispatcher based on kernel size
    template <typename CudaT, typename HostT>
    void conv1d_compute_impl_direct(
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
      const int kernel_size = weight.dim(2);

      const CudaT* input_ptr = reinterpret_cast<const CudaT*>(input.data<HostT>());
      const CudaT* weight_ptr = reinterpret_cast<const CudaT*>(weight.data<HostT>());
      const CudaT* bias_ptr = bias ? reinterpret_cast<const CudaT*>(bias->data<HostT>()) : nullptr;
      CudaT* output_ptr = reinterpret_cast<CudaT*>(output.data<HostT>());

      // Check if this is causal convolution (SSM-style)
      const bool is_causal = (stride == 1 && padding == kernel_size - 1 && 
                             dilation == 1 && groups == 1 && 
                             in_channels == out_channels);

      if (is_causal && kernel_size >= 3 && kernel_size <= 8) {
        // Optimized path for SSM/Mamba causal convolution
        const int threads = 128;
        const int blocks_x = (in_channels + threads - 1) / threads;
        const int blocks_y = batch_size;
        dim3 blocks(blocks_x, blocks_y);
        
        switch (kernel_size) {
          case 3:
            causal_conv1d_kernel<CudaT, 3><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, input_length);
            break;
          case 4:
            causal_conv1d_kernel<CudaT, 4><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, input_length);
            break;
          case 5:
            causal_conv1d_kernel<CudaT, 5><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, input_length);
            break;
          case 6:
            causal_conv1d_kernel<CudaT, 6><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, input_length);
            break;
          case 7:
            causal_conv1d_kernel<CudaT, 7><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, input_length);
            break;
          case 8:
            causal_conv1d_kernel<CudaT, 8><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, input_length);
            break;
        }
      } else {
        // General path for arbitrary stride/padding/dilation/groups
        const int threads = 256;
        dim3 blocks(out_channels, batch_size);
        
        switch (kernel_size) {
          case 1:
            direct_conv1d_kernel<CudaT, 1><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels, input_length, output_length,
                stride, padding, dilation, groups);
            break;
          case 2:
            direct_conv1d_kernel<CudaT, 2><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels, input_length, output_length,
                stride, padding, dilation, groups);
            break;
          case 3:
            direct_conv1d_kernel<CudaT, 3><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels, input_length, output_length,
                stride, padding, dilation, groups);
            break;
          case 4:
            direct_conv1d_kernel<CudaT, 4><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels, input_length, output_length,
                stride, padding, dilation, groups);
            break;
          case 5:
            direct_conv1d_kernel<CudaT, 5><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels, input_length, output_length,
                stride, padding, dilation, groups);
            break;
          case 6:
            direct_conv1d_kernel<CudaT, 6><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels, input_length, output_length,
                stride, padding, dilation, groups);
            break;
          case 7:
            direct_conv1d_kernel<CudaT, 7><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels, input_length, output_length,
                stride, padding, dilation, groups);
            break;
          case 8:
            direct_conv1d_kernel<CudaT, 8><<<blocks, threads>>>(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                batch_size, in_channels, out_channels, input_length, output_length,
                stride, padding, dilation, groups);
            break;
          default:
            throw std::runtime_error(
                "Unsupported kernel size: " + std::to_string(kernel_size) + 
                ". Supported: 1-8. For larger kernels, use im2col+GEMM implementation");
        }
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
        conv1d_compute_impl_direct<CudaT, HostT>(                     \
            input, weight, bias, output,                              \
            _stride, _padding, _dilation, _groups);                   \
      }

    CONV1D_COMPUTE_SPECIALIZATION(float, float)
    CONV1D_COMPUTE_SPECIALIZATION(float16_t, __half)
    CONV1D_COMPUTE_SPECIALIZATION(bfloat16_t, __nv_bfloat16)

    #undef CONV1D_COMPUTE_SPECIALIZATION

  }
}
