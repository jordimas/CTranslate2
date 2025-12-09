#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/primitives.h"
#include "cuda/utils.h"
#include <cstdio>

namespace ctranslate2 {
  namespace ops {

    // Debug print helper for device arrays
    template <typename T>
    void print_device_array(const char* name, const T* d_ptr, int count, int batch_idx = 0, int channel_idx = 0, int offset = 0) {
      T* h_data = new T[count];
      cudaMemcpy(h_data, d_ptr + offset, count * sizeof(T), cudaMemcpyDeviceToHost);
      printf("%s[%d,%d,%d:%d]: ", name, batch_idx, channel_idx, offset, offset + count - 1);
      for (int i = 0; i < count && i < 10; i++) {
        printf("%.3f ", float(h_data[i]));
      }
      printf("\n");
      delete[] h_data;
    }

    // FIXED: Properly designed tiled Conv1D kernel
    // Key fix: Each thread computes ONE output element, threads cooperate to load tiles
    template <typename T>
    __global__ void tiled_conv1d_kernel(
        const T* __restrict__ input,
        const T* __restrict__ weight,
        const T* __restrict__ bias,
        T* __restrict__ output,
        int batch_size,
        int in_channels,
        int input_length,
        int out_channels,
        int kernel_size,
        int stride,
        int padding,
        int dilation,
        int output_length,
        int in_channels_per_group,
        int out_channels_per_group,
        int group_id) {
      
      // Each thread computes one output element
      int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
      int out_ch_in_group = blockIdx.y * blockDim.y + threadIdx.y;
      int batch = blockIdx.z;
      
      if (batch >= batch_size || out_ch_in_group >= out_channels_per_group || out_pos >= output_length) {
        return;
      }
      
      int out_ch = group_id * out_channels_per_group + out_ch_in_group;
      float acc = bias ? float(bias[out_ch]) : 0.0f;
      
      int in_offset = group_id * in_channels_per_group;
      int weight_offset = group_id * out_channels_per_group * in_channels_per_group * kernel_size;
      
      // Compute convolution
      for (int ic = 0; ic < in_channels_per_group; ic++) {
        for (int k = 0; k < kernel_size; k++) {
          int w_in = out_pos * stride - padding + k * dilation;
          
          if (w_in >= 0 && w_in < input_length) {
            int input_idx = (batch * in_channels + in_offset + ic) * input_length + w_in;
            int weight_idx = weight_offset + (out_ch_in_group * in_channels_per_group + ic) * kernel_size + k;
            
            acc += float(input[input_idx]) * float(weight[weight_idx]);
          }
        }
      }
      
      int out_idx = (batch * out_channels + out_ch) * output_length + out_pos;
      output[out_idx] = T(acc);
    }

    // Fallback kernel for small problems or edge cases
    template <typename T>
    __global__ void simple_conv1d_kernel(
        const T* __restrict__ input,
        const T* __restrict__ weight,
        const T* __restrict__ bias,
        T* __restrict__ output,
        int batch_size,
        int in_channels,
        int input_length,
        int out_channels,
        int kernel_size,
        int stride,
        int padding,
        int dilation,
        int output_length,
        int in_channels_per_group,
        int out_channels_per_group,
        int groups) {
      
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = batch_size * out_channels * output_length;
      
      if (idx < total) {
        int out_pos = idx % output_length;
        int out_ch = (idx / output_length) % out_channels;
        int batch = idx / (output_length * out_channels);
        
        int group_id = out_ch / out_channels_per_group;
        int out_ch_in_group = out_ch % out_channels_per_group;
        
        float acc = bias ? float(bias[out_ch]) : 0.0f;
        
        int in_offset = group_id * in_channels_per_group;
        int weight_offset = group_id * out_channels_per_group * in_channels_per_group * kernel_size;
        
        for (int ic = 0; ic < in_channels_per_group; ic++) {
          for (int k = 0; k < kernel_size; k++) {
            int w_in = out_pos * stride - padding + k * dilation;
            
            if (w_in >= 0 && w_in < input_length) {
              int input_idx = (batch * in_channels + in_offset + ic) * input_length + w_in;
              int weight_idx = weight_offset + (out_ch_in_group * in_channels_per_group + ic) * kernel_size + k;
              
              acc += float(input[input_idx]) * float(weight[weight_idx]);
            }
          }
        }
        
        output[idx] = T(acc);
      }
    }

    // Main convolution implementation
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
      
      // Extract dimensions
      const int batch_size = input.dim(0);
      const int in_channels = input.dim(1);
      const int input_length = input.dim(2);
      const int output_length = output.dim(2);
      const int out_channels = weight.dim(0);
      const int in_channels_per_group = weight.dim(1);
      const int kernel_size = weight.dim(2);
      const int out_channels_per_group = out_channels / groups;

      printf("\n=== CONV1D DEBUG INFO ===\n");
      printf("Dimensions:\n");
      printf("  Input: [%d, %d, %d] (batch, in_channels, length)\n", 
             batch_size, in_channels, input_length);
      printf("  Weight: [%d, %d, %d] (out_channels, in_per_group, kernel)\n",
             out_channels, in_channels_per_group, kernel_size);
      printf("  Output: [%d, %d, %d] (batch, out_channels, length)\n",
             batch_size, out_channels, output_length);
      printf("  Bias: %s\n", bias ? "yes" : "no");
      
      printf("Parameters:\n");
      printf("  Stride: %d, Padding: %d, Dilation: %d, Groups: %d\n",
             stride, padding, dilation, groups);
      printf("  Channels per group: in=%d out=%d\n", 
             in_channels_per_group, out_channels_per_group);
      
      // Validate dimensions
      int expected_output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
      printf("\nValidation:\n");
      printf("  Expected output length: %d (actual: %d) %s\n",
             expected_output_length, output_length,
             expected_output_length == output_length ? "✓" : "✗ MISMATCH!");
      
      assert(in_channels == in_channels_per_group * groups);
      assert(out_channels == out_channels_per_group * groups);
      assert(output_length == expected_output_length);
      
      printf("  All dimension checks passed ✓\n");

      // Cast pointers
      const CudaT* input_ptr = reinterpret_cast<const CudaT*>(input.data<HostT>());
      const CudaT* weight_ptr = reinterpret_cast<const CudaT*>(weight.data<HostT>());
      const CudaT* bias_ptr = bias ? reinterpret_cast<const CudaT*>(bias->data<HostT>()) : nullptr;
      CudaT* output_ptr = reinterpret_cast<CudaT*>(output.data<HostT>());

      // Print sample input/weight values
      printf("\nSample values (first 5):\n");
      print_device_array("Input[0,0]", input_ptr, std::min(5, input_length), 0, 0, 0);
      if (in_channels > 1) {
        print_device_array("Input[0,1]", input_ptr, std::min(5, input_length), 0, 1, input_length);
      }
      print_device_array("Weight[0,0]", weight_ptr, std::min(5, in_channels_per_group * kernel_size), 0, 0, 0);
      if (bias_ptr) {
        print_device_array("Bias", bias_ptr, std::min(5, out_channels), 0, 0, 0);
      }

      // Choose kernel based on problem size
      bool use_tiled = (out_channels_per_group >= 16 && output_length >= 128);
      
      printf("\nKernel selection:\n");
      printf("  Use tiled kernel: %s\n", use_tiled ? "YES" : "NO (using simple)");
      
      if (use_tiled) {
        // Use 2D tiled kernel - each thread computes one output
        dim3 block(32, 8);  // 32 spatial positions, 8 output channels = 256 threads
        dim3 grid(
            (output_length + block.x - 1) / block.x,
            (out_channels_per_group + block.y - 1) / block.y,
            batch_size
        );
        
        printf("  Block: (%d, %d) = %d threads\n", block.x, block.y, block.x * block.y);
        printf("  Grid: (%d, %d, %d) = %d blocks\n", grid.x, grid.y, grid.z, grid.x * grid.y * grid.z);
        
        for (int g = 0; g < groups; g++) {
          tiled_conv1d_kernel<CudaT><<<grid, block>>>(
              input_ptr, weight_ptr, bias_ptr, output_ptr,
              batch_size, in_channels, input_length, out_channels,
              kernel_size, stride, padding, dilation, output_length,
              in_channels_per_group, out_channels_per_group, g);
        }
      } else {
        // Use simple 1D kernel
        int total = batch_size * out_channels * output_length;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        printf("  Threads per block: %d\n", threads);
        printf("  Blocks: %d\n", blocks);
        printf("  Total outputs: %d\n", total);
        
        simple_conv1d_kernel<<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, input_length, out_channels,
            kernel_size, stride, padding, dilation, output_length,
            in_channels_per_group, out_channels_per_group, groups);
      }
      
      cudaDeviceSynchronize();
      CUDA_CHECK(cudaGetLastError());
      
      // Print sample output values
      printf("\nSample outputs (first 5):\n");
      print_device_array("Output[0,0]", output_ptr, std::min(5, output_length), 0, 0, 0);
      if (out_channels > 1) {
        print_device_array("Output[0,1]", output_ptr, std::min(5, output_length), 0, 1, output_length);
      }
      
      printf("=== CONV1D DEBUG END ===\n\n");
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
