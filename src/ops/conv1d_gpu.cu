#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/primitives.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    // Fused im2col-GEMM kernel - directly computes convolution output
    // Each thread block computes a tile of the output
    template <typename T, int TILE_M = 64, int TILE_N = 64, int TILE_K = 16>
    __global__ void fused_conv1d_kernel(
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
      
      // Shared memory for tiles
      __shared__ T tile_weight[TILE_M][TILE_K];
      __shared__ T tile_input[TILE_K][TILE_N];
      
      // Thread and block indices
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int bx = blockIdx.x;
      int by = blockIdx.y;
      int bz = blockIdx.z;
      
      // Output position: batch and spatial
      int batch = bz;
      int out_row = by * TILE_M + ty;  // output channel (within group)
      int out_col_base = bx * TILE_N + tx;  // output spatial position
      
      // Bounds check
      if (batch >= batch_size) return;
      
      // Accumulator
      float acc = 0.0f;
      
      // K dimension: in_channels_per_group * kernel_size
      int K = in_channels_per_group * kernel_size;
      
      // Loop over K dimension in tiles
      for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        int k_base = k_tile * TILE_K;
        
        // Load weight tile (each thread loads one element)
        int weight_row = out_row;
        int weight_col = k_base + tx;
        
        if (weight_row < out_channels_per_group && weight_col < K && ty == 0) {
          int weight_offset = group_id * out_channels_per_group * K;
          tile_weight[ty][tx] = weight[weight_offset + weight_row * K + weight_col];
        }
        
        // Broadcast weight across ty dimension
        if (ty == 0 && weight_row < out_channels_per_group && weight_col < K) {
          T val = tile_weight[0][tx];
          for (int i = 1; i < TILE_M && (by * TILE_M + i) < out_channels_per_group; i++) {
            tile_weight[i][tx] = val;
          }
        }
        
        // Load input tile with im2col logic
        int k_idx = k_base + ty;
        int out_col = out_col_base;
        
        if (k_idx < K && out_col < output_length) {
          // Decompose k_idx into (input_channel, kernel_position)
          int kernel_pos = k_idx % kernel_size;
          int in_ch = k_idx / kernel_size;
          
          // Compute input spatial position
          int w_in = out_col * stride - padding + kernel_pos * dilation;
          
          // Load from input or use zero padding
          if (w_in >= 0 && w_in < input_length) {
            int in_offset = group_id * in_channels_per_group;
            int input_idx = (batch * in_channels + in_offset + in_ch) * input_length + w_in;
            tile_input[ty][tx] = input[input_idx];
          } else {
            tile_input[ty][tx] = T(0);
          }
        } else {
          tile_input[ty][tx] = T(0);
        }
        
        __syncthreads();
        
        // Compute partial dot product
        if (out_row < out_channels_per_group && out_col_base < output_length) {
          for (int k = 0; k < TILE_K && (k_base + k) < K; k++) {
            acc += float(tile_weight[ty][k]) * float(tile_input[k][tx]);
          }
        }
        
        __syncthreads();
      }
      
      // Write output with bias
      if (out_row < out_channels_per_group && out_col_base < output_length) {
        int out_ch = group_id * out_channels_per_group + out_row;
        int out_idx = (batch * out_channels + out_ch) * output_length + out_col_base;
        
        if (bias) {
          acc += float(bias[out_ch]);
        }
        
        output[out_idx] = T(acc);
      }
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

    // Main convolution implementation with fused kernel
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

      // Cast pointers
      const CudaT* input_ptr = reinterpret_cast<const CudaT*>(input.data<HostT>());
      const CudaT* weight_ptr = reinterpret_cast<const CudaT*>(weight.data<HostT>());
      const CudaT* bias_ptr = bias ? reinterpret_cast<const CudaT*>(bias->data<HostT>()) : nullptr;
      CudaT* output_ptr = reinterpret_cast<CudaT*>(output.data<HostT>());

      // Choose kernel based on problem size
      constexpr int TILE_M = 64;
      constexpr int TILE_N = 64;
      constexpr int TILE_K = 16;
      
      bool use_fused = (out_channels_per_group >= 32 && output_length >= 32);
      
      if (use_fused) {
        // Use fused tiled kernel
        dim3 block(TILE_N / 4, TILE_M / 16);  // 16x4 = 64 threads
        dim3 grid(
            (output_length + TILE_N - 1) / TILE_N,
            (out_channels_per_group + TILE_M - 1) / TILE_M,
            batch_size
        );
        
        for (int g = 0; g < groups; g++) {
          fused_conv1d_kernel<CudaT, TILE_M, TILE_N, TILE_K><<<grid, block>>>(
              input_ptr, weight_ptr, bias_ptr, output_ptr,
              batch_size, in_channels, input_length, out_channels,
              kernel_size, stride, padding, dilation, output_length,
              in_channels_per_group, out_channels_per_group, g);
        }
      } else {
        // Use simple kernel for small problems
        int total = batch_size * out_channels * output_length;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        simple_conv1d_kernel<<<blocks, threads>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, input_length, out_channels,
            kernel_size, stride, padding, dilation, output_length,
            in_channels_per_group, out_channels_per_group, groups);
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
