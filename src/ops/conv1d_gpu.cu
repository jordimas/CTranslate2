#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/primitives.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

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
          col_buffer[col_idx] = T(0);
        }
      }
    }

    // Broadcast bias to output (used when we need to prepare output before GEMM)
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

    // Main implementation - using primitives for GEMM operations
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

      // Allocate im2col buffer
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
      
      // Perform grouped GEMM using primitives
      int out_channels_per_group = out_channels / groups;
      
      if (groups == 1) {
        // Single group: use batched GEMM from primitives
        
        // Broadcast bias first if present
        if (bias_ptr) {
          int total_elems = batch_size * out_channels * output_length;
          int bias_threads = 256;
          int bias_blocks = (total_elems + bias_threads - 1) / bias_threads;
          broadcast_bias_kernel<CudaT><<<bias_blocks, bias_threads>>>(
              output_ptr, bias_ptr, batch_size, out_channels, output_length);
        }
        
        // GEMM: output = weight @ col_buffer (+ bias if present)
        // Dimensions: (out_channels, output_length) = (out_channels, in_channels*kernel) @ (in_channels*kernel, output_length)
        // Using batched strided GEMM from primitives
        float alpha = 1.0f;
        float beta = bias_ptr ? 1.0f : 0.0f;
        
        dim_t m = out_channels_per_group;
        dim_t n = output_length;
        dim_t k = in_channels_per_group * kernel_size;
        dim_t lda = k;  // weight is (out_channels, k)
        dim_t ldb = n;  // col_buffer is (k, output_length)
        dim_t ldc = n;  // output is (out_channels, output_length)
        
        dim_t stride_a = 0;  // weight is shared across batches
        dim_t stride_b = k * n;  // col_buffer strides by batch
        dim_t stride_c = m * n;  // output strides by batch
        
        primitives<Device::CUDA>::gemm_batch_strided<HostT, HostT>(
            false, false,  // no transpose
            m, n, k,
            alpha,
            reinterpret_cast<const HostT*>(weight_ptr), lda, stride_a,
            reinterpret_cast<const HostT*>(col_buffer), ldb, stride_b,
            beta,
            reinterpret_cast<HostT*>(output_ptr), ldc, stride_c,
            batch_size);
        
      } else {
        // Multiple groups: broadcast bias first if present
        if (bias_ptr) {
          int total_elems = batch_size * out_channels * output_length;
          int bias_threads = 256;
          int bias_blocks = (total_elems + bias_threads - 1) / bias_threads;
          broadcast_bias_kernel<CudaT><<<bias_blocks, bias_threads>>>(
              output_ptr, bias_ptr, batch_size, out_channels, output_length);
        }
        
        // Process each group using primitives
        for (int g = 0; g < groups; g++) {
          const CudaT* group_weight = weight_ptr + 
              g * out_channels_per_group * in_channels_per_group * kernel_size;
          CudaT* group_output = output_ptr + 
              g * out_channels_per_group * output_length;
          const CudaT* group_col = col_buffer + 
              g * in_channels_per_group * kernel_size * output_length;
          
          float alpha = 1.0f;
          float beta = bias_ptr ? 1.0f : 0.0f;
          
          dim_t m = out_channels_per_group;
          dim_t n = output_length;
          dim_t k = in_channels_per_group * kernel_size;
          dim_t lda = k;
          dim_t ldb = n;
          dim_t ldc = n;
          
          dim_t stride_a = 0;
          dim_t stride_b = in_channels * kernel_size * output_length;
          dim_t stride_c = out_channels * output_length;
          
          primitives<Device::CUDA>::gemm_batch_strided<HostT, HostT>(
              false, false,
              m, n, k,
              alpha,
              reinterpret_cast<const HostT*>(group_weight), lda, stride_a,
              reinterpret_cast<const HostT*>(group_col), ldb, stride_b,
              beta,
              reinterpret_cast<HostT*>(group_output), ldc, stride_c,
              batch_size);
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
