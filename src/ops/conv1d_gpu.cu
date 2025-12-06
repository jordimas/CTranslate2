#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/primitives.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    // Optimized im2col kernel with optional bias initialization
    template <typename T>
    __global__ void im2col_1d_kernel(
        const T* __restrict__ input,
        T* __restrict__ col_buffer,
        T* __restrict__ output,
        const T* __restrict__ bias,
        int batch_size,
        int in_channels,
        int input_length,
        int kernel_size,
        int stride,
        int padding,
        int dilation,
        int output_length,
        int out_channels) {
      
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int total = batch_size * in_channels * kernel_size * output_length;
      
      if (idx < total) {
        // Unpack indices (innermost: output_length for coalesced writes)
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
        
        // Initialize output with bias (first thread per output position)
        if (bias && ic == 0 && k == 0) {
          for (int oc = 0; oc < out_channels; oc++) {
            int out_idx = (b * out_channels + oc) * output_length + out_pos;
            output[out_idx] = bias[oc];
          }
        }
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

      // Cast pointers
      const CudaT* input_ptr = reinterpret_cast<const CudaT*>(input.data<HostT>());
      const CudaT* weight_ptr = reinterpret_cast<const CudaT*>(weight.data<HostT>());
      const CudaT* bias_ptr = bias ? reinterpret_cast<const CudaT*>(bias->data<HostT>()) : nullptr;
      CudaT* output_ptr = reinterpret_cast<CudaT*>(output.data<HostT>());

      // Allocate and populate im2col buffer
      size_t col_size = batch_size * in_channels_per_group * kernel_size * output_length * groups;
      CudaT* col_buffer = static_cast<CudaT*>(
          get_allocator<Device::CUDA>().allocate(col_size * sizeof(CudaT)));
      
      int total = batch_size * in_channels * kernel_size * output_length;
      int threads = 256;
      int blocks = (total + threads - 1) / threads;
      
      im2col_1d_kernel<<<blocks, threads>>>(
          input_ptr, col_buffer, output_ptr, bias_ptr, batch_size, in_channels, 
          input_length, kernel_size, stride, padding, dilation, output_length, out_channels);
      
      // Setup GEMM parameters
      float alpha = 1.0f;
      float beta = bias_ptr ? 1.0f : 0.0f;
      
      dim_t m = out_channels_per_group;
      dim_t n = output_length;
      dim_t k = in_channels_per_group * kernel_size;
      
      // Perform grouped convolution via GEMM
      for (int g = 0; g < groups; g++) {
        const CudaT* group_weight = weight_ptr + g * m * k;
        const CudaT* group_col = col_buffer + g * k * n;
        CudaT* group_output = output_ptr + g * m * n;
        
        dim_t stride_b = in_channels * kernel_size * output_length;
        dim_t stride_c = out_channels * output_length;
        
        primitives<Device::CUDA>::gemm_batch_strided<HostT, HostT>(
            false, false,
            m, n, k,
            alpha,
            reinterpret_cast<const HostT*>(group_weight), k, 0,
            reinterpret_cast<const HostT*>(group_col), n, stride_b,
            beta,
            reinterpret_cast<HostT*>(group_output), n, stride_c,
            batch_size);
      }
      
      get_allocator<Device::CUDA>().free(col_buffer);
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
