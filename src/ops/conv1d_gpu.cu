#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/primitives.h"
#include "cuda/utils.h"
#include <cublas_v2.h>
#include <algorithm>

namespace ctranslate2 {
  namespace ops {

    // --- Helper for cuBLAS Handle Management ---
    inline cublasHandle_t get_cublas_handle() {
        // In a real production environment, you should pull this from 
        // the global execution context/resource pool.
        static cublasHandle_t handle = nullptr;
        if (!handle) cublasCreate(&handle);
        return handle;
    }

    // --- CUDA Type Mapping ---
    template <typename T> constexpr cublasDataType_t CUDA_GET_DATA_TYPE();
    template <> constexpr cublasDataType_t CUDA_GET_DATA_TYPE<float>() { return CUDA_R_32F; }
    template <> constexpr cublasDataType_t CUDA_GET_DATA_TYPE<__half>() { return CUDA_R_16F; }
    template <> constexpr cublasDataType_t CUDA_GET_DATA_TYPE<__nv_bfloat16>() { return CUDA_R_16BF; }

    // --- Optimized Im2Col 1D Kernel ---
    template <typename T>
    __global__ void im2col_1d_kernel(
        const T* __restrict__ input,
        T* __restrict__ output_col,
        int batch_size, int in_channels, int input_length,
        int kernel_size, int stride, int padding, int dilation,
        int output_length, int group_id, int in_channels_per_group) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_elements = output_length * batch_size * in_channels_per_group * kernel_size;

        if (idx >= total_elements) return;

        const int output_batch_len = output_length * batch_size;
        const int Ck_index = idx / output_batch_len; 
        const int LB_index = idx % output_batch_len; 

        const int k = Ck_index % kernel_size;
        const int ic = Ck_index / kernel_size; 
        const int batch = LB_index / output_length;
        const int out_pos = LB_index % output_length;

        const int w_in = out_pos * stride - padding + k * dilation;
        const int in_ch = group_id * in_channels_per_group + ic; 

        if (w_in < 0 || w_in >= input_length) {
            output_col[idx] = T(0.0);
        } else {
            const long long input_idx = (long long)batch * in_channels * input_length + (long long)in_ch * input_length + w_in;
            output_col[idx] = input[input_idx];
        }
    }

    // --- Bias Addition Kernel ---
    template <typename T>
    __global__ void add_bias_kernel(T* output, const T* bias, int channels, int out_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < channels * out_dim) {
            output[idx] += bias[idx / out_dim];
        }
    }

    // --- Implementation ---
    template <typename CudaT, typename HostT>
    void conv1d_compute_impl(
        const StorageView& input, const StorageView& weight, const StorageView* bias,
        StorageView& output, int stride, int padding, int dilation, int groups) {
      
      using ComputeT = float; // Accumulation precision
      cublasHandle_t handle = get_cublas_handle();
      
      const int batch_size = input.dim(0);
      const int in_channels = input.dim(1);
      const int input_length = input.dim(2);
      const int output_length = output.dim(2);
      const int out_channels = weight.dim(0);
      const int in_channels_per_group = weight.dim(1);
      const int kernel_size = weight.dim(2);
      const int out_channels_per_group = out_channels / groups;

      const CudaT* input_ptr = reinterpret_cast<const CudaT*>(input.data<HostT>());
      const CudaT* weight_ptr = reinterpret_cast<const CudaT*>(weight.data<HostT>());
      CudaT* output_ptr = reinterpret_cast<CudaT*>(output.data<HostT>());
      
      // GEMM: (OutGroups, OutBatch) = (Weights) * (Im2Col_Input)
      const int K = out_channels_per_group;
      const int N = in_channels_per_group * kernel_size;
      const int M = output_length * batch_size; 

      StorageView input_col_view({(long long)N, (long long)M}, output.dtype(), output.device());
      CudaT* input_col_ptr = reinterpret_cast<CudaT*>(input_col_view.data<HostT>());

      const ComputeT alpha = 1.0f;
      const ComputeT beta = 0.0f;

      for (int g = 0; g < groups; g++) {
        // 1. Prepare Workspace (Im2Col)
        const int total_im2col = N * M;
        im2col_1d_kernel<CudaT><<<(total_im2col + 255) / 256, 256>>>(
            input_ptr, input_col_ptr, batch_size, in_channels, input_length,
            kernel_size, stride, padding, dilation, output_length, g, in_channels_per_group);

        const CudaT* current_w = weight_ptr + (long long)g * K * N;
        CudaT* current_out = output_ptr + (long long)g * K * M;

        // 2. Compute GEMM
        // Note: cuBLAS is column-major. We use Transpose to simulate Row-Major logic.
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            M, K, N, &alpha, 
            input_col_ptr, CUDA_GET_DATA_TYPE<CudaT>(), M, 
            current_w, CUDA_GET_DATA_TYPE<CudaT>(), N, 
            &beta, current_out, CUDA_GET_DATA_TYPE<CudaT>(), M, 
            CUDA_GET_DATA_TYPE<ComputeT>(), CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // 3. Add Bias
        if (bias) {
            const CudaT* b_ptr = reinterpret_cast<const CudaT*>(bias->data<HostT>()) + g * K;
            add_bias_kernel<CudaT><<<(K * M + 255) / 256, 256>>>(current_out, b_ptr, K, M);
        }
      }
    }

    // --- Specializations (Required for CTranslate2 linking) ---
    #define CONV1D_SPEC(HostT, CudaT) \
      template <> void Conv1D::compute<Device::CUDA, HostT>( \
          const StorageView& in, const StorageView& w, const StorageView* b, \
          StorageView& out, const StorageView* q) const { \
        if (q) throw std::runtime_error("Quantization not supported"); \
        conv1d_compute_impl<CudaT, HostT>(in, w, b, out, _stride, _padding, _dilation, _groups); \
      }

    CONV1D_SPEC(float, float)
    CONV1D_SPEC(float16_t, __half)
    CONV1D_SPEC(bfloat16_t, __nv_bfloat16)
  }
}
