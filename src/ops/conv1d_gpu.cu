#include "ctranslate2/ops/conv1d.h"
#include "cuda/utils.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace ctranslate2 {
  namespace ops {

    // Helper to convert between CTranslate2 types and CUDA native types
    template<typename T> struct CudaNativeType { using type = T; };
    template<> struct CudaNativeType<float16_t> { using type = __half; };
    template<> struct CudaNativeType<bfloat16_t> { using type = __nv_bfloat16; };

    // Helper functions for type conversion
    template<typename T>
    __device__ __forceinline__ typename CudaNativeType<T>::type to_cuda_type(const T& val) {
      return val;
    }

    template<>
    __device__ __forceinline__ __half to_cuda_type<float16_t>(const float16_t& val) {
      return *reinterpret_cast<const __half*>(&val);
    }

    template<>
    __device__ __forceinline__ __nv_bfloat16 to_cuda_type<bfloat16_t>(const bfloat16_t& val) {
      return *reinterpret_cast<const __nv_bfloat16*>(&val);
    }

    template<typename T>
    __device__ __forceinline__ void store_cuda_type(T* ptr, const typename CudaNativeType<T>::type& val) {
      *ptr = val;
    }

    template<>
    __device__ __forceinline__ void store_cuda_type<float16_t>(float16_t* ptr, const __half& val) {
      *reinterpret_cast<__half*>(ptr) = val;
    }

    template<>
    __device__ __forceinline__ void store_cuda_type<bfloat16_t>(bfloat16_t* ptr, const __nv_bfloat16& val) {
      *reinterpret_cast<__nv_bfloat16*>(ptr) = val;
    }

    // Zero value helpers
    template<typename CudaT>
    __device__ __forceinline__ CudaT cuda_zero() {
      return CudaT(0.0f);
    }

    template<>
    __device__ __forceinline__ __half cuda_zero<__half>() {
      return __float2half(0.0f);
    }

    template<>
    __device__ __forceinline__ __nv_bfloat16 cuda_zero<__nv_bfloat16>() {
      return __float2bfloat16(0.0f);
    }

    // CUDA kernel for 1D convolution
    template <typename T>
    __global__ void conv1d_kernel(
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
        int groups,
        int in_channels_per_group,
        int out_channels_per_group) {
      
      using CudaT = typename CudaNativeType<T>::type;
      
      // Each block handles one output channel for one batch
      int batch_idx = blockIdx.x;
      int out_ch = blockIdx.y;
      int out_pos = threadIdx.x + blockIdx.z * blockDim.x;
      
      if (batch_idx >= batch_size || out_ch >= out_channels || out_pos >= output_length)
        return;
      
      int group_idx = out_ch / out_channels_per_group;
      int in_ch_start = group_idx * in_channels_per_group;
      int in_ch_end = in_ch_start + in_channels_per_group;
      
      CudaT sum = bias ? to_cuda_type(bias[out_ch]) : cuda_zero<CudaT>();
      
      // Compute convolution
      for (int k = 0; k < kernel_size; ++k) {
        int in_pos = out_pos * stride - padding + k * dilation;
        
        if (in_pos >= 0 && in_pos < input_length) {
          for (int in_ch = in_ch_start; in_ch < in_ch_end; ++in_ch) {
            int input_idx = ((batch_idx * in_channels + in_ch) * input_length) + in_pos;
            int weight_idx = ((out_ch * in_channels_per_group + (in_ch - in_ch_start)) * kernel_size) + k;
            
            CudaT a = to_cuda_type(input[input_idx]);
            CudaT b = to_cuda_type(weight[weight_idx]);
            sum = sum + a * b;
          }
        }
      }
      
      int output_idx = ((batch_idx * out_channels + out_ch) * output_length) + out_pos;
      store_cuda_type(&output[output_idx], sum);
    }

    // Optimized kernel using shared memory for small kernels
    template <typename T, int TILE_SIZE = 256>
    __global__ void conv1d_shared_kernel(
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
        int groups,
        int in_channels_per_group,
        int out_channels_per_group) {
      
      using CudaT = typename CudaNativeType<T>::type;
      
      extern __shared__ char shared_mem[];
      CudaT* shared_input = reinterpret_cast<CudaT*>(shared_mem);
      
      int batch_idx = blockIdx.x;
      int out_ch = blockIdx.y;
      int out_pos = threadIdx.x + blockIdx.z * blockDim.x;
      
      if (batch_idx >= batch_size || out_ch >= out_channels)
        return;
      
      int group_idx = out_ch / out_channels_per_group;
      int in_ch_start = group_idx * in_channels_per_group;
      int in_ch_end = in_ch_start + in_channels_per_group;
      
      // Load input tile to shared memory
      int tile_start = blockIdx.z * blockDim.x * stride - padding;
      int tile_end = tile_start + blockDim.x * stride + (kernel_size - 1) * dilation;
      int tile_size = tile_end - tile_start + 1;
      
      for (int in_ch = in_ch_start; in_ch < in_ch_end; ++in_ch) {
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
          int in_pos = tile_start + i;
          int shared_idx = (in_ch - in_ch_start) * tile_size + i;
          
          if (in_pos >= 0 && in_pos < input_length) {
            int input_idx = ((batch_idx * in_channels + in_ch) * input_length) + in_pos;
            shared_input[shared_idx] = to_cuda_type(input[input_idx]);
          } else {
            shared_input[shared_idx] = cuda_zero<CudaT>();
          }
        }
      }
      
      __syncthreads();
      
      if (out_pos >= output_length)
        return;
      
      CudaT sum = bias ? to_cuda_type(bias[out_ch]) : cuda_zero<CudaT>();
      
      // Compute convolution from shared memory
      for (int k = 0; k < kernel_size; ++k) {
        int in_pos = out_pos * stride - padding + k * dilation;
        int shared_offset = in_pos - tile_start;
        
        if (shared_offset >= 0 && shared_offset < tile_size) {
          for (int in_ch = in_ch_start; in_ch < in_ch_end; ++in_ch) {
            int shared_idx = (in_ch - in_ch_start) * tile_size + shared_offset;
            int weight_idx = ((out_ch * in_channels_per_group + (in_ch - in_ch_start)) * kernel_size) + k;
            
            CudaT a = shared_input[shared_idx];
            CudaT b = to_cuda_type(weight[weight_idx]);
            sum = sum + a * b;
          }
        }
      }
      
      int output_idx = ((batch_idx * out_channels + out_ch) * output_length) + out_pos;
      store_cuda_type(&output[output_idx], sum);
    }

    template <Device D, typename T>
    void Conv1D::compute(const StorageView& input,
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
      const int in_channels_per_group = weight.dim(1);
      const int kernel_size = weight.dim(2);
      const int out_channels_per_group = out_channels / _groups;

      // Calculate grid and block dimensions
      const int BLOCK_SIZE = 256;
      const int num_blocks_per_channel = (output_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
      
      dim3 grid(batch_size, out_channels, num_blocks_per_channel);
      dim3 block(BLOCK_SIZE);

      // Determine if we should use shared memory optimization
      bool use_shared = (kernel_size <= 32 && in_channels_per_group <= 64);
      
      const T* input_ptr = input.data<T>();
      const T* weight_ptr = weight.data<T>();
      const T* bias_ptr = bias ? bias->data<T>() : nullptr;
      T* output_ptr = output.data<T>();

      using CudaT = typename CudaNativeType<T>::type;

      if (use_shared) {
        // Calculate shared memory size
        int tile_start = -_padding;
        int tile_end = BLOCK_SIZE * _stride + (kernel_size - 1) * _dilation - _padding;
        int tile_size = tile_end - tile_start + 1;
        size_t shared_mem_size = in_channels_per_group * tile_size * sizeof(CudaT);
        
        // Check if shared memory is available
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        if (shared_mem_size <= prop.sharedMemPerBlock) {
          conv1d_shared_kernel<T><<<grid, block, shared_mem_size>>>(
              input_ptr, weight_ptr, bias_ptr, output_ptr,
              batch_size, in_channels, out_channels,
              input_length, output_length, kernel_size,
              _stride, _padding, _dilation, _groups,
              in_channels_per_group, out_channels_per_group);
        } else {
          // Fall back to non-shared version
          conv1d_kernel<T><<<grid, block>>>(
              input_ptr, weight_ptr, bias_ptr, output_ptr,
              batch_size, in_channels, out_channels,
              input_length, output_length, kernel_size,
              _stride, _padding, _dilation, _groups,
              in_channels_per_group, out_channels_per_group);
        }
      } else {
        conv1d_kernel<T><<<grid, block>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, out_channels,
            input_length, output_length, kernel_size,
            _stride, _padding, _dilation, _groups,
            in_channels_per_group, out_channels_per_group);
      }

      // Check for errors
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + 
                                std::string(cudaGetErrorString(err)));
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Conv1D::compute<Device::CUDA, T>(const StorageView& input,          \
                                     const StorageView& weight,         \
                                     const StorageView* bias,           \
                                     StorageView& output,               \
                                     const StorageView* qscale) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
