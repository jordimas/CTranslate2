#include "ctranslate2/ops/causal_depthwise_conv1d.h"

#include <algorithm>
#include "ctranslate2/primitives.h"
#include <cstdint>
#include <stdexcept>

#include "device_dispatch.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      bool overlaps(const StorageView& a, dim_t a_bytes, const StorageView& b, dim_t b_bytes) {
        if (!a.buffer() || !b.buffer() || a_bytes == 0 || b_bytes == 0)
          return false;
        const auto x = reinterpret_cast<uintptr_t>(a.buffer());
        const auto y = reinterpret_cast<uintptr_t>(b.buffer());
        return x <= y ? y - x < static_cast<uintptr_t>(a_bytes)
                      : x - y < static_cast<uintptr_t>(b_bytes);
      }
    }

    CausalDepthwiseConv1D::CausalDepthwiseConv1D(bool apply_silu)
      : _apply_silu(apply_silu) {
    }

    void CausalDepthwiseConv1D::validate_input(const StorageView& input, const StorageView& weight,
                                               const StorageView* bias, const StorageView& history,
                                               const StorageView& output) const {
      const StorageView* tensors[] = {&input, &weight, bias, &history, &output};
      for (const auto* tensor : tensors) {
        if (tensor && (tensor->device() != input.device()
                       || tensor->device_index() != input.device_index()
                       || tensor->dtype() != DataType::FLOAT32))
          throw std::invalid_argument("CausalDepthwiseConv1D requires float32 tensors on the same device");
      }
      if (input.rank() != 3 || input.dim(0) <= 0 || input.dim(1) < 0 || input.dim(2) <= 0)
        throw std::invalid_argument("CausalDepthwiseConv1D input must be [batch,time,channels]");
      const dim_t batch = input.dim(0);
      const dim_t channels = input.dim(2);
      if (weight.rank() != 2 || weight.dim(0) != channels || weight.dim(1) < 1)
        throw std::invalid_argument("CausalDepthwiseConv1D weights must be [channels,kernel>=1]");
      const dim_t kernel = weight.dim(1);
      const dim_t past = kernel - 1;
      if (history.shape() != Shape{batch, channels, past})
        throw std::invalid_argument(
            "CausalDepthwiseConv1D history must be [batch,channels,kernel-1]");
      if (bias && bias->shape() != Shape{channels})
        throw std::invalid_argument("CausalDepthwiseConv1D bias must be [channels]");
      if (&history == &output
          || overlaps(history, history.reserved_memory(), output, output.reserved_memory()))
        throw std::invalid_argument("CausalDepthwiseConv1D history and output must not alias");
      for (const auto* tensor : {&input, &weight, bias}) {
        if (!tensor)
          continue;
        for (const auto* mutable_tensor : {&history, &output}) {
          if (tensor == mutable_tensor
              || overlaps(*tensor, tensor->size() * tensor->item_size(), *mutable_tensor,
                          mutable_tensor->reserved_memory()))
            throw std::invalid_argument(
                "CausalDepthwiseConv1D inputs must not alias history or output");
        }
      }
    }

    void CausalDepthwiseConv1D::operator()(const StorageView& input, const StorageView& weight,
                                           const StorageView* bias, StorageView& history,
                                           StorageView& output) const {
      PROFILE("CausalDepthwiseConv1D");
      validate_input(input, weight, bias, history, output);
      const ScopedDeviceSetter setter(input.device(), input.device_index());
      DEVICE_DISPATCH(input.device(), (compute<D, float>(input, weight, bias, history, output)));
    }

  }
}
