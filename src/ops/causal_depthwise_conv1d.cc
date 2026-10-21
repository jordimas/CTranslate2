#include "ctranslate2/ops/causal_depthwise_conv1d.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cpu/parallel.h"

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
        if (tensor && (tensor->device() != Device::CPU || tensor->dtype() != DataType::FLOAT32))
          throw std::invalid_argument("CausalDepthwiseConv1D requires CPU float32 tensors");
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
      for (const auto* tensor :
           {&input, &weight, bias, static_cast<const StorageView*>(&history)}) {
        if (tensor) {
          const auto* data = tensor->data<float>();
          for (dim_t i = 0; i < tensor->size(); ++i) {
            if (!std::isfinite(data[i]))
              throw std::invalid_argument(
                  "CausalDepthwiseConv1D inputs and history must be finite");
          }
        }
      }
    }

    void CausalDepthwiseConv1D::operator()(const StorageView& input, const StorageView& weight,
                                           const StorageView* bias, StorageView& history,
                                           StorageView& output) const {
      PROFILE("CausalDepthwiseConv1D");
      validate_input(input, weight, bias, history, output);
      const dim_t batch = input.dim(0), steps = input.dim(1), channels = input.dim(2);
      const dim_t kernel = weight.dim(1), past = kernel - 1;
      output.resize_as(input);
      if (steps == 0)
        return;
      const auto* x = input.data<float>();
      const auto* w = weight.data<float>();
      const auto* b = bias ? bias->data<float>() : nullptr;
      auto* state = history.data<float>();
      auto* y = output.data<float>();
      cpu::parallel_for(0, batch * channels, 1, [&](dim_t begin, dim_t end) {
        std::vector<float> ring(past);
        for (dim_t bc = begin; bc < end; ++bc) {
          const dim_t batch_index = bc / channels;
          const dim_t c = bc % channels;
          for (dim_t i = 0; i < past; ++i)
            ring[i] = state[bc * past + i];
          dim_t cursor = 0;
          for (dim_t t = 0; t < steps; ++t) {
            const dim_t offset = (batch_index * steps + t) * channels + c;
            float value = b ? b[c] : 0.f;
            for (dim_t k = 0; k < past; ++k)
              value += ring[(cursor + k) % past] * w[c * kernel + k];
            value += x[offset] * w[c * kernel + past];
            y[offset] = _apply_silu ? value / (1.f + std::exp(-value)) : value;
            if (past) {
              ring[cursor] = x[offset];
              cursor = (cursor + 1) % past;
            }
          }
          for (dim_t i = 0; i < past; ++i)
            state[bc * past + i] = ring[(cursor + i) % past];
        }
      });
    }

  }
}
