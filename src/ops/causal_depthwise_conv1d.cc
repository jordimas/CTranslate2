#include "ctranslate2/ops/causal_depthwise_conv1d.h"

#include <cmath>
#include <algorithm>
#include "ctranslate2/primitives.h"
#include "finite_check.h"
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
          if (!all_finite(data, tensor->size()))
            throw std::invalid_argument(
                "CausalDepthwiseConv1D inputs and history must be finite");
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
      // The tiled path amortizes its separate vector activation on wide, long chunks.
      if (kernel > 4 || channels < 1024 || (steps > 1 && steps < 64)) {
        const dim_t grain = input.size() <= cpu::GRAIN_SIZE / 8 / kernel ? batch * channels : 1;
        cpu::parallel_for(0, batch * channels, grain, [&](dim_t begin, dim_t end) {
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
        return;
      }
      constexpr dim_t channel_tile = 64;
      const dim_t tiles = (channels + channel_tile - 1) / channel_tile;
      const dim_t grain = input.size() <= cpu::GRAIN_SIZE ? batch * tiles : 1;
      cpu::parallel_for(0, batch * tiles, grain, [&](dim_t begin, dim_t end) {
        for (dim_t task = begin; task < end; ++task) {
          const dim_t n = task / tiles;
          const dim_t first = (task % tiles) * channel_tile;
          const dim_t last = std::min(first + channel_tile, channels);
          for (dim_t t = 0; t < steps; ++t) {
            auto* out = y + (n * steps + t) * channels;
            for (dim_t c = first; c < last; ++c)
              out[c] = b ? b[c] : 0.f;
            for (dim_t k = 0; k < kernel; ++k) {
              const dim_t source = t + k - past;
              if (source < 0) {
                for (dim_t c = first; c < last; ++c)
                  out[c] += state[(n * channels + c) * past + source + past] * w[c * kernel + k];
              } else {
                const auto* in = x + (n * steps + source) * channels;
                for (dim_t c = first; c < last; ++c)
                  out[c] += in[c] * w[c * kernel + k];
              }
            }
          }
          for (dim_t c = first; past && c < last; ++c) {
            auto* channel_history = state + (n * channels + c) * past;
            for (dim_t i = 0; i < past; ++i) {
              const dim_t source = steps + i;
              channel_history[i] = source < past ? channel_history[source]
                                                : x[(n * steps + source - past) * channels + c];
            }
          }
        }
      });
      if (_apply_silu)
        primitives<Device::CPU>::swish(y, y, output.size());
    }

  }
}
