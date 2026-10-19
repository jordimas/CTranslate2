#include "ctranslate2/ops/snake.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      bool overlaps_allocation(const StorageView& input, const StorageView& output) {
        if (input.empty() || !output.buffer() || output.reserved_memory() == 0)
          return false;
        const auto a = reinterpret_cast<uintptr_t>(input.buffer());
        const auto b = reinterpret_cast<uintptr_t>(output.buffer());
        const auto a_size = static_cast<uintptr_t>(input.size() * input.item_size());
        const auto b_size = static_cast<uintptr_t>(output.reserved_memory());
        return a <= b ? b - a < a_size : a - b < b_size;
      }

    }

    void Snake::operator()(const StorageView& x, const StorageView& alpha, StorageView& y) const {
      PROFILE("Snake");
      for (const auto* tensor : {&x, &alpha, static_cast<const StorageView*>(&y)}) {
        if (tensor->device() != Device::CPU || tensor->dtype() != DataType::FLOAT32)
          throw std::invalid_argument("Snake requires CPU float32 tensors");
      }
      if (x.rank() != 3 || x.dim(0) <= 0 || x.dim(1) <= 0 || x.dim(2) < 0)
        throw std::invalid_argument("Snake expects [batch, channels, time]");
      const dim_t channels = x.dim(1);
      const dim_t time = x.dim(2);
      if (alpha.shape() != Shape{channels} && alpha.shape() != Shape{1, channels, 1})
        throw std::invalid_argument("Snake alpha must have shape [channels] or [1, channels, 1]");
      if (&y == &x || &y == &alpha || overlaps_allocation(x, y) || overlaps_allocation(alpha, y))
        throw std::invalid_argument("Snake output must not alias inputs");
      const auto* data = x.data<float>();
      const auto* coefficients = alpha.data<float>();
      constexpr float epsilon = 1e-9f;
      for (dim_t c = 0; c < channels; ++c) {
        if (!std::isfinite(coefficients[c]) || coefficients[c] + epsilon == 0.f)
          throw std::invalid_argument("Snake alpha must be finite with nonzero alpha + 1e-9");
      }
      for (dim_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(data[i])
            || !std::isfinite(data[i] * coefficients[(i / time) % channels]))
          throw std::invalid_argument("Snake input and alpha * input must be finite");
      }
      y.resize_as(x);
      auto* output = y.data<float>();
      cpu::parallel_for(0, x.size(), cpu::GRAIN_SIZE, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const float a = coefficients[(i / time) % channels];
          const float sine = std::sin(a * data[i]);
          output[i] = data[i] + (1.f / (a + epsilon)) * (sine * sine);
        }
      });
    }

  }
}
