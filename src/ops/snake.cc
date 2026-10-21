#include "ctranslate2/ops/snake.h"

#include <cstdint>
#include <stdexcept>

#include "device_dispatch.h"

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
        if (tensor->device() != x.device() || tensor->device_index() != x.device_index()
            || tensor->dtype() != DataType::FLOAT32)
          throw std::invalid_argument("Snake requires float32 tensors on the same device");
      }
      if (x.rank() != 3 || x.dim(0) <= 0 || x.dim(1) <= 0 || x.dim(2) < 0)
        throw std::invalid_argument("Snake expects [batch, channels, time]");
      const dim_t channels = x.dim(1);
      if (alpha.shape() != Shape{channels} && alpha.shape() != Shape{1, channels, 1})
        throw std::invalid_argument("Snake alpha must have shape [channels] or [1, channels, 1]");
      if (&y == &x || &y == &alpha || overlaps_allocation(x, y) || overlaps_allocation(alpha, y))
        throw std::invalid_argument("Snake output must not alias inputs");
      const ScopedDeviceSetter setter(x.device(), x.device_index());
      DEVICE_DISPATCH(x.device(), (compute<D, float>(x, alpha, y)));
    }

  }
}
