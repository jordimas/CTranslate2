#include "ctranslate2/ops/gated_delta_net.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "device_dispatch.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      bool overlaps(const StorageView& a, const StorageView& b) {
        const dim_t a_bytes = std::max(a.reserved_memory(), a.size() * a.item_size());
        const dim_t b_bytes = std::max(b.reserved_memory(), b.size() * b.item_size());
        if (a_bytes == 0 || b_bytes == 0)
          return false;
        const auto a_begin = reinterpret_cast<uintptr_t>(a.buffer());
        const auto b_begin = reinterpret_cast<uintptr_t>(b.buffer());
        return a_begin < b_begin + b_bytes && b_begin < a_begin + a_bytes;
      }
    }

    void GatedDeltaNet::validate_input(const StorageView& query, const StorageView& key,
                                       const StorageView& value, const StorageView& log_decay,
                                       const StorageView& beta, const StorageView& state,
                                       const StorageView& output) const {
      const StorageView* inputs[] = {&query, &key, &value, &log_decay, &beta};
      for (const auto* tensor :
           {&query, &key, &value, &log_decay, &beta, static_cast<const StorageView*>(&state),
            static_cast<const StorageView*>(&output)}) {
        if (tensor->device() != query.device()
            || tensor->device_index() != query.device_index()
            || tensor->dtype() != DataType::FLOAT32)
          throw std::invalid_argument("GatedDeltaNet requires float32 tensors on the same device");
      }
      if (query.rank() != 4 || key.shape() != query.shape() || value.rank() != 4)
        throw std::invalid_argument(
            "GatedDeltaNet expects matching rank-4 query/key and rank-4 value");
      const dim_t batch = query.dim(0);
      const dim_t steps = query.dim(1);
      const dim_t key_heads = query.dim(2);
      const dim_t key_dim = query.dim(3);
      const dim_t heads = value.dim(2);
      const dim_t value_dim = value.dim(3);
      if (batch <= 0 || steps < 0 || key_heads <= 0 || key_dim <= 0 || heads <= 0 || value_dim <= 0
          || heads % key_heads != 0 || value.dim(0) != batch || value.dim(1) != steps)
        throw std::invalid_argument("GatedDeltaNet has incompatible query/value dimensions");
      if (log_decay.shape() != Shape{batch, steps, heads} || beta.shape() != log_decay.shape())
        throw std::invalid_argument(
            "GatedDeltaNet gates must have shape [batch, time, value_heads]");
      if (state.shape() != Shape{batch, heads, key_dim, value_dim})
        throw std::invalid_argument(
            "GatedDeltaNet state must have shape [batch, value_heads, key_dim, value_dim]");
      if (&state == &output || overlaps(state, output))
        throw std::invalid_argument("GatedDeltaNet state and output must not alias");
      for (const auto* input : inputs) {
        if (&output == input || overlaps(*input, state) || overlaps(*input, output))
          throw std::invalid_argument("GatedDeltaNet inputs must not alias state or output");
      }
    }

    void GatedDeltaNet::operator()(const StorageView& query, const StorageView& key,
                                   const StorageView& value, const StorageView& log_decay,
                                   const StorageView& beta, StorageView& state,
                                   StorageView& output) const {
      PROFILE("GatedDeltaNet");
      validate_input(query, key, value, log_decay, beta, state, output);
      const ScopedDeviceSetter setter(query.device(), query.device_index());
      DEVICE_DISPATCH(query.device(),
                      (compute<D, float>(query, key, value, log_decay, beta, state, output)));
    }

  }
}
