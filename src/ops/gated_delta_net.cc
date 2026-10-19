#include "ctranslate2/ops/gated_delta_net.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cpu/parallel.h"

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

    void GatedDeltaNet::operator()(const StorageView& query, const StorageView& key,
                                   const StorageView& value, const StorageView& log_decay,
                                   const StorageView& beta, StorageView& state,
                                   StorageView& output) const {
      PROFILE("GatedDeltaNet");
      const StorageView* inputs[] = {&query, &key, &value, &log_decay, &beta};
      for (const auto* tensor :
           {&query, &key, &value, &log_decay, &beta, static_cast<const StorageView*>(&state),
            static_cast<const StorageView*>(&output)}) {
        if (tensor->device() != Device::CPU || tensor->dtype() != DataType::FLOAT32)
          throw std::invalid_argument("GatedDeltaNet currently requires CPU float32 tensors");
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

      output.resize({batch, steps, heads, value_dim});
      const auto* q = query.data<float>();
      const auto* k = key.data<float>();
      const auto* v = value.data<float>();
      const auto* g = log_decay.data<float>();
      const auto* b = beta.data<float>();
      auto* s = state.data<float>();
      auto* out = output.data<float>();
      const float scale = 1.f / std::sqrt(static_cast<float>(key_dim));
      const dim_t heads_per_key = heads / key_heads;

      cpu::parallel_for(0, batch * heads, 1, [&](dim_t begin, dim_t end) {
        std::vector<float> residual(value_dim);
        for (dim_t bh = begin; bh < end; ++bh) {
          const dim_t batch_index = bh / heads;
          const dim_t head = bh % heads;
          auto* memory = s + bh * key_dim * value_dim;
          for (dim_t t = 0; t < steps; ++t) {
            const dim_t row = (batch_index * steps + t) * heads + head;
            const dim_t key_row = (batch_index * steps + t) * key_heads + head / heads_per_key;
            const auto* qt = q + key_row * key_dim;
            const auto* kt = k + key_row * key_dim;
            const auto* vt = v + row * value_dim;
            auto* ot = out + row * value_dim;
            const float decay = std::exp(g[row]);
            for (dim_t i = 0; i < key_dim * value_dim; ++i)
              memory[i] *= decay;
            // Accumulate each column in the original key-dimension order while
            // traversing contiguous state rows, allowing SIMD across values.
            std::fill(residual.begin(), residual.end(), 0.f);
            for (dim_t i = 0; i < key_dim; ++i) {
              const auto* memory_row = memory + i * value_dim;
              for (dim_t j = 0; j < value_dim; ++j)
                residual[j] += kt[i] * memory_row[j];
            }
            for (dim_t j = 0; j < value_dim; ++j) {
              residual[j] = b[row] * (vt[j] - residual[j]);
              ot[j] = 0;
            }
            for (dim_t i = 0; i < key_dim; ++i) {
              for (dim_t j = 0; j < value_dim; ++j) {
                auto& cell = memory[i * value_dim + j];
                cell += kt[i] * residual[j];
                ot[j] += qt[i] * scale * cell;
              }
            }
          }
        }
      });
    }

  }
}
