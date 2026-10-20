#include "ops/drama/moe_topk.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      bool overlaps(const StorageView& a, dim_t a_bytes, const StorageView& b, dim_t b_bytes) {
        if (a_bytes == 0 || b_bytes == 0 || !a.buffer() || !b.buffer())
          return false;
        const auto a_ptr = reinterpret_cast<uintptr_t>(a.buffer());
        const auto b_ptr = reinterpret_cast<uintptr_t>(b.buffer());
        return a_ptr <= b_ptr ? b_ptr - a_ptr < static_cast<uintptr_t>(a_bytes)
                              : a_ptr - b_ptr < static_cast<uintptr_t>(b_bytes);
      }

    }

    MoeTopK::MoeTopK(dim_t k)
      : _k(k) {
      if (k <= 0)
        throw std::invalid_argument("MoeTopK k must be positive");
    }

    void MoeTopK::operator()(const StorageView& router_logits, StorageView& expert_indices,
                             StorageView& routing_weights) const {
      PROFILE("MoeTopK");
      if (router_logits.device() != Device::CPU || expert_indices.device() != Device::CPU
          || routing_weights.device() != Device::CPU || router_logits.dtype() != DataType::FLOAT32
          || expert_indices.dtype() != DataType::INT32
          || routing_weights.dtype() != DataType::FLOAT32)
        throw std::invalid_argument(
            "MoeTopK requires CPU float32 logits/weights and int32 indices");
      if (router_logits.rank() != 2 || router_logits.dim(0) < 0 || router_logits.dim(1) <= 0
          || _k > router_logits.dim(1)
          || router_logits.dim(1) > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument(
            "MoeTopK expects [rows, experts] with 0 < k <= experts <= INT32_MAX");
      const auto input_bytes = router_logits.size() * router_logits.item_size();
      for (const auto* output : {&expert_indices, &routing_weights}) {
        if (&router_logits == output
            || overlaps(router_logits, input_bytes, *output, output->reserved_memory()))
          throw std::invalid_argument("MoeTopK outputs must not alias logits");
      }
      if (&expert_indices == &routing_weights
          || overlaps(expert_indices, expert_indices.reserved_memory(), routing_weights,
                      routing_weights.reserved_memory()))
        throw std::invalid_argument("MoeTopK outputs must not alias each other");
      const auto* logits = router_logits.data<float>();
      for (dim_t i = 0; i < router_logits.size(); ++i) {
        if (!std::isfinite(logits[i]))
          throw std::invalid_argument("MoeTopK logits must be finite");
      }

      const dim_t rows = router_logits.dim(0);
      const dim_t experts = router_logits.dim(1);
      expert_indices.resize({rows, _k});
      routing_weights.resize({rows, _k});
      auto* indices = expert_indices.data<int32_t>();
      auto* weights = routing_weights.data<float>();
      cpu::parallel_for(0, rows, 1, [&](dim_t begin, dim_t end) {
        std::vector<int32_t> order(experts);
        std::vector<double> probabilities(_k);
        for (dim_t row = begin; row < end; ++row) {
          const auto* values = logits + row * experts;
          std::iota(order.begin(), order.end(), int32_t(0));
          std::partial_sort(order.begin(), order.begin() + _k, order.end(),
                            [&](int32_t a, int32_t b) {
                              return values[a] > values[b] || (values[a] == values[b] && a < b);
                            });
          const double maximum = values[order[0]];
          double sum = 0;
          for (dim_t i = 0; i < _k; ++i) {
            probabilities[i] = std::exp(static_cast<double>(values[order[i]]) - maximum);
            sum += probabilities[i];
          }
          for (dim_t i = 0; i < _k; ++i) {
            indices[row * _k + i] = order[i];
            weights[row * _k + i] = static_cast<float>(probabilities[i] / sum);
          }
        }
      });
    }

  }
}
