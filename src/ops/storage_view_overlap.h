#pragma once

#include <algorithm>
#include <cstdint>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace ops {

    inline bool storage_views_overlap(const StorageView& a, dim_t a_bytes,
                                      const StorageView& b, dim_t b_bytes) {
      if (!a.buffer() || !b.buffer() || a_bytes == 0 || b_bytes == 0)
        return false;
      const auto a_ptr = reinterpret_cast<uintptr_t>(a.buffer());
      const auto b_ptr = reinterpret_cast<uintptr_t>(b.buffer());
      return a_ptr <= b_ptr ? b_ptr - a_ptr < static_cast<uintptr_t>(a_bytes)
                            : a_ptr - b_ptr < static_cast<uintptr_t>(b_bytes);
    }

    inline bool storage_views_overlap(const StorageView& a, const StorageView& b) {
      return storage_views_overlap(a, std::max(a.reserved_memory(), a.size() * a.item_size()),
                                   b, std::max(b.reserved_memory(), b.size() * b.item_size()));
    }

  }
}
