#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
namespace experimental {
// Diagnostic prototype for oneDNN 3.1.1 / AVX2, not a supported backend API.
bool planned_attention(const StorageView& queries, const StorageView& keys,
                       const StorageView& values, float scale, StorageView& output);
}
}
