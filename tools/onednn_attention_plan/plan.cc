// Portions of the PV scheduling policy are adapted from oneDNN's
// src/cpu/x64/gemm/f32/jit_avx_gemm_f32.cpp, Copyright 2016-2022 Intel
// Corporation, licensed under Apache License 2.0:
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software under
// that license is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the license for the
// specific language governing permissions and limitations under the license.
//
// Experimental adapter to oneDNN 3.1.1's private SGEMM kernels. Never linked by
// the normal CT2 build. See README.md for the arithmetic and eligibility limits.
#include "plan.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <oneapi/dnnl/dnnl.h>
#include "cpu/x64/gemm/gemm_info.hpp"
#include "cpu/x64/gemm/f32/jit_avx_gemm_f32.hpp"
#include "cpu/cpu_isa.h"
#include "cpu/kernels.h"
#include "cpu/parallel.h"

namespace ctranslate2 {
namespace experimental {
namespace {
namespace x64 = dnnl::impl::cpu::x64;
using Info = x64::gemm_info_t<float, float, float>;
using D = dnnl::impl::dim_t;
struct Tile { D begin, size; };

// The reference PV no-copy driver chooses NN column blocks of 48, coalescing
// the final short remainder. Build that schedule ONCE for all 1500 queries.
// Outer tiles group these blocks; no tile is allowed to change a boundary.
std::vector<Tile> pv_blocks(D queries) {
  std::vector<Tile> blocks;
  for (D q = 0; q < queries;) {
    D n = queries - q;
    if (n >= 96) n = 48;
    else if (n > 72) n = (n + 1) / 2;
    blocks.push_back({q, n});
    q += n;
  }
  return blocks;
}

struct Plan {
  D t = 1500, depth = 64;
  float alpha = .125f, zero = 0.f;
  Info qk;
  std::vector<Tile> blocks = pv_blocks(t);
  std::vector<Tile> tiles;
  explicit Plan(D target)
    : qk("T", "N", nullptr, &t, &t, &depth, &alpha,
         nullptr, &depth, nullptr, nullptr, &depth, nullptr, &zero,
         nullptr, &t, nullptr, false, x64::pack_type::none, nullptr, false) {
    // QK is copy-based at the full shape on AVX2. PV is no-copy (m=64).
    // Restrict K to one complete QK reduction; no new K partition is created.
    if (!qk.hasKernels() || qk.bk_traditional < depth)
      throw std::runtime_error("unexpected oneDNN QK kernel configuration");
    for (const auto& block : blocks) {
      if (tiles.empty() || tiles.back().size + block.size > target)
        tiles.push_back(block);
      else
        tiles.back().size += block.size;
    }
  }
};
}

bool planned_attention(const StorageView& queries, const StorageView& keys,
                       const StorageView& values, float scale, StorageView& output) {
  const char* enabled = std::getenv("CT2_DIAG_DNNL_PLAN");
  if (!enabled || enabled[0] == '0') return false;
  const auto* version = dnnl_version();
  if (version->major != 3 || version->minor != 1 || version->patch != 1
      || dnnl_get_effective_cpu_isa() != dnnl_cpu_isa_avx2
      || queries.dim(2) != 1500 || keys.dim(2) != 1500
      || queries.dim(3) != 64 || values.dim(3) != 64 || scale != .125f
      || queries.dim(0) * queries.dim(1) < 2)
    return false;

  // A single immutable full-shape plan is shared by all heads and layers.
  // Input-dependent packed keys remain local to the current head invocation.
  static const Plan plan([] {
    const char* value = std::getenv("CT2_DIAG_PLAN_TARGET");
    return value ? std::max(96, std::min(1500, std::atoi(value))) : 256;
  }());
  static const bool logged = [] {
    std::fprintf(stderr, "CT2 planned SGEMM prototype: full=1500, depth=64, "
                         "QK um=%ld un=%ld; PV blocks=%zu; outer tiles=%zu\n",
                 long(plan.qk.um), long(plan.qk.un), plan.blocks.size(), plan.tiles.size());
    return true;
  }();
  (void)logged;
  output.resize({queries.dim(0), queries.dim(1), plan.t, plan.depth});
  const D heads = queries.dim(0) * queries.dim(1);
  D max_tile = 0;
  for (const auto& tile : plan.tiles) max_tile = std::max(max_tile, tile.size);
  const bool verify = std::getenv("CT2_DIAG_PLAN_VERIFY") != nullptr;
  std::atomic<size_t> qk_changes{0}, pv_changes{0};
  cpu::parallel_for(0, heads, 1, [&](dim_t begin, dim_t end) {
    StorageView packed_keys({plan.t, plan.depth}, DataType::FLOAT32);
    StorageView packed_queries({max_tile, plan.depth}, DataType::FLOAT32);
    StorageView scores({max_tile, plan.t}, DataType::FLOAT32);
    const D ld = plan.depth, t = plan.t, k = plan.depth;
    const float one = 1.f, zero = 0.f;
    for (D h = begin; h < end; ++h) {
      const float* q = queries.data<float>() + h * t * k;
      const float* key = keys.data<float>() + h * t * k;
      const float* v = values.data<float>() + h * t * k;
      float* out = output.data<float>() + h * t * k;
      StorageView reference;
      if (verify) {
        reference.resize({t, t});
        const auto status = dnnl_sgemm('N', 'T', t, t, k, plan.alpha,
                                      q, k, key, k, zero, reference.data<float>(), t);
        if (status != dnnl_success) std::abort();
      }
      // Same alpha placement and copy kernel as the full SGEMM, once per head.
      plan.qk.copyA(&k, &t, key, &ld, &plan.alpha, packed_keys.data<float>(),
                    nullptr, nullptr, nullptr);
      size_t pv_index = 0;
      for (const auto& tile : plan.tiles) {
        const D nq = tile.size;
        plan.qk.copyB(&k, &nq, q + tile.begin * k, &ld, &one,
                      packed_queries.data<float>(), nullptr, nullptr, nullptr);
        for (D m0 = 0; m0 < t; m0 += plan.qk.um) {
          const D m = std::min(plan.qk.um, t - m0);
          plan.qk.kernel[x64::do_beta0][0][0](
            &m, &nq, &k, &plan.alpha, packed_keys.data<float>() + m0 * k,
            packed_queries.data<float>(), scores.data<float>() + m0,
            t, nullptr, nullptr);
        }
        if (verify) {
          const float* ref = reference.data<float>() + tile.begin * t;
          size_t changed = 0;
          for (D i = 0; i < nq * t; ++i)
            changed += std::memcmp(ref + i, scores.data<float>() + i, sizeof(float)) != 0;
          qk_changes.fetch_add(changed, std::memory_order_relaxed);
        }
        CPU_ISA_DISPATCH((cpu::softmax<ISA>(scores.data<float>(), nullptr,
                                          scores.data<float>(), nq, t, false)));
        // Execute precisely the full-shape PV schedule, including its tail.
        // This driver is explicitly no-copy; generic SGEMM cannot reselect it.
        while (pv_index < plan.blocks.size()
               && plan.blocks[pv_index].begin < tile.begin + tile.size) {
          const auto& block = plan.blocks[pv_index++];
          const auto status = x64::avx_gemm_f32::sgemm_nocopy_driver(
            "N", "N", k, block.size, t, &one, v, k,
            scores.data<float>() + (block.begin - tile.begin) * t,
            t, &zero, out + block.begin * k, k, nullptr);
          if (status != dnnl_success) std::abort();
        }
      }
      if (verify) {
        CPU_ISA_DISPATCH((cpu::softmax<ISA>(reference.data<float>(), nullptr,
                                          reference.data<float>(), t, t, false)));
        StorageView ref_out({t, k}, DataType::FLOAT32);
        const auto status = dnnl_sgemm('N', 'N', t, k, t, one,
                                      reference.data<float>(), t, v, k,
                                      zero, ref_out.data<float>(), k);
        if (status != dnnl_success) std::abort();
        size_t changed = 0;
        for (D i = 0; i < t * k; ++i)
          changed += std::memcmp(ref_out.data<float>() + i, out + i, sizeof(float)) != 0;
        pv_changes.fetch_add(changed, std::memory_order_relaxed);
      }
    }
  });
  if (verify) {
    std::fprintf(stderr, "planned stage verification: QK changed=%zu; final changed=%zu\n",
                 qk_changes.load(), pv_changes.load());
    if (qk_changes || pv_changes) std::abort();
  }
  return true;
}
}
}
