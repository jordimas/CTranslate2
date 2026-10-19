#include "ctranslate2/ops/conv_transpose1d.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "cpu/parallel.h"
#include "cpu/backend.h"
#include "ctranslate2/ops/gemm.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      bool overlaps(const StorageView& a, const StorageView& b) {
        const auto an = std::max(a.reserved_memory(), a.size() * a.item_size());
        const auto bn = std::max(b.reserved_memory(), b.size() * b.item_size());
        if (!an || !bn)
          return false;
        const auto ap = reinterpret_cast<uintptr_t>(a.buffer());
        const auto bp = reinterpret_cast<uintptr_t>(b.buffer());
        return ap <= bp ? bp - ap < static_cast<uintptr_t>(an)
                        : ap - bp < static_cast<uintptr_t>(bn);
      }

      dim_t checked_product(dim_t a, dim_t b) {
        if (a && b > std::numeric_limits<dim_t>::max() / a)
          throw std::invalid_argument("ConvTranspose1D dimensions overflow");
        return a * b;
      }

      dim_t checked_sum(dim_t a, dim_t b) {
        if (b > std::numeric_limits<dim_t>::max() - a)
          throw std::invalid_argument("ConvTranspose1D dimensions overflow");
        return a + b;
      }

    }

    ConvTranspose1D::ConvTranspose1D(dim_t stride, dim_t padding, dim_t output_padding,
                                     dim_t dilation, dim_t groups)
      : _stride(stride)
      , _padding(padding)
      , _output_padding(output_padding)
      , _dilation(dilation)
      , _groups(groups) {
      if (stride <= 0 || dilation <= 0 || groups <= 0 || padding < 0 || output_padding < 0
          || (output_padding >= stride && output_padding >= dilation))
        throw std::invalid_argument("ConvTranspose1D invalid convolution parameters");
    }

    void ConvTranspose1D::operator()(const StorageView& input, const StorageView& weight,
                                     StorageView& output) const {
      compute(input, weight, nullptr, output);
    }

    void ConvTranspose1D::operator()(const StorageView& input, const StorageView& weight,
                                     const StorageView& bias, StorageView& output) const {
      compute(input, weight, &bias, output);
    }

    void ConvTranspose1D::compute(const StorageView& input, const StorageView& weight,
                                  const StorageView* bias, StorageView& output) const {
      PROFILE("ConvTranspose1D");
      for (const auto* tensor : {&input, &weight, static_cast<const StorageView*>(&output), bias}) {
        if (tensor && (tensor->device() != Device::CPU || tensor->dtype() != DataType::FLOAT32))
          throw std::invalid_argument("ConvTranspose1D requires CPU float32 tensors");
      }
      if (input.rank() != 3 || weight.rank() != 3)
        throw std::invalid_argument("ConvTranspose1D expects rank-3 input and weight");
      for (const auto* tensor : {&input, &weight}) {
        for (const auto dim : tensor->shape()) {
          if (dim <= 0)
            throw std::invalid_argument("ConvTranspose1D dimensions must be positive");
        }
      }
      const dim_t batch = input.dim(0), cin = input.dim(1), time = input.dim(2);
      const dim_t cout_group = weight.dim(1), kernel = weight.dim(2);
      if (weight.dim(0) != cin || cin % _groups != 0)
        throw std::invalid_argument("ConvTranspose1D incompatible input channels/groups");
      const dim_t cout = checked_product(cout_group, _groups);
      dim_t length =
          checked_sum(checked_product(time - 1, _stride), checked_product(kernel - 1, _dilation));
      length = checked_sum(length, checked_sum(_output_padding, 1));
      const auto crop = checked_product(_padding, 2);
      if (length <= crop)
        throw std::invalid_argument("ConvTranspose1D output length must be positive");
      length -= crop;
      checked_product(checked_product(checked_product(batch, cout), length), sizeof(float));
      if (bias && bias->shape() != Shape{cout})
        throw std::invalid_argument("ConvTranspose1D bias must have shape [Cout]");
      for (const auto* tensor : {&input, &weight, bias}) {
        if (tensor && (tensor == &output || overlaps(*tensor, output)))
          throw std::invalid_argument("ConvTranspose1D output must not alias inputs");
      }

      output.resize({batch, cout, length});
      const auto* x = input.data<float>();
      const auto* w = weight.data<float>();
      const auto* b = bias ? bias->data<float>() : nullptr;
      auto* y = output.data<float>();
      const dim_t cin_group = cin / _groups;
      // W is already [Cin/group, Cout/group * kernel] in each group.
      // Multiplying W^T by X produces the overlapping output contributions;
      // col2im below adds those contributions without reversing the kernel.
      // Bound both the GEMM result and a possible input-tile copy to 16 MiB.
      constexpr dim_t workspace_elements = 16 * 1024 * 1024 / sizeof(float);
      const dim_t contribution_rows = checked_product(cout_group, kernel);
      const dim_t tile_rows = checked_sum(cin_group, contribution_rows);
      if (cin_group >= 8 && time >= 2 && tile_rows <= workspace_elements
          && cpu::has_gemm_backend(ComputeType::FLOAT32)) {
        const dim_t tile_size = std::min(time, workspace_elements / tile_rows);
        StorageView contributions, input_tile;
        const Gemm gemm(1.f, 0.f, true, false);
        cpu::parallel_for(0, batch * cout, 1, [&](dim_t begin, dim_t end) {
          for (dim_t row = begin; row < end; ++row)
            std::fill(y + row * length, y + (row + 1) * length, b ? b[row % cout] : 0.f);
        });
        for (dim_t n = 0; n < batch; ++n) {
          for (dim_t group = 0; group < _groups; ++group) {
            StorageView weight_matrix(
                {cin_group, contribution_rows},
                const_cast<float*>(w + group * cin_group * contribution_rows));
            const auto* group_input = x + (n * cin + group * cin_group) * time;
            for (dim_t start = 0; start < time; start += tile_size) {
              const dim_t count = std::min(tile_size, time - start);
              StorageView input_matrix;
              if (count == time) {
                input_matrix.view(const_cast<float*>(group_input), {cin_group, count});
              } else {
                input_tile.resize({cin_group, count});
                for (dim_t ic = 0; ic < cin_group; ++ic)
                  std::copy_n(group_input + ic * time + start, count,
                              input_tile.data<float>() + ic * count);
                input_matrix.view(input_tile.data<float>(), {cin_group, count});
              }
              gemm(weight_matrix, input_matrix, contributions);
              const auto* values = contributions.data<float>();
              cpu::parallel_for(0, cout_group, 1, [&](dim_t begin, dim_t end) {
                for (dim_t oc = begin; oc < end; ++oc) {
                  auto* out = y + (n * cout + group * cout_group + oc) * length;
                  for (dim_t k = 0; k < kernel; ++k) {
                    const auto* column = values + (oc * kernel + k) * count;
                    for (dim_t t = 0; t < count; ++t) {
                      const dim_t pos = (start + t) * _stride - _padding + k * _dilation;
                      if (pos >= 0 && pos < length)
                        out[pos] += column[t];
                    }
                  }
                }
              });
            }
          }
        }
        return;
      }
      // Each worker owns one output channel; all scatter additions are local.
      cpu::parallel_for(0, batch * cout, 1, [&](dim_t begin, dim_t end) {
        for (dim_t row = begin; row < end; ++row) {
          const dim_t n = row / cout, oc = row % cout;
          const dim_t group = oc / cout_group, local_oc = oc % cout_group;
          auto* out = y + row * length;
          std::fill(out, out + length, b ? b[oc] : 0.f);
          for (dim_t local_ic = 0; local_ic < cin_group; ++local_ic) {
            const auto ic = group * cin_group + local_ic;
            const auto* in = x + (n * cin + ic) * time;
            const auto* filter = w + (ic * cout_group + local_oc) * kernel;
            for (dim_t t = 0; t < time; ++t) {
              for (dim_t k = 0; k < kernel; ++k) {
                const dim_t pos = t * _stride - _padding + k * _dilation;
                if (pos >= 0 && pos < length)
                  out[pos] += in[t] * filter[k];
              }
            }
          }
        }
      });
    }

  }
}
