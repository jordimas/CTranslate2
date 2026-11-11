#include "ctranslate2/ops/gemm.h"

#include "ctranslate2/ops/bias_add.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // activation_type(x + bias + residual)
    void apply_bias_and_activation(StorageView& x,
                                   const StorageView* bias,
                                   const ActivationType* activation_type,
                                   const StorageView* residual,
                                   const dim_t axis) {
      if (bias) {
        const BiasAdd bias_add_op(activation_type, axis);
        bias_add_op(x, *bias, x, residual);
      } else {
        if (residual)
          Add()(*residual, x, x);
        if (activation_type)
          get_activation_op(*activation_type)(x, x);
      }
    }


    Gemm::Gemm(float alpha,
               float beta,
               bool trans_a,
               bool trans_b,
               bool a_is_packed,
               bool b_is_packed,
               const ActivationType* activation_type)
      : _activation_type(activation_type)
      , _alpha(alpha)
      , _beta(beta)
      , _trans_a(trans_a)
      , _trans_b(trans_b)
      , _a_is_packed(a_is_packed)
      , _b_is_packed(b_is_packed)
    {
    }

    template<>
    bool Gemm::try_compute_with_epilogue<Device::CPU>(const StorageView&,
                                                       const StorageView&,
                                                       StorageView&,
                                                       const StorageView*,
                                                       const StorageView*,
                                                       const StorageView*) const {
      return false;
    }

    template<>
    bool Gemm::try_compute_with_epilogue<Device::CUDA>(const StorageView&,
                                                        const StorageView&,
                                                        StorageView&,
                                                        const StorageView*,
                                                        const StorageView*,
                                                        const StorageView*) const {
      return false;
    }

    template<>
    bool Gemm::try_weight_only_int8<Device::CPU>(const StorageView& input,
                                                  const StorageView& weight,
                                                  const StorageView& scale,
                                                  StorageView& output,
                                                  const ActivationType* activation,
                                                  const StorageView* bias,
                                                  const StorageView* residual) const {
     if (input.dtype() != DataType::FLOAT32
          || weight.dtype() != DataType::INT8
          || scale.dtype() != DataType::FLOAT32
          || (bias && bias->dtype() != DataType::FLOAT32)
          || (residual && residual->dtype() != DataType::FLOAT32)
          || _beta != 0.0f)
        return false;

      const dim_t k = input.dim(_trans_a ? -2 : -1);
      const dim_t n = weight.dim(_trans_b ? -2 : -1);
      const dim_t m = input.size() / k;

      // Only take the weight-only int8 path for M=1 (decode / memory-bound).
      // For larger M, oneDNN's packed INT8 GEMM is faster than this kernel.
      if (m != 1)
        return false;

      if (scale.size() != n)
        return false;

      Shape output_shape(input.shape());
      output_shape[output_shape.size() - 2] = input.dim(_trans_a ? -1 : -2);
      output_shape[output_shape.size() - 1] = n;
      output.resize(std::move(output_shape));

      const float* input_data = input.data<float>();
      const int8_t* weight_data = weight.data<int8_t>();
      const float* scale_data = scale.data<float>();
      float* output_data = output.data<float>();

      for (dim_t m_i = 0; m_i < m; ++m_i) {
        #pragma omp parallel for
        for (dim_t n_i = 0; n_i < n; ++n_i) {
          float acc = 0.0f;
          for (dim_t k_i = 0; k_i < k; ++k_i) {
            const float a = _trans_a
              ? input_data[k_i * m + m_i]
              : input_data[m_i * k + k_i];
            const int8_t w = _trans_b
              ? weight_data[n_i * k + k_i]
              : weight_data[k_i * n + n_i];
            acc += a * static_cast<float>(w);
          }
          output_data[m_i * n + n_i] = _alpha * acc / scale_data[n_i];
        }
      }

      apply_bias_and_activation(output, bias, activation, residual);
      return true;
    }

    template<>
    bool Gemm::try_weight_only_int8<Device::CUDA>(const StorageView&,
                                                   const StorageView&,
                                                   const StorageView&,
                                                   StorageView&,
                                                   const ActivationType*,
                                                   const StorageView*,
                                                   const StorageView*) const {
      return false;
    }

    bool Gemm::try_weight_only_int8(const StorageView& input,
                                    const StorageView& weight,
                                    const StorageView& scale,
                                    StorageView& output,
                                    const ActivationType* activation,
                                    const StorageView* bias,
                                    const StorageView* residual) const {
      bool supported = false;
      DEVICE_DISPATCH(input.device(),
                      supported = try_weight_only_int8<D>(input,
                                                          weight,
                                                          scale,
                                                          output,
                                                          activation,
                                                          bias,
                                                          residual));
      return supported;
    }

    void Gemm::operator()(const StorageView& a,
                          const StorageView& b,
                          StorageView& c,
                          const StorageView* a_shift_compensation,
                          const StorageView* bias,
                          const StorageView* residual) const {
      PROFILE("Gemm");

      bool fused = false;
      DEVICE_DISPATCH(a.device(),
                      fused = try_compute_with_epilogue<D>(a,
                                                           b,
                                                           c,
                                                           a_shift_compensation,
                                                           bias,
                                                           residual));
      if (fused)
        return;

      switch (a.dtype()) {
      case DataType::INT8:
        DEVICE_DISPATCH(a.device(), (compute<D, int8_t, int32_t>(a, b, c, a_shift_compensation)));
        break;

      case DataType::INT16:
        if (a.device() != Device::CPU)
          throw std::invalid_argument("INT16 GEMM is only supported on CPU");
        compute<Device::CPU, int16_t, int32_t>(a, b, c, a_shift_compensation);
        break;

      case DataType::FLOAT32:
      case DataType::FLOAT16:
      case DataType::BFLOAT16: {
        DEVICE_AND_FLOAT_DISPATCH("Gemm", a.device(), a.dtype(),
                                  (compute<D, T, T>(a, b, c, a_shift_compensation)));
        break;
      }

      default:
        throw std::invalid_argument("Gemm: unsupported input type " + dtype_name(a.dtype()));
      }

      apply_bias_and_activation(c, bias, _activation_type, residual);
    }

    template <Device D, typename In, typename Out>
    void Gemm::compute(const StorageView& a,
                       const StorageView& b,
                       StorageView& c,
                       const StorageView* a_shift_compensation) const {
      const dim_t k = a.dim(_trans_a ? -2 : -1);
      const dim_t n = b.dim(_trans_b ? -2 : -1);
      const dim_t m = a.size() / k;  // Collapse leading dimensions.
      const dim_t lda = _trans_a ? m : k;
      const dim_t ldb = _trans_b ? k : n;
      const dim_t ldc = n;

      {
        Shape output_shape(a.shape());
        output_shape[output_shape.size() - 2] = a.dim(_trans_a ? -1 : -2); // m
        output_shape[output_shape.size() - 1] = n;
        c.resize(std::move(output_shape));
      }

      primitives<D>::gemm(_a_is_packed, _b_is_packed,
                          _trans_a, _trans_b,
                          m, n, k,
                          _alpha,
                          a.data<In>(), lda,
                          b.data<In>(), ldb,
                          _beta,
                          c.data<Out>(), ldc,
                          a_shift_compensation ? a_shift_compensation->data<Out>() : nullptr);
    }

    template <typename T>
    static void pack_b(const StorageView& b,
                       const bool transpose,
                       const dim_t k,
                       const dim_t n,
                       const float alpha,
                       StorageView& packed) {
      const T* src = b.data<T>();
      const dim_t pack_bytes = primitives<Device::CPU>::gemm_pack_b(src,
                                                                    transpose,
                                                                    k, n,
                                                                    alpha);

      if (pack_bytes == 0)  // Packed Gemm is not supported.
        throw std::runtime_error("Packed GEMM APIs are not supported by this GEMM backend");

      const dim_t pack_size = pack_bytes / sizeof (T);
      const dim_t b_size = b.size();

      // We want the packed storage to have the same shape as the original weight
      // so that operators can query its shape, but also have enough space to store
      // the packed data.
      packed.reserve(std::max(b_size, pack_size));
      packed.resize_as(b);

      primitives<Device::CPU>::gemm_pack_b(src,
                                           transpose,
                                           k, n,
                                           alpha,
                                           packed.data<T>());
    }

    StorageView Gemm::pack_b_input(const StorageView& b,
                                   const bool transpose,
                                   const dim_t k,
                                   const dim_t n,
                                   const float alpha) {
      if (b.device() != Device::CPU)
        throw std::invalid_argument("Packed GEMM APIs are only defined on CPU");

      DataType dtype = b.dtype();
      StorageView packed(dtype);

      switch (dtype) {
      case DataType::FLOAT32:
        pack_b<float>(b, transpose, k, n, alpha, packed);
        break;
      case DataType::INT16:
        pack_b<int16_t>(b, transpose, k, n, alpha, packed);
        break;
      case DataType::INT8:
        pack_b<int8_t>(b, transpose, k, n, alpha, packed);
        break;
      default:
        throw std::invalid_argument("Cannot pack GEMM input of type " + dtype_name(dtype));
        break;
      }

      return packed;
    }

    StorageView Gemm::compensate_u8_input(const StorageView& b,
                                          const bool transpose,
                                          const dim_t k,
                                          const dim_t n,
                                          const float alpha) {
      if (b.device() != Device::CPU && b.dtype() != DataType::INT8)
        throw std::invalid_argument("Unsigned input compensation is only defined for "
                                    "INT8 GEMM on CPU");

      StorageView compensation({n}, DataType::INT32);
      primitives<Device::CPU>::compute_u8_compensation(b.data<int8_t>(),
                                                       transpose,
                                                       k, n,
                                                       alpha,
                                                       compensation.data<int32_t>());
      return compensation;
    }

  }
}
