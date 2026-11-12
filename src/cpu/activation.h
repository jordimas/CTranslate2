#pragma once

#include "ctranslate2/ops/activation.h"
#include "cpu/vec.h"

namespace ctranslate2 {
  namespace cpu {

    template <CpuIsa ISA>
    struct relu_func {
      vec_type<float, ISA> operator()(vec_type<float, ISA> v) const {
        using VecType = Vec<float, ISA>;
        return VecType::max(v, VecType::load(0.f));
      }
    };

    template <CpuIsa ISA>
    struct gelu_func {
      vec_type<float, ISA> operator()(vec_type<float, ISA> v) const {
        using VecType = Vec<float, ISA>;
        auto u = VecType::mul(VecType::load(0.7071067811865475f), v);
        u = VecType::add(VecType::load(1.f), VecType::erf(u));
        u = VecType::mul(v, u);
        u = VecType::mul(VecType::load(0.5f), u);
        return u;
      }
    };

    template <CpuIsa ISA>
    struct gelu_tanh_func {
      vec_type<float, ISA> operator()(vec_type<float, ISA> v) const {
        using VecType = Vec<float, ISA>;
        auto u = VecType::mul(VecType::mul(v, v), v);
        u = VecType::mul_add(VecType::load(0.044715f), u, v);
        u = VecType::mul(VecType::load(0.7978845608028654f), u);
        u = VecType::tanh(u);
        u = VecType::add(VecType::load(1.f), u);
        u = VecType::mul(v, u);
        u = VecType::mul(VecType::load(0.5f), u);
        return u;
      }
    };

    template <CpuIsa ISA>
    struct gelu_sigmoid_func {
      vec_type<float, ISA> operator()(vec_type<float, ISA> v) const {
        using VecType = Vec<float, ISA>;
        return VecType::div(v, VecType::add(VecType::load(1.f),
                                            VecType::exp(VecType::mul(VecType::load(-1.702f), v))));
      }
    };

    template <CpuIsa ISA>
    struct sigmoid_func {
      vec_type<float, ISA> operator()(vec_type<float, ISA> v) const {
        using VecType = Vec<float, ISA>;
        return VecType::div(VecType::load(1.f), VecType::add(VecType::load(1.f), VecType::exp(VecType::neg(v))));
      }
    };

    template <CpuIsa ISA>
    struct swish_func {
      vec_type<float, ISA> operator()(vec_type<float, ISA> v) const {
        using VecType = Vec<float, ISA>;
        return VecType::div(v, VecType::add(VecType::load(1.f), VecType::exp(VecType::neg(v))));
      }
    };

    template <CpuIsa ISA>
    struct tanh_func {
      vec_type<float, ISA> operator()(vec_type<float, ISA> v) const {
        using VecType = Vec<float, ISA>;
        return VecType::tanh(v);
      }
    };


    template <CpuIsa ISA, typename Function>
    static void with_activation(const ops::ActivationType* activation, const Function& func) {
      if (!activation) {
        func(identity());
      } else {
        switch (*activation) {
        case ops::ActivationType::ReLU:        func(relu_func<ISA>()); break;
        case ops::ActivationType::GELU:        func(gelu_func<ISA>()); break;
        case ops::ActivationType::GELUTanh:    func(gelu_tanh_func<ISA>()); break;
        case ops::ActivationType::GELUSigmoid: func(gelu_sigmoid_func<ISA>()); break;
        case ops::ActivationType::Sigmoid:     func(sigmoid_func<ISA>()); break;
        case ops::ActivationType::Swish:       func(swish_func<ISA>()); break;
        case ops::ActivationType::Tanh:        func(tanh_func<ISA>()); break;
        }
      }
    }

  }
}
