#include "cpu_isa.h"

#include <stdexcept>

#include "cpu_info.h"
#include "env.h"

#if defined(CT2_X86_BUILD)
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif

namespace ctranslate2 {
  namespace cpu {

    static CpuIsa try_isa(const std::string& name, CpuIsa cpu_isa, bool supported) {
#ifdef CT2_WITH_CPU_DISPATCH
      if (!supported) {
        throw std::invalid_argument("The CPU does not support " + name);
      } else {
        return cpu_isa;
      }
#else
      (void)cpu_isa;
      (void)supported;
      throw std::invalid_argument("This CTranslate2 binary was not compiled with "
                                  + name + " support");
#endif
    }

    std::string isa_to_str(CpuIsa isa) {
      switch (isa) {
#if defined(CT2_X86_BUILD)
      case CpuIsa::AVX:
        return "AVX";
      case CpuIsa::AVX2:
        return "AVX2";
      case CpuIsa::AVX2_VNNI:
        return "AVX2_VNNI";
      case CpuIsa::AVX512:
        return "AVX512";
#elif defined(CT2_ARM64_BUILD)
      case CpuIsa::NEON:
        return "NEON";
#endif
      default:
        return "GENERIC";
      }
    }

    static CpuIsa init_isa() {
      const std::string env_isa = read_string_from_env("CT2_FORCE_CPU_ISA");
      if (!env_isa.empty()) {
#if defined(CT2_X86_BUILD)
        if (env_isa == "AVX512")
          return try_isa(env_isa, CpuIsa::AVX512, cpu_supports_avx512());
        if (env_isa == "AVX2")
          return try_isa(env_isa, CpuIsa::AVX2, cpu_supports_avx2());
        if (env_isa == "AVX")
          return try_isa(env_isa, CpuIsa::AVX, cpu_supports_avx());
#elif defined(CT2_ARM64_BUILD)
        if (env_isa == "NEON")
          return try_isa(env_isa, CpuIsa::NEON, cpu_supports_neon());
#endif
        if (env_isa == "GENERIC")
          return CpuIsa::GENERIC;

        throw std::invalid_argument("Invalid CPU ISA: " + env_isa);
      }

#ifdef CT2_WITH_CPU_DISPATCH
#  if defined(CT2_X86_BUILD)
      // Note that AVX512 can only be enabled with the environment variable at this time.
      if (cpu_supports_avx2()) {
        // AVX-VNNI (256-bit vpdpbusd): CPUID.(EAX=7,ECX=1):EAX[bit 4].
#  if defined(_MSC_VER)
        int regs[4]; __cpuidex(regs, 7, 1);
        if (regs[0] & (1 << 4)) return CpuIsa::AVX2_VNNI;
#  else
        unsigned int a, b, c, d;
        if (__get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & (1u << 4)))
          return CpuIsa::AVX2_VNNI;
#  endif
        return CpuIsa::AVX2;
      }
      if (cpu_supports_avx())
        return CpuIsa::AVX;
#  elif defined(CT2_ARM64_BUILD)
      if (cpu_supports_neon())
        return CpuIsa::NEON;
#  endif
#endif

      return CpuIsa::GENERIC;
    }

    CpuIsa get_cpu_isa() {
      static const CpuIsa cpu_isa = init_isa();
      return cpu_isa;
    }

  }
}
