#pragma once

#include <string>

namespace ctranslate2 {
  namespace cpu {

    enum class CpuIsa {
      GENERIC,
#if defined(CT2_X86_BUILD)
      AVX,
      AVX2,
      AVX2_VNNI,
      AVX512,
#elif defined(CT2_ARM64_BUILD)
      NEON,
#endif
    };

    std::string isa_to_str(CpuIsa isa);

    // Returns the CPU ISA to dispatch to.
    CpuIsa get_cpu_isa();

  }
}

#define CPU_ISA_CASE(CPU_ISA, STMTS)            \
  case CPU_ISA: {                               \
    constexpr cpu::CpuIsa ISA = CPU_ISA;        \
    STMTS;                                      \
    break;                                      \
  }

#define CPU_ISA_DEFAULT(CPU_ISA, STMTS)                 \
  default: {                                            \
    constexpr cpu::CpuIsa ISA = CPU_ISA;                \
    STMTS;                                              \
    break;                                              \
  }

#define SINGLE_ARG(...) __VA_ARGS__
#ifdef CT2_WITH_CPU_DISPATCH
#if defined(CT2_X86_BUILD)
#  define CPU_ISA_DISPATCH(STMTS)                             \
  switch (cpu::get_cpu_isa()) {                               \
    CPU_ISA_CASE(cpu::CpuIsa::AVX512, SINGLE_ARG(STMTS))      \
    CPU_ISA_CASE(cpu::CpuIsa::AVX2_VNNI, SINGLE_ARG(STMTS))   \
    CPU_ISA_CASE(cpu::CpuIsa::AVX2, SINGLE_ARG(STMTS))        \
    CPU_ISA_CASE(cpu::CpuIsa::AVX, SINGLE_ARG(STMTS))         \
    CPU_ISA_DEFAULT(cpu::CpuIsa::GENERIC, SINGLE_ARG(STMTS))  \
  }
#elif defined(CT2_ARM64_BUILD)
#  define CPU_ISA_DISPATCH(STMTS)                             \
  switch (cpu::get_cpu_isa()) {                               \
    CPU_ISA_CASE(cpu::CpuIsa::NEON, SINGLE_ARG(STMTS))        \
    CPU_ISA_DEFAULT(cpu::CpuIsa::GENERIC, SINGLE_ARG(STMTS))  \
  }
#endif
#else
#  if defined(__AVX512F__)
#    define CT2_COMPILED_CPU_ISA cpu::CpuIsa::AVX512
#  elif defined(__AVX2__) && defined(__AVXVNNI__)
#    define CT2_COMPILED_CPU_ISA cpu::CpuIsa::AVX2_VNNI
#  elif defined(__AVX2__)
#    define CT2_COMPILED_CPU_ISA cpu::CpuIsa::AVX2
#  elif defined(__AVX__)
#    define CT2_COMPILED_CPU_ISA cpu::CpuIsa::AVX
#  elif defined(__ARM_NEON)
#    define CT2_COMPILED_CPU_ISA cpu::CpuIsa::NEON
#  else
#    define CT2_COMPILED_CPU_ISA cpu::CpuIsa::GENERIC
#  endif

#  define CPU_ISA_DISPATCH(STMTS)                             \
  switch (cpu::get_cpu_isa()) {                               \
    CPU_ISA_DEFAULT(CT2_COMPILED_CPU_ISA, SINGLE_ARG(STMTS))   \
  }
#endif
