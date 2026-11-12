#include "cpu_info.h"

#if defined(CT2_X86_BUILD)

#include <cstring>

#include <cpuinfo_x86.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <cpuid.h>
#endif

namespace ctranslate2 {
  namespace cpu {

    static const cpu_features::X86Info info = cpu_features::GetX86Info();

    const char* cpu_vendor() {
      return info.vendor;
    }

    bool cpu_is_genuine_intel() {
      return strcmp(cpu_vendor(), CPU_FEATURES_VENDOR_GENUINE_INTEL) == 0;
    }

    bool cpu_supports_sse41() {
      return info.features.sse4_1;
    }

    bool cpu_supports_avx() {
      return info.features.avx;
    }

    bool cpu_supports_avx2() {
      return info.features.avx2;
    }

    bool cpu_supports_avx2_vnni() {
      if (!cpu_supports_avx2())
        return false;

      // AVX-VNNI (256-bit vpdpbusd): CPUID.(EAX=7,ECX=1):EAX[bit 4].
#if defined(_MSC_VER)
      int regs[4];
      __cpuidex(regs, 7, 1);
      return (regs[0] & (1 << 4)) != 0;
#else
      unsigned int a, b, c, d;
      return __get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & (1u << 4));
#endif
    }

    bool cpu_supports_avx512() {
      return (info.features.avx512f
              && info.features.avx512cd
              && info.features.avx512vl
              && info.features.avx512dq
              && info.features.avx512bw);
    }

  }
}

#elif defined(CT2_ARM64_BUILD)

namespace ctranslate2 {
  namespace cpu {

    const char* cpu_vendor() {
      return "ARM";
    }

    bool cpu_supports_neon() {
      return true;
    }

  }
}

#endif
