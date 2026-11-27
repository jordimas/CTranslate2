#include "ctranslate2/allocator.h"

#include <algorithm>

#ifdef _WIN32
#  include <malloc.h>
#else
#  include <cstdlib>
#endif

#ifdef __linux__
#  include <sys/mman.h>
#endif

#ifdef CT2_WITH_MKL
#  include <mkl.h>
#endif

namespace ctranslate2 {
  namespace cpu {

    // Large buffers (weights, KV cache) are walked in full on every decoding step. Backing
    // them with transparent huge pages cuts TLB misses. Disable with CT2_CPU_HUGEPAGES=0.
    constexpr size_t huge_page_size = size_t(2) << 20;
    constexpr size_t huge_page_threshold = size_t(4) << 20;

    static bool huge_pages_enabled() {
#ifdef __linux__
      static const bool enabled = []() {
        const char* env = std::getenv("CT2_CPU_HUGEPAGES");
        return !(env && env[0] == '0');
      }();
      return enabled;
#else
      return false;
#endif
    }

    static size_t effective_alignment(size_t size, size_t alignment) {
      return (size >= huge_page_threshold && huge_pages_enabled()
              ? std::max(alignment, huge_page_size)
              : alignment);
    }

    static void advise_huge_pages(void* ptr, size_t size) {
#ifdef __linux__
      if (ptr && size >= huge_page_threshold && huge_pages_enabled())
        madvise(ptr, size, MADV_HUGEPAGE);  // Best effort, failure is harmless.
#else
      (void)ptr;
      (void)size;
#endif
    }

    class AlignedAllocator : public Allocator {
    public:
      AlignedAllocator(size_t alignment)
        : _alignment(alignment)
      {
      }

      void* allocate(size_t size, int) override {
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(size, _alignment);
#else
        if (posix_memalign(&ptr, effective_alignment(size, _alignment), size) != 0)
          ptr = nullptr;
        advise_huge_pages(ptr, size);
#endif
        if (!ptr)
          throw std::runtime_error("aligned_alloc: failed to allocate memory");
        return ptr;
      }

      void free(void* ptr, int) override {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
      }

    private:
      size_t _alignment;
    };

#ifdef CT2_WITH_MKL
    class MklAllocator : public Allocator {
    public:
      MklAllocator(size_t alignment)
        : _alignment(alignment)
      {
      }

      void* allocate(size_t size, int) override {
        void* ptr = mkl_malloc(size, effective_alignment(size, _alignment));
        advise_huge_pages(ptr, size);
        if (!ptr)
          throw std::runtime_error("mkl_malloc: failed to allocate memory");
        return ptr;
      }

      void free(void* ptr, int) override {
        mkl_free(ptr);
      }

      void clear_cache() override {
        mkl_free_buffers();
      }

    private:
      size_t _alignment;
    };
#endif

  }

  template<>
  Allocator& get_allocator<Device::CPU>() {
    constexpr size_t alignment = 64;
#ifdef CT2_WITH_MKL
    static cpu::MklAllocator allocator(alignment);
#else
    static cpu::AlignedAllocator allocator(alignment);
#endif
    return allocator;
  }

}
