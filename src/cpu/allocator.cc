#include "ctranslate2/allocator.h"

#ifdef _WIN32
#  include <malloc.h>
#else
#  include <cstdlib>
#endif

#ifdef __APPLE__
#  include <mutex>
#  include <unordered_map>
#  include <vector>

#  include "env.h"
#endif

#ifdef CT2_WITH_MKL
#  include <mkl.h>
#endif

namespace ctranslate2 {
  namespace cpu {

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
        if (posix_memalign(&ptr, _alignment, size) != 0)
          ptr = nullptr;
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
        void* ptr = mkl_malloc(size, _alignment);
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

#ifdef __APPLE__
    // Keeps freed large blocks for reuse: returning them to the system on every free and
    // faulting the pages back in on the next allocation is expensive for the many
    // temporary tensors of a decoding step. The block size is stored in a header.
    class CachingAllocator : public Allocator {
    public:
      CachingAllocator(Allocator& base, size_t max_cached_bytes)
        : _base(base)
        , _max_cached_bytes(max_cached_bytes)
      {
      }

      void* allocate(size_t size, int device_index) override {
        size += header_size;
        if (size >= min_cached_size) {
          // Round up to 4 sizes per power of 2 so that growing tensors can reuse blocks.
          size_t step = 1;
          while (step * 8 <= size)
            step *= 2;
          size = (size + step - 1) / step * step;
        }

        void* block = nullptr;
        if (size >= min_cached_size) {
          std::lock_guard<std::mutex> lock(_mutex);
          auto& blocks = _free_blocks[size];
          if (!blocks.empty()) {
            block = blocks.back();
            blocks.pop_back();
            _cached_bytes -= size;
          }
        }
        if (!block)
          block = _base.allocate(size, device_index);

        *static_cast<size_t*>(block) = size;
        return static_cast<char*>(block) + header_size;
      }

      void free(void* ptr, int device_index) override {
        void* block = static_cast<char*>(ptr) - header_size;
        const size_t size = *static_cast<size_t*>(block);
        if (size >= min_cached_size) {
          std::lock_guard<std::mutex> lock(_mutex);
          if (_cached_bytes + size <= _max_cached_bytes) {
            _free_blocks[size].push_back(block);
            _cached_bytes += size;
            return;
          }
        }
        _base.free(block, device_index);
      }

      void clear_cache() override {
        {
          std::lock_guard<std::mutex> lock(_mutex);
          for (auto& pair : _free_blocks)
            for (void* block : pair.second)
              _base.free(block, -1);
          _free_blocks.clear();
          _cached_bytes = 0;
        }
        _base.clear_cache();
      }

    private:
      static constexpr size_t header_size = 64;  // Keeps the user pointer aligned.
      static constexpr size_t min_cached_size = 16 * 1024;

      Allocator& _base;
      const size_t _max_cached_bytes;
      std::mutex _mutex;
      std::unordered_map<size_t, std::vector<void*>> _free_blocks;
      size_t _cached_bytes = 0;
    };
#endif

  }

  template<>
  Allocator& get_allocator<Device::CPU>() {
    constexpr size_t alignment = 64;
#ifdef CT2_WITH_MKL
    static cpu::MklAllocator base_allocator(alignment);
#else
    static cpu::AlignedAllocator base_allocator(alignment);
#endif
#ifdef __APPLE__
    // CT2_CPU_ALLOCATOR_CACHE_MB: maximum size of the cache of freed blocks (0 to disable).
    static cpu::CachingAllocator allocator(
      base_allocator,
      static_cast<size_t>(read_int_from_env("CT2_CPU_ALLOCATOR_CACHE_MB", 1024)) << 20);
    return allocator;
#else
    return base_allocator;
#endif
  }

}
