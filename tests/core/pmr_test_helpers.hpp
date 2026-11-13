#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory_resource>
#include <new>

#ifdef _WIN32
#include <malloc.h>
#endif

#ifndef __has_feature
#define __has_feature(x) 0
#endif

namespace algoat::testing {

/**
 * @brief PMR memory resource wrapper counting allocations, deallocations, and bytes.
 */
class CountingMemoryResource final : public std::pmr::memory_resource {
public:
    explicit CountingMemoryResource(
        std::pmr::memory_resource* upstream = std::pmr::new_delete_resource())
        : upstream_(upstream) {}

    [[nodiscard]] std::size_t allocate_calls() const noexcept {
        return allocate_calls_;
    }

    [[nodiscard]] std::size_t deallocate_calls() const noexcept {
        return deallocate_calls_;
    }

    [[nodiscard]] std::size_t total_bytes_allocated() const noexcept {
        return total_bytes_allocated_;
    }

    [[nodiscard]] std::size_t last_alignment() const noexcept {
        return last_alignment_;
    }

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        ++allocate_calls_;
        total_bytes_allocated_ += bytes;
        last_alignment_ = alignment;
        return upstream_->allocate(bytes, alignment);
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        ++deallocate_calls_;
        upstream_->deallocate(p, bytes, alignment);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }

private:
    std::pmr::memory_resource* upstream_;
    std::size_t allocate_calls_{0};
    std::size_t deallocate_calls_{0};
    std::size_t total_bytes_allocated_{0};
    std::size_t last_alignment_{0};
};

inline std::atomic<std::size_t>& global_alloc_calls() noexcept {
    static std::atomic<std::size_t> calls{0};
    return calls;
}

inline std::atomic<bool>& track_global_allocs() noexcept {
    static std::atomic<bool> tracking{false};
    return tracking;
}

/**
 * @brief RAII scope tracking global operator new calls within a non-ASan test section.
 */
class ScopedGlobalAllocTracker {
public:
    ScopedGlobalAllocTracker() noexcept {
        global_alloc_calls().store(0, std::memory_order_relaxed);
        track_global_allocs().store(true, std::memory_order_seq_cst);
    }

    ~ScopedGlobalAllocTracker() {
        stop();
    }

    ScopedGlobalAllocTracker(const ScopedGlobalAllocTracker&) = delete;
    ScopedGlobalAllocTracker& operator=(const ScopedGlobalAllocTracker&) = delete;

    [[nodiscard]] std::size_t count() noexcept {
        stop();
        return global_alloc_calls().load(std::memory_order_relaxed);
    }

private:
    void stop() noexcept {
        if (active_) {
            track_global_allocs().store(false, std::memory_order_seq_cst);
            active_ = false;
        }
    }

    bool active_{true};
};

namespace detail {

#ifdef _WIN32
inline void* aligned_alloc_impl(std::size_t alignment, std::size_t size) noexcept {
    return ::_aligned_malloc(size, alignment);
}

inline void aligned_free_impl(void* p) noexcept {
    ::_aligned_free(p);
}
#else
inline void* aligned_alloc_impl(std::size_t alignment, std::size_t size) noexcept {
    return std::aligned_alloc(alignment, size);
}

inline void aligned_free_impl(void* p) noexcept {
    std::free(p);
}
#endif

} // namespace detail

} // namespace algoat::testing

#if !defined(__SANITIZE_ADDRESS__) && (!defined(__has_feature) || !__has_feature(address_sanitizer))
#define ALGOAT_INSTALL_GLOBAL_ALLOC_COUNTER()                                                      \
    void* operator new(std::size_t size) {                                                         \
        if (::algoat::testing::track_global_allocs().load(std::memory_order_relaxed)) {            \
            ::algoat::testing::global_alloc_calls().fetch_add(1, std::memory_order_relaxed);       \
        }                                                                                          \
        void* p = std::malloc(size);                                                               \
        if (!p) {                                                                                  \
            throw std::bad_alloc();                                                                \
        }                                                                                          \
        return p;                                                                                  \
    }                                                                                              \
    void* operator new(std::size_t size, std::align_val_t align) {                                 \
        if (::algoat::testing::track_global_allocs().load(std::memory_order_relaxed)) {            \
            ::algoat::testing::global_alloc_calls().fetch_add(1, std::memory_order_relaxed);       \
        }                                                                                          \
        const std::size_t raw_align = static_cast<std::size_t>(align);                             \
        const std::size_t adj_align = raw_align < sizeof(void*) ? sizeof(void*) : raw_align;       \
        if (raw_align == 0 || (raw_align & (raw_align - 1)) != 0 ||                                \
            size > std::numeric_limits<std::size_t>::max() - (adj_align - 1)) {                    \
            throw std::bad_alloc();                                                                \
        }                                                                                          \
        const std::size_t req =                                                                    \
            size == 0 ? adj_align : ((size + adj_align - 1) & ~(adj_align - 1));                   \
        void* p = ::algoat::testing::detail::aligned_alloc_impl(adj_align, req);                   \
        if (!p) {                                                                                  \
            throw std::bad_alloc();                                                                \
        }                                                                                          \
        return p;                                                                                  \
    }                                                                                              \
    void operator delete(void* p) noexcept {                                                       \
        std::free(p);                                                                              \
    }                                                                                              \
    void operator delete(void* p, std::size_t) noexcept {                                          \
        std::free(p);                                                                              \
    }                                                                                              \
    void operator delete(void* p, std::align_val_t) noexcept {                                     \
        ::algoat::testing::detail::aligned_free_impl(p);                                           \
    }                                                                                              \
    void operator delete(void* p, std::size_t, std::align_val_t) noexcept {                        \
        ::algoat::testing::detail::aligned_free_impl(p);                                           \
    }
#else
#define ALGOAT_INSTALL_GLOBAL_ALLOC_COUNTER()
#endif
