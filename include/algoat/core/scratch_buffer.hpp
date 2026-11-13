/**
 * @file scratch_buffer.hpp
 * @brief PMR monotonic stack arena and RAII scratchpad buffer utilities.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <memory_resource>
#include <span>
#include <utility>

namespace algoat::core {

/**
 * @brief RAII monotonic stack scratchpad memory resource.
 *
 * Wraps a stack-allocated byte buffer with @c std::pmr::monotonic_buffer_resource.
 * Defaults to @c std::pmr::null_memory_resource() as its upstream to strictly forbid
 * silent heap fallbacks unless the caller explicitly provides an upstream resource.
 *
 * @tparam StackCapacityBytes Capacity of the inline stack buffer in bytes.
 */
template <std::size_t StackCapacityBytes = 65536> class StackArena {
public:
    /**
     * @brief Constructs a stack arena backed by an optional upstream resource.
     * @param upstream Fallback resource on buffer exhaustion (defaults to null_memory_resource).
     */
    explicit StackArena(std::pmr::memory_resource* upstream = std::pmr::null_memory_resource())
        : resource_(StackCapacityBytes > 0 ? buffer_.data() : nullptr, StackCapacityBytes,
                    upstream ? upstream : std::pmr::null_memory_resource()) {}

    StackArena(const StackArena&) = delete;
    StackArena& operator=(const StackArena&) = delete;
    StackArena(StackArena&&) = delete;
    StackArena& operator=(StackArena&&) = delete;
    ~StackArena() = default;

    /**
     * @brief Returns a pointer to the underlying polymorphic memory resource.
     * @return Pointer to the embedded @c std::pmr::monotonic_buffer_resource.
     */
    [[nodiscard]] std::pmr::memory_resource* resource() noexcept {
        return &resource_;
    }

    /**
     * @brief Returns the inline stack buffer capacity in bytes.
     * @return Stack buffer size in bytes (@p StackCapacityBytes).
     */
    [[nodiscard]] constexpr std::size_t capacity() const noexcept {
        return StackCapacityBytes;
    }

    /**
     * @brief Releases any upstream chunks and resets the bump pointer to the stack buffer.
     *
     * @warning Callers must ensure all objects/buffers allocated from this arena are already
     * destroyed before calling @c release().
     */
    void release() {
        resource_.release();
    }

private:
    // 64-byte alignment prevents initial padding waste for SIMD/cache-line allocations,
    // and declaration before resource_ guarantees buffer_ outlives resource_.
    // Intentionally uninitialized to keep StackArena construction O(1) without a 64 KiB memset.
    alignas(std::max(alignof(std::max_align_t),
                     std::size_t{64})) std::array<std::byte, StackCapacityBytes> buffer_;
    std::pmr::monotonic_buffer_resource resource_;
};

/**
 * @brief RAII scratchpad buffer allocating contiguous temporary spans from a PMR memory resource.
 *
 * Constructs live @c T elements using @c std::uninitialized_default_construct_n (a zero-cost
 * no-op for trivially default-constructible types) and destroys them deterministically on scope
 * exit before returning memory to @c mr.
 *
 * @tparam T Element type stored in the scratch buffer.
 */
template <typename T> class ScratchBuffer {
public:
    using value_type = T;

    /**
     * @brief Allocates and default-constructs @p count elements using @p mr.
     * @param count Number of elements to allocate. Zero performs no allocation.
     * @param mr Memory resource to allocate from (defaults to @c std::pmr::get_default_resource()).
     */
    explicit ScratchBuffer(std::size_t count,
                           std::pmr::memory_resource* mr = std::pmr::get_default_resource())
        : mr_(mr ? mr : std::pmr::get_default_resource()), count_(count) {
        if (count_ == 0) {
            return;
        }
        std::pmr::polymorphic_allocator<T> alloc(mr_);
        data_ = alloc.template allocate_object<T>(count_);
        try {
            std::uninitialized_default_construct_n(data_, count_);
        } catch (...) {
            alloc.deallocate_object(data_, count_);
            data_ = nullptr;
            count_ = 0;
            throw;
        }
    }

    ScratchBuffer(const ScratchBuffer&) = delete;
    ScratchBuffer& operator=(const ScratchBuffer&) = delete;

    ScratchBuffer(ScratchBuffer&& other) noexcept
        : mr_(other.mr_), data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    ScratchBuffer& operator=(ScratchBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        mr_ = other.mr_;
        data_ = std::exchange(other.data_, nullptr);
        count_ = std::exchange(other.count_, 0);
        return *this;
    }

    ~ScratchBuffer() {
        reset();
    }

    /**
     * @brief Returns a mutable contiguous view over the scratch buffer.
     * @return Mutable @c std::span<T> covering @c size() elements.
     */
    [[nodiscard]] std::span<T> span() noexcept {
        return {data_, count_};
    }

    /**
     * @brief Returns a read-only contiguous view over the scratch buffer.
     * @return Const @c std::span<const T> covering @c size() elements.
     */
    [[nodiscard]] std::span<const T> span() const noexcept {
        return {data_, count_};
    }

    /**
     * @brief Returns a pointer to the underlying contiguous buffer, or @c nullptr if empty.
     * @return Mutable pointer to the first element, or @c nullptr when empty.
     */
    [[nodiscard]] T* data() noexcept {
        return data_;
    }

    /**
     * @brief Returns a const pointer to the underlying contiguous buffer, or @c nullptr if empty.
     * @return Const pointer to the first element, or @c nullptr when empty.
     */
    [[nodiscard]] const T* data() const noexcept {
        return data_;
    }

    /**
     * @brief Returns the number of elements in the scratch buffer.
     * @return Element count.
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return count_;
    }

    /**
     * @brief Returns @c true if the buffer holds zero elements.
     * @return @c true when @c size() == 0, otherwise @c false.
     */
    [[nodiscard]] bool empty() const noexcept {
        return count_ == 0;
    }

    /**
     * @brief Returns the memory resource bound to this buffer.
     * @return Non-null pointer to the bound @c std::pmr::memory_resource.
     */
    [[nodiscard]] std::pmr::memory_resource* memory_resource() const noexcept {
        return mr_;
    }

private:
    void reset() noexcept {
        if (data_ != nullptr) {
            std::destroy_n(data_, count_);
            std::pmr::polymorphic_allocator<T> alloc(mr_);
            alloc.deallocate_object(data_, count_);
            data_ = nullptr;
            count_ = 0;
        }
    }

    std::pmr::memory_resource* mr_;
    T* data_{nullptr};
    std::size_t count_{0};
};

} // namespace algoat::core
