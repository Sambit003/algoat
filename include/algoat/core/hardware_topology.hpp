/**
 * @file hardware_topology.hpp
 * @brief Cross-platform hardware cache topology and SIMD ISA capability inspection.
 *
 * Provides @c algoat::core::HardwareTopology, a zero-overhead cached singleton that
 * probes L1D/L2/L3 cache capacities, cache-line width, and vector extensions at
 * startup using CPUID on x86-64 and native OS APIs across Linux, macOS, and Windows.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

namespace algoat::core {

/**
 * @struct CacheProperties
 * @brief Architectural geometry of a single CPU cache level.
 */
struct CacheProperties {
    std::size_t level{0};           ///< Cache hierarchy level (1 = L1, 2 = L2, 3 = L3).
    std::size_t size_bytes{0};      ///< Total capacity in bytes.
    std::size_t line_size_bytes{0}; ///< Coherency cache-line width in bytes.
    std::size_t associativity{0};   ///< Set-associativity ways (0 if unknown).
};

/**
 * @class HardwareTopology
 * @brief Thread-safe singleton reporting runtime CPU cache topology and SIMD extensions.
 *
 * Evaluated once upon first call to @c instance(), exposing inline @c noexcept
 * accessors with sub-5 nanosecond query overhead for dynamic buffer auto-tuning.
 */
class HardwareTopology {
public:
    /**
     * @brief Returns the process-wide cached hardware topology singleton.
     * @return Const reference to the initialized @c HardwareTopology instance.
     */
    static const HardwareTopology& instance() noexcept;

    HardwareTopology(const HardwareTopology&) = delete;
    HardwareTopology& operator=(const HardwareTopology&) = delete;
    HardwareTopology(HardwareTopology&&) = delete;
    HardwareTopology& operator=(HardwareTopology&&) = delete;

    /// @brief Returns L1 Data Cache capacity in bytes.
    [[nodiscard]] std::size_t l1d_cache_size() const noexcept {
        return l1d_props_.size_bytes;
    }

    /// @brief Returns L2 Cache capacity in bytes.
    [[nodiscard]] std::size_t l2_cache_size() const noexcept {
        return l2_props_.size_bytes;
    }

    /// @brief Returns L3 Cache capacity in bytes (0 if absent on platform).
    [[nodiscard]] std::size_t l3_cache_size() const noexcept {
        return l3_props_.size_bytes;
    }

    /// @brief Returns system coherency cache-line width in bytes (typically 64 or 128).
    [[nodiscard]] std::size_t cache_line_size() const noexcept {
        return cache_line_size_;
    }

    /// @brief Returns detailed properties for the L1 Data Cache.
    [[nodiscard]] const CacheProperties& l1d_properties() const noexcept {
        return l1d_props_;
    }

    /// @brief Returns detailed properties for the L2 Cache.
    [[nodiscard]] const CacheProperties& l2_properties() const noexcept {
        return l2_props_;
    }

    /// @brief Returns detailed properties for the L3 Cache.
    [[nodiscard]] const CacheProperties& l3_properties() const noexcept {
        return l3_props_;
    }

    /// @brief Returns true if both CPU and OS support AVX2 (XCR0 SSE+AVX verified).
    [[nodiscard]] bool has_avx2() const noexcept {
        return has_avx2_;
    }

    /// @brief Returns true if both CPU and OS support AVX-512F (XCR0 opmask/ZMM verified).
    [[nodiscard]] bool has_avx512f() const noexcept {
        return has_avx512f_;
    }

    /// @brief Returns true if the CPU supports BMI2 scalar bit-manipulation instructions.
    [[nodiscard]] bool has_bmi2() const noexcept {
        return has_bmi2_;
    }

    /// @brief Returns true if ARM Advanced SIMD (NEON) is available.
    [[nodiscard]] bool has_neon() const noexcept {
        return has_neon_;
    }

    /**
     * @brief Computes optimal elements-per-block so a buffer of @p T fits in @p l1_fraction of L1D.
     *
     * Matches `(l1d_cache_size() * l1_fraction) / sizeof(T)` while guarding against non-finite,
     * non-positive, or @c std::size_t -overflowing inputs to prevent floating-point cast UB.
     *
     * @tparam T Element type stored in the cache-resident working buffer.
     * @param l1_fraction Target multiplier of L1D cache capacity (default: 0.5).
     * @return Number of elements of type @p T fitting within the target budget.
     */
    template <typename T>
    [[nodiscard]] std::size_t optimal_l1_block_elements(double l1_fraction = 0.5) const noexcept {
        static_assert(sizeof(T) > 0, "Element type T must have non-zero size");
        if (!std::isfinite(l1_fraction) || l1_fraction <= 0.0) {
            return 0;
        }
        const double elements =
            (static_cast<double>(l1d_cache_size()) * l1_fraction) / static_cast<double>(sizeof(T));
        constexpr double kMaxSizeTAsDouble =
            static_cast<double>(std::numeric_limits<std::size_t>::max());
        if (!std::isfinite(elements) || elements >= kMaxSizeTAsDouble) {
            return std::numeric_limits<std::size_t>::max();
        }
        return static_cast<std::size_t>(elements);
    }

private:
    HardwareTopology() noexcept;
    void detect_x86_cpuid() noexcept;
    void detect_posix_sysconf() noexcept;
    void record_cache(std::size_t level, std::size_t size_bytes, std::size_t line_bytes,
                      std::size_t assoc) noexcept;

    CacheProperties l1d_props_{1, 32768, 64, 0};
    CacheProperties l2_props_{2, 524288, 64, 0};
    CacheProperties l3_props_{3, 8388608, 64, 0};
    std::size_t cache_line_size_{64};

    bool has_avx2_{false};
    bool has_avx512f_{false};
    bool has_bmi2_{false};
    bool has_neon_{false};
    bool detected_any_cache_{false};
    bool detected_l3_cache_{false};
};

} // namespace algoat::core
