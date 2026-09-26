#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace algoat::simd {

/**
 * @brief 2-way SIMD partition using Google Highway.
 *
 * Utilizes Highway's scalable vector model (hn::ScalableTag) which automatically
 * emits vectorized kernels utilizing AVX-512 (512-bit lanes), AVX2 (256-bit lanes),
 * or ARM NEON (128-bit lanes) depending on compile target and host capabilities.
 *
 * Partitions the data into elements < pivot and elements >= pivot.
 *
 * @param data Pointer to the contiguous array to partition.
 * @param size Number of elements in the array.
 * @param pivot Pivot value to partition against.
 * @return std::optional<std::size_t> containing the split point index (number of elements < pivot),
 *         or std::nullopt if SIMD execution is unavailable on the host architecture (scalar
 * fallback).
 */
template <typename T> std::optional<std::size_t> partition_simd(T* data, std::size_t size, T pivot);

// Explicit instantiations for supported 32-bit and 64-bit primitives
template <typename T>
inline constexpr bool is_simd_supported_v =
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
    std::is_same_v<T, uint64_t> || std::is_same_v<T, float> || std::is_same_v<T, double>;

extern template std::optional<std::size_t> partition_simd<int32_t>(int32_t*, std::size_t, int32_t);
extern template std::optional<std::size_t> partition_simd<uint32_t>(uint32_t*, std::size_t,
                                                                    uint32_t);
extern template std::optional<std::size_t> partition_simd<int64_t>(int64_t*, std::size_t, int64_t);
extern template std::optional<std::size_t> partition_simd<uint64_t>(uint64_t*, std::size_t,
                                                                    uint64_t);
extern template std::optional<std::size_t> partition_simd<float>(float*, std::size_t, float);
extern template std::optional<std::size_t> partition_simd<double>(double*, std::size_t, double);

} // namespace algoat::simd
