#pragma once

#include <cstddef>
#include <cstdint>

namespace algoat::simd {

/**
 * @brief 2-way SIMD partition using Google Highway.
 * Partitions the data into elements < pivot and elements >= pivot.
 * @return The index of the first element >= pivot (the split point).
 */
template <typename T> std::size_t partition_simd(T* data, std::size_t size, T pivot);

// Explicit instantiations for supported 32-bit and 64-bit primitives
extern template std::size_t partition_simd<int32_t>(int32_t*, std::size_t, int32_t);
extern template std::size_t partition_simd<uint32_t>(uint32_t*, std::size_t, uint32_t);
extern template std::size_t partition_simd<int64_t>(int64_t*, std::size_t, int64_t);
extern template std::size_t partition_simd<uint64_t>(uint64_t*, std::size_t, uint64_t);
extern template std::size_t partition_simd<float>(float*, std::size_t, float);
extern template std::size_t partition_simd<double>(double*, std::size_t, double);

} // namespace algoat::simd
