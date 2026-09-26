#include "algoat/simd/partition.hpp"

#include "algoat/simd/isa.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/partition.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace algoat::simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename T> std::size_t partition_simd_impl(T* data, std::size_t size, T pivot) {
    const hn::ScalableTag<T> tag;
    const std::size_t N = hn::Lanes(tag);
    const auto v_pivot = hn::Set(tag, pivot);

    std::size_t left_idx = 0;
    std::size_t right_idx = size; // exclusive right bound

    while (true) {
        // Fast-skip left elements < pivot using vector loads
        while (left_idx + 2 * N <= right_idx) {
            auto vec_data = hn::LoadU(tag, data + left_idx);
            if (hn::AllTrue(tag, hn::Lt(vec_data, v_pivot))) {
                left_idx += N;
            } else {
                break;
            }
        }

        // Fast-skip right elements >= pivot using vector loads
        while (left_idx + 2 * N <= right_idx) {
            auto vec_data = hn::LoadU(tag, data + right_idx - N);
            if (hn::AllTrue(tag, hn::Not(hn::Lt(vec_data, v_pivot)))) {
                right_idx -= N;
            } else {
                break;
            }
        }

        if (left_idx + 2 * N > right_idx)
            break; // Ensure left and right blocks do not overlap

        // Both blocks have mismatches. Count them exactly.
        auto vec_left = hn::LoadU(tag, data + left_idx);
        std::size_t count_left = hn::CountTrue(tag, hn::Not(hn::Lt(vec_left, v_pivot)));

        auto vec_right = hn::LoadU(tag, data + right_idx - N);
        std::size_t count_right = hn::CountTrue(tag, hn::Lt(vec_right, v_pivot));

        std::size_t swaps = std::min(count_left, count_right);
        std::size_t i = left_idx;
        std::size_t j = right_idx - 1;

        // Perform exactly 'swaps' swaps between the left and right block mismatches
        for (std::size_t k = 0; k < swaps; ++k) {
            while (data[i] < pivot)
                i++;
            while (data[j] >= pivot)
                j--;
            std::swap(data[i], data[j]);
            i++;
            j--;
        }

        // Advance boundaries if the block is fully resolved
        if (count_left == swaps)
            left_idx += N;
        if (count_right == swaps)
            right_idx -= N;
    }

    // Scalar drain: process remaining elements when they're too close for vector loads
    while (left_idx < right_idx) {
        if (data[left_idx] < pivot) {
            left_idx++;
        } else if (data[right_idx - 1] >= pivot) {
            right_idx--;
        } else {
            std::swap(data[left_idx], data[right_idx - 1]);
            left_idx++;
            right_idx--;
        }
    }

    return left_idx;
}

} // namespace HWY_NAMESPACE
} // namespace algoat::simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace algoat::simd {

template <typename T>
std::optional<std::size_t> partition_simd(T* data, std::size_t size, T pivot) {
    // Guard: if the host ISA is scalar-only, signal to the caller to use the
    // scalar partition path instead by returning std::nullopt. This wires
    // detect_host_isa() into actual dispatch logic with zero ambiguity.
    if (detect_host_isa() == InstructionSet::Scalar) {
        return std::nullopt;
    }
    return HWY_STATIC_DISPATCH(partition_simd_impl)(data, size, pivot);
}

// Explicit instantiations for 32-bit and 64-bit primitives
template std::optional<std::size_t> partition_simd<int32_t>(int32_t*, std::size_t, int32_t);
template std::optional<std::size_t> partition_simd<uint32_t>(uint32_t*, std::size_t, uint32_t);
template std::optional<std::size_t> partition_simd<int64_t>(int64_t*, std::size_t, int64_t);
template std::optional<std::size_t> partition_simd<uint64_t>(uint64_t*, std::size_t, uint64_t);
template std::optional<std::size_t> partition_simd<float>(float*, std::size_t, float);
template std::optional<std::size_t> partition_simd<double>(double*, std::size_t, double);

} // namespace algoat::simd
#endif
