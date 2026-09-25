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

    std::size_t l = 0;
    std::size_t r = size; // exclusive right bound

    // SIMD fast-skipping partition: in-place, O(1) auxiliary space.
    while (true) {
        // Fast-skip left elements < pivot using vector loads
        while (l + N <= r) {
            auto vec_data = hn::LoadU(tag, data + l);
            auto mask_lt = hn::Lt(vec_data, v_pivot);
            if (hn::AllTrue(tag, mask_lt)) {
                l += N;
            } else {
                break;
            }
        }

        // Fast-skip right elements >= pivot using vector loads
        while (l + N <= r) {
            auto vec_data = hn::LoadU(tag, data + r - N);
            auto mask_ge = hn::Not(hn::Lt(vec_data, v_pivot));
            if (hn::AllTrue(tag, mask_ge)) {
                r -= N;
            } else {
                break;
            }
        }

        // Scalar fallback for the mismatch boundary
        if (l < r) {
            if (data[l] < pivot) {
                l++;
            } else if (data[r - 1] >= pivot) {
                r--;
            } else {
                std::swap(data[l], data[r - 1]);
                l++;
                r--;
            }
        } else {
            break;
        }
    }

    return l;
}

} // namespace HWY_NAMESPACE
} // namespace algoat::simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace algoat::simd {

template <typename T> std::size_t partition_simd(T* data, std::size_t size, T pivot) {
    // HWY_STATIC_DISPATCH provides safe fallback architecture context.
    return HWY_STATIC_DISPATCH(partition_simd_impl)(data, size, pivot);
}

// instantiations for 32-bit and 64-bit primitives
template std::size_t partition_simd<int32_t>(int32_t*, std::size_t, int32_t);
template std::size_t partition_simd<uint32_t>(uint32_t*, std::size_t, uint32_t);
template std::size_t partition_simd<int64_t>(int64_t*, std::size_t, int64_t);
template std::size_t partition_simd<uint64_t>(uint64_t*, std::size_t, uint64_t);

} // namespace algoat::simd
#endif
