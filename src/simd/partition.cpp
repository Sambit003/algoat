#include "algoat/simd/partition.hpp"

#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/partition.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace algoat::simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename T> std::size_t partition_simd_impl(T* data, std::size_t size, T pivot) {
    std::vector<T> left(size);
    std::vector<T> right(size);
    std::size_t left_idx = 0;
    std::size_t right_idx = 0;

    const hn::ScalableTag<T> d;
    const std::size_t N = hn::Lanes(d);
    const auto v_pivot = hn::Set(d, pivot);

    std::size_t i = 0;
    for (; i + N <= size; i += N) {
        auto v = hn::LoadU(d, data + i);
        auto mask_lt = hn::Lt(v, v_pivot);
        auto mask_ge = hn::Not(mask_lt);

        hn::CompressStore(v, mask_lt, d, left.data() + left_idx);
        left_idx += hn::CountTrue(d, mask_lt);

        hn::CompressStore(v, mask_ge, d, right.data() + right_idx);
        right_idx += hn::CountTrue(d, mask_ge);
    }

    for (; i < size; ++i) {
        if (data[i] < pivot) {
            left[left_idx++] = data[i];
        } else {
            right[right_idx++] = data[i];
        }
    }

    for (std::size_t j = 0; j < left_idx; ++j)
        data[j] = left[j];
    for (std::size_t j = 0; j < right_idx; ++j)
        data[left_idx + j] = right[j];

    return left_idx;
}

} // namespace HWY_NAMESPACE
} // namespace algoat::simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace algoat::simd {

template <typename T> std::size_t partition_simd(T* data, std::size_t size, T pivot) {
    return HWY_STATIC_DISPATCH(partition_simd_impl)(data, size, pivot);
}

// instantiations
template std::size_t partition_simd<int32_t>(int32_t*, std::size_t, int32_t);
template std::size_t partition_simd<uint32_t>(uint32_t*, std::size_t, uint32_t);
template std::size_t partition_simd<int64_t>(int64_t*, std::size_t, int64_t);
template std::size_t partition_simd<uint64_t>(uint64_t*, std::size_t, uint64_t);
template std::size_t partition_simd<float>(float*, std::size_t, float);
template std::size_t partition_simd<double>(double*, std::size_t, double);

} // namespace algoat::simd
#endif
