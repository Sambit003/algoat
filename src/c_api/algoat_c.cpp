/**
 * @file algoat_c.cpp
 * @brief Implementation of C-ABI entry points for Algoat algorithm suite.
 */

#include "algoat/c_api/algoat_c.h"

#include "algoat/searching/branchless_binary_search.hpp"
#include "algoat/selection/top_k.hpp"
#include "algoat/sorting/ips4o.hpp"
#include "algoat/sorting/pdqsort.hpp"
#include "algoat/sorting/radixsort.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>

namespace {

template <typename T> void dispatch_sort(T* data, size_t len, int parallel) noexcept {
    if (!data || len <= 1) {
        return;
    }
    try {
        if (parallel && len >= 1024) {
            algoat::sorting::ips4o_parallel_sort(data, data + len);
        } else {
            algoat::sorting::pdqsort(data, data + len);
        }
    } catch (...) {
        // Guarantee no C++ exceptions cross C-ABI boundary
    }
}

template <typename T> void dispatch_radix_sort(T* data, size_t len) noexcept {
    if (!data || len <= 1) {
        return;
    }
    try {
        algoat::sorting::radixsort_lsd(std::span<T>(data, len));
    } catch (...) {
        // Guarantee no C++ exceptions cross C-ABI boundary
    }
}

template <typename T> int64_t dispatch_search(const T* data, size_t len, T target) noexcept {
    if (!data || len == 0) {
        return 0;
    }
    try {
        const auto it = algoat::searching::branchless_lower_bound(data, data + len, target);
        return static_cast<int64_t>(std::distance(data, it));
    } catch (...) {
        return 0;
    }
}

template <typename T>
void dispatch_search_batch(const T* data, size_t len, const T* targets, size_t num_targets,
                           int64_t* results) noexcept {
    if (!results || num_targets == 0) {
        return;
    }
    if (!data || len == 0) {
        std::fill_n(results, num_targets, 0);
        return;
    }
    try {
        for (size_t i = 0; i < num_targets; ++i) {
            const auto it = algoat::searching::branchless_lower_bound(data, data + len, targets[i]);
            results[i] = static_cast<int64_t>(std::distance(data, it));
        }
    } catch (...) {
        // Guarantee no C++ exceptions cross C-ABI boundary
    }
}

template <typename T> void dispatch_topk(T* data, size_t len, size_t k) noexcept {
    if (!data || len <= 1 || k == 0) {
        return;
    }
    try {
        if (k >= len) {
            algoat::sorting::pdqsort(data, data + len);
            return;
        }
        algoat::selection::top_k_sorted(std::span<T>(data, len), k);
    } catch (...) {
        // Guarantee no C++ exceptions cross C-ABI boundary
    }
}

} // namespace

extern "C" {

void algoat_sort_int64(int64_t* data, size_t len, int parallel) {
    dispatch_sort(data, len, parallel);
}

void algoat_sort_uint64(uint64_t* data, size_t len, int parallel) {
    dispatch_sort(data, len, parallel);
}

void algoat_sort_int32(int32_t* data, size_t len, int parallel) {
    dispatch_sort(data, len, parallel);
}

void algoat_sort_uint32(uint32_t* data, size_t len, int parallel) {
    dispatch_sort(data, len, parallel);
}

void algoat_sort_int16(int16_t* data, size_t len, int parallel) {
    dispatch_sort(data, len, parallel);
}

void algoat_sort_uint16(uint16_t* data, size_t len, int parallel) {
    dispatch_sort(data, len, parallel);
}

void algoat_sort_int8(int8_t* data, size_t len, int parallel) {
    dispatch_sort(data, len, parallel);
}

void algoat_sort_uint8(uint8_t* data, size_t len, int parallel) {
    dispatch_sort(data, len, parallel);
}

void algoat_sort_float64(double* data, size_t len, int parallel) {
    dispatch_sort(data, len, parallel);
}

void algoat_sort_float32(float* data, size_t len, int parallel) {
    dispatch_sort(data, len, parallel);
}

void algoat_radix_sort_int64(int64_t* data, size_t len) {
    dispatch_radix_sort(data, len);
}

void algoat_radix_sort_uint64(uint64_t* data, size_t len) {
    dispatch_radix_sort(data, len);
}

void algoat_radix_sort_int32(int32_t* data, size_t len) {
    dispatch_radix_sort(data, len);
}

void algoat_radix_sort_uint32(uint32_t* data, size_t len) {
    dispatch_radix_sort(data, len);
}

void algoat_radix_sort_int16(int16_t* data, size_t len) {
    dispatch_radix_sort(data, len);
}

void algoat_radix_sort_uint16(uint16_t* data, size_t len) {
    dispatch_radix_sort(data, len);
}

void algoat_radix_sort_int8(int8_t* data, size_t len) {
    dispatch_radix_sort(data, len);
}

void algoat_radix_sort_uint8(uint8_t* data, size_t len) {
    dispatch_radix_sort(data, len);
}

int64_t algoat_search_int64(const int64_t* data, size_t len, int64_t target) {
    return dispatch_search(data, len, target);
}

int64_t algoat_search_uint64(const uint64_t* data, size_t len, uint64_t target) {
    return dispatch_search(data, len, target);
}

int64_t algoat_search_int32(const int32_t* data, size_t len, int32_t target) {
    return dispatch_search(data, len, target);
}

int64_t algoat_search_uint32(const uint32_t* data, size_t len, uint32_t target) {
    return dispatch_search(data, len, target);
}

int64_t algoat_search_int16(const int16_t* data, size_t len, int16_t target) {
    return dispatch_search(data, len, target);
}

int64_t algoat_search_uint16(const uint16_t* data, size_t len, uint16_t target) {
    return dispatch_search(data, len, target);
}

int64_t algoat_search_int8(const int8_t* data, size_t len, int8_t target) {
    return dispatch_search(data, len, target);
}

int64_t algoat_search_uint8(const uint8_t* data, size_t len, uint8_t target) {
    return dispatch_search(data, len, target);
}

int64_t algoat_search_float64(const double* data, size_t len, double target) {
    return dispatch_search(data, len, target);
}

int64_t algoat_search_float32(const float* data, size_t len, float target) {
    return dispatch_search(data, len, target);
}

void algoat_search_batch_int64(const int64_t* data, size_t len, const int64_t* targets,
                               size_t num_targets, int64_t* results) {
    dispatch_search_batch(data, len, targets, num_targets, results);
}

void algoat_search_batch_uint64(const uint64_t* data, size_t len, const uint64_t* targets,
                                size_t num_targets, int64_t* results) {
    dispatch_search_batch(data, len, targets, num_targets, results);
}

void algoat_search_batch_int32(const int32_t* data, size_t len, const int32_t* targets,
                               size_t num_targets, int64_t* results) {
    dispatch_search_batch(data, len, targets, num_targets, results);
}

void algoat_search_batch_uint32(const uint32_t* data, size_t len, const uint32_t* targets,
                                size_t num_targets, int64_t* results) {
    dispatch_search_batch(data, len, targets, num_targets, results);
}

void algoat_search_batch_int16(const int16_t* data, size_t len, const int16_t* targets,
                               size_t num_targets, int64_t* results) {
    dispatch_search_batch(data, len, targets, num_targets, results);
}

void algoat_search_batch_uint16(const uint16_t* data, size_t len, const uint16_t* targets,
                                size_t num_targets, int64_t* results) {
    dispatch_search_batch(data, len, targets, num_targets, results);
}

void algoat_search_batch_int8(const int8_t* data, size_t len, const int8_t* targets,
                              size_t num_targets, int64_t* results) {
    dispatch_search_batch(data, len, targets, num_targets, results);
}

void algoat_search_batch_uint8(const uint8_t* data, size_t len, const uint8_t* targets,
                               size_t num_targets, int64_t* results) {
    dispatch_search_batch(data, len, targets, num_targets, results);
}

void algoat_search_batch_float64(const double* data, size_t len, const double* targets,
                                 size_t num_targets, int64_t* results) {
    dispatch_search_batch(data, len, targets, num_targets, results);
}

void algoat_search_batch_float32(const float* data, size_t len, const float* targets,
                                 size_t num_targets, int64_t* results) {
    dispatch_search_batch(data, len, targets, num_targets, results);
}

void algoat_topk_int64(int64_t* data, size_t len, size_t k) {
    dispatch_topk(data, len, k);
}

void algoat_topk_uint64(uint64_t* data, size_t len, size_t k) {
    dispatch_topk(data, len, k);
}

void algoat_topk_int32(int32_t* data, size_t len, size_t k) {
    dispatch_topk(data, len, k);
}

void algoat_topk_uint32(uint32_t* data, size_t len, size_t k) {
    dispatch_topk(data, len, k);
}

void algoat_topk_int16(int16_t* data, size_t len, size_t k) {
    dispatch_topk(data, len, k);
}

void algoat_topk_uint16(uint16_t* data, size_t len, size_t k) {
    dispatch_topk(data, len, k);
}

void algoat_topk_int8(int8_t* data, size_t len, size_t k) {
    dispatch_topk(data, len, k);
}

void algoat_topk_uint8(uint8_t* data, size_t len, size_t k) {
    dispatch_topk(data, len, k);
}

void algoat_topk_float64(double* data, size_t len, size_t k) {
    dispatch_topk(data, len, k);
}

void algoat_topk_float32(float* data, size_t len, size_t k) {
    dispatch_topk(data, len, k);
}

} // extern "C"
