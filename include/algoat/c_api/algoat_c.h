/**
 * @file algoat_c.h
 * @brief Pure C-ABI interface for Algoat high-performance algorithms.
 *
 * Exposes zero-copy C entry points for FFI/interop bridges (including Go,
 * Python, Rust, and C#). Ensures ABI stability and pure C99/C11 compatibility.
 */

#ifndef ALGOAT_C_H
#define ALGOAT_C_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(ALGOAT_C_BUILD_SHARED)
#define ALGOAT_C_EXPORT __declspec(dllexport)
#else
#define ALGOAT_C_EXPORT
#endif
#else
#if defined(__GNUC__) && __GNUC__ >= 4
#define ALGOAT_C_EXPORT __attribute__((visibility("default")))
#else
#define ALGOAT_C_EXPORT
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * In-place sorting entry points.
 * If parallel is non-zero and len is sufficiently large, multi-threaded
 * In-Place Parallel Super Scalar Samplesort (IPS⁴o) is used.
 * Otherwise, pattern-defeating quicksort (PDQSort) is used.
 */
ALGOAT_C_EXPORT void algoat_sort_int64(int64_t* data, size_t len, int parallel);
ALGOAT_C_EXPORT void algoat_sort_uint64(uint64_t* data, size_t len, int parallel);
ALGOAT_C_EXPORT void algoat_sort_int32(int32_t* data, size_t len, int parallel);
ALGOAT_C_EXPORT void algoat_sort_uint32(uint32_t* data, size_t len, int parallel);
ALGOAT_C_EXPORT void algoat_sort_int16(int16_t* data, size_t len, int parallel);
ALGOAT_C_EXPORT void algoat_sort_uint16(uint16_t* data, size_t len, int parallel);
ALGOAT_C_EXPORT void algoat_sort_int8(int8_t* data, size_t len, int parallel);
ALGOAT_C_EXPORT void algoat_sort_uint8(uint8_t* data, size_t len, int parallel);
ALGOAT_C_EXPORT void algoat_sort_float64(double* data, size_t len, int parallel);
ALGOAT_C_EXPORT void algoat_sort_float32(float* data, size_t len, int parallel);

/*
 * Linear-time LSD Radix sort entry points for integers.
 */
ALGOAT_C_EXPORT void algoat_radix_sort_int64(int64_t* data, size_t len);
ALGOAT_C_EXPORT void algoat_radix_sort_uint64(uint64_t* data, size_t len);
ALGOAT_C_EXPORT void algoat_radix_sort_int32(int32_t* data, size_t len);
ALGOAT_C_EXPORT void algoat_radix_sort_uint32(uint32_t* data, size_t len);
ALGOAT_C_EXPORT void algoat_radix_sort_int16(int16_t* data, size_t len);
ALGOAT_C_EXPORT void algoat_radix_sort_uint16(uint16_t* data, size_t len);
ALGOAT_C_EXPORT void algoat_radix_sort_int8(int8_t* data, size_t len);
ALGOAT_C_EXPORT void algoat_radix_sort_uint8(uint8_t* data, size_t len);

/*
 * Branchless lower-bound searching entry points.
 * Returns the index of the first element not less than target,
 * or len if all elements are less than target.
 */
ALGOAT_C_EXPORT int64_t algoat_search_int64(const int64_t* data, size_t len, int64_t target);
ALGOAT_C_EXPORT int64_t algoat_search_uint64(const uint64_t* data, size_t len, uint64_t target);
ALGOAT_C_EXPORT int64_t algoat_search_int32(const int32_t* data, size_t len, int32_t target);
ALGOAT_C_EXPORT int64_t algoat_search_uint32(const uint32_t* data, size_t len, uint32_t target);
ALGOAT_C_EXPORT int64_t algoat_search_int16(const int16_t* data, size_t len, int16_t target);
ALGOAT_C_EXPORT int64_t algoat_search_uint16(const uint16_t* data, size_t len, uint16_t target);
ALGOAT_C_EXPORT int64_t algoat_search_int8(const int8_t* data, size_t len, int8_t target);
ALGOAT_C_EXPORT int64_t algoat_search_uint8(const uint8_t* data, size_t len, uint8_t target);
ALGOAT_C_EXPORT int64_t algoat_search_float64(const double* data, size_t len, double target);
ALGOAT_C_EXPORT int64_t algoat_search_float32(const float* data, size_t len, float target);

/*
 * Batch branchless lower-bound searching entry points.
 * Amortizes Cgo context-switch overhead across multiple query lookups.
 */
ALGOAT_C_EXPORT void algoat_search_batch_int64(const int64_t* data, size_t len,
                                               const int64_t* targets, size_t num_targets,
                                               int64_t* results);
ALGOAT_C_EXPORT void algoat_search_batch_uint64(const uint64_t* data, size_t len,
                                                const uint64_t* targets, size_t num_targets,
                                                int64_t* results);
ALGOAT_C_EXPORT void algoat_search_batch_int32(const int32_t* data, size_t len,
                                               const int32_t* targets, size_t num_targets,
                                               int64_t* results);
ALGOAT_C_EXPORT void algoat_search_batch_uint32(const uint32_t* data, size_t len,
                                                const uint32_t* targets, size_t num_targets,
                                                int64_t* results);
ALGOAT_C_EXPORT void algoat_search_batch_int16(const int16_t* data, size_t len,
                                               const int16_t* targets, size_t num_targets,
                                               int64_t* results);
ALGOAT_C_EXPORT void algoat_search_batch_uint16(const uint16_t* data, size_t len,
                                                const uint16_t* targets, size_t num_targets,
                                                int64_t* results);
ALGOAT_C_EXPORT void algoat_search_batch_int8(const int8_t* data, size_t len, const int8_t* targets,
                                              size_t num_targets, int64_t* results);
ALGOAT_C_EXPORT void algoat_search_batch_uint8(const uint8_t* data, size_t len,
                                               const uint8_t* targets, size_t num_targets,
                                               int64_t* results);
ALGOAT_C_EXPORT void algoat_search_batch_float64(const double* data, size_t len,
                                                 const double* targets, size_t num_targets,
                                                 int64_t* results);
ALGOAT_C_EXPORT void algoat_search_batch_float32(const float* data, size_t len,
                                                 const float* targets, size_t num_targets,
                                                 int64_t* results);

/*
 * Top-K selection entry points.
 * Rearranges data such that the subrange [0, k) contains the k smallest
 * elements in strictly sorted ascending order.
 */
ALGOAT_C_EXPORT void algoat_topk_int64(int64_t* data, size_t len, size_t k);
ALGOAT_C_EXPORT void algoat_topk_uint64(uint64_t* data, size_t len, size_t k);
ALGOAT_C_EXPORT void algoat_topk_int32(int32_t* data, size_t len, size_t k);
ALGOAT_C_EXPORT void algoat_topk_uint32(uint32_t* data, size_t len, size_t k);
ALGOAT_C_EXPORT void algoat_topk_int16(int16_t* data, size_t len, size_t k);
ALGOAT_C_EXPORT void algoat_topk_uint16(uint16_t* data, size_t len, size_t k);
ALGOAT_C_EXPORT void algoat_topk_int8(int8_t* data, size_t len, size_t k);
ALGOAT_C_EXPORT void algoat_topk_uint8(uint8_t* data, size_t len, size_t k);
ALGOAT_C_EXPORT void algoat_topk_float64(double* data, size_t len, size_t k);
ALGOAT_C_EXPORT void algoat_topk_float32(float* data, size_t len, size_t k);

#ifdef __cplusplus
}
#endif

#endif /* ALGOAT_C_H */
