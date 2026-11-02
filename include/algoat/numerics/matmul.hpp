/**
 * @file matmul.hpp
 * @brief Cache-Oblivious GEMM & Strassen Sub-Cubic Matrix Multiplication Engine.
 *
 * Implements high-performance dense matrix multiplication algorithms for floating-point matrices:
 * 1. Cache-Oblivious Recursive GEMM (Frigo et al., FOCS 1999) using recursive 3-way
 *    quad-tree subdivision (Morton Z-order traversal) with AVX2/AVX-512 FMA micro-kernels.
 * 2. Strassen Sub-Cubic Multiplication (Strassen 1969) reducing O(N^3) arithmetic complexity
 *    to O(N^2.807) with single-allocation scratchpad memory reuse and dynamic peeling for
 *    arbitrary non-power-of-two matrix dimensions.
 * 3. Parallelized Divide-and-Conquer GEMM utilizing OpenMP tasking across independent
 *    submatrices without data races.
 * 4. Cache-line-aligned (64-byte) matrix allocation helper.
 *
 * @author Algoat Development Team
 * @date 2026
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace algoat::numerics {

/// \brief 64-byte aligned allocator ensuring cache-line alignment.
template <typename T, size_t Alignment = 64> struct AlignedAllocator {
    using value_type = T;

    template <typename U> struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(size_t n) {
        if (n == 0)
            return nullptr;
        size_t bytes = n * sizeof(T);
        constexpr size_t align = (Alignment < sizeof(void*)) ? sizeof(void*) : Alignment;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, align, bytes) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t) noexcept {
        free(p);
    }

    bool operator==(const AlignedAllocator&) const noexcept = default;
};

/// \brief Row-major matrix view (non-owning).
template <typename T> struct MatrixView {
    T* data{nullptr};
    size_t rows{0};
    size_t cols{0};
    size_t row_stride{0}; ///< Elements per row (may differ from cols for sub-matrices)

    constexpr MatrixView() noexcept = default;

    constexpr MatrixView(T* ptr, size_t r, size_t c, size_t stride) noexcept
        : data(ptr), rows(r), cols(c), row_stride(stride) {}

    constexpr MatrixView(T* ptr, size_t r, size_t c) noexcept
        : data(ptr), rows(r), cols(c), row_stride(c) {}

    // Implicit conversion from non-const to const MatrixView
    template <typename U>
        requires(std::is_same_v<T, const U>)
    constexpr MatrixView(const MatrixView<U>& other) noexcept
        : data(other.data), rows(other.rows), cols(other.cols), row_stride(other.row_stride) {}

    [[nodiscard]] constexpr T& at(size_t r, size_t c) noexcept {
        return data[r * row_stride + c];
    }

    [[nodiscard]] constexpr const T& at(size_t r, size_t c) const noexcept {
        return data[r * row_stride + c];
    }

    /// \brief Subview slicing
    [[nodiscard]] constexpr MatrixView subview(size_t r_offset, size_t c_offset, size_t r_len,
                                               size_t c_len) const noexcept {
        return MatrixView(data + r_offset * row_stride + c_offset, r_len, c_len, row_stride);
    }
};

/// \brief Helper to allocate a cache-line-aligned (64-byte) matrix.
template <typename T, typename Alloc = AlignedAllocator<T, 64>>
[[nodiscard]] std::vector<T, Alloc> make_matrix(size_t rows, size_t cols) {
    return std::vector<T, Alloc>(rows * cols, T{0});
}

namespace detail {

#if defined(__AVX512F__)
inline void microkernel_8x16_fp32_avx512(size_t K, const float* A_ptr, size_t lda,
                                         const float* B_ptr, size_t ldb, float* C_ptr,
                                         size_t ldc) noexcept {
    __m512 c0 = _mm512_loadu_ps(C_ptr + 0 * ldc);
    __m512 c1 = _mm512_loadu_ps(C_ptr + 1 * ldc);
    __m512 c2 = _mm512_loadu_ps(C_ptr + 2 * ldc);
    __m512 c3 = _mm512_loadu_ps(C_ptr + 3 * ldc);
    __m512 c4 = _mm512_loadu_ps(C_ptr + 4 * ldc);
    __m512 c5 = _mm512_loadu_ps(C_ptr + 5 * ldc);
    __m512 c6 = _mm512_loadu_ps(C_ptr + 6 * ldc);
    __m512 c7 = _mm512_loadu_ps(C_ptr + 7 * ldc);

    for (size_t p = 0; p < K; ++p) {
        __m512 b = _mm512_loadu_ps(B_ptr + p * ldb);
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(A_ptr[0 * lda + p]), b, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(A_ptr[1 * lda + p]), b, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(A_ptr[2 * lda + p]), b, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(A_ptr[3 * lda + p]), b, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(A_ptr[4 * lda + p]), b, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(A_ptr[5 * lda + p]), b, c5);
        c6 = _mm512_fmadd_ps(_mm512_set1_ps(A_ptr[6 * lda + p]), b, c6);
        c7 = _mm512_fmadd_ps(_mm512_set1_ps(A_ptr[7 * lda + p]), b, c7);
    }

    _mm512_storeu_ps(C_ptr + 0 * ldc, c0);
    _mm512_storeu_ps(C_ptr + 1 * ldc, c1);
    _mm512_storeu_ps(C_ptr + 2 * ldc, c2);
    _mm512_storeu_ps(C_ptr + 3 * ldc, c3);
    _mm512_storeu_ps(C_ptr + 4 * ldc, c4);
    _mm512_storeu_ps(C_ptr + 5 * ldc, c5);
    _mm512_storeu_ps(C_ptr + 6 * ldc, c6);
    _mm512_storeu_ps(C_ptr + 7 * ldc, c7);
}
#endif

#if defined(__AVX2__) && defined(__FMA__)
inline void microkernel_6x16_fp32(size_t K, const float* A_ptr, size_t lda, const float* B_ptr,
                                  size_t ldb, float* C_ptr, size_t ldc) noexcept {
    __m256 c00 = _mm256_loadu_ps(C_ptr + 0 * ldc + 0);
    __m256 c01 = _mm256_loadu_ps(C_ptr + 0 * ldc + 8);
    __m256 c10 = _mm256_loadu_ps(C_ptr + 1 * ldc + 0);
    __m256 c11 = _mm256_loadu_ps(C_ptr + 1 * ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C_ptr + 2 * ldc + 0);
    __m256 c21 = _mm256_loadu_ps(C_ptr + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C_ptr + 3 * ldc + 0);
    __m256 c31 = _mm256_loadu_ps(C_ptr + 3 * ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C_ptr + 4 * ldc + 0);
    __m256 c41 = _mm256_loadu_ps(C_ptr + 4 * ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C_ptr + 5 * ldc + 0);
    __m256 c51 = _mm256_loadu_ps(C_ptr + 5 * ldc + 8);

    for (size_t p = 0; p < K; ++p) {
        __m256 b0 = _mm256_loadu_ps(B_ptr + p * ldb + 0);
        __m256 b1 = _mm256_loadu_ps(B_ptr + p * ldb + 8);

        __m256 a0 = _mm256_set1_ps(A_ptr[0 * lda + p]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        __m256 a1 = _mm256_set1_ps(A_ptr[1 * lda + p]);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        __m256 a2 = _mm256_set1_ps(A_ptr[2 * lda + p]);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        __m256 a3 = _mm256_set1_ps(A_ptr[3 * lda + p]);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);

        __m256 a4 = _mm256_set1_ps(A_ptr[4 * lda + p]);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);

        __m256 a5 = _mm256_set1_ps(A_ptr[5 * lda + p]);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    _mm256_storeu_ps(C_ptr + 0 * ldc + 0, c00);
    _mm256_storeu_ps(C_ptr + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C_ptr + 1 * ldc + 0, c10);
    _mm256_storeu_ps(C_ptr + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C_ptr + 2 * ldc + 0, c20);
    _mm256_storeu_ps(C_ptr + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C_ptr + 3 * ldc + 0, c30);
    _mm256_storeu_ps(C_ptr + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C_ptr + 4 * ldc + 0, c40);
    _mm256_storeu_ps(C_ptr + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C_ptr + 5 * ldc + 0, c50);
    _mm256_storeu_ps(C_ptr + 5 * ldc + 8, c51);
}

inline void microkernel_6x8_fp64(size_t K, const double* A_ptr, size_t lda, const double* B_ptr,
                                 size_t ldb, double* C_ptr, size_t ldc) noexcept {
    __m256d c00 = _mm256_loadu_pd(C_ptr + 0 * ldc + 0);
    __m256d c01 = _mm256_loadu_pd(C_ptr + 0 * ldc + 4);
    __m256d c10 = _mm256_loadu_pd(C_ptr + 1 * ldc + 0);
    __m256d c11 = _mm256_loadu_pd(C_ptr + 1 * ldc + 4);
    __m256d c20 = _mm256_loadu_pd(C_ptr + 2 * ldc + 0);
    __m256d c21 = _mm256_loadu_pd(C_ptr + 2 * ldc + 4);
    __m256d c30 = _mm256_loadu_pd(C_ptr + 3 * ldc + 0);
    __m256d c31 = _mm256_loadu_pd(C_ptr + 3 * ldc + 4);
    __m256d c40 = _mm256_loadu_pd(C_ptr + 4 * ldc + 0);
    __m256d c41 = _mm256_loadu_pd(C_ptr + 4 * ldc + 4);
    __m256d c50 = _mm256_loadu_pd(C_ptr + 5 * ldc + 0);
    __m256d c51 = _mm256_loadu_pd(C_ptr + 5 * ldc + 4);

    for (size_t p = 0; p < K; ++p) {
        __m256d b0 = _mm256_loadu_pd(B_ptr + p * ldb + 0);
        __m256d b1 = _mm256_loadu_pd(B_ptr + p * ldb + 4);

        __m256d a0 = _mm256_set1_pd(A_ptr[0 * lda + p]);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c01 = _mm256_fmadd_pd(a0, b1, c01);

        __m256d a1 = _mm256_set1_pd(A_ptr[1 * lda + p]);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c11 = _mm256_fmadd_pd(a1, b1, c11);

        __m256d a2 = _mm256_set1_pd(A_ptr[2 * lda + p]);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c21 = _mm256_fmadd_pd(a2, b1, c21);

        __m256d a3 = _mm256_set1_pd(A_ptr[3 * lda + p]);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c31 = _mm256_fmadd_pd(a3, b1, c31);

        __m256d a4 = _mm256_set1_pd(A_ptr[4 * lda + p]);
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c41 = _mm256_fmadd_pd(a4, b1, c41);

        __m256d a5 = _mm256_set1_pd(A_ptr[5 * lda + p]);
        c50 = _mm256_fmadd_pd(a5, b0, c50);
        c51 = _mm256_fmadd_pd(a5, b1, c51);
    }

    _mm256_storeu_pd(C_ptr + 0 * ldc + 0, c00);
    _mm256_storeu_pd(C_ptr + 0 * ldc + 4, c01);
    _mm256_storeu_pd(C_ptr + 1 * ldc + 0, c10);
    _mm256_storeu_pd(C_ptr + 1 * ldc + 4, c11);
    _mm256_storeu_pd(C_ptr + 2 * ldc + 0, c20);
    _mm256_storeu_pd(C_ptr + 2 * ldc + 4, c21);
    _mm256_storeu_pd(C_ptr + 3 * ldc + 0, c30);
    _mm256_storeu_pd(C_ptr + 3 * ldc + 4, c31);
    _mm256_storeu_pd(C_ptr + 4 * ldc + 0, c40);
    _mm256_storeu_pd(C_ptr + 4 * ldc + 4, c41);
    _mm256_storeu_pd(C_ptr + 5 * ldc + 0, c50);
    _mm256_storeu_pd(C_ptr + 5 * ldc + 4, c51);
}
#endif

template <std::floating_point T>
inline void base_gemm(MatrixView<const T> A, MatrixView<const T> B, MatrixView<T> C) noexcept {
    const size_t M = A.rows;
    const size_t K = A.cols;
    const size_t N = B.cols;

#if defined(__AVX512F__)
    if constexpr (std::is_same_v<T, float>) {
        size_t i = 0;
        for (; i + 8 <= M; i += 8) {
            size_t j = 0;
            for (; j + 16 <= N; j += 16) {
                microkernel_8x16_fp32_avx512(K, &A.at(i, 0), A.row_stride, &B.at(0, j),
                                             B.row_stride, &C.at(i, j), C.row_stride);
            }
            for (; j < N; ++j) {
                for (size_t r = 0; r < 8; ++r) {
                    float sum = 0.0f;
                    for (size_t p = 0; p < K; ++p) {
                        sum += A.at(i + r, p) * B.at(p, j);
                    }
                    C.at(i + r, j) += sum;
                }
            }
        }
        for (; i < M; ++i) {
            for (size_t p = 0; p < K; ++p) {
                float a_ip = A.at(i, p);
                for (size_t j = 0; j < N; ++j) {
                    C.at(i, j) += a_ip * B.at(p, j);
                }
            }
        }
        return;
    }
#elif defined(__AVX2__) && defined(__FMA__)
    if constexpr (std::is_same_v<T, float>) {
        size_t i = 0;
        for (; i + 6 <= M; i += 6) {
            size_t j = 0;
            for (; j + 16 <= N; j += 16) {
                microkernel_6x16_fp32(K, &A.at(i, 0), A.row_stride, &B.at(0, j), B.row_stride,
                                      &C.at(i, j), C.row_stride);
            }
            for (; j < N; ++j) {
                for (size_t r = 0; r < 6; ++r) {
                    float sum = 0.0f;
                    for (size_t p = 0; p < K; ++p) {
                        sum += A.at(i + r, p) * B.at(p, j);
                    }
                    C.at(i + r, j) += sum;
                }
            }
        }
        for (; i < M; ++i) {
            for (size_t p = 0; p < K; ++p) {
                float a_ip = A.at(i, p);
                for (size_t j = 0; j < N; ++j) {
                    C.at(i, j) += a_ip * B.at(p, j);
                }
            }
        }
        return;
    } else if constexpr (std::is_same_v<T, double>) {
        size_t i = 0;
        for (; i + 6 <= M; i += 6) {
            size_t j = 0;
            for (; j + 8 <= N; j += 8) {
                microkernel_6x8_fp64(K, &A.at(i, 0), A.row_stride, &B.at(0, j), B.row_stride,
                                     &C.at(i, j), C.row_stride);
            }
            for (; j < N; ++j) {
                for (size_t r = 0; r < 6; ++r) {
                    double sum = 0.0;
                    for (size_t p = 0; p < K; ++p) {
                        sum += A.at(i + r, p) * B.at(p, j);
                    }
                    C.at(i + r, j) += sum;
                }
            }
        }
        for (; i < M; ++i) {
            for (size_t p = 0; p < K; ++p) {
                double a_ip = A.at(i, p);
                for (size_t j = 0; j < N; ++j) {
                    C.at(i, j) += a_ip * B.at(p, j);
                }
            }
        }
        return;
    }
#endif

    // Portable fallback (IKJ loop order)
    for (size_t i = 0; i < M; ++i) {
        for (size_t p = 0; p < K; ++p) {
            T a_ip = A.at(i, p);
            for (size_t j = 0; j < N; ++j) {
                C.at(i, j) += a_ip * B.at(p, j);
            }
        }
    }
}

template <typename T> struct ScratchBuffer {
    std::unique_ptr<T[]> storage;
    size_t capacity{0};
    size_t offset{0};

    explicit ScratchBuffer(size_t max_elements)
        : storage(std::make_unique<T[]>(max_elements)), capacity(max_elements) {}

    MatrixView<T> alloc(size_t r, size_t c) {
        size_t count = r * c;
        if (offset + count > capacity) {
            overflow_blocks.push_back(std::make_unique<T[]>(count));
            return MatrixView<T>(overflow_blocks.back().get(), r, c, c);
        }
        MatrixView<T> view(storage.get() + offset, r, c, c);
        offset += count;
        return view;
    }

    void reset(size_t mark) noexcept {
        offset = mark;
    }

    [[nodiscard]] size_t mark() const noexcept {
        return offset;
    }

private:
    std::vector<std::unique_ptr<T[]>> overflow_blocks;
};

template <typename T>
inline void mat_add(MatrixView<const T> A, MatrixView<const T> B, MatrixView<T> Out) noexcept {
    const size_t R = A.rows;
    const size_t C = A.cols;
    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            Out.at(i, j) = A.at(i, j) + B.at(i, j);
        }
    }
}

template <typename T>
inline void mat_sub(MatrixView<const T> A, MatrixView<const T> B, MatrixView<T> Out) noexcept {
    const size_t R = A.rows;
    const size_t C = A.cols;
    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            Out.at(i, j) = A.at(i, j) - B.at(i, j);
        }
    }
}

} // namespace detail

/// \brief Cache-oblivious recursive GEMM: C += A * B.
/// \tparam T Must be float or double.
/// \details Optimal cache complexity O(N^3 / B sqrt(M)) without block-size tuning.
///          Base case dispatches to AVX2/AVX-512 FMA micro-kernel.
template <std::floating_point T>
void matmul_cache_oblivious(MatrixView<const T> A, MatrixView<const T> B,
                            MatrixView<T> C) noexcept {
    const size_t M = A.rows;
    const size_t K = A.cols;
    const size_t N = B.cols;

    if (M == 0 || K == 0 || N == 0)
        return;

    // Cache-oblivious base-case threshold
    constexpr size_t BASE_THRESHOLD = 48;

    if (M <= BASE_THRESHOLD && N <= BASE_THRESHOLD && K <= BASE_THRESHOLD) {
        detail::base_gemm(A, B, C);
        return;
    }

    if (M >= K && M >= N) {
        // Halve M: A split horizontally, C split horizontally
        const size_t M1 = M / 2;
        const size_t M2 = M - M1;
        matmul_cache_oblivious(A.subview(0, 0, M1, K), B, C.subview(0, 0, M1, N));
        matmul_cache_oblivious(A.subview(M1, 0, M2, K), B, C.subview(M1, 0, M2, N));
    } else if (N >= M && N >= K) {
        // Halve N: B split vertically, C split vertically
        const size_t N1 = N / 2;
        const size_t N2 = N - N1;
        matmul_cache_oblivious(A, B.subview(0, 0, K, N1), C.subview(0, 0, M, N1));
        matmul_cache_oblivious(A, B.subview(0, N1, K, N2), C.subview(0, N1, M, N2));
    } else {
        // Halve K: A split vertically, B split horizontally (sequential accumulation)
        const size_t K1 = K / 2;
        const size_t K2 = K - K1;
        matmul_cache_oblivious(A.subview(0, 0, M, K1), B.subview(0, 0, K1, N), C);
        matmul_cache_oblivious(A.subview(0, K1, M, K2), B.subview(K1, 0, K2, N), C);
    }
}

namespace detail {

template <std::floating_point T>
void matmul_strassen_impl(MatrixView<const T> A, MatrixView<const T> B, MatrixView<T> C,
                          size_t threshold, ScratchBuffer<T>& scratch) {
    const size_t M = A.rows;
    const size_t K = A.cols;
    const size_t N = B.cols;

    if (M != N || K != N || N <= threshold) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C.at(i, j) = T{0};
            }
        }
        matmul_cache_oblivious(A, B, C);
        return;
    }

    if (N % 2 != 0) {
        // Dynamic peeling for odd dimension N = 2m + 1
        const size_t N_even = N - 1;
        auto A_prime = A.subview(0, 0, N_even, N_even);
        auto B_prime = B.subview(0, 0, N_even, N_even);
        auto C_prime = C.subview(0, 0, N_even, N_even);

        auto a_col = A.subview(0, N_even, N_even, 1);
        auto a_row = A.subview(N_even, 0, 1, N_even);
        T alpha = A.at(N_even, N_even);

        auto b_col = B.subview(0, N_even, N_even, 1);
        auto b_row = B.subview(N_even, 0, 1, N_even);
        T beta = B.at(N_even, N_even);

        auto c_col = C.subview(0, N_even, N_even, 1);
        auto c_row = C.subview(N_even, 0, 1, N_even);

        // 1. Recurse on even submatrix C' = A' * B'
        matmul_strassen_impl(A_prime, B_prime, C_prime, threshold, scratch);

        // 2. Rank-1 update C' += a_col * b_row^T
        for (size_t i = 0; i < N_even; ++i) {
            T a_val = a_col.at(i, 0);
            for (size_t j = 0; j < N_even; ++j) {
                C_prime.at(i, j) += a_val * b_row.at(0, j);
            }
        }

        // 3. Compute right column c_col = A' * b_col + beta * a_col
        for (size_t i = 0; i < N_even; ++i) {
            T sum = beta * a_col.at(i, 0);
            for (size_t p = 0; p < N_even; ++p) {
                sum += A_prime.at(i, p) * b_col.at(p, 0);
            }
            c_col.at(i, 0) = sum;
        }

        // 4. Compute bottom row c_row = a_row * B' + alpha * b_row
        for (size_t j = 0; j < N_even; ++j) {
            T sum = alpha * b_row.at(0, j);
            for (size_t p = 0; p < N_even; ++p) {
                sum += a_row.at(0, p) * B_prime.at(p, j);
            }
            c_row.at(0, j) = sum;
        }

        // 5. Compute corner gamma = a_row * b_col + alpha * beta
        T gamma = alpha * beta;
        for (size_t p = 0; p < N_even; ++p) {
            gamma += a_row.at(0, p) * b_col.at(p, 0);
        }
        C.at(N_even, N_even) = gamma;
        return;
    }

    const size_t m = N / 2;

    auto A11 = A.subview(0, 0, m, m);
    auto A12 = A.subview(0, m, m, m);
    auto A21 = A.subview(m, 0, m, m);
    auto A22 = A.subview(m, m, m, m);

    auto B11 = B.subview(0, 0, m, m);
    auto B12 = B.subview(0, m, m, m);
    auto B21 = B.subview(m, 0, m, m);
    auto B22 = B.subview(m, m, m, m);

    auto C11 = C.subview(0, 0, m, m);
    auto C12 = C.subview(0, m, m, m);
    auto C21 = C.subview(m, 0, m, m);
    auto C22 = C.subview(m, m, m, m);

    const size_t base_mark = scratch.mark();

    auto M1 = scratch.alloc(m, m);
    auto M2 = scratch.alloc(m, m);
    auto M3 = scratch.alloc(m, m);
    auto M4 = scratch.alloc(m, m);
    auto M5 = scratch.alloc(m, m);
    auto M6 = scratch.alloc(m, m);
    auto M7 = scratch.alloc(m, m);

    // M1 = (A11 + A22) * (B11 + B22)
    {
        const size_t op_mark = scratch.mark();
        auto S_A = scratch.alloc(m, m);
        auto S_B = scratch.alloc(m, m);
        mat_add(A11, A22, S_A);
        mat_add(B11, B22, S_B);
        matmul_strassen_impl(MatrixView<const T>(S_A), MatrixView<const T>(S_B), M1, threshold,
                             scratch);
        scratch.reset(op_mark);
    }

    // M2 = (A21 + A22) * B11
    {
        const size_t op_mark = scratch.mark();
        auto S_A = scratch.alloc(m, m);
        mat_add(A21, A22, S_A);
        matmul_strassen_impl(MatrixView<const T>(S_A), B11, M2, threshold, scratch);
        scratch.reset(op_mark);
    }

    // M3 = A11 * (B12 - B22)
    {
        const size_t op_mark = scratch.mark();
        auto S_B = scratch.alloc(m, m);
        mat_sub(B12, B22, S_B);
        matmul_strassen_impl(A11, MatrixView<const T>(S_B), M3, threshold, scratch);
        scratch.reset(op_mark);
    }

    // M4 = A22 * (B21 - B11)
    {
        const size_t op_mark = scratch.mark();
        auto S_B = scratch.alloc(m, m);
        mat_sub(B21, B11, S_B);
        matmul_strassen_impl(A22, MatrixView<const T>(S_B), M4, threshold, scratch);
        scratch.reset(op_mark);
    }

    // M5 = (A11 + A12) * B22
    {
        const size_t op_mark = scratch.mark();
        auto S_A = scratch.alloc(m, m);
        mat_add(A11, A12, S_A);
        matmul_strassen_impl(MatrixView<const T>(S_A), B22, M5, threshold, scratch);
        scratch.reset(op_mark);
    }

    // M6 = (A21 - A11) * (B11 + B12)
    {
        const size_t op_mark = scratch.mark();
        auto S_A = scratch.alloc(m, m);
        auto S_B = scratch.alloc(m, m);
        mat_sub(A21, A11, S_A);
        mat_add(B11, B12, S_B);
        matmul_strassen_impl(MatrixView<const T>(S_A), MatrixView<const T>(S_B), M6, threshold,
                             scratch);
        scratch.reset(op_mark);
    }

    // M7 = (A12 - A22) * (B21 + B22)
    {
        const size_t op_mark = scratch.mark();
        auto S_A = scratch.alloc(m, m);
        auto S_B = scratch.alloc(m, m);
        mat_sub(A12, A22, S_A);
        mat_add(B21, B22, S_B);
        matmul_strassen_impl(MatrixView<const T>(S_A), MatrixView<const T>(S_B), M7, threshold,
                             scratch);
        scratch.reset(op_mark);
    }

    // Recombine products into C
    // C11 = M1 + M4 - M5 + M7
    // C12 = M3 + M5
    // C21 = M2 + M4
    // C22 = M1 - M2 + M3 + M6
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            const T m1 = M1.at(i, j);
            const T m2 = M2.at(i, j);
            const T m3 = M3.at(i, j);
            const T m4 = M4.at(i, j);
            const T m5 = M5.at(i, j);
            const T m6 = M6.at(i, j);
            const T m7 = M7.at(i, j);

            C11.at(i, j) = m1 + m4 - m5 + m7;
            C12.at(i, j) = m3 + m5;
            C21.at(i, j) = m2 + m4;
            C22.at(i, j) = m1 - m2 + m3 + m6;
        }
    }

    scratch.reset(base_mark);
}

} // namespace detail

/// \brief Strassen sub-cubic GEMM: C = A * B (out-of-place).
/// \details O(N^2.807) by recursing with 7 sub-problems of half size.
/// \param strassen_threshold Switch to cache-oblivious below this dimension.
template <std::floating_point T>
void matmul_strassen(MatrixView<const T> A, MatrixView<const T> B, MatrixView<T> C,
                     size_t strassen_threshold = 128) {
    const size_t N = A.rows;
    detail::ScratchBuffer<T> scratch(3 * N * N + 1024);
    detail::matmul_strassen_impl(A, B, C, strassen_threshold, scratch);
}

namespace detail {

#if defined(_OPENMP)
template <std::floating_point T>
void matmul_parallel_omp_task(MatrixView<const T> A, MatrixView<const T> B, MatrixView<T> C) {
    const size_t M = A.rows;
    const size_t K = A.cols;
    const size_t N = B.cols;

    if (M == 0 || K == 0 || N == 0)
        return;

    constexpr size_t PARALLEL_CUTOFF = 64;

    if (M <= PARALLEL_CUTOFF && N <= PARALLEL_CUTOFF && K <= PARALLEL_CUTOFF) {
        matmul_cache_oblivious(A, B, C);
        return;
    }

    if (M >= K && M >= N) {
        const size_t M1 = M / 2;
        const size_t M2 = M - M1;
#pragma omp task
        matmul_parallel_omp_task(A.subview(0, 0, M1, K), B, C.subview(0, 0, M1, N));
#pragma omp task
        matmul_parallel_omp_task(A.subview(M1, 0, M2, K), B, C.subview(M1, 0, M2, N));
#pragma omp taskwait
    } else if (N >= M && N >= K) {
        const size_t N1 = N / 2;
        const size_t N2 = N - N1;
#pragma omp task
        matmul_parallel_omp_task(A, B.subview(0, 0, K, N1), C.subview(0, 0, M, N1));
#pragma omp task
        matmul_parallel_omp_task(A, B.subview(0, N1, K, N2), C.subview(0, N1, M, N2));
#pragma omp taskwait
    } else {
        const size_t K1 = K / 2;
        const size_t K2 = K - K1;
        matmul_parallel_omp_task(A.subview(0, 0, M, K1), B.subview(0, 0, K1, N), C);
        matmul_parallel_omp_task(A.subview(0, K1, M, K2), B.subview(K1, 0, K2, N), C);
    }
}
#endif

template <std::floating_point T>
void matmul_parallel_thread_impl(MatrixView<const T> A, MatrixView<const T> B, MatrixView<T> C,
                                 size_t depth = 0) {
    const size_t M = A.rows;
    const size_t K = A.cols;
    const size_t N = B.cols;

    if (M == 0 || K == 0 || N == 0)
        return;

    constexpr size_t MAX_ASYNC_DEPTH = 4;
    constexpr size_t PARALLEL_CUTOFF = 64;

    if (depth >= MAX_ASYNC_DEPTH || (M <= PARALLEL_CUTOFF && N <= PARALLEL_CUTOFF)) {
        matmul_cache_oblivious(A, B, C);
        return;
    }

    if (M >= K && M >= N) {
        const size_t M1 = M / 2;
        const size_t M2 = M - M1;
        auto fut = std::async(std::launch::async, [=]() {
            matmul_parallel_thread_impl(A.subview(0, 0, M1, K), B, C.subview(0, 0, M1, N),
                                        depth + 1);
        });
        matmul_parallel_thread_impl(A.subview(M1, 0, M2, K), B, C.subview(M1, 0, M2, N), depth + 1);
        fut.get();
    } else if (N >= M && N >= K) {
        const size_t N1 = N / 2;
        const size_t N2 = N - N1;
        auto fut = std::async(std::launch::async, [=]() {
            matmul_parallel_thread_impl(A, B.subview(0, 0, K, N1), C.subview(0, 0, M, N1),
                                        depth + 1);
        });
        matmul_parallel_thread_impl(A, B.subview(0, N1, K, N2), C.subview(0, N1, M, N2), depth + 1);
        fut.get();
    } else {
        const size_t K1 = K / 2;
        const size_t K2 = K - K1;
        matmul_parallel_thread_impl(A.subview(0, 0, M, K1), B.subview(0, 0, K1, N), C, depth + 1);
        matmul_parallel_thread_impl(A.subview(0, K1, M, K2), B.subview(K1, 0, K2, N), C, depth + 1);
    }
}

} // namespace detail

/// \brief Parallelized cache-oblivious GEMM using divide-and-conquer tasking.
template <std::floating_point T>
void matmul_parallel(MatrixView<const T> A, MatrixView<const T> B, MatrixView<T> C) {
#if defined(_OPENMP)
    if (omp_in_parallel()) {
        detail::matmul_parallel_omp_task(A, B, C);
    } else {
#pragma omp parallel
        {
#pragma omp single
            {
                detail::matmul_parallel_omp_task(A, B, C);
            }
        }
    }
#else
    detail::matmul_parallel_thread_impl(A, B, C, 0);
#endif
}

} // namespace algoat::numerics
