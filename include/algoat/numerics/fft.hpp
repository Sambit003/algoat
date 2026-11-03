/**
 * @file fft.hpp
 * @brief Fast Fourier Transform (FFT) Engine — 1D/2D Cooley-Tukey with SIMD Butterfly.
 *
 * Implements high-performance Fast Fourier Transforms:
 * 1. 1D Complex FFT / IFFT using iterative Cooley-Tukey Radix-2 Decimation-in-Time (DIT)
 *    accelerated with AVX2/FMA SIMD complex butterflies.
 * 2. 2D FFT via row-column decomposition.
 * 3. Real FFT / IRFFT exploiting Hermitian symmetry for real-valued signals.
 * 4. FFT-accelerated linear convolution (convolve_fft).
 * 5. Precomputed 64-byte cache-line aligned twiddle factor tables.
 *
 * @author Algoat Development Team
 * @date 2026
 */

#pragma once

#include <atomic>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numbers>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace algoat::numerics {

/// \brief Direction of transform.
enum class FFTDirection : int { Forward = -1, Inverse = +1 };

/// \brief Returns the smallest power of 2 >= n.
[[nodiscard]] constexpr std::size_t next_pow2(std::size_t n) noexcept {
    if (n <= 1) {
        return 1;
    }
    return std::bit_ceil(n);
}

namespace detail {

/// \brief 64-byte aligned allocator ensuring cache-line alignment for SIMD vector operations.
template <typename T, std::size_t Alignment = 64> struct AlignedAllocator {
    using value_type = T;

    template <typename U> struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;
    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        std::size_t bytes = n * sizeof(T);
#if defined(_MSC_VER)
        void* ptr = _aligned_malloc(bytes, Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, bytes) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
#endif
    }

    void deallocate(T* p, std::size_t) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    bool operator==(const AlignedAllocator&) const noexcept {
        return true;
    }
    bool operator!=(const AlignedAllocator&) const noexcept {
        return false;
    }
};

/// \brief Branchless 32-bit integer bit reversal.
[[nodiscard]] inline constexpr uint32_t bit_reverse32(uint32_t x, int bits) noexcept {
    x = (((x & 0xaaaaaaaaU) >> 1) | ((x & 0x55555555U) << 1));
    x = (((x & 0xccccccccU) >> 2) | ((x & 0x33333333U) << 2));
    x = (((x & 0xf0f0f0f0U) >> 4) | ((x & 0x0f0f0f0fU) << 4));
    x = (((x & 0xff00ff00U) >> 8) | ((x & 0x00ff00ffU) << 8));
    return ((x >> 16) | (x << 16)) >> (32 - bits);
}

#if defined(__AVX2__)
/// \brief AVX2 vectorized bit reversal for 8 packed 32-bit integers.
[[nodiscard]] inline __m256i bit_reverse_avx2(__m256i v, int bits) noexcept {
    const __m256i bswap_mask =
        _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5,
                         4, 11, 10, 9, 8, 15, 14, 13, 12);
    v = _mm256_shuffle_epi8(v, bswap_mask);

    const __m256i m_nib = _mm256_set1_epi32(0x0f0f0f0f);
    __m256i v1 = _mm256_slli_epi32(_mm256_and_si256(v, m_nib), 4);
    __m256i v2 = _mm256_srli_epi32(_mm256_andnot_si256(m_nib, v), 4);
    v = _mm256_or_si256(v1, v2);

    const __m256i m_pair = _mm256_set1_epi32(0x33333333);
    v1 = _mm256_slli_epi32(_mm256_and_si256(v, m_pair), 2);
    v2 = _mm256_srli_epi32(_mm256_andnot_si256(m_pair, v), 2);
    v = _mm256_or_si256(v1, v2);

    const __m256i m_bit = _mm256_set1_epi32(0x55555555);
    v1 = _mm256_slli_epi32(_mm256_and_si256(v, m_bit), 1);
    v2 = _mm256_srli_epi32(_mm256_andnot_si256(m_bit, v), 1);
    v = _mm256_or_si256(v1, v2);

    return _mm256_srli_epi32(v, 32 - bits);
}
#endif

/// \brief In-place bit-reversal permutation for an array of size N = 2^bits.
template <typename T> inline void bit_reversal_permute(std::span<T> data) noexcept {
    const std::size_t n = data.size();
    if (n <= 2) {
        return;
    }
    const int bits = std::countr_zero(n);

    std::size_t i = 0;
#if defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        __m256i indices = _mm256_setr_epi32(static_cast<int>(i), static_cast<int>(i + 1),
                                            static_cast<int>(i + 2), static_cast<int>(i + 3),
                                            static_cast<int>(i + 4), static_cast<int>(i + 5),
                                            static_cast<int>(i + 6), static_cast<int>(i + 7));
        __m256i reversed = bit_reverse_avx2(indices, bits);
        alignas(32) uint32_t rev_buf[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(rev_buf), reversed);

        for (std::size_t v = 0; v < 8; ++v) {
            std::size_t cur = i + v;
            std::size_t rev = rev_buf[v];
            if (cur < rev) {
                std::swap(data[cur], data[rev]);
            }
        }
    }
#endif
    for (; i < n; ++i) {
        std::size_t rev = bit_reverse32(static_cast<uint32_t>(i), bits);
        if (i < rev) {
            std::swap(data[i], data[rev]);
        }
    }
}

/// \brief Precomputed, 64-byte cache-aligned stage-contiguous twiddle factor tables.
class TwiddleTable {
public:
    using AlignedComplexFloat =
        std::vector<std::complex<float>, AlignedAllocator<std::complex<float>, 64>>;
    using AlignedComplexDouble =
        std::vector<std::complex<double>, AlignedAllocator<std::complex<double>, 64>>;

    static TwiddleTable& instance() {
        static TwiddleTable inst;
        return inst;
    }

    void ensure_capacity(std::size_t n) {
        if (n <= current_capacity_.load(std::memory_order_acquire)) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (n <= current_capacity_.load(std::memory_order_relaxed)) {
            return;
        }
        std::size_t target = next_pow2(n);
        int max_stage = std::countr_zero(target);

        if (stage_twiddles_float_.size() <= static_cast<std::size_t>(max_stage)) {
            stage_twiddles_float_.resize(max_stage + 1);
            stage_twiddles_double_.resize(max_stage + 1);
        }

        constexpr double pi2 = 2.0 * std::numbers::pi_v<double>;

        for (int s = 1; s <= max_stage; ++s) {
            std::size_t L = std::size_t{1} << s;
            std::size_t h = L >> 1;
            if (stage_twiddles_float_[s].size() < h) {
                stage_twiddles_float_[s].resize(h);
                stage_twiddles_double_[s].resize(h);
                const double angle_step = -pi2 / static_cast<double>(L);
                for (std::size_t k = 0; k < h; ++k) {
                    double angle = angle_step * static_cast<double>(k);
                    double c = std::cos(angle);
                    double s_val = std::sin(angle);
                    stage_twiddles_double_[s][k] = {c, s_val};
                    stage_twiddles_float_[s][k] = {static_cast<float>(c),
                                                   static_cast<float>(s_val)};
                }
            }
        }

        current_capacity_.store(target, std::memory_order_release);
    }

    [[nodiscard]] std::size_t capacity() const noexcept {
        return current_capacity_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] const AlignedComplexFloat& stage_float(int stage) const noexcept {
        return stage_twiddles_float_[stage];
    }

    [[nodiscard]] const AlignedComplexDouble& stage_double(int stage) const noexcept {
        return stage_twiddles_double_[stage];
    }

private:
    TwiddleTable() {
        ensure_capacity(1024);
    }

    std::mutex mutex_;
    std::atomic<std::size_t> current_capacity_{0};
    std::vector<AlignedComplexFloat> stage_twiddles_float_;
    std::vector<AlignedComplexDouble> stage_twiddles_double_;
};

#if defined(__AVX2__) && defined(__FMA__)

inline void butterfly_stage_simd_avx2(std::span<std::complex<float>> data, std::size_t L,
                                      std::size_t h, FFTDirection dir,
                                      const std::complex<float>* stage_tw) noexcept {
    const std::size_t n = data.size();
    const bool is_inverse = (dir == FFTDirection::Inverse);
    const __m256 neg_imag = _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);

    for (std::size_t i = 0; i < n; i += L) {
        std::size_t k = 0;
        // Process 4 complex numbers at once (8 floats) with unit-stride streaming load
        for (; k + 4 <= h; k += 4) {
            __m256 w = _mm256_load_ps(reinterpret_cast<const float*>(stage_tw + k));
            if (is_inverse) {
                w = _mm256_xor_ps(w, neg_imag);
            }

            // Load 4 complex elements from upper half (v)
            __m256 v_vec = _mm256_loadu_ps(reinterpret_cast<const float*>(data.data() + i + k + h));

            // Complex multiplication: v * w = (v_r*w_r - v_i*w_i) + i*(v_r*w_i + v_i*w_r)
            __m256 v_re = _mm256_moveldup_ps(v_vec);
            __m256 v_im = _mm256_movehdup_ps(v_vec);
            __m256 w_swap = _mm256_permute_ps(w, 0xB1);
            __m256 prod = _mm256_mul_ps(v_im, w_swap);
            __m256 t = _mm256_fmaddsub_ps(v_re, w, prod);

            // Load 4 complex elements from lower half (u)
            __m256 u_vec = _mm256_loadu_ps(reinterpret_cast<const float*>(data.data() + i + k));

            __m256 out_u = _mm256_add_ps(u_vec, t);
            __m256 out_v = _mm256_sub_ps(u_vec, t);

            _mm256_storeu_ps(reinterpret_cast<float*>(data.data() + i + k), out_u);
            _mm256_storeu_ps(reinterpret_cast<float*>(data.data() + i + k + h), out_v);
        }

        // Remainder scalar loop
        const float sign = is_inverse ? -1.0f : 1.0f;
        for (; k < h; ++k) {
            auto tw = stage_tw[k];
            std::complex<float> w{tw.real(), tw.imag() * sign};
            auto u = data[i + k];
            auto v = data[i + k + h] * w;
            data[i + k] = u + v;
            data[i + k + h] = u - v;
        }
    }
}

inline void butterfly_stage_simd_avx2(std::span<std::complex<double>> data, std::size_t L,
                                      std::size_t h, FFTDirection dir,
                                      const std::complex<double>* stage_tw) noexcept {
    const std::size_t n = data.size();
    const bool is_inverse = (dir == FFTDirection::Inverse);
    const __m256d neg_imag = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);

    for (std::size_t i = 0; i < n; i += L) {
        std::size_t k = 0;
        // Process 2 complex doubles at once (4 doubles)
        for (; k + 2 <= h; k += 2) {
            __m256d w = _mm256_load_pd(reinterpret_cast<const double*>(stage_tw + k));
            if (is_inverse) {
                w = _mm256_xor_pd(w, neg_imag);
            }

            __m256d v_vec =
                _mm256_loadu_pd(reinterpret_cast<const double*>(data.data() + i + k + h));

            __m256d v_re = _mm256_movedup_pd(v_vec);
            __m256d v_im = _mm256_permute_pd(v_vec, 0xF);
            __m256d w_swap = _mm256_permute_pd(w, 0x5);
            __m256d prod = _mm256_mul_pd(v_im, w_swap);
            __m256d t = _mm256_fmaddsub_pd(v_re, w, prod);

            __m256d u_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(data.data() + i + k));

            __m256d out_u = _mm256_add_pd(u_vec, t);
            __m256d out_v = _mm256_sub_pd(u_vec, t);

            _mm256_storeu_pd(reinterpret_cast<double*>(data.data() + i + k), out_u);
            _mm256_storeu_pd(reinterpret_cast<double*>(data.data() + i + k + h), out_v);
        }

        const double sign = is_inverse ? -1.0 : 1.0;
        for (; k < h; ++k) {
            auto tw = stage_tw[k];
            std::complex<double> w{tw.real(), tw.imag() * sign};
            auto u = data[i + k];
            auto v = data[i + k + h] * w;
            data[i + k] = u + v;
            data[i + k + h] = u - v;
        }
    }
}

#endif // AVX2 && FMA

template <typename T>
inline void butterfly_stage_scalar(std::span<std::complex<T>> data, std::size_t L, std::size_t h,
                                   FFTDirection dir, const std::complex<T>* stage_tw) noexcept {
    const std::size_t n = data.size();
    const T sign = (dir == FFTDirection::Forward) ? static_cast<T>(1) : static_cast<T>(-1);

    for (std::size_t i = 0; i < n; i += L) {
        for (std::size_t k = 0; k < h; ++k) {
            auto tw = stage_tw[k];
            std::complex<T> w{tw.real(), tw.imag() * sign};
            auto u = data[i + k];
            auto v = data[i + k + h] * w;
            data[i + k] = u + v;
            data[i + k + h] = u - v;
        }
    }
}

template <typename T> inline void fft_1d_impl(std::span<std::complex<T>> data, FFTDirection dir) {
    const std::size_t n = data.size();
    if (n <= 1) {
        return;
    }

    // Step 1: Bit-reversal permutation
    bit_reversal_permute(data);

    // Ensure twiddle factor capacity
    auto& tw_table = TwiddleTable::instance();
    tw_table.ensure_capacity(n);

    // Step 2: Stage 1 (L = 2, h = 1): W_2^0 = 1, zero trigonometric multiplications
    for (std::size_t i = 0; i < n; i += 2) {
        auto u = data[i];
        auto v = data[i + 1];
        data[i] = u + v;
        data[i + 1] = u - v;
    }

    if (n == 2) {
        return;
    }

    // Step 3: Stage 2 (L = 4, h = 2): W_4^0 = 1, W_4^1 = -i (Forward) or +i (Inverse)
    const T imag_sign = (dir == FFTDirection::Forward) ? static_cast<T>(-1) : static_cast<T>(1);
    for (std::size_t i = 0; i < n; i += 4) {
        auto u0 = data[i];
        auto v0 = data[i + 2];
        data[i] = u0 + v0;
        data[i + 2] = u0 - v0;

        auto u1 = data[i + 1];
        auto v1 = data[i + 3];
        std::complex<T> v1w{-imag_sign * v1.imag(), imag_sign * v1.real()};
        data[i + 1] = u1 + v1w;
        data[i + 3] = u1 - v1w;
    }

    // Step 4: Subsequent stages L = 8, 16, ..., n
    int stage = 3;
    for (std::size_t L = 8; L <= n; L <<= 1, ++stage) {
        const std::size_t h = L >> 1;
        if constexpr (std::is_same_v<T, float>) {
            const auto* stage_tw = tw_table.stage_float(stage).data();
#if defined(__AVX2__) && defined(__FMA__)
            if (h >= 4) {
                butterfly_stage_simd_avx2(data, L, h, dir, stage_tw);
                continue;
            }
#endif
            butterfly_stage_scalar(data, L, h, dir, stage_tw);
        } else {
            const auto* stage_tw = tw_table.stage_double(stage).data();
#if defined(__AVX2__) && defined(__FMA__)
            if (h >= 2) {
                butterfly_stage_simd_avx2(data, L, h, dir, stage_tw);
                continue;
            }
#endif
            butterfly_stage_scalar(data, L, h, dir, stage_tw);
        }
    }
}

template <typename T>
inline void fft_2d_impl(std::span<std::complex<T>> data, std::size_t rows, std::size_t cols,
                        FFTDirection dir) {
    // Step 1: 1D FFT along each row
    for (std::size_t r = 0; r < rows; ++r) {
        std::span<std::complex<T>> row_span(data.data() + r * cols, cols);
        fft_1d_impl(row_span, dir);
    }

    // Step 2: 1D FFT along each column
    std::vector<std::complex<T>> col_buf(rows);
    for (std::size_t c = 0; c < cols; ++c) {
        for (std::size_t r = 0; r < rows; ++r) {
            col_buf[r] = data[r * cols + c];
        }
        fft_1d_impl(std::span<std::complex<T>>(col_buf), dir);
        for (std::size_t r = 0; r < rows; ++r) {
            data[r * cols + c] = col_buf[r];
        }
    }

    // Step 3: Normalization for inverse transform
    if (dir == FFTDirection::Inverse) {
        const T inv_total = static_cast<T>(1) / static_cast<T>(rows * cols);
        for (auto& val : data) {
            val *= inv_total;
        }
    }
}

} // namespace detail

/// \brief Pre-computes and caches twiddle factor tables for a given N.
inline void precompute_twiddles(std::size_t N) {
    if (N > 0) {
        detail::TwiddleTable::instance().ensure_capacity(next_pow2(N));
    }
}

/// \brief In-place 1D DFT of complex float data. N must be a power of 2.
/// \throws std::invalid_argument If data.size() is not a power of 2 (including size 0).
inline void fft_1d(std::span<std::complex<float>> data, FFTDirection dir = FFTDirection::Forward) {
    if (!std::has_single_bit(data.size())) {
        throw std::invalid_argument("fft_1d: data size must be a power of 2");
    }
    detail::fft_1d_impl(data, dir);
}

/// \brief In-place 1D DFT of complex double data. N must be a power of 2.
/// \throws std::invalid_argument If data.size() is not a power of 2 (including size 0).
inline void fft_1d(std::span<std::complex<double>> data, FFTDirection dir = FFTDirection::Forward) {
    if (!std::has_single_bit(data.size())) {
        throw std::invalid_argument("fft_1d: data size must be a power of 2");
    }
    detail::fft_1d_impl(data, dir);
}

/// \brief Inverse FFT (performs forward + normalization by 1/N).
/// \throws std::invalid_argument If data.size() is not a power of 2 (including size 0).
inline void ifft_1d(std::span<std::complex<float>> data) {
    if (!std::has_single_bit(data.size())) {
        throw std::invalid_argument("ifft_1d: data size must be a power of 2");
    }
    fft_1d(data, FFTDirection::Inverse);
    const float inv_n = 1.0f / static_cast<float>(data.size());
#if defined(__AVX2__)
    const std::size_t n = data.size();
    std::size_t i = 0;
    __m256 v_scale = _mm256_set1_ps(inv_n);
    for (; i + 4 <= n; i += 4) {
        __m256 v = _mm256_loadu_ps(reinterpret_cast<const float*>(data.data() + i));
        _mm256_storeu_ps(reinterpret_cast<float*>(data.data() + i), _mm256_mul_ps(v, v_scale));
    }
    for (; i < n; ++i) {
        data[i] *= inv_n;
    }
#else
    for (auto& x : data) {
        x *= inv_n;
    }
#endif
}

/// \brief Inverse FFT (performs forward + normalization by 1/N).
/// \throws std::invalid_argument If data.size() is not a power of 2 (including size 0).
inline void ifft_1d(std::span<std::complex<double>> data) {
    if (!std::has_single_bit(data.size())) {
        throw std::invalid_argument("ifft_1d: data size must be a power of 2");
    }
    fft_1d(data, FFTDirection::Inverse);
    const double inv_n = 1.0 / static_cast<double>(data.size());
#if defined(__AVX2__)
    const std::size_t n = data.size();
    std::size_t i = 0;
    __m256d v_scale = _mm256_set1_pd(inv_n);
    for (; i + 2 <= n; i += 2) {
        __m256d v = _mm256_loadu_pd(reinterpret_cast<const double*>(data.data() + i));
        _mm256_storeu_pd(reinterpret_cast<double*>(data.data() + i), _mm256_mul_pd(v, v_scale));
    }
    for (; i < n; ++i) {
        data[i] *= inv_n;
    }
#else
    for (auto& x : data) {
        x *= inv_n;
    }
#endif
}

/// \brief 2D in-place DFT via row-column decomposition.
/// \param data Flattened row-major complex matrix of size rows x cols.
/// \throws std::invalid_argument If dimensions are not powers of 2 or size != rows * cols.
inline void fft_2d(std::span<std::complex<float>> data, std::size_t rows, std::size_t cols,
                   FFTDirection dir = FFTDirection::Forward) {
    if (!std::has_single_bit(rows) || !std::has_single_bit(cols)) {
        throw std::invalid_argument("fft_2d: rows and cols must be powers of 2");
    }
    if (data.size() != rows * cols) {
        throw std::invalid_argument("fft_2d: data size does not match rows * cols");
    }
    detail::fft_2d_impl(data, rows, cols, dir);
}

/// \brief 2D in-place DFT via row-column decomposition.
/// \param data Flattened row-major complex matrix of size rows x cols.
/// \throws std::invalid_argument If dimensions are not powers of 2 or size != rows * cols.
inline void fft_2d(std::span<std::complex<double>> data, std::size_t rows, std::size_t cols,
                   FFTDirection dir = FFTDirection::Forward) {
    if (!std::has_single_bit(rows) || !std::has_single_bit(cols)) {
        throw std::invalid_argument("fft_2d: rows and cols must be powers of 2");
    }
    if (data.size() != rows * cols) {
        throw std::invalid_argument("fft_2d: data size does not match rows * cols");
    }
    detail::fft_2d_impl(data, rows, cols, dir);
}

/// \brief Real-to-complex FFT: exploits Hermitian symmetry for 2x speedup.
/// \param real_in Input of N real values.
/// \param complex_out Output of N/2 + 1 complex values.
/// \throws std::invalid_argument If real_in size is not a power of 2 or complex_out is too small.
inline void rfft_1d(std::span<const float> real_in, std::span<std::complex<float>> complex_out) {
    const std::size_t n = real_in.size();
    if (!std::has_single_bit(n)) {
        throw std::invalid_argument("rfft_1d: real_in size must be a power of 2");
    }
    if (complex_out.size() < n / 2 + 1) {
        throw std::invalid_argument("rfft_1d: complex_out buffer must be at least N/2 + 1");
    }

    if (n == 1) {
        complex_out[0] = {real_in[0], 0.0f};
        return;
    }
    if (n == 2) {
        complex_out[0] = {real_in[0] + real_in[1], 0.0f};
        complex_out[1] = {real_in[0] - real_in[1], 0.0f};
        return;
    }

    const std::size_t half_n = n / 2;
    std::vector<std::complex<float>> z(half_n);
    for (std::size_t i = 0; i < half_n; ++i) {
        z[i] = {real_in[2 * i], real_in[2 * i + 1]};
    }

    fft_1d(z, FFTDirection::Forward);

    // DC and Nyquist components
    complex_out[0] = {z[0].real() + z[0].imag(), 0.0f};
    complex_out[half_n] = {z[0].real() - z[0].imag(), 0.0f};

    constexpr double pi2 = 2.0 * std::numbers::pi_v<double>;
    const double angle_step = -pi2 / static_cast<double>(n);

    for (std::size_t k = 1; k <= half_n / 2; ++k) {
        const auto z_forward = z[k];
        const auto z_reverse = z[half_n - k];

        const float z_r = z_forward.real();
        const float z_i = z_forward.imag();
        const float z_rev_r = z_reverse.real();
        const float z_rev_i = z_reverse.imag();

        const float even_real = 0.5f * (z_r + z_rev_r);
        const float even_imag = 0.5f * (z_i - z_rev_i);
        const float odd_real = 0.5f * (z_i + z_rev_i);
        const float odd_imag = -0.5f * (z_r - z_rev_r);

        const double angle = angle_step * static_cast<double>(k);
        const float twiddle_r = static_cast<float>(std::cos(angle));
        const float twiddle_i = static_cast<float>(std::sin(angle));

        const float term_r = twiddle_r * odd_real - twiddle_i * odd_imag;
        const float term_i = twiddle_r * odd_imag + twiddle_i * odd_real;

        complex_out[k] = {even_real + term_r, even_imag + term_i};
        complex_out[half_n - k] = {even_real - term_r, term_i - even_imag};
    }
}

/// \brief Complex-to-real inverse FFT (irfft).
/// \param complex_in Input of N/2 + 1 complex values.
/// \param real_out Output of N real values.
/// \throws std::invalid_argument If real_out size is not a power of 2 or complex_in is too small.
inline void irfft_1d(std::span<const std::complex<float>> complex_in, std::span<float> real_out) {
    const std::size_t n = real_out.size();
    if (!std::has_single_bit(n)) {
        throw std::invalid_argument("irfft_1d: real_out size must be a power of 2");
    }
    if (complex_in.size() < n / 2 + 1) {
        throw std::invalid_argument("irfft_1d: complex_in buffer must be at least N/2 + 1");
    }

    if (n == 1) {
        real_out[0] = complex_in[0].real();
        return;
    }
    if (n == 2) {
        real_out[0] = 0.5f * (complex_in[0].real() + complex_in[1].real());
        real_out[1] = 0.5f * (complex_in[0].real() - complex_in[1].real());
        return;
    }

    const std::size_t half_n = n / 2;
    std::vector<std::complex<float>> z(half_n);

    // Reconstruct Z[0]
    z[0] = {0.5f * (complex_in[0].real() + complex_in[half_n].real()),
            0.5f * (complex_in[0].real() - complex_in[half_n].real())};

    constexpr double pi2 = 2.0 * std::numbers::pi_v<double>;
    const double angle_step = pi2 / static_cast<double>(n);

    for (std::size_t k = 1; k <= half_n / 2; ++k) {
        const auto x_forward = complex_in[k];
        const auto x_reverse = complex_in[half_n - k];

        const float avg_real = 0.5f * (x_forward.real() + x_reverse.real());
        const float avg_imag = 0.5f * (x_forward.imag() - x_reverse.imag());
        const float diff_real = 0.5f * (x_forward.real() - x_reverse.real());
        const float diff_imag = 0.5f * (x_forward.imag() + x_reverse.imag());

        const double angle = angle_step * static_cast<double>(k);
        const float twiddle_r = static_cast<float>(std::cos(angle));
        const float twiddle_i = static_cast<float>(std::sin(angle));

        // i * W_N^{-k} = -twiddle_i + i * twiddle_r
        const float factor_r = -twiddle_i;
        const float factor_i = twiddle_r;

        const float term_r = factor_r * diff_real - factor_i * diff_imag;
        const float term_i = factor_r * diff_imag + factor_i * diff_real;

        z[k] = {avg_real + term_r, avg_imag + term_i};
        z[half_n - k] = {avg_real - term_r, term_i - avg_imag};
    }

    // Run N/2-point IFFT (which already normalizes by 1 / (N/2) = 2/N)
    ifft_1d(z);

    for (std::size_t i = 0; i < half_n; ++i) {
        real_out[2 * i] = z[i].real();
        real_out[2 * i + 1] = z[i].imag();
    }
}

/// \brief Convolution via FFT: computes out = a conv b.
/// \details Pads to next power of 2, FFTs, multiplies pointwise, IFFTs.
template <typename T>
    requires std::floating_point<T> || std::integral<T>
[[nodiscard]] std::vector<T> convolve_fft(std::span<const T> a, std::span<const T> b) {
    if (a.empty() || b.empty()) {
        return {};
    }

    const std::size_t result_len = a.size() + b.size() - 1;
    const std::size_t n = next_pow2(result_len);

    using ComputeType = std::conditional_t<std::is_same_v<T, float>, float, double>;

    std::vector<std::complex<ComputeType>> fa(n, {0, 0});
    std::vector<std::complex<ComputeType>> fb(n, {0, 0});

    for (std::size_t i = 0; i < a.size(); ++i) {
        fa[i] = {static_cast<ComputeType>(a[i]), 0};
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        fb[i] = {static_cast<ComputeType>(b[i]), 0};
    }

    fft_1d(fa, FFTDirection::Forward);
    fft_1d(fb, FFTDirection::Forward);

    for (std::size_t i = 0; i < n; ++i) {
        fa[i] *= fb[i];
    }

    ifft_1d(fa);

    std::vector<T> result(result_len);
    for (std::size_t i = 0; i < result_len; ++i) {
        if constexpr (std::is_integral_v<T>) {
            result[i] = static_cast<T>(std::round(fa[i].real()));
        } else {
            result[i] = static_cast<T>(fa[i].real());
        }
    }
    return result;
}

} // namespace algoat::numerics
