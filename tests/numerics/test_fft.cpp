#include <algoat/numerics/fft.hpp>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace algoat::numerics;

TEST(FFTTest, NextPow2) {
    EXPECT_EQ(next_pow2(0), 1);
    EXPECT_EQ(next_pow2(1), 1);
    EXPECT_EQ(next_pow2(2), 2);
    EXPECT_EQ(next_pow2(3), 4);
    EXPECT_EQ(next_pow2(4), 4);
    EXPECT_EQ(next_pow2(5), 8);
    EXPECT_EQ(next_pow2(7), 8);
    EXPECT_EQ(next_pow2(8), 8);
    EXPECT_EQ(next_pow2(9), 16);
    EXPECT_EQ(next_pow2(1023), 1024);
    EXPECT_EQ(next_pow2(1024), 1024);
    EXPECT_EQ(next_pow2(1025), 2048);
}

TEST(FFTTest, NonPowerOfTwoThrows) {
    std::vector<std::complex<float>> data3(3);
    EXPECT_THROW(fft_1d(data3), std::invalid_argument);

    std::vector<std::complex<float>> data5(5);
    EXPECT_THROW(fft_1d(data5), std::invalid_argument);

    std::vector<std::complex<double>> ddata7(7);
    EXPECT_THROW(fft_1d(ddata7), std::invalid_argument);

    std::vector<std::complex<float>> data_empty;
    EXPECT_THROW(fft_1d(data_empty), std::invalid_argument);
}

TEST(FFTTest, KnownTransformSizes) {
    // 1-point
    std::vector<std::complex<float>> data1 = {{3.0f, 4.0f}};
    fft_1d(data1);
    EXPECT_FLOAT_EQ(data1[0].real(), 3.0f);
    EXPECT_FLOAT_EQ(data1[0].imag(), 4.0f);
    ifft_1d(data1);
    EXPECT_FLOAT_EQ(data1[0].real(), 3.0f);
    EXPECT_FLOAT_EQ(data1[0].imag(), 4.0f);

    // 2-point
    std::vector<std::complex<float>> data2 = {{1.0f, 0.0f}, {1.0f, 0.0f}};
    fft_1d(data2);
    EXPECT_NEAR(data2[0].real(), 2.0f, 1e-6f);
    EXPECT_NEAR(data2[0].imag(), 0.0f, 1e-6f);
    EXPECT_NEAR(data2[1].real(), 0.0f, 1e-6f);
    EXPECT_NEAR(data2[1].imag(), 0.0f, 1e-6f);
    ifft_1d(data2);
    EXPECT_NEAR(data2[0].real(), 1.0f, 1e-6f);
    EXPECT_NEAR(data2[1].real(), 1.0f, 1e-6f);

    // 4-point impulse
    std::vector<std::complex<float>> data4 = {
        {1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
    fft_1d(data4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(data4[i].real(), 1.0f, 1e-6f);
        EXPECT_NEAR(data4[i].imag(), 0.0f, 1e-6f);
    }
    ifft_1d(data4);
    EXPECT_NEAR(data4[0].real(), 1.0f, 1e-6f);
    EXPECT_NEAR(data4[1].real(), 0.0f, 1e-6f);
    EXPECT_NEAR(data4[2].real(), 0.0f, 1e-6f);
    EXPECT_NEAR(data4[3].real(), 0.0f, 1e-6f);
}

template <typename T>
static double compute_l2_relative_error(const std::vector<std::complex<T>>& original,
                                        const std::vector<std::complex<T>>& reconstructed) {
    double diff_sq = 0.0;
    double norm_sq = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        double dr = static_cast<double>(reconstructed[i].real() - original[i].real());
        double di = static_cast<double>(reconstructed[i].imag() - original[i].imag());
        diff_sq += dr * dr + di * di;
        double or_ = static_cast<double>(original[i].real());
        double oi = static_cast<double>(original[i].imag());
        norm_sq += or_ * or_ + oi * oi;
    }
    return (norm_sq > 0.0) ? std::sqrt(diff_sq / norm_sq) : std::sqrt(diff_sq);
}

TEST(FFTTest, RoundtripReconstructionFloat) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::vector<size_t> sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (size_t N : sizes) {
        std::vector<std::complex<float>> original(N);
        for (size_t i = 0; i < N; ++i) {
            original[i] = {dist(rng), dist(rng)};
        }
        std::vector<std::complex<float>> copy = original;
        fft_1d(copy, FFTDirection::Forward);
        ifft_1d(copy);

        double rel_err = compute_l2_relative_error(original, copy);
        EXPECT_LT(rel_err, 1e-5) << "Failed for N = " << N;
    }
}

TEST(FFTTest, RoundtripReconstructionDouble) {
    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    const std::vector<size_t> sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    for (size_t N : sizes) {
        std::vector<std::complex<double>> original(N);
        for (size_t i = 0; i < N; ++i) {
            original[i] = {dist(rng), dist(rng)};
        }
        std::vector<std::complex<double>> copy = original;
        fft_1d(copy, FFTDirection::Forward);
        ifft_1d(copy);

        double rel_err = compute_l2_relative_error(original, copy);
        EXPECT_LT(rel_err, 1e-12) << "Failed for N = " << N;
    }
}

TEST(FFTTest, AcceptanceCriteriaLargeScaleFloatRoundtrip) {
    size_t N = 1 << 18; // 262,144 points
    std::mt19937_64 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::complex<float>> original(N);
    for (size_t i = 0; i < N; ++i) {
        original[i] = {dist(rng), dist(rng)};
    }
    std::vector<std::complex<float>> copy = original;
    fft_1d(copy, FFTDirection::Forward);
    ifft_1d(copy);

    double rel_err = compute_l2_relative_error(original, copy);
    EXPECT_LT(rel_err, 1e-5);
}

TEST(FFTTest, RealFFTMatchesComplexFFT) {
    std::mt19937_64 rng(555);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::vector<size_t> sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (size_t N : sizes) {
        std::vector<float> real_in(N);
        std::vector<std::complex<float>> complex_ref(N);
        for (size_t i = 0; i < N; ++i) {
            real_in[i] = dist(rng);
            complex_ref[i] = {real_in[i], 0.0f};
        }

        std::vector<std::complex<float>> complex_out(N / 2 + 1);
        rfft_1d(real_in, complex_out);

        fft_1d(complex_ref, FFTDirection::Forward);

        for (size_t k = 0; k <= N / 2; ++k) {
            EXPECT_NEAR(complex_out[k].real(), complex_ref[k].real(), 1e-4f)
                << "Mismatch real at k=" << k << " for N=" << N;
            EXPECT_NEAR(complex_out[k].imag(), complex_ref[k].imag(), 1e-4f)
                << "Mismatch imag at k=" << k << " for N=" << N;
        }
    }
}

TEST(FFTTest, RealFFTRoundtrip) {
    std::mt19937_64 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const std::vector<size_t> sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    for (size_t N : sizes) {
        std::vector<float> original(N);
        for (size_t i = 0; i < N; ++i) {
            original[i] = dist(rng);
        }

        std::vector<std::complex<float>> complex_freq(N / 2 + 1);
        rfft_1d(original, complex_freq);

        std::vector<float> reconstructed(N);
        irfft_1d(complex_freq, reconstructed);

        double diff_sq = 0.0;
        double norm_sq = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double d = static_cast<double>(reconstructed[i] - original[i]);
            diff_sq += d * d;
            double o = static_cast<double>(original[i]);
            norm_sq += o * o;
        }
        double rel_err = std::sqrt(diff_sq / norm_sq);
        EXPECT_LT(rel_err, 1e-5) << "RFFT roundtrip failed for N = " << N;
    }
}

TEST(FFTTest, RealFFTInvalidArguments) {
    std::vector<float> non_pow2(6, 1.0f);
    std::vector<std::complex<float>> out(4);
    EXPECT_THROW(rfft_1d(non_pow2, out), std::invalid_argument);

    std::vector<float> pow2(8, 1.0f);
    std::vector<std::complex<float>> small_out(3);
    EXPECT_THROW(rfft_1d(pow2, small_out), std::invalid_argument);

    std::vector<std::complex<float>> cin(5);
    std::vector<float> non_pow2_out(6);
    EXPECT_THROW(irfft_1d(cin, non_pow2_out), std::invalid_argument);

    std::vector<std::complex<float>> small_cin(3);
    std::vector<float> rout(8);
    EXPECT_THROW(irfft_1d(small_cin, rout), std::invalid_argument);
}

TEST(FFTTest, FFT2DMatchesReferenceDecomposition) {
    const size_t rows = 16;
    const size_t cols = 32;
    std::mt19937_64 rng(101);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<std::complex<float>> matrix(rows * cols);
    for (auto& val : matrix) {
        val = {dist(rng), dist(rng)};
    }
    std::vector<std::complex<float>> original = matrix;
    std::vector<std::complex<float>> ref = matrix;

    // Reference 2D DFT via explicit 1D row FFTs then 1D col FFTs
    for (size_t r = 0; r < rows; ++r) {
        std::span<std::complex<float>> row_span(ref.data() + r * cols, cols);
        fft_1d(row_span, FFTDirection::Forward);
    }
    for (size_t c = 0; c < cols; ++c) {
        std::vector<std::complex<float>> col_buf(rows);
        for (size_t r = 0; r < rows; ++r) {
            col_buf[r] = ref[r * cols + c];
        }
        fft_1d(col_buf, FFTDirection::Forward);
        for (size_t r = 0; r < rows; ++r) {
            ref[r * cols + c] = col_buf[r];
        }
    }

    // Call fft_2d
    fft_2d(matrix, rows, cols, FFTDirection::Forward);

    for (size_t i = 0; i < rows * cols; ++i) {
        EXPECT_NEAR(matrix[i].real(), ref[i].real(), 1e-4f) << "At index " << i;
        EXPECT_NEAR(matrix[i].imag(), ref[i].imag(), 1e-4f) << "At index " << i;
    }

    // Test 2D inverse roundtrip back to original
    fft_2d(matrix, rows, cols, FFTDirection::Inverse);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            EXPECT_NEAR(matrix[r * cols + c].real(), original[r * cols + c].real(), 1e-4f);
            EXPECT_NEAR(matrix[r * cols + c].imag(), original[r * cols + c].imag(), 1e-4f);
        }
    }
}

TEST(FFTTest, FFT2DAcceptanceCriteria512x512) {
    const size_t rows = 512;
    const size_t cols = 512;
    std::mt19937_64 rng(202);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::complex<float>> matrix(rows * cols);
    for (auto& val : matrix) {
        val = {dist(rng), dist(rng)};
    }
    std::vector<std::complex<float>> original = matrix;
    std::vector<std::complex<float>> ref = matrix;

    // Reference DFT computed via 512 x 1D row FFTs followed by 512 x 1D col FFTs
    for (size_t r = 0; r < rows; ++r) {
        std::span<std::complex<float>> row_span(ref.data() + r * cols, cols);
        fft_1d(row_span, FFTDirection::Forward);
    }
    std::vector<std::complex<float>> col_buf(rows);
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) {
            col_buf[r] = ref[r * cols + c];
        }
        fft_1d(col_buf, FFTDirection::Forward);
        for (size_t r = 0; r < rows; ++r) {
            ref[r * cols + c] = col_buf[r];
        }
    }

    // Call fft_2d and compare with reference
    fft_2d(matrix, rows, cols, FFTDirection::Forward);
    double rel_err_fwd = compute_l2_relative_error(ref, matrix);
    EXPECT_LT(rel_err_fwd, 1e-5);

    // Inverse roundtrip back to original
    fft_2d(matrix, rows, cols, FFTDirection::Inverse);
    double rel_err_inv = compute_l2_relative_error(original, matrix);
    EXPECT_LT(rel_err_inv, 1e-5);
}

TEST(FFTTest, FFT2DInvalidArguments) {
    std::vector<std::complex<float>> mat(16);
    EXPECT_THROW(fft_2d(mat, 3, 5), std::invalid_argument);
    EXPECT_THROW(fft_2d(mat, 4, 3), std::invalid_argument);
    EXPECT_THROW(fft_2d(mat, 4, 8), std::invalid_argument); // size 16 != 4*8
}

template <typename T>
static std::vector<T> naive_poly_multiply(std::span<const T> a, std::span<const T> b) {
    if (a.empty() || b.empty()) {
        return {};
    }
    std::vector<T> res(a.size() + b.size() - 1, T{});
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            res[i + j] += a[i] * b[j];
        }
    }
    return res;
}

TEST(FFTTest, ConvolveFFTMatchesDirectPolynomialMultiplication) {
    std::mt19937_64 rng(303);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    // Degree <= 100 as per Acceptance Criteria
    const size_t deg_a = 75;
    const size_t deg_b = 95;

    std::vector<double> a(deg_a + 1);
    std::vector<double> b(deg_b + 1);
    for (auto& v : a)
        v = dist(rng);
    for (auto& v : b)
        v = dist(rng);

    auto expected = naive_poly_multiply<double>(a, b);
    auto actual = convolve_fft<double>(a, b);

    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-9) << "At index " << i;
    }
}

TEST(FFTTest, ConvolveFFTIntegerExactness) {
    std::vector<int> a = {1, 2, 3, 4};
    std::vector<int> b = {5, 6, 7};
    auto expected = naive_poly_multiply<int>(a, b);
    auto actual = convolve_fft<int>(a, b);

    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual[i], expected[i]) << "At index " << i;
    }
}

TEST(FFTTest, ConvolveFFTEmptyInputs) {
    std::vector<float> empty_vec;
    std::vector<float> non_empty = {1.0f, 2.0f};

    EXPECT_TRUE(convolve_fft<float>(empty_vec, non_empty).empty());
    EXPECT_TRUE(convolve_fft<float>(non_empty, empty_vec).empty());
    EXPECT_TRUE(convolve_fft<float>(empty_vec, empty_vec).empty());
}
