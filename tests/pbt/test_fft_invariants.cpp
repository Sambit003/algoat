#include "algoat/numerics/fft.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <numbers>
#include <rapidcheck.h>
#include <vector>

using namespace algoat::numerics;

TEST(FFTPBT, ParsevalTheoremInvariant) {
    rc::check("Energy in time domain equals energy in frequency domain divided by N",
              [](size_t p_raw) {
                  // Test power-of-two sizes from 2^1 (2) to 2^10 (1024)
                  size_t power = (p_raw % 10) + 1;
                  size_t N = size_t{1} << power;

                  std::vector<std::complex<double>> time_data(N);
                  for (size_t i = 0; i < N; ++i) {
                      double r = static_cast<double>(static_cast<int>(i % 31) - 15) * 0.125;
                      double im = static_cast<double>(static_cast<int>((i * 7) % 31) - 15) * 0.125;
                      time_data[i] = {r, im};
                  }

                  double time_energy = 0.0;
                  for (const auto& val : time_data) {
                      time_energy += std::norm(val); // |val|^2
                  }

                  std::vector<std::complex<double>> freq_data = time_data;
                  fft_1d(freq_data, FFTDirection::Forward);

                  double freq_energy = 0.0;
                  for (const auto& val : freq_data) {
                      freq_energy += std::norm(val);
                  }
                  freq_energy /= static_cast<double>(N);

                  RC_ASSERT(std::abs(time_energy - freq_energy) < 1e-9 * (time_energy + 1.0));
              });
}

TEST(FFTPBT, LinearityInvariant) {
    rc::check("FFT(alpha * x + beta * y) == alpha * FFT(x) + beta * FFT(y)",
              [](size_t p_raw, int alpha_raw, int beta_raw) {
                  size_t power = (p_raw % 9) + 1; // 2..512
                  size_t N = size_t{1} << power;

                  double alpha = (static_cast<double>(alpha_raw % 10) + 1.0) * 0.25;
                  double beta = (static_cast<double>(beta_raw % 10) + 1.0) * 0.25;

                  std::vector<std::complex<double>> x(N);
                  std::vector<std::complex<double>> y(N);
                  std::vector<std::complex<double>> combined(N);

                  for (size_t i = 0; i < N; ++i) {
                      double rx = static_cast<double>(static_cast<int>(i % 17) - 8);
                      double ix = static_cast<double>(static_cast<int>((i * 3) % 17) - 8);
                      double ry = static_cast<double>(static_cast<int>((i * 5) % 19) - 9);
                      double iy = static_cast<double>(static_cast<int>((i * 11) % 19) - 9);

                      x[i] = {rx, ix};
                      y[i] = {ry, iy};
                      combined[i] = alpha * x[i] + beta * y[i];
                  }

                  fft_1d(x, FFTDirection::Forward);
                  fft_1d(y, FFTDirection::Forward);
                  fft_1d(combined, FFTDirection::Forward);

                  for (size_t k = 0; k < N; ++k) {
                      std::complex<double> expected = alpha * x[k] + beta * y[k];
                      RC_ASSERT(std::abs(combined[k] - expected) < 1e-9);
                  }
              });
}

TEST(FFTPBT, TimeShiftTheoremInvariant) {
    rc::check("Time shift by m corresponds to multiplication by exp(-2*pi*i*k*m/N)",
              [](size_t p_raw, size_t m_raw) {
                  size_t power = (p_raw % 8) + 1; // 2..256
                  size_t N = size_t{1} << power;
                  size_t m = m_raw % N;

                  std::vector<std::complex<double>> x(N);
                  std::vector<std::complex<double>> x_shifted(N);

                  for (size_t i = 0; i < N; ++i) {
                      double r = static_cast<double>(static_cast<int>(i % 13) - 6);
                      double im = static_cast<double>(static_cast<int>((i * 4) % 13) - 6);
                      x[i] = {r, im};
                  }

                  for (size_t i = 0; i < N; ++i) {
                      x_shifted[i] = x[(i + N - m) % N];
                  }

                  fft_1d(x, FFTDirection::Forward);
                  fft_1d(x_shifted, FFTDirection::Forward);

                  constexpr double pi2 = 2.0 * std::numbers::pi_v<double>;
                  for (size_t k = 0; k < N; ++k) {
                      double angle = -pi2 * static_cast<double>(k * m) / static_cast<double>(N);
                      std::complex<double> twiddle{std::cos(angle), std::sin(angle)};
                      std::complex<double> expected = x[k] * twiddle;
                      RC_ASSERT(std::abs(x_shifted[k] - expected) < 1e-9);
                  }
              });
}

TEST(FFTPBT, ConvolutionCommutativityInvariant) {
    rc::check("a * b == b * a", [](const std::vector<int>& raw_a, const std::vector<int>& raw_b) {
        if (raw_a.empty() || raw_b.empty()) {
            return;
        }
        size_t len_a = std::min<size_t>(raw_a.size(), 32);
        size_t len_b = std::min<size_t>(raw_b.size(), 32);

        std::vector<int> a(len_a);
        std::vector<int> b(len_b);
        for (size_t i = 0; i < len_a; ++i)
            a[i] = (raw_a[i] % 100) - 50;
        for (size_t i = 0; i < len_b; ++i)
            b[i] = (raw_b[i] % 100) - 50;

        auto ab = convolve_fft<int>(a, b);
        auto ba = convolve_fft<int>(b, a);

        RC_ASSERT(ab == ba);
    });
}

TEST(FFTPBT, RealFFTRoundtripInvariant) {
    rc::check("irfft(rfft(x)) == x for all power of 2 sizes", [](size_t p_raw) {
        size_t power = (p_raw % 9) + 1; // 2..512
        size_t N = size_t{1} << power;

        std::vector<float> original(N);
        for (size_t i = 0; i < N; ++i) {
            original[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.125f;
        }

        std::vector<std::complex<float>> freq(N / 2 + 1);
        rfft_1d(original, freq);

        std::vector<float> reconstructed(N);
        irfft_1d(freq, reconstructed);

        for (size_t i = 0; i < N; ++i) {
            RC_ASSERT(std::abs(reconstructed[i] - original[i]) < 1e-4f);
        }
    });
}
