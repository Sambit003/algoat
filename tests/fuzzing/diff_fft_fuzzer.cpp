#include "algoat/numerics/fft.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

using namespace algoat::numerics;

namespace {

template <typename T>
std::vector<std::complex<T>> naive_dft(const std::vector<std::complex<T>>& x) {
    const size_t N = x.size();
    std::vector<std::complex<T>> X(N, {0, 0});
    constexpr double pi2 = 2.0 * std::numbers::pi_v<double>;

    for (size_t k = 0; k < N; ++k) {
        for (size_t n = 0; n < N; ++n) {
            double angle = -pi2 * static_cast<double>(k * n) / static_cast<double>(N);
            std::complex<T> twiddle{static_cast<T>(std::cos(angle)),
                                    static_cast<T>(std::sin(angle))};
            X[k] += x[n] * twiddle;
        }
    }
    return X;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) {
        return 0;
    }

    // Power of 2 between 2^1 (2) and 2^7 (128)
    const size_t power = (data[0] % 7) + 1;
    const size_t N = size_t{1} << power;

    if (size < 1 + 2 * N) {
        return 0;
    }

    std::vector<std::complex<float>> input(N);
    for (size_t i = 0; i < N; ++i) {
        float r = (static_cast<float>(data[1 + 2 * i]) - 128.0f) / 64.0f;
        float im = (static_cast<float>(data[1 + 2 * i + 1]) - 128.0f) / 64.0f;
        input[i] = {r, im};
    }

    auto naive_result = naive_dft(input);

    auto fft_result = input;
    fft_1d(fft_result, FFTDirection::Forward);

    for (size_t k = 0; k < N; ++k) {
        float diff_r = std::abs(fft_result[k].real() - naive_result[k].real());
        float diff_i = std::abs(fft_result[k].imag() - naive_result[k].imag());
        float tol = 1e-3f * (std::abs(naive_result[k]) + 1.0f);
        assert(diff_r <= tol);
        assert(diff_i <= tol);
    }

    // Roundtrip verification
    ifft_1d(fft_result);
    for (size_t i = 0; i < N; ++i) {
        float diff_r = std::abs(fft_result[i].real() - input[i].real());
        float diff_i = std::abs(fft_result[i].imag() - input[i].imag());
        assert(diff_r <= 1e-4f);
        assert(diff_i <= 1e-4f);
    }

    // Real FFT fuzzing
    std::vector<float> real_input(N);
    for (size_t i = 0; i < N; ++i) {
        real_input[i] = input[i].real();
    }
    std::vector<std::complex<float>> rfft_out(N / 2 + 1);
    rfft_1d(real_input, rfft_out);

    std::vector<float> irfft_out(N);
    irfft_1d(rfft_out, irfft_out);

    for (size_t i = 0; i < N; ++i) {
        assert(std::abs(irfft_out[i] - real_input[i]) <= 1e-4f);
    }

    return 0;
}
