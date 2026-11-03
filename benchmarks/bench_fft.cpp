#include "algoat/numerics/fft.hpp"

#include <benchmark/benchmark.h>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

using namespace algoat::numerics;

namespace {

static void BM_FFT_1D_Float_SIMD(benchmark::State& state) {
    const size_t N = state.range(0);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::complex<float>> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = {dist(rng), dist(rng)};
    }
    precompute_twiddles(N);

    std::vector<std::complex<float>> buf = data;

    for (auto _ : state) {
        state.PauseTiming();
        buf = data;
        state.ResumeTiming();

        fft_1d(buf, FFTDirection::Forward);
        benchmark::DoNotOptimize(buf.data());
    }

    // 5 * N * log2(N) floating-point operations
    const double log2_n = std::log2(static_cast<double>(N));
    const double flops = 5.0 * static_cast<double>(N) * log2_n;
    state.counters["MFLOPS"] = benchmark::Counter(
        flops * static_cast<double>(state.iterations()) / 1e6, benchmark::Counter::kIsRate);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(N * sizeof(std::complex<float>)));
}
BENCHMARK(BM_FFT_1D_Float_SIMD)->RangeMultiplier(4)->Range(1024, 1 << 20);

static void BM_FFT_1D_Float_Scalar(benchmark::State& state) {
    const size_t N = state.range(0);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::complex<float>> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = {dist(rng), dist(rng)};
    }
    precompute_twiddles(N);
    auto& tw_table = detail::TwiddleTable::instance();

    std::vector<std::complex<float>> buf = data;

    for (auto _ : state) {
        state.PauseTiming();
        buf = data;
        state.ResumeTiming();

        // Run scalar DIT stages explicitly
        detail::bit_reversal_permute(std::span<std::complex<float>>(buf));
        for (size_t i = 0; i < N; i += 2) {
            auto u = buf[i];
            auto v = buf[i + 1];
            buf[i] = u + v;
            buf[i + 1] = u - v;
        }
        int stage = 2;
        for (size_t L = 4; L <= N; L <<= 1, ++stage) {
            const auto* stage_tw = tw_table.stage_float(stage).data();
            detail::butterfly_stage_scalar(std::span<std::complex<float>>(buf), L, L >> 1,
                                           FFTDirection::Forward, stage_tw);
        }
        benchmark::DoNotOptimize(buf.data());
    }

    const double log2_n = std::log2(static_cast<double>(N));
    const double flops = 5.0 * static_cast<double>(N) * log2_n;
    state.counters["MFLOPS"] = benchmark::Counter(
        flops * static_cast<double>(state.iterations()) / 1e6, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_FFT_1D_Float_Scalar)->RangeMultiplier(4)->Range(1024, 1 << 20);

static void BM_RealFFT_1D_Float(benchmark::State& state) {
    const size_t N = state.range(0);
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> real_data(N);
    for (size_t i = 0; i < N; ++i) {
        real_data[i] = dist(rng);
    }
    std::vector<std::complex<float>> out(N / 2 + 1);

    for (auto _ : state) {
        rfft_1d(real_data, out);
        benchmark::DoNotOptimize(out.data());
    }

    const double log2_n = std::log2(static_cast<double>(N));
    const double flops = 2.5 * static_cast<double>(N) * log2_n;
    state.counters["MFLOPS"] = benchmark::Counter(
        flops * static_cast<double>(state.iterations()) / 1e6, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_RealFFT_1D_Float)->RangeMultiplier(4)->Range(1024, 1 << 20);

static void BM_FFT_2D_Float(benchmark::State& state) {
    const size_t N = state.range(0); // N x N
    std::mt19937_64 rng(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::complex<float>> mat(N * N);
    for (auto& v : mat) {
        v = {dist(rng), dist(rng)};
    }
    std::vector<std::complex<float>> buf = mat;

    for (auto _ : state) {
        state.PauseTiming();
        buf = mat;
        state.ResumeTiming();

        fft_2d(buf, N, N, FFTDirection::Forward);
        benchmark::DoNotOptimize(buf.data());
    }
}
BENCHMARK(BM_FFT_2D_Float)->RangeMultiplier(2)->Range(64, 512);

} // namespace
