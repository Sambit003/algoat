#include "algoat/numerics/matmul.hpp"

#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

using namespace algoat::numerics;

namespace {

template <typename T>
void naive_gemm(MatrixView<const T> A, MatrixView<const T> B, MatrixView<T> C) {
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t k = 0; k < A.cols; ++k) {
            T a_ik = A.at(i, k);
            for (size_t j = 0; j < B.cols; ++j) {
                C.at(i, j) += a_ik * B.at(k, j);
            }
        }
    }
}

} // namespace

static void BM_Naive_GEMM(benchmark::State& state) {
    const size_t N = state.range(0);
    std::vector<float> A_buf(N * N, 0.5f);
    std::vector<float> B_buf(N * N, 0.25f);
    std::vector<float> C_buf(N * N, 0.0f);

    MatrixView<const float> A(A_buf.data(), N, N);
    MatrixView<const float> B(B_buf.data(), N, N);
    MatrixView<float> C(C_buf.data(), N, N);

    for (auto _ : state) {
        naive_gemm<float>(A, B, C);
        benchmark::DoNotOptimize(C.data);
    }

    const double flops =
        2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    state.counters["GFLOPS"] = benchmark::Counter(
        flops * static_cast<double>(state.iterations()) / 1e9, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Naive_GEMM)->RangeMultiplier(2)->Range(128, 1024);

static void BM_CacheOblivious_GEMM(benchmark::State& state) {
    const size_t N = state.range(0);
    std::vector<float> A_buf(N * N, 0.5f);
    std::vector<float> B_buf(N * N, 0.25f);
    std::vector<float> C_buf(N * N, 0.0f);

    MatrixView<const float> A(A_buf.data(), N, N);
    MatrixView<const float> B(B_buf.data(), N, N);
    MatrixView<float> C(C_buf.data(), N, N);

    for (auto _ : state) {
        matmul_cache_oblivious<float>(A, B, C);
        benchmark::DoNotOptimize(C.data);
    }

    const double flops =
        2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    state.counters["GFLOPS"] = benchmark::Counter(
        flops * static_cast<double>(state.iterations()) / 1e9, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_CacheOblivious_GEMM)->RangeMultiplier(2)->Range(128, 2048);

static void BM_Strassen_GEMM(benchmark::State& state) {
    const size_t N = state.range(0);
    std::vector<float> A_buf(N * N, 0.5f);
    std::vector<float> B_buf(N * N, 0.25f);
    std::vector<float> C_buf(N * N, 0.0f);

    MatrixView<const float> A(A_buf.data(), N, N);
    MatrixView<const float> B(B_buf.data(), N, N);
    MatrixView<float> C(C_buf.data(), N, N);

    for (auto _ : state) {
        matmul_strassen<float>(A, B, C, 128);
        benchmark::DoNotOptimize(C.data);
    }

    const double flops =
        2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    state.counters["GFLOPS"] = benchmark::Counter(
        flops * static_cast<double>(state.iterations()) / 1e9, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Strassen_GEMM)->RangeMultiplier(2)->Range(128, 2048);

static void BM_Parallel_GEMM(benchmark::State& state) {
    const size_t N = state.range(0);
    std::vector<float> A_buf(N * N, 0.5f);
    std::vector<float> B_buf(N * N, 0.25f);
    std::vector<float> C_buf(N * N, 0.0f);

    MatrixView<const float> A(A_buf.data(), N, N);
    MatrixView<const float> B(B_buf.data(), N, N);
    MatrixView<float> C(C_buf.data(), N, N);

    for (auto _ : state) {
        matmul_parallel<float>(A, B, C);
        benchmark::DoNotOptimize(C.data);
    }

    const double flops =
        2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    state.counters["GFLOPS"] = benchmark::Counter(
        flops * static_cast<double>(state.iterations()) / 1e9, benchmark::Counter::kIsRate);
}
BENCHMARK(BM_Parallel_GEMM)->RangeMultiplier(2)->Range(128, 2048);
