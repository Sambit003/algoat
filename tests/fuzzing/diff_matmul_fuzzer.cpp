#include "algoat/numerics/matmul.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

template <typename T>
double relative_diff(MatrixView<const T> expected, MatrixView<const T> actual) {
    double diff_sq = 0.0;
    double norm_sq = 0.0;
    for (size_t i = 0; i < expected.rows; ++i) {
        for (size_t j = 0; j < expected.cols; ++j) {
            double diff =
                static_cast<double>(actual.at(i, j)) - static_cast<double>(expected.at(i, j));
            diff_sq += diff * diff;
            norm_sq +=
                static_cast<double>(expected.at(i, j)) * static_cast<double>(expected.at(i, j));
        }
    }
    if (norm_sq == 0.0)
        return std::sqrt(diff_sq);
    return std::sqrt(diff_sq) / std::sqrt(norm_sq);
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4)
        return 0;

    size_t M = (data[0] % 20) + 1;
    size_t K = (data[1] % 20) + 1;
    size_t N = (data[2] % 20) + 1;

    size_t total_elements = M * K + K * N;
    if (size < 3 + total_elements)
        return 0;

    std::vector<float> a_data(M * K);
    std::vector<float> b_data(K * N);
    std::vector<float> c_naive(M * N, 0.0f);
    std::vector<float> c_co(M * N, 0.0f);
    std::vector<float> c_par(M * N, 0.0f);

    size_t offset = 3;
    for (size_t i = 0; i < M * K; ++i) {
        a_data[i] = static_cast<float>(static_cast<int>(data[offset++]) - 128) * 0.05f;
    }
    for (size_t i = 0; i < K * N; ++i) {
        b_data[i] = static_cast<float>(static_cast<int>(data[offset++]) - 128) * 0.05f;
    }

    MatrixView<const float> A(a_data.data(), M, K);
    MatrixView<const float> B(b_data.data(), K, N);
    MatrixView<float> C_naive(c_naive.data(), M, N);
    MatrixView<float> C_co(c_co.data(), M, N);
    MatrixView<float> C_par(c_par.data(), M, N);

    naive_gemm<float>(A, B, C_naive);
    matmul_cache_oblivious<float>(A, B, C_co);
    matmul_parallel<float>(A, B, C_par);

    assert(relative_diff<float>(C_naive, C_co) < 1e-4);
    assert(relative_diff<float>(C_naive, C_par) < 1e-4);

    // If square, also test Strassen
    if (M == K && K == N) {
        std::vector<float> c_strassen(M * M, -99.0f);
        MatrixView<float> C_str(c_strassen.data(), M, M);
        matmul_strassen<float>(A, B, C_str, 8);
        assert(relative_diff<float>(C_naive, C_str) < 1e-4);
    }

    return 0;
}
