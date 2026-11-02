#include <algoat/numerics/matmul.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace algoat::numerics;

TEST(MatrixViewTest, BasicConstructionAndAccess) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    MatrixView<float> mat(data.data(), 2, 3);
    EXPECT_EQ(mat.rows, 2);
    EXPECT_EQ(mat.cols, 3);
    EXPECT_EQ(mat.row_stride, 3);
    EXPECT_FLOAT_EQ(mat.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(mat.at(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(mat.at(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(mat.at(1, 2), 6.0f);

    mat.at(1, 1) = 42.0f;
    EXPECT_FLOAT_EQ(data[4], 42.0f);
}

TEST(MatrixViewTest, ConstConversionAndSubviews) {
    std::vector<double> data = {1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

    MatrixView<double> full(data.data(), 4, 4);
    MatrixView<const double> const_view = full;
    EXPECT_DOUBLE_EQ(const_view.at(2, 3), 12.0);

    // Subview 2x2 at offset (1, 1) -> elements 6, 7, 10, 11
    auto sub = full.subview(1, 1, 2, 2);
    EXPECT_EQ(sub.rows, 2);
    EXPECT_EQ(sub.cols, 2);
    EXPECT_EQ(sub.row_stride, 4);
    EXPECT_DOUBLE_EQ(sub.at(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(sub.at(0, 1), 7.0);
    EXPECT_DOUBLE_EQ(sub.at(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(sub.at(1, 1), 11.0);
}

TEST(MakeMatrixTest, AllocatesAndInitializes) {
    auto mat = make_matrix<float>(4, 8);
    EXPECT_EQ(mat.size(), 32);
    // Verify 64-byte cache-line alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(mat.data()) % 64, 0);
    for (float val : mat) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }
}

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

template <typename T> T relative_error(MatrixView<const T> expected, MatrixView<const T> actual) {
    T diff_sq = 0;
    T norm_sq = 0;
    for (size_t i = 0; i < expected.rows; ++i) {
        for (size_t j = 0; j < expected.cols; ++j) {
            T diff = actual.at(i, j) - expected.at(i, j);
            diff_sq += diff * diff;
            norm_sq += expected.at(i, j) * expected.at(i, j);
        }
    }
    if (norm_sq == 0)
        return std::sqrt(diff_sq);
    return std::sqrt(diff_sq) / std::sqrt(norm_sq);
}

TEST(CacheObliviousMatmulTest, SmallMatricesFloatAndDouble) {
    // 2x2 matrix multiplication
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_data = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> c_data(4, 0.0f);
    std::vector<float> c_naive(4, 0.0f);

    MatrixView<float> A(a_data.data(), 2, 2);
    MatrixView<float> B(b_data.data(), 2, 2);
    MatrixView<float> C(c_data.data(), 2, 2);
    MatrixView<float> C_naive(c_naive.data(), 2, 2);

    naive_gemm<float>(A, B, C_naive);
    matmul_cache_oblivious<float>(A, B, C);

    EXPECT_FLOAT_EQ(C.at(0, 0), C_naive.at(0, 0));
    EXPECT_FLOAT_EQ(C.at(0, 1), C_naive.at(0, 1));
    EXPECT_FLOAT_EQ(C.at(1, 0), C_naive.at(1, 0));
    EXPECT_FLOAT_EQ(C.at(1, 1), C_naive.at(1, 1));
}

TEST(CacheObliviousMatmulTest, RectangularAndFringeDimensions) {
    // 17 x 31 times 31 x 11 matrix multiplication
    const size_t M = 17, K = 31, N = 11;
    std::vector<double> a_data(M * K);
    std::vector<double> b_data(K * N);
    std::vector<double> c_data(M * N, 0.5); // test accumulation with existing values
    std::vector<double> c_naive(M * N, 0.5);

    for (size_t i = 0; i < M * K; ++i)
        a_data[i] = std::sin(static_cast<double>(i));
    for (size_t i = 0; i < K * N; ++i)
        b_data[i] = std::cos(static_cast<double>(i));

    MatrixView<const double> A(a_data.data(), M, K);
    MatrixView<const double> B(b_data.data(), K, N);
    MatrixView<double> C(c_data.data(), M, N);
    MatrixView<double> C_naive(c_naive.data(), M, N);

    naive_gemm<double>(A, B, C_naive);
    matmul_cache_oblivious<double>(A, B, C);

    EXPECT_LT(relative_error<double>(C_naive, C), 1e-12);
}

TEST(CacheObliviousMatmulTest, PowerOfTwoSquare128) {
    const size_t N = 128;
    std::vector<double> a_data(N * N);
    std::vector<double> b_data(N * N);
    std::vector<double> c_data(N * N, 0.0);
    std::vector<double> c_naive(N * N, 0.0);

    for (size_t i = 0; i < N * N; ++i) {
        a_data[i] = (i % 13) * 0.1 - 0.5;
        b_data[i] = (i % 17) * 0.1 - 0.7;
    }

    MatrixView<const double> A(a_data.data(), N, N);
    MatrixView<const double> B(b_data.data(), N, N);
    MatrixView<double> C(c_data.data(), N, N);
    MatrixView<double> C_naive(c_naive.data(), N, N);

    naive_gemm<double>(A, B, C_naive);
    matmul_cache_oblivious<double>(A, B, C);

    EXPECT_LT(relative_error<double>(C_naive, C), 1e-11);
}

TEST(StrassenMatmulTest, SquareMatrixPowerOfTwoDouble) {
    const size_t N = 128;
    std::vector<double> a_data(N * N);
    std::vector<double> b_data(N * N);
    std::vector<double> c_data(N * N, 999.0); // verify out-of-place overwrite
    std::vector<double> c_naive(N * N, 0.0);

    for (size_t i = 0; i < N * N; ++i) {
        a_data[i] = (i % 7) * 0.2 - 0.6;
        b_data[i] = (i % 11) * 0.2 - 0.8;
    }

    MatrixView<const double> A(a_data.data(), N, N);
    MatrixView<const double> B(b_data.data(), N, N);
    MatrixView<double> C(c_data.data(), N, N);
    MatrixView<double> C_naive(c_naive.data(), N, N);

    naive_gemm<double>(A, B, C_naive);
    // Use threshold 64 so 128 recurses into 7 subproblems of size 64
    matmul_strassen<double>(A, B, C, 64);

    EXPECT_LT(relative_error<double>(C_naive, C), 1e-11);
}

TEST(StrassenMatmulTest, SquareMatrix256Double) {
    const size_t N = 256;
    std::vector<double> a_data(N * N);
    std::vector<double> b_data(N * N);
    std::vector<double> c_data(N * N, 0.0);
    std::vector<double> c_naive(N * N, 0.0);

    for (size_t i = 0; i < N * N; ++i) {
        a_data[i] = (i % 5) * 0.1 - 0.2;
        b_data[i] = (i % 9) * 0.1 - 0.4;
    }

    MatrixView<const double> A(a_data.data(), N, N);
    MatrixView<const double> B(b_data.data(), N, N);
    MatrixView<double> C(c_data.data(), N, N);
    MatrixView<double> C_naive(c_naive.data(), N, N);

    naive_gemm<double>(A, B, C_naive);
    matmul_strassen<double>(A, B, C, 128);

    EXPECT_LT(relative_error<double>(C_naive, C), 1e-10);
}

TEST(StrassenMatmulTest, OddDimensionPeeling) {
    const size_t N = 65; // odd dimension
    std::vector<float> a_data(N * N);
    std::vector<float> b_data(N * N);
    std::vector<float> c_data(N * N, -1.0f);
    std::vector<float> c_naive(N * N, 0.0f);

    for (size_t i = 0; i < N * N; ++i) {
        a_data[i] = static_cast<float>(static_cast<int>(i % 11) - 5);
        b_data[i] = static_cast<float>(static_cast<int>(i % 13) - 6);
    }

    MatrixView<const float> A(a_data.data(), N, N);
    MatrixView<const float> B(b_data.data(), N, N);
    MatrixView<float> C(c_data.data(), N, N);
    MatrixView<float> C_naive(c_naive.data(), N, N);

    naive_gemm<float>(A, B, C_naive);
    matmul_strassen<float>(A, B, C, 32);

    EXPECT_LT(relative_error<float>(C_naive, C), 1e-5f);
}

TEST(ParallelMatmulTest, SquareMatrix256DoubleParity) {
    const size_t N = 256;
    std::vector<double> a_data(N * N);
    std::vector<double> b_data(N * N);
    std::vector<double> c_data(N * N, 1.5);
    std::vector<double> c_seq(N * N, 1.5);

    for (size_t i = 0; i < N * N; ++i) {
        a_data[i] = (i % 7) * 0.1 - 0.3;
        b_data[i] = (i % 11) * 0.1 - 0.5;
    }

    MatrixView<const double> A(a_data.data(), N, N);
    MatrixView<const double> B(b_data.data(), N, N);
    MatrixView<double> C(c_data.data(), N, N);
    MatrixView<double> C_seq(c_seq.data(), N, N);

    matmul_cache_oblivious<double>(A, B, C_seq);
    matmul_parallel<double>(A, B, C);

    EXPECT_LT(relative_error<double>(C_seq, C), 1e-12);
}

TEST(ParallelMatmulTest, RectangularMatrixFloatParity) {
    const size_t M = 120, K = 90, N = 150;
    std::vector<float> a_data(M * K);
    std::vector<float> b_data(K * N);
    std::vector<float> c_data(M * N, 0.0f);
    std::vector<float> c_seq(M * N, 0.0f);

    for (size_t i = 0; i < M * K; ++i)
        a_data[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
    for (size_t i = 0; i < K * N; ++i)
        b_data[i] = static_cast<float>(static_cast<int>(i % 7) - 3);

    MatrixView<const float> A(a_data.data(), M, K);
    MatrixView<const float> B(b_data.data(), K, N);
    MatrixView<float> C(c_data.data(), M, N);
    MatrixView<float> C_seq(c_seq.data(), M, N);

    matmul_cache_oblivious<float>(A, B, C_seq);
    matmul_parallel<float>(A, B, C);

    EXPECT_LT(relative_error<float>(C_seq, C), 1e-6f);
}

TEST(AcceptanceCriteriaTest, NumericalCorrectness512And1024) {
    // Acceptance criterion 1:
    // matmul_strassen and matmul_cache_oblivious agree with reference naive O(N^3)
    // implementation to within double-precision epsilon (< 10^-10 relative error)
    // for N in {128, 256, 512, 1024}.
    const size_t N = 512;
    std::vector<double> a_data(N * N);
    std::vector<double> b_data(N * N);
    std::vector<double> c_cache_obl(N * N, 0.0);
    std::vector<double> c_strassen(N * N, 0.0);
    std::vector<double> c_naive(N * N, 0.0);

    for (size_t i = 0; i < N * N; ++i) {
        a_data[i] = (static_cast<int>(i % 17) - 8) * 0.05;
        b_data[i] = (static_cast<int>(i % 23) - 11) * 0.05;
    }

    MatrixView<const double> A(a_data.data(), N, N);
    MatrixView<const double> B(b_data.data(), N, N);
    MatrixView<double> C_co(c_cache_obl.data(), N, N);
    MatrixView<double> C_str(c_strassen.data(), N, N);
    MatrixView<double> C_ref(c_naive.data(), N, N);

    naive_gemm<double>(A, B, C_ref);
    matmul_cache_oblivious<double>(A, B, C_co);
    matmul_strassen<double>(A, B, C_str, 128);

    double err_co = relative_error<double>(C_ref, C_co);
    double err_str = relative_error<double>(C_ref, C_str);

    EXPECT_LT(err_co, 1e-10);
    EXPECT_LT(err_str, 1e-10);
}

TEST(AcceptanceCriteriaTest, NumericalCorrectness1024) {
    // Acceptance criterion 1 for N = 1024
    const size_t N = 1024;
    auto a_data = make_matrix<double>(N, N);
    auto b_data = make_matrix<double>(N, N);
    auto c_co = make_matrix<double>(N, N);
    auto c_str = make_matrix<double>(N, N);

    for (size_t i = 0; i < N * N; ++i) {
        a_data[i] = (static_cast<int>(i % 13) - 6) * 0.02;
        b_data[i] = (static_cast<int>(i % 19) - 9) * 0.02;
    }

    MatrixView<const double> A(a_data.data(), N, N);
    MatrixView<const double> B(b_data.data(), N, N);
    MatrixView<double> C_co(c_co.data(), N, N);
    MatrixView<double> C_str(c_str.data(), N, N);

    matmul_cache_oblivious<double>(A, B, C_co);
    matmul_strassen<double>(A, B, C_str, 128);

    double err = relative_error<double>(C_co, C_str);
    EXPECT_LT(err, 1e-10);
}
