#include "algoat/numerics/matmul.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <vector>

using namespace algoat::numerics;

template <typename T>
static void naive_gemm(MatrixView<const T> A, MatrixView<const T> B, MatrixView<T> C) {
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t k = 0; k < A.cols; ++k) {
            T a_ik = A.at(i, k);
            for (size_t j = 0; j < B.cols; ++j) {
                C.at(i, j) += a_ik * B.at(k, j);
            }
        }
    }
}

template <typename T> static double max_abs_diff(MatrixView<const T> A, MatrixView<const T> B) {
    double max_d = 0.0;
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < A.cols; ++j) {
            double d = std::abs(static_cast<double>(A.at(i, j)) - static_cast<double>(B.at(i, j)));
            if (d > max_d)
                max_d = d;
        }
    }
    return max_d;
}

TEST(MatmulPBT, IdentityMatrixInvariant) {
    rc::check("A * I == A and I * A == A", [](size_t n_raw) {
        size_t n = (n_raw % 32) + 1;

        std::vector<double> a_data(n * n);
        std::vector<double> i_data(n * n, 0.0);
        for (size_t k = 0; k < n; ++k) {
            i_data[k * n + k] = 1.0;
        }
        for (size_t k = 0; k < n * n; ++k) {
            a_data[k] = (static_cast<int>(k % 19) - 9) * 0.25;
        }

        std::vector<double> c_data(n * n, 0.0);
        MatrixView<const double> A(a_data.data(), n, n);
        MatrixView<const double> I(i_data.data(), n, n);
        MatrixView<double> C(c_data.data(), n, n);

        matmul_cache_oblivious<double>(A, I, C);
        RC_ASSERT(max_abs_diff<double>(A, C) < 1e-12);

        std::vector<double> c2_data(n * n, 0.0);
        MatrixView<double> C2(c2_data.data(), n, n);
        matmul_cache_oblivious<double>(I, A, C2);
        RC_ASSERT(max_abs_diff<double>(A, C2) < 1e-12);
    });
}

TEST(MatmulPBT, ZeroMatrixInvariant) {
    rc::check("A * 0 == 0", [](size_t m_raw, size_t k_raw, size_t n_raw) {
        size_t m = (m_raw % 20) + 1;
        size_t k = (k_raw % 20) + 1;
        size_t n = (n_raw % 20) + 1;

        std::vector<double> a_data(m * k);
        for (size_t idx = 0; idx < m * k; ++idx) {
            a_data[idx] = (static_cast<int>(idx % 7) - 3) * 0.5;
        }
        std::vector<double> b_zero(k * n, 0.0);
        std::vector<double> c_data(m * n, 0.0);

        MatrixView<const double> A(a_data.data(), m, k);
        MatrixView<const double> B(b_zero.data(), k, n);
        MatrixView<double> C(c_data.data(), m, n);

        matmul_cache_oblivious<double>(A, B, C);
        for (size_t i = 0; i < m * n; ++i) {
            RC_ASSERT(c_data[i] == 0.0);
        }
    });
}

TEST(MatmulPBT, DistributiveLaw) {
    rc::check("A * (B + C) == A * B + A * C", [](size_t n_raw) {
        size_t n = (n_raw % 24) + 1;

        std::vector<double> a_data(n * n);
        std::vector<double> b_data(n * n);
        std::vector<double> c_data(n * n);
        std::vector<double> b_plus_c(n * n);

        for (size_t idx = 0; idx < n * n; ++idx) {
            a_data[idx] = (static_cast<int>(idx % 5) - 2) * 0.2;
            b_data[idx] = (static_cast<int>(idx % 7) - 3) * 0.2;
            c_data[idx] = (static_cast<int>(idx % 11) - 5) * 0.2;
            b_plus_c[idx] = b_data[idx] + c_data[idx];
        }

        MatrixView<const double> A(a_data.data(), n, n);
        MatrixView<const double> B(b_data.data(), n, n);
        MatrixView<const double> C_in(c_data.data(), n, n);
        MatrixView<const double> BC(b_plus_c.data(), n, n);

        std::vector<double> left_data(n * n, 0.0);
        MatrixView<double> Left(left_data.data(), n, n);
        matmul_cache_oblivious<double>(A, BC, Left);

        std::vector<double> right_data(n * n, 0.0);
        MatrixView<double> Right(right_data.data(), n, n);
        matmul_cache_oblivious<double>(A, B, Right);
        matmul_cache_oblivious<double>(A, C_in, Right);

        RC_ASSERT(max_abs_diff<double>(Left, Right) < 1e-10);
    });
}

TEST(MatmulPBT, CacheObliviousMatchesNaiveArbitraryDimensions) {
    rc::check("CacheOblivious matches naive GEMM", [](size_t m_raw, size_t k_raw, size_t n_raw) {
        size_t m = (m_raw % 35) + 1;
        size_t k = (k_raw % 35) + 1;
        size_t n = (n_raw % 35) + 1;

        std::vector<double> a_data(m * k);
        std::vector<double> b_data(k * n);
        for (size_t idx = 0; idx < m * k; ++idx) {
            a_data[idx] = (static_cast<int>(idx % 13) - 6) * 0.1;
        }
        for (size_t idx = 0; idx < k * n; ++idx) {
            b_data[idx] = (static_cast<int>(idx % 17) - 8) * 0.1;
        }

        std::vector<double> c_actual(m * n, 0.0);
        std::vector<double> c_expected(m * n, 0.0);

        MatrixView<const double> A(a_data.data(), m, k);
        MatrixView<const double> B(b_data.data(), k, n);
        MatrixView<double> C_act(c_actual.data(), m, n);
        MatrixView<double> C_exp(c_expected.data(), m, n);

        naive_gemm<double>(A, B, C_exp);
        matmul_cache_oblivious<double>(A, B, C_act);

        RC_ASSERT(max_abs_diff<double>(C_exp, C_act) < 1e-11);
    });
}

TEST(MatmulPBT, StrassenMatchesCacheObliviousArbitraryN) {
    rc::check("Strassen matches CacheOblivious for arbitrary square N", [](size_t n_raw) {
        size_t n = (n_raw % 40) + 1;

        std::vector<double> a_data(n * n);
        std::vector<double> b_data(n * n);
        for (size_t idx = 0; idx < n * n; ++idx) {
            a_data[idx] = (static_cast<int>(idx % 7) - 3) * 0.1;
            b_data[idx] = (static_cast<int>(idx % 11) - 5) * 0.1;
        }

        std::vector<double> c_strassen(n * n, -99.0);
        std::vector<double> c_co(n * n, 0.0);

        MatrixView<const double> A(a_data.data(), n, n);
        MatrixView<const double> B(b_data.data(), n, n);
        MatrixView<double> C_str(c_strassen.data(), n, n);
        MatrixView<double> C_co_view(c_co.data(), n, n);

        matmul_cache_oblivious<double>(A, B, C_co_view);
        matmul_strassen<double>(A, B, C_str, 16); // force small threshold to trigger recursion

        RC_ASSERT(max_abs_diff<double>(C_co_view, C_str) < 1e-10);
    });
}
