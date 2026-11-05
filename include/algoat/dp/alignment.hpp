/**
 * @file alignment.hpp
 * @brief Needleman-Wunsch & Smith-Waterman Sequence Alignment with Hirschberg Linear Space
 * and Farrar Striped SIMD Vectorization.
 *
 * Implements high-performance global and local sequence alignment:
 * - Needleman-Wunsch (1970) global alignment with Gotoh (1982) affine gap penalties.
 * - Hirschberg (1975) divide-and-conquer linear-space O(M + N) global alignment (Myers-Miller
 * 1988).
 * - Smith-Waterman (1981) local alignment with traceback and CIGAR string generation.
 * - Farrar (2007) Striped SIMD (AVX2 / SSE2) local alignment score engine.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#if defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#endif
#endif
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace algoat::dp {

namespace detail {

// Standard BLOSUM62 score matrix (Henikoff & Henikoff 1992)
inline constexpr std::string_view kBlosum62Alphabet = "ARNDCQEGHILKMFPSTWYVBZX*";
inline constexpr int8_t kBlosum62Scores[24][24] = {
    {4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4}, // A
    {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4}, // R
    {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4},      // N
    {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4}, // D
    {0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3,
     -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},                                       // C
    {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4},  // Q
    {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4}, // E
    {0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2,
     -3, -3, -2, 0,  -2, -2, -3, -3, -1, -2, -1, -4},                                         // G
    {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4}, // H
    {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3,
     1,  0,  -3, -2, -1, -3, -1, 3,  -3, -3, -1, -4}, // I
    {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2,
     2,  0,  -3, -2, -1, -2, -1, 1,  -4, -3, -1, -4},                                           // L
    {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4},   // K
    {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4}, // M
    {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4}, // F
    {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1,
     -2, -4, 7,  -1, -1, -4, -3, -2, -2, -1, -2, -4},                                           // P
    {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4},       // S
    {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4}, // T
    {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3,
     -1, 1,  -4, -3, -2, 11, 2,  -3, -4, -3, -2, -4}, // W
    {-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2,
     -1, 3,  -3, -2, -2, 2,  7,  -1, -3, -2, -1, -4},                                           // Y
    {0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4}, // V
    {-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4},    // B
    {-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},     // Z
    {0,  -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -2, 0,  0,  -2, -1, -1, -1, -1, -1, -4}, // X
    {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
     -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}, // *
};

// Standard PAM250 score matrix (Dayhoff 1978)
inline constexpr std::string_view kPam250Alphabet = "ARNDCQEGHILKMFPSTWYVBZX*";
inline constexpr int8_t kPam250Scores[24][24] = {
    {2, -2, 0, 0, -2, 0, 0, 1, -1, -1, -2, -1, -1, -3, 1, 1, 1, -6, -3, 0, 0, 0, 0, -8},    // A
    {-2, 6, 0, -1, -4, 1, -1, -3, 2, -2, -3, 3, 0, -4, 0, 0, -1, 2, -4, -2, -1, 0, -1, -8}, // R
    {0, 0, 2, 2, -4, 1, 1, 0, 2, -2, -3, 1, -2, -3, 0, 1, 0, -4, -2, -2, 2, 1, 0, -8},      // N
    {0, -1, 2, 4, -5, 2, 3, 1, 1, -2, -4, 0, -3, -6, -1, 0, 0, -7, -4, -2, 3, 3, -1, -8},   // D
    {-2, -4, -4, -5, 12, -5, -5, -3, -3, -2, -6, -5,
     -5, -4, -3, 0,  -2, -8, 0,  -2, -4, -5, -3, -8},                                           // C
    {0, 1, 1, 2, -5, 4, 2, -1, 3, -2, -2, 1, -1, -5, 0, -1, -1, -5, -4, -2, 1, 3, -1, -8},      // Q
    {0, -1, 1, 3, -5, 2, 4, 0, 1, -2, -3, 0, -2, -5, -1, 0, 0, -7, -4, -2, 3, 3, -1, -8},       // E
    {1, -3, 0, 1, -3, -1, 0, 5, -2, -3, -4, -2, -3, -5, 0, 1, 0, -7, -5, -1, 0, 0, -1, -8},     // G
    {-1, 2, 2, 1, -3, 3, 1, -2, 6, -2, -2, 0, -2, -2, 0, -1, -1, -3, 0, -2, 1, 2, -1, -8},      // H
    {-1, -2, -2, -2, -2, -2, -2, -3, -2, 5, 2, -2, 2, 1, -2, -1, 0, -5, -1, 4, -2, -2, -1, -8}, // I
    {-2, -3, -3, -4, -6, -2, -3, -4, -2, 2,  6,  -3,
     4,  2,  -3, -3, -2, -2, -1, 2,  -3, -3, -1, -8},                                           // L
    {-1, 3, 1, 0, -5, 1, 0, -2, 0, -2, -3, 5, 0, -5, -1, 0, 0, -3, -4, -2, 1, 0, -1, -8},       // K
    {-1, 0, -2, -3, -5, -1, -2, -3, -2, 2, 4, 0, 6, 0, -2, -2, -1, -4, -2, 2, -2, -2, -1, -8},  // M
    {-3, -4, -3, -6, -4, -5, -5, -5, -2, 1, 2, -5, 0, 9, -5, -3, -3, 0, 7, -1, -4, -5, -2, -8}, // F
    {1, 0, 0, -1, -3, 0, -1, 0, 0, -2, -3, -1, -2, -5, 6, 1, 0, -6, -5, -1, -1, 0, -1, -8},     // P
    {1, 0, 1, 0, 0, -1, 0, 1, -1, -1, -3, 0, -2, -3, 1, 2, 1, -2, -3, -1, 0, 0, 0, -8},         // S
    {1, -1, 0, 0, -2, -1, 0, 0, -1, 0, -2, 0, -1, -3, 0, 1, 3, -5, -3, 0, 0, -1, 0, -8},        // T
    {-6, 2, -4, -7, -8, -5, -7, -7, -3, -5, -2, -3,
     -4, 0, -6, -2, -5, 17, 0,  -6, -5, -6, -4, -8}, // W
    {-3, -4, -2, -4, 0,  -4, -4, -5, 0,  -1, -1, -4,
     -2, 7,  -5, -3, -3, 0,  10, -2, -3, -4, -2, -8},                                           // Y
    {0, -2, -2, -2, -2, -2, -2, -1, -2, 4, 2, -2, 2, -1, -1, -1, 0, -6, -2, 4, -2, -2, -1, -8}, // V
    {0, -1, 2, 3, -4, 1, 3, 0, 1, -2, -3, 1, -2, -4, -1, 0, 0, -5, -3, -2, 3, 2, -1, -8},       // B
    {0, 0, 1, 3, -5, 3, 3, 0, 2, -2, -3, 0, -2, -5, 0, 0, -1, -6, -4, -2, 2, 3, -1, -8},        // Z
    {0,  -1, 0,  -1, -3, -1, -1, -1, -1, -1, -1, -1,
     -1, -2, -1, 0,  0,  -4, -2, -1, -1, -1, -1, -8}, // X
    {-8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8,
     -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, 1}, // *
};

inline void populate_matrix(std::array<std::array<int32_t, 256>, 256>& matrix,
                            std::string_view alphabet, const int8_t scores[24][24],
                            int32_t default_val) noexcept {
    for (size_t i = 0; i < 256; ++i) {
        for (size_t j = 0; j < 256; ++j) {
            matrix[i][j] = default_val;
        }
    }
    const size_t n = alphabet.size();
    for (size_t i = 0; i < n; ++i) {
        const auto c1_up = static_cast<uint8_t>(alphabet[i]);
        const auto c1_lo = static_cast<uint8_t>(std::tolower(alphabet[i]));
        for (size_t j = 0; j < n; ++j) {
            const auto c2_up = static_cast<uint8_t>(alphabet[j]);
            const auto c2_lo = static_cast<uint8_t>(std::tolower(alphabet[j]));
            const int32_t score = scores[i][j];

            matrix[c1_up][c2_up] = score;
            matrix[c1_up][c2_lo] = score;
            matrix[c1_lo][c2_up] = score;
            matrix[c1_lo][c2_lo] = score;
        }
    }
}

inline constexpr int32_t kNegInf = -1'000'000'000;

inline std::string compress_cigar(std::span<const char> ops) {
    if (ops.empty()) {
        return "";
    }
    std::string res;
    char current_op = ops[0];
    size_t count = 0;
    for (char op : ops) {
        if (op == current_op) {
            ++count;
        } else {
            res += std::to_string(count);
            res.push_back(current_op);
            current_op = op;
            count = 1;
        }
    }
    res += std::to_string(count);
    res.push_back(current_op);
    return res;
}

} // namespace detail

/// \brief Substitution/scoring matrix for biological sequence alignment.
struct ScoringMatrix {
    int32_t match{2};
    int32_t mismatch{-1};
    std::optional<std::array<std::array<int8_t, 128>, 128>> custom_matrix{std::nullopt};
    std::array<std::array<int32_t, 256>, 256> matrix{};

    constexpr ScoringMatrix() noexcept : ScoringMatrix(2, -1) {}

    constexpr ScoringMatrix(
        int32_t m, int32_t mis,
        std::optional<std::array<std::array<int8_t, 128>, 128>> custom = std::nullopt) noexcept
        : match(m), mismatch(mis), custom_matrix(custom) {
        if (custom_matrix.has_value()) {
            for (size_t i = 0; i < 256; ++i) {
                for (size_t j = 0; j < 256; ++j) {
                    if (i < 128 && j < 128) {
                        matrix[i][j] = (*custom_matrix)[i][j];
                    } else {
                        matrix[i][j] = (i == j && i != 0) ? match : mismatch;
                    }
                }
            }
        } else {
            for (size_t i = 0; i < 256; ++i) {
                for (size_t j = 0; j < 256; ++j) {
                    if (i == j && i != 0) {
                        matrix[i][j] = match;
                    } else {
                        matrix[i][j] = mismatch;
                    }
                }
            }
            for (char c = 'A'; c <= 'Z'; ++c) {
                const auto up = static_cast<uint8_t>(c);
                const auto lo = static_cast<uint8_t>(std::tolower(c));
                matrix[up][lo] = match;
                matrix[lo][up] = match;
                matrix[lo][lo] = match;
            }
        }
    }

    [[nodiscard]] static ScoringMatrix blosum62() noexcept {
        ScoringMatrix sm(1, -4);
        detail::populate_matrix(sm.matrix, detail::kBlosum62Alphabet, detail::kBlosum62Scores, -4);
        return sm;
    }

    [[nodiscard]] static ScoringMatrix pam250() noexcept {
        ScoringMatrix sm(1, -8);
        detail::populate_matrix(sm.matrix, detail::kPam250Alphabet, detail::kPam250Scores, -8);
        return sm;
    }

    [[nodiscard]] static ScoringMatrix dna(int32_t match_score = 2,
                                           int32_t mismatch_score = -3) noexcept {
        ScoringMatrix sm(match_score, mismatch_score);
        for (size_t i = 0; i < 256; ++i) {
            for (size_t j = 0; j < 256; ++j) {
                sm.matrix[i][j] = mismatch_score;
            }
        }
        constexpr std::string_view bases = "ACGTUN";
        for (char b : bases) {
            const auto up = static_cast<uint8_t>(b);
            const auto lo = static_cast<uint8_t>(std::tolower(b));
            sm.matrix[up][up] = match_score;
            sm.matrix[up][lo] = match_score;
            sm.matrix[lo][up] = match_score;
            sm.matrix[lo][lo] = match_score;
        }
        return sm;
    }

    [[nodiscard]] constexpr int32_t score(char a, char b) const noexcept {
        return matrix[static_cast<uint8_t>(a)][static_cast<uint8_t>(b)];
    }
};

/// \brief Affine gap penalty parameters: Cost(k) = -(open + (k - 1) * extend).
struct GapPenalty {
    int32_t open{-5};
    int32_t extend{-1};
};

/// \brief Alignment outcome containing score, aligned strings, SAM CIGAR, and coordinates.
struct AlignmentResult {
    int64_t score{0};
    std::string cigar;
    size_t query_start{0};
    size_t query_end{0};
    size_t ref_start{0};
    size_t ref_end{0};
    std::string aligned_query;
    std::string aligned_ref;
};

namespace detail {

// Tabular Gotoh DP for leaf subproblems in Hirschberg NW and for local alignment SW
struct LeafGotohResult {
    int64_t score{0};
    std::vector<char> ops; // 'M', 'I', 'D'
};

inline LeafGotohResult gotoh_tabular_global(std::string_view q, std::string_view r,
                                            const ScoringMatrix& matrix, int32_t g_open,
                                            int32_t g_ext) {
    const size_t M = q.size();
    const size_t N = r.size();

    if (M == 0 && N == 0) {
        return {0, {}};
    }
    if (M == 0) {
        return {-(g_open + static_cast<int64_t>(N - 1) * g_ext), std::vector<char>(N, 'D')};
    }
    if (N == 0) {
        return {-(g_open + static_cast<int64_t>(M - 1) * g_ext), std::vector<char>(M, 'I')};
    }

    const size_t stride = N + 1;
    std::vector<int32_t> H((M + 1) * stride, kNegInf);
    std::vector<int32_t> E((M + 1) * stride, kNegInf);
    std::vector<int32_t> F((M + 1) * stride, kNegInf);

    auto at = [stride](auto& table, size_t i, size_t j) -> int32_t& {
        return table[i * stride + j];
    };

    at(H, 0, 0) = 0;
    for (size_t j = 1; j <= N; ++j) {
        const int32_t val = -(g_open + static_cast<int32_t>(j - 1) * g_ext);
        at(H, 0, j) = val;
        at(E, 0, j) = val;
    }
    for (size_t i = 1; i <= M; ++i) {
        const int32_t val = -(g_open + static_cast<int32_t>(i - 1) * g_ext);
        at(H, i, 0) = val;
        at(F, i, 0) = val;
    }

    for (size_t i = 1; i <= M; ++i) {
        const uint8_t qi = static_cast<uint8_t>(q[i - 1]);
        for (size_t j = 1; j <= N; ++j) {
            const uint8_t rj = static_cast<uint8_t>(r[j - 1]);
            const int32_t e_val = std::max(at(H, i, j - 1) - g_open, at(E, i, j - 1) - g_ext);
            const int32_t f_val = std::max(at(H, i - 1, j) - g_open, at(F, i - 1, j) - g_ext);
            at(E, i, j) = e_val;
            at(F, i, j) = f_val;

            const int32_t diag = at(H, i - 1, j - 1) + matrix.matrix[qi][rj];
            at(H, i, j) = std::max({diag, e_val, f_val});
        }
    }

    // Traceback
    size_t i = M;
    size_t j = N;
    int32_t state = 0; // 0: H, 1: E, 2: F
    std::vector<char> ops;
    ops.reserve(M + N);

    while (i > 0 || j > 0) {
        if (state == 0) {
            if (i > 0 && j > 0) {
                const uint8_t qi = static_cast<uint8_t>(q[i - 1]);
                const uint8_t rj = static_cast<uint8_t>(r[j - 1]);
                if (at(H, i, j) == at(H, i - 1, j - 1) + matrix.matrix[qi][rj]) {
                    ops.push_back('M');
                    --i;
                    --j;
                    continue;
                }
            }
            if (at(H, i, j) == at(E, i, j)) {
                state = 1;
                continue;
            }
            if (at(H, i, j) == at(F, i, j)) {
                state = 2;
                continue;
            }
            if (i > 0) {
                ops.push_back('I');
                --i;
            } else {
                ops.push_back('D');
                --j;
            }
        } else if (state == 1) { // in E (gap in query, character in ref consumed -> D)
            ops.push_back('D');
            if (j > 0 && at(E, i, j) == at(H, i, j - 1) - g_open) {
                state = 0;
            }
            --j;
        } else if (state == 2) { // in F (character in query consumed, gap in ref -> I)
            ops.push_back('I');
            if (i > 0 && at(F, i, j) == at(H, i - 1, j) - g_open) {
                state = 0;
            }
            --i;
        }
    }

    std::reverse(ops.begin(), ops.end());
    return {at(H, M, N), std::move(ops)};
}

// Compute the final row of Gotoh DP using O(N) space without allocating string copies
inline void gotoh_forward_row(std::string_view q, std::string_view r, const ScoringMatrix& matrix,
                              int32_t g_open, int32_t g_ext, bool reverse_q, bool reverse_r,
                              std::vector<int32_t>& out_H, std::vector<int32_t>& prev_H,
                              std::vector<int32_t>& E) {
    const size_t N = r.size();
    out_H.resize(N + 1);
    prev_H.resize(N + 1);
    E.assign(N + 1, kNegInf);

    out_H[0] = 0;
    for (size_t j = 1; j <= N; ++j) {
        const int32_t val = -(g_open + static_cast<int32_t>(j - 1) * g_ext);
        out_H[j] = val;
        E[j] = val;
    }

    const size_t M = q.size();
    for (size_t i = 1; i <= M; ++i) {
        prev_H = out_H;
        const char q_ch = reverse_q ? q[M - i] : q[i - 1];
        const uint8_t qi = static_cast<uint8_t>(q_ch);
        out_H[0] = -(g_open + static_cast<int32_t>(i - 1) * g_ext);
        int32_t F_curr = out_H[0];

        for (size_t j = 1; j <= N; ++j) {
            const char r_ch = reverse_r ? r[N - j] : r[j - 1];
            const uint8_t rj = static_cast<uint8_t>(r_ch);
            E[j] = std::max(out_H[j - 1] - g_open, E[j - 1] - g_ext);
            F_curr = std::max(prev_H[j] - g_open, F_curr - g_ext);
            const int32_t diag = prev_H[j - 1] + matrix.matrix[qi][rj];
            out_H[j] = std::max({diag, E[j], F_curr});
        }
    }
}

// Preallocated workspace buffers reused across the recursive Hirschberg divide-and-conquer tree
struct HirschbergWorkspace {
    std::vector<int32_t> buf_fwd;
    std::vector<int32_t> buf_rev;
    std::vector<int32_t> buf_prev;
    std::vector<int32_t> buf_E;

    explicit HirschbergWorkspace(size_t n)
        : buf_fwd(n + 1), buf_rev(n + 1), buf_prev(n + 1), buf_E(n + 1) {}
};

inline void hirschberg_nw_recursive(std::string_view q, std::string_view r,
                                    const ScoringMatrix& matrix, int32_t g_open, int32_t g_ext,
                                    std::vector<char>& ops_accum, HirschbergWorkspace& ws) {
    const size_t M = q.size();
    const size_t N = r.size();

    // Base leaf condition: switch to tabular Gotoh DP
    if (M <= 16 || N <= 16 || M * N <= 2048) {
        auto leaf = gotoh_tabular_global(q, r, matrix, g_open, g_ext);
        ops_accum.insert(ops_accum.end(), leaf.ops.begin(), leaf.ops.end());
        return;
    }

    const size_t mid = M / 2;
    std::string_view q_top = q.substr(0, mid);
    std::string_view q_bot = q.substr(mid);

    gotoh_forward_row(q_top, r, matrix, g_open, g_ext, false, false, ws.buf_fwd, ws.buf_prev,
                      ws.buf_E);
    gotoh_forward_row(q_bot, r, matrix, g_open, g_ext, true, true, ws.buf_rev, ws.buf_prev,
                      ws.buf_E);

    int32_t best_score = kNegInf;
    size_t best_k = 0;
    for (size_t k = 0; k <= N; ++k) {
        const int32_t score = ws.buf_fwd[k] + ws.buf_rev[N - k];
        if (score > best_score) {
            best_score = score;
            best_k = k;
        }
    }

    hirschberg_nw_recursive(q_top, r.substr(0, best_k), matrix, g_open, g_ext, ops_accum, ws);
    hirschberg_nw_recursive(q_bot, r.substr(best_k), matrix, g_open, g_ext, ops_accum, ws);
}

} // namespace detail

/**
 * @brief Needleman-Wunsch global alignment with Gotoh affine gap penalties and Hirschberg linear
 * space.
 *
 * @param query Query biological sequence.
 * @param ref Reference biological sequence.
 * @param matrix Substitution/scoring matrix (default: match=2, mismatch=-1).
 * @param gap Affine gap penalty parameters (default: open=-5, extend=-1).
 * @return AlignmentResult containing score, CIGAR string, and aligned sequences.
 */
[[nodiscard]] inline AlignmentResult needleman_wunsch(std::string_view query, std::string_view ref,
                                                      const ScoringMatrix& matrix = ScoringMatrix{},
                                                      const GapPenalty& gap = GapPenalty{}) {
    const int32_t g_open = std::abs(gap.open);
    const int32_t g_ext = std::abs(gap.extend);

    if (query.empty() && ref.empty()) {
        return AlignmentResult{0, "", 0, 0, 0, 0, "", ""};
    }

    // Sequence dimension swapping: keep the recursion row dimension <= min(M, N)
    const bool need_swap = (ref.size() > query.size());
    std::string_view eff_query = need_swap ? ref : query;
    std::string_view eff_ref = need_swap ? query : ref;

    const size_t min_len = eff_ref.size();
    detail::HirschbergWorkspace ws(min_len);

    std::vector<char> ops;
    detail::hirschberg_nw_recursive(eff_query, eff_ref, matrix, g_open, g_ext, ops, ws);

    if (need_swap) {
        for (char& op : ops) {
            if (op == 'I') {
                op = 'D';
            } else if (op == 'D') {
                op = 'I';
            }
        }
    }

    // Calculate exact alignment score from CIGAR edit operations
    int64_t score = 0;
    size_t qi = 0;
    size_t ri = 0;
    std::string aligned_q;
    std::string aligned_r;
    aligned_q.reserve(ops.size());
    aligned_r.reserve(ops.size());

    bool in_ins = false;
    bool in_del = false;
    for (char op : ops) {
        if (op == 'M') {
            in_ins = false;
            in_del = false;
            score += matrix.matrix[static_cast<uint8_t>(query[qi])][static_cast<uint8_t>(ref[ri])];
            aligned_q.push_back(query[qi++]);
            aligned_r.push_back(ref[ri++]);
        } else if (op == 'I') { // insertion in ref: query char, '-' in ref
            in_del = false;
            score -= (!in_ins ? g_open : g_ext);
            in_ins = true;
            aligned_q.push_back(query[qi++]);
            aligned_r.push_back('-');
        } else if (op == 'D') { // deletion in ref: '-' in query, ref char
            in_ins = false;
            score -= (!in_del ? g_open : g_ext);
            in_del = true;
            aligned_q.push_back('-');
            aligned_r.push_back(ref[ri++]);
        }
    }

    AlignmentResult res;
    res.score = score;
    res.cigar = detail::compress_cigar(ops);
    res.query_start = 0;
    res.query_end = query.size();
    res.ref_start = 0;
    res.ref_end = ref.size();
    res.aligned_query = std::move(aligned_q);
    res.aligned_ref = std::move(aligned_r);
    return res;
}

/**
 * @brief Smith-Waterman local alignment with Gotoh affine gap penalties and traceback.
 */
[[nodiscard]] inline AlignmentResult smith_waterman(std::string_view query, std::string_view ref,
                                                    const ScoringMatrix& matrix = ScoringMatrix{},
                                                    const GapPenalty& gap = GapPenalty{}) {
    const int32_t g_open = std::abs(gap.open);
    const int32_t g_ext = std::abs(gap.extend);

    const size_t M = query.size();
    const size_t N = ref.size();
    if (M == 0 || N == 0) {
        return AlignmentResult{};
    }

    const size_t stride = N + 1;
    std::vector<int32_t> H((M + 1) * stride, 0);
    std::vector<int32_t> E((M + 1) * stride, detail::kNegInf);
    std::vector<int32_t> F((M + 1) * stride, detail::kNegInf);

    auto at = [stride](auto& table, size_t i, size_t j) -> int32_t& {
        return table[i * stride + j];
    };

    int32_t max_score = 0;
    size_t max_i = 0;
    size_t max_j = 0;

    for (size_t i = 1; i <= M; ++i) {
        const uint8_t qi = static_cast<uint8_t>(query[i - 1]);
        for (size_t j = 1; j <= N; ++j) {
            const uint8_t rj = static_cast<uint8_t>(ref[j - 1]);
            const int32_t e_val = std::max(at(H, i, j - 1) - g_open, at(E, i, j - 1) - g_ext);
            const int32_t f_val = std::max(at(H, i - 1, j) - g_open, at(F, i - 1, j) - g_ext);
            at(E, i, j) = e_val;
            at(F, i, j) = f_val;

            const int32_t diag = at(H, i - 1, j - 1) + matrix.matrix[qi][rj];
            const int32_t h_val = std::max({0, diag, e_val, f_val});
            at(H, i, j) = h_val;

            if (h_val > max_score) {
                max_score = h_val;
                max_i = i;
                max_j = j;
            }
        }
    }

    if (max_score == 0) {
        return AlignmentResult{};
    }

    // Traceback from (max_i, max_j) until H == 0
    size_t i = max_i;
    size_t j = max_j;
    int32_t state = 0;
    std::vector<char> ops;

    while (i > 0 && j > 0 && at(H, i, j) > 0) {
        if (state == 0) {
            const uint8_t qi = static_cast<uint8_t>(query[i - 1]);
            const uint8_t rj = static_cast<uint8_t>(ref[j - 1]);
            if (at(H, i, j) == at(H, i - 1, j - 1) + matrix.matrix[qi][rj]) {
                ops.push_back('M');
                --i;
                --j;
                continue;
            }
            if (at(H, i, j) == at(E, i, j)) {
                state = 1;
                continue;
            }
            if (at(H, i, j) == at(F, i, j)) {
                state = 2;
                continue;
            }
            break;
        } else if (state == 1) { // gap in query, ref consumed -> D
            ops.push_back('D');
            if (j > 0 && at(E, i, j) == at(H, i, j - 1) - g_open) {
                state = 0;
            }
            --j;
        } else if (state == 2) { // query consumed, gap in ref -> I
            ops.push_back('I');
            if (i > 0 && at(F, i, j) == at(H, i - 1, j) - g_open) {
                state = 0;
            }
            --i;
        }
    }

    const size_t start_q = i;
    const size_t start_r = j;
    std::reverse(ops.begin(), ops.end());

    std::string aligned_q;
    std::string aligned_r;
    size_t cur_q = start_q;
    size_t cur_r = start_r;
    for (char op : ops) {
        if (op == 'M') {
            aligned_q.push_back(query[cur_q++]);
            aligned_r.push_back(ref[cur_r++]);
        } else if (op == 'I') {
            aligned_q.push_back(query[cur_q++]);
            aligned_r.push_back('-');
        } else if (op == 'D') {
            aligned_q.push_back('-');
            aligned_r.push_back(ref[cur_r++]);
        }
    }

    AlignmentResult res;
    res.score = max_score;
    res.cigar = detail::compress_cigar(ops);
    res.query_start = start_q;
    res.query_end = max_i;
    res.ref_start = start_r;
    res.ref_end = max_j;
    res.aligned_query = std::move(aligned_q);
    res.aligned_ref = std::move(aligned_r);
    return res;
}

namespace detail {

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)

inline int16_t hmax_epi16_avx2(__m256i v) noexcept {
    const __m128i low = _mm256_castsi256_si128(v);
    const __m128i high = _mm256_extracti128_si256(v, 1);
    __m128i m = _mm_max_epi16(low, high);
    m = _mm_max_epi16(m, _mm_srli_si128(m, 8));
    m = _mm_max_epi16(m, _mm_srli_si128(m, 4));
    m = _mm_max_epi16(m, _mm_srli_si128(m, 2));
    return static_cast<int16_t>(_mm_extract_epi16(m, 0));
}

inline __m256i shift_left_1word_avx2(__m256i a) noexcept {
    const __m256i perm = _mm256_permute2x128_si256(a, a, 0x08);
    return _mm256_alignr_epi8(a, perm, 14);
}

// Farrar 2007 Striped Smith-Waterman query profile structure
struct FarrarProfileAVX2 {
    size_t seg_len{0};
    size_t padded_len{0};
    std::array<int16_t, 256> char_to_slot{};
    std::vector<int16_t> profile; // [num_slots * seg_len * 16]
};

inline FarrarProfileAVX2 make_farrar_profile_avx2(std::string_view query,
                                                  const ScoringMatrix& matrix,
                                                  std::string_view ref = "") {
    constexpr size_t p = 16;
    const size_t seg_len = (query.size() + p - 1) / p;
    const size_t padded_len = seg_len * p;

    FarrarProfileAVX2 prof;
    prof.seg_len = seg_len;
    prof.padded_len = padded_len;
    prof.char_to_slot.fill(-1);

    int16_t slot_count = 0;
    if (ref.empty()) {
        for (size_t c = 0; c < 256; ++c) {
            prof.char_to_slot[c] = slot_count++;
        }
    } else {
        for (char ch : ref) {
            const uint8_t c = static_cast<uint8_t>(ch);
            if (prof.char_to_slot[c] < 0) {
                prof.char_to_slot[c] = slot_count++;
            }
        }
    }

    prof.profile.resize(static_cast<size_t>(slot_count) * seg_len * p, 0);

    for (size_t c = 0; c < 256; ++c) {
        const int16_t slot = prof.char_to_slot[c];
        if (slot < 0) {
            continue;
        }
        for (size_t s = 0; s < seg_len; ++s) {
            for (size_t k = 0; k < p; ++k) {
                const size_t q_idx = s + k * seg_len;
                const size_t out_idx = (static_cast<size_t>(slot) * seg_len + s) * p + k;
                if (q_idx < query.size()) {
                    prof.profile[out_idx] =
                        static_cast<int16_t>(matrix.matrix[static_cast<uint8_t>(query[q_idx])][c]);
                } else {
                    prof.profile[out_idx] = 0;
                }
            }
        }
    }
    return prof;
}

inline int64_t farrar_striped_avx2(const FarrarProfileAVX2& prof, std::string_view ref,
                                   int32_t g_open, int32_t g_ext) noexcept {
    const size_t seg_len = prof.seg_len;
    alignas(32) int16_t vH_stack0[2048]{};
    alignas(32) int16_t vH_stack1[2048]{};
    alignas(32) int16_t vE_stack[2048]{};
    std::vector<int16_t> vH_heap0;
    std::vector<int16_t> vH_heap1;
    std::vector<int16_t> vE_heap;
    int16_t* vH_load = vH_stack0;
    int16_t* vH_store = vH_stack1;
    int16_t* vE_ptr = vE_stack;

    if (seg_len * 16 > 2048) {
        vH_heap0.assign(seg_len * 16, 0);
        vH_heap1.assign(seg_len * 16, 0);
        vE_heap.assign(seg_len * 16, 0);
        vH_load = vH_heap0.data();
        vH_store = vH_heap1.data();
        vE_ptr = vE_heap.data();
    }

    const __m256i vZero = _mm256_setzero_si256();
    const __m256i vGapOpen = _mm256_set1_epi16(static_cast<int16_t>(g_open));
    const __m256i vGapExt = _mm256_set1_epi16(static_cast<int16_t>(g_ext));

    __m256i vMaxScore = vZero;

    for (size_t j = 0; j < ref.size(); ++j) {
        const uint8_t r_char = static_cast<uint8_t>(ref[j]);
        const int16_t slot = prof.char_to_slot[r_char];
        if (slot < 0) {
            continue;
        }
        const int16_t* prof_base = prof.profile.data() + (static_cast<size_t>(slot) * seg_len) * 16;

        // Shift previous column's bottom vector into top
        const __m256i last_vH =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(vH_load + (seg_len - 1) * 16));
        __m256i vH = shift_left_1word_avx2(last_vH);
        __m256i vF = vZero;

        for (size_t s = 0; s < seg_len; ++s) {
            __m256i vE = _mm256_load_si256(reinterpret_cast<const __m256i*>(vE_ptr + s * 16));
            __m256i vP = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prof_base + s * 16));
            vH = _mm256_adds_epi16(vH, vP);
            vH = _mm256_max_epi16(vH, vE);
            vH = _mm256_max_epi16(vH, vF);
            vH = _mm256_max_epi16(vH, vZero);

            __m256i vH_open = _mm256_subs_epi16(vH, vGapOpen);
            vE = _mm256_subs_epi16(vE, vGapExt);
            vE = _mm256_max_epi16(vE, vH_open);
            _mm256_store_si256(reinterpret_cast<__m256i*>(vE_ptr + s * 16), vE);

            vF = _mm256_subs_epi16(vF, vGapExt);
            vF = _mm256_max_epi16(vF, vH_open);

            vMaxScore = _mm256_max_epi16(vMaxScore, vH);

            _mm256_store_si256(reinterpret_cast<__m256i*>(vH_store + s * 16), vH);
            vH = _mm256_load_si256(reinterpret_cast<const __m256i*>(vH_load + s * 16));
        }

        // Lazy-F loop: propagates vertical gap carries across stripe boundaries
        for (size_t pass = 0; pass < 16; ++pass) {
            vF = shift_left_1word_avx2(vF);
            bool pass_had_carry = false;
            for (size_t s = 0; s < seg_len; ++s) {
                __m256i cur_H =
                    _mm256_load_si256(reinterpret_cast<const __m256i*>(vH_store + s * 16));
                const __m256i cmp = _mm256_cmpgt_epi16(vF, cur_H);
                if (_mm256_movemask_epi8(cmp) == 0) {
                    const __m256i vH_open = _mm256_subs_epi16(cur_H, vGapOpen);
                    vF = _mm256_subs_epi16(vF, vGapExt);
                    const __m256i cmp_ext = _mm256_cmpgt_epi16(vF, vH_open);
                    if (_mm256_movemask_epi8(cmp_ext) == 0) {
                        goto end_lazy_f;
                    }
                } else {
                    pass_had_carry = true;
                    cur_H = _mm256_max_epi16(cur_H, vF);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(vH_store + s * 16), cur_H);
                    vMaxScore = _mm256_max_epi16(vMaxScore, cur_H);
                    const __m256i vH_open = _mm256_subs_epi16(cur_H, vGapOpen);
                    vF = _mm256_subs_epi16(vF, vGapExt);
                    vF = _mm256_max_epi16(vF, vH_open);
                }
            }
            if (!pass_had_carry) {
                break;
            }
        }
    end_lazy_f:
        std::swap(vH_load, vH_store);
    }

    return static_cast<int64_t>(hmax_epi16_avx2(vMaxScore));
}

#endif // AVX2

inline int64_t smith_waterman_scalar_score(std::string_view query, std::string_view ref,
                                           const ScoringMatrix& matrix, int32_t g_open,
                                           int32_t g_ext) noexcept {
    const size_t M = query.size();
    const size_t N = ref.size();
    if (M == 0 || N == 0) {
        return 0;
    }

    // Column-major scalar DP
    std::vector<int32_t> H(M + 1, 0);
    std::vector<int32_t> E(M + 1, kNegInf);
    int32_t max_score = 0;

    for (size_t j = 1; j <= N; ++j) {
        const uint8_t rj = static_cast<uint8_t>(ref[j - 1]);
        int32_t diag = 0;
        int32_t F_curr = kNegInf;

        for (size_t i = 1; i <= M; ++i) {
            const uint8_t qi = static_cast<uint8_t>(query[i - 1]);
            const int32_t next_diag = H[i];

            E[i] = std::max(H[i] - g_open, E[i] - g_ext);
            const int32_t match_val = diag + matrix.matrix[qi][rj];
            const int32_t h_val = std::max({0, match_val, E[i], F_curr});
            H[i] = h_val;
            diag = next_diag;

            F_curr = std::max(h_val - g_open, F_curr - g_ext);

            if (h_val > max_score) {
                max_score = h_val;
            }
        }
    }
    return max_score;
}

} // namespace detail

/**
 * @brief Farrar (2007) Striped SIMD Smith-Waterman local alignment score calculation.
 *
 * @param query Query biological sequence.
 * @param ref Reference sequence.
 * @param matrix Scoring matrix (default match=2, mismatch=-1).
 * @param gap Affine gap penalty parameters.
 * @return Optimal local alignment score.
 */
[[nodiscard]] inline int64_t smith_waterman_score(std::string_view query, std::string_view ref,
                                                  const ScoringMatrix& matrix = ScoringMatrix{},
                                                  const GapPenalty& gap = GapPenalty{}) noexcept {
    const int32_t g_open = std::abs(gap.open);
    const int32_t g_ext = std::abs(gap.extend);

    if (query.empty() || ref.empty()) {
        return 0;
    }

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
    // AVX2 Striped SIMD for sequences of length >= 16
    if (query.size() >= 16 && ref.size() >= 16) {
        auto prof = detail::make_farrar_profile_avx2(query, matrix, ref);
        return detail::farrar_striped_avx2(prof, ref, g_open, g_ext);
    }
#endif

    return detail::smith_waterman_scalar_score(query, ref, matrix, g_open, g_ext);
}

/**
 * @brief Batch Smith-Waterman alignment against multiple reference targets reusing query profile.
 */
[[nodiscard]] inline std::vector<int64_t>
smith_waterman_batch(std::string_view query, std::span<const std::string_view> refs,
                     const ScoringMatrix& matrix = ScoringMatrix{},
                     const GapPenalty& gap = GapPenalty{}) {
    const int32_t g_open = std::abs(gap.open);
    const int32_t g_ext = std::abs(gap.extend);

    std::vector<int64_t> results;
    results.reserve(refs.size());

    if (query.empty()) {
        results.assign(refs.size(), 0);
        return results;
    }

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
    if (query.size() >= 16) {
        auto prof = detail::make_farrar_profile_avx2(query, matrix);
        for (std::string_view ref : refs) {
            if (ref.empty()) {
                results.push_back(0);
            } else if (ref.size() < 16) {
                results.push_back(
                    detail::smith_waterman_scalar_score(query, ref, matrix, g_open, g_ext));
            } else {
                results.push_back(detail::farrar_striped_avx2(prof, ref, g_open, g_ext));
            }
        }
        return results;
    }
#endif

    for (std::string_view ref : refs) {
        results.push_back(detail::smith_waterman_scalar_score(query, ref, matrix, g_open, g_ext));
    }
    return results;
}

} // namespace algoat::dp
