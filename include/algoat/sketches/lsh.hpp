#pragma once

#include "algoat/sketches/detail/hash.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#endif

namespace algoat::sketches {

namespace detail {

struct SimHashSignsLUT {
    alignas(32) float table[256][8];

    constexpr SimHashSignsLUT() : table{} {
        for (int b = 0; b < 256; ++b) {
            for (int j = 0; j < 8; ++j) {
                table[b][j] = ((b >> j) & 1) ? 1.0f : -1.0f;
            }
        }
    }
};

inline constexpr SimHashSignsLUT SIMHASH_SIGNS_LUT{};

} // namespace detail

/// \brief b-Bit MinHash signature generator for Jaccard similarity estimation.
/// \tparam K Number of hash functions (e.g., 64, 128, 256, 512).
/// \tparam B Bits retained per hash (1, 2, or 4).
template <size_t K = 256, size_t B = 2>
    requires(B == 1 || B == 2 || B == 4) && (K % (64 / B) == 0)
class BBitMinHash {
public:
    static constexpr size_t kNumWords = (K * B) / 64;
    static constexpr size_t kHashesPerWord = 64 / B;
    static constexpr uint64_t kBitMask = (B == 64) ? ~0ULL : ((1ULL << B) - 1ULL);

    struct alignas(64) Signature {
        std::array<uint64_t, kNumWords> words{};

        [[nodiscard]] double jaccard_similarity(const Signature& other) const noexcept {
            constexpr double kInvK = 1.0 / static_cast<double>(K);
            constexpr double base = 1.0 / static_cast<double>(1ULL << B);
            constexpr double inv_one_minus_base = 1.0 / (1.0 - base);
            constexpr double factor = kInvK * inv_one_minus_base;
            constexpr double offset = base * inv_one_minus_base;

            const size_t mismatches = hamming_distance(other);
            const size_t matches = K - mismatches;
            const double est = static_cast<double>(matches) * factor - offset;
            return std::clamp(est, 0.0, 1.0);
        }

        [[nodiscard]] size_t hamming_distance(const Signature& other) const noexcept {
#if defined(__AVX2__)
            if constexpr (B == 2 && kNumWords % 4 == 0) {
                const __m256i mask = _mm256_set1_epi64x(0x5555555555555555ULL);
                size_t mismatches = 0;
                for (size_t w = 0; w < kNumWords; w += 4) {
                    __m256i a =
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(words.data() + w));
                    __m256i b =
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(other.words.data() + w));
                    __m256i diff = _mm256_xor_si256(a, b);
                    __m256i any_diff = _mm256_or_si256(_mm256_srli_epi64(diff, 1), diff);
                    __m256i m = _mm256_and_si256(any_diff, mask);
                    mismatches += static_cast<size_t>(
                        __builtin_popcountll(static_cast<uint64_t>(_mm256_extract_epi64(m, 0))) +
                        __builtin_popcountll(static_cast<uint64_t>(_mm256_extract_epi64(m, 1))) +
                        __builtin_popcountll(static_cast<uint64_t>(_mm256_extract_epi64(m, 2))) +
                        __builtin_popcountll(static_cast<uint64_t>(_mm256_extract_epi64(m, 3))));
                }
                return mismatches;
            } else if constexpr (B == 1 && kNumWords % 4 == 0) {
                size_t mismatches = 0;
                for (size_t w = 0; w < kNumWords; w += 4) {
                    __m256i a =
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(words.data() + w));
                    __m256i b =
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(other.words.data() + w));
                    __m256i diff = _mm256_xor_si256(a, b);
                    mismatches += static_cast<size_t>(
                        __builtin_popcountll(static_cast<uint64_t>(_mm256_extract_epi64(diff, 0))) +
                        __builtin_popcountll(static_cast<uint64_t>(_mm256_extract_epi64(diff, 1))) +
                        __builtin_popcountll(static_cast<uint64_t>(_mm256_extract_epi64(diff, 2))) +
                        __builtin_popcountll(static_cast<uint64_t>(_mm256_extract_epi64(diff, 3))));
                }
                return mismatches;
            }
#endif
            size_t mismatches = 0;
            for (size_t w = 0; w < kNumWords; ++w) {
                uint64_t diff = words[w] ^ other.words[w];
                if constexpr (B == 1) {
                    mismatches += static_cast<size_t>(__builtin_popcountll(diff));
                } else if constexpr (B == 2) {
                    uint64_t any_diff = (diff >> 1) | diff;
                    uint64_t mismatch_mask = any_diff & 0x5555555555555555ULL;
                    mismatches += static_cast<size_t>(__builtin_popcountll(mismatch_mask));
                } else if constexpr (B == 4) {
                    uint64_t t1 = (diff >> 2) | diff;
                    uint64_t t2 = (t1 >> 1) | t1;
                    uint64_t mismatch_mask = t2 & 0x1111111111111111ULL;
                    mismatches += static_cast<size_t>(__builtin_popcountll(mismatch_mask));
                }
            }
            return mismatches;
        }

        [[nodiscard]] uint64_t get_hash(size_t index) const noexcept {
            size_t word_idx = index / kHashesPerWord;
            size_t bit_offset = (index % kHashesPerWord) * B;
            return (words[word_idx] >> bit_offset) & kBitMask;
        }

        bool operator==(const Signature& other) const noexcept = default;
    };

    explicit BBitMinHash(uint64_t seed = 0x9E3779B97F4A7C15ULL) {
        uint64_t s = seed;
        for (size_t i = 0; i < K; ++i) {
            s = detail::splitmix64(s + 0x9E3779B97F4A7C15ULL);
            hash_seeds_a_[i] = s | 1ULL; // Coprime to 2^64 ensures full permutation
            s = detail::splitmix64(s + 0x9E3779B97F4A7C15ULL);
            hash_seeds_b_[i] = s;
        }
    }

    /// \brief Computes signature from a stream of hashed 64-bit tokens.
    [[nodiscard]] Signature compute_signature(std::span<const uint64_t> tokens) const noexcept {
        Signature sig{};
        if (tokens.empty()) {
            return sig;
        }

        for (size_t i = 0; i < K; ++i) {
            const uint64_t a = hash_seeds_a_[i];
            const uint64_t b = hash_seeds_b_[i];
            uint64_t min_h = std::numeric_limits<uint64_t>::max();

            for (uint64_t token : tokens) {
                uint64_t h = detail::splitmix64(token * a + b);
                if (h < min_h) {
                    min_h = h;
                }
            }

            uint64_t val = min_h & kBitMask;
            size_t word_idx = i / kHashesPerWord;
            size_t bit_offset = (i % kHashesPerWord) * B;
            sig.words[word_idx] |= (val << bit_offset);
        }

        return sig;
    }

    /// \brief Unbiased Jaccard similarity estimator from match ratio S_b.
    [[nodiscard]] static double estimate_jaccard(double match_ratio) noexcept {
        constexpr double base = 1.0 / static_cast<double>(1ULL << B);
        constexpr double inv_one_minus_base = 1.0 / (1.0 - base);
        double est = (match_ratio - base) * inv_one_minus_base;
        return std::clamp(est, 0.0, 1.0);
    }

private:
    std::array<uint64_t, K> hash_seeds_a_{};
    std::array<uint64_t, K> hash_seeds_b_{};
};

/// \brief 128-bit SimHash representation for extended angular cosine similarity.
struct SimHash128 {
    uint64_t low{0};
    uint64_t high{0};

    bool operator==(const SimHash128&) const noexcept = default;
};

/// \brief 64-bit / 128-bit Vectorized SimHash for Cosine Similarity.
class SimHash {
public:
    /// \brief Generates 64-bit SimHash fingerprint from weighted token hashes.
    [[nodiscard]] static uint64_t compute_64(std::span<const uint64_t> token_hashes,
                                             std::span<const float> weights = {}) noexcept {
        const size_t count = token_hashes.size();
        if (count == 0) {
            return 0ULL;
        }

#if defined(__AVX2__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        const auto& lut = detail::SIMHASH_SIGNS_LUT.table;
        const uint64_t* h_ptr = token_hashes.data();
        const uint64_t* const h_end = h_ptr + count;

        if (!weights.empty()) {
            const float* w_ptr = weights.data();
            const float* const w_end = w_ptr + weights.size();
            for (; h_ptr != h_end; ++h_ptr) {
                const float w = (w_ptr < w_end) ? *w_ptr++ : 1.0f;
                if (w == 0.0f) {
                    continue;
                }
                const uint64_t h = *h_ptr;
                const __m256 v_w = _mm256_set1_ps(w);
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&h);

                acc0 = _mm256_fmadd_ps(v_w, _mm256_load_ps(lut[bytes[0]]), acc0);
                acc1 = _mm256_fmadd_ps(v_w, _mm256_load_ps(lut[bytes[1]]), acc1);
                acc2 = _mm256_fmadd_ps(v_w, _mm256_load_ps(lut[bytes[2]]), acc2);
                acc3 = _mm256_fmadd_ps(v_w, _mm256_load_ps(lut[bytes[3]]), acc3);
                acc4 = _mm256_fmadd_ps(v_w, _mm256_load_ps(lut[bytes[4]]), acc4);
                acc5 = _mm256_fmadd_ps(v_w, _mm256_load_ps(lut[bytes[5]]), acc5);
                acc6 = _mm256_fmadd_ps(v_w, _mm256_load_ps(lut[bytes[6]]), acc6);
                acc7 = _mm256_fmadd_ps(v_w, _mm256_load_ps(lut[bytes[7]]), acc7);
            }
        } else {
            for (; h_ptr != h_end; ++h_ptr) {
                const uint64_t h = *h_ptr;
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&h);

                acc0 = _mm256_add_ps(acc0, _mm256_load_ps(lut[bytes[0]]));
                acc1 = _mm256_add_ps(acc1, _mm256_load_ps(lut[bytes[1]]));
                acc2 = _mm256_add_ps(acc2, _mm256_load_ps(lut[bytes[2]]));
                acc3 = _mm256_add_ps(acc3, _mm256_load_ps(lut[bytes[3]]));
                acc4 = _mm256_add_ps(acc4, _mm256_load_ps(lut[bytes[4]]));
                acc5 = _mm256_add_ps(acc5, _mm256_load_ps(lut[bytes[5]]));
                acc6 = _mm256_add_ps(acc6, _mm256_load_ps(lut[bytes[6]]));
                acc7 = _mm256_add_ps(acc7, _mm256_load_ps(lut[bytes[7]]));
            }
        }

        const __m256 v_zero = _mm256_setzero_ps();
        const uint64_t m0 =
            static_cast<uint8_t>(_mm256_movemask_ps(_mm256_cmp_ps(acc0, v_zero, _CMP_GT_OQ)));
        const uint64_t m1 =
            static_cast<uint8_t>(_mm256_movemask_ps(_mm256_cmp_ps(acc1, v_zero, _CMP_GT_OQ)));
        const uint64_t m2 =
            static_cast<uint8_t>(_mm256_movemask_ps(_mm256_cmp_ps(acc2, v_zero, _CMP_GT_OQ)));
        const uint64_t m3 =
            static_cast<uint8_t>(_mm256_movemask_ps(_mm256_cmp_ps(acc3, v_zero, _CMP_GT_OQ)));
        const uint64_t m4 =
            static_cast<uint8_t>(_mm256_movemask_ps(_mm256_cmp_ps(acc4, v_zero, _CMP_GT_OQ)));
        const uint64_t m5 =
            static_cast<uint8_t>(_mm256_movemask_ps(_mm256_cmp_ps(acc5, v_zero, _CMP_GT_OQ)));
        const uint64_t m6 =
            static_cast<uint8_t>(_mm256_movemask_ps(_mm256_cmp_ps(acc6, v_zero, _CMP_GT_OQ)));
        const uint64_t m7 =
            static_cast<uint8_t>(_mm256_movemask_ps(_mm256_cmp_ps(acc7, v_zero, _CMP_GT_OQ)));

        return m0 | (m1 << 8) | (m2 << 16) | (m3 << 24) | (m4 << 32) | (m5 << 40) | (m6 << 48) |
               (m7 << 56);
#else
        std::array<float, 64> acc{};
        const size_t num_weights = weights.size();
        for (size_t i = 0; i < count; ++i) {
            const float w = (i < num_weights) ? weights[i] : 1.0f;
            if (w == 0.0f) {
                continue;
            }
            const uint64_t h = token_hashes[i];
            for (size_t j = 0; j < 64; ++j) {
                acc[j] += ((h >> j) & 1ULL) ? w : -w;
            }
        }

        uint64_t fp = 0;
        for (size_t j = 0; j < 64; ++j) {
            if (acc[j] > 0.0f) {
                fp |= (1ULL << j);
            }
        }
        return fp;
#endif
    }

    /// \brief Generates 128-bit SimHash fingerprint from paired low and high token hashes.
    [[nodiscard]] static SimHash128 compute_128(std::span<const uint64_t> token_hashes_lo,
                                                std::span<const uint64_t> token_hashes_hi,
                                                std::span<const float> weights = {}) noexcept {
        return SimHash128{
            .low = compute_64(token_hashes_lo, weights),
            .high = compute_64(token_hashes_hi, weights),
        };
    }

    /// \brief Computes angular cosine distance between two 64-bit fingerprints.
    [[nodiscard]] static double cosine_similarity(uint64_t fp1, uint64_t fp2) noexcept {
        const uint32_t dist = hamming_distance(fp1, fp2);
        return std::cos(static_cast<double>(dist) * (std::numbers::pi / 64.0));
    }

    /// \brief Computes angular cosine distance between two 128-bit fingerprints.
    [[nodiscard]] static double cosine_similarity(SimHash128 fp1, SimHash128 fp2) noexcept {
        const uint32_t dist = hamming_distance(fp1, fp2);
        return std::cos(static_cast<double>(dist) * (std::numbers::pi / 128.0));
    }

    /// \brief Hardware popcount Hamming distance for 64-bit fingerprints.
    [[nodiscard]] static inline uint32_t hamming_distance(uint64_t fp1, uint64_t fp2) noexcept {
        return static_cast<uint32_t>(__builtin_popcountll(fp1 ^ fp2));
    }

    /// \brief Hardware popcount Hamming distance for 128-bit fingerprints.
    [[nodiscard]] static inline uint32_t hamming_distance(SimHash128 fp1, SimHash128 fp2) noexcept {
        return static_cast<uint32_t>(__builtin_popcountll(fp1.low ^ fp2.low) +
                                     __builtin_popcountll(fp1.high ^ fp2.high));
    }
};

/// \brief Banded LSH Table for fast similarity candidate pair retrieval.
template <typename SignatureType, size_t Bands, size_t RowsPerBand> class LSHIndex {
public:
    static_assert(Bands > 0, "Bands must be greater than 0");
    static_assert(RowsPerBand > 0, "RowsPerBand must be greater than 0");

    LSHIndex() {
        for (size_t b = 0; b < Bands; ++b) {
            band_seeds_[b] = detail::splitmix64(b ^ 0x9E3779B97F4A7C15ULL);
        }
    }

    void insert(uint64_t id, const SignatureType& sig) {
        for (size_t b = 0; b < Bands; ++b) {
            const uint64_t key = compute_band_hash(sig, b);
            tables_[b][key].push_back(id);
        }
    }

    void query_candidates(const SignatureType& sig, std::vector<uint64_t>& out_candidates) const {
        for (size_t b = 0; b < Bands; ++b) {
            const uint64_t key = compute_band_hash(sig, b);
            auto it = tables_[b].find(key);
            if (it != tables_[b].end()) {
                out_candidates.insert(out_candidates.end(), it->second.begin(), it->second.end());
            }
        }

        if (!out_candidates.empty()) {
            std::sort(out_candidates.begin(), out_candidates.end());
            out_candidates.erase(std::unique(out_candidates.begin(), out_candidates.end()),
                                 out_candidates.end());
        }
    }

    /// \brief Multi-probe candidate retrieval supporting Radius-1 perturbation probing.
    void query_candidates_multiprobe(const SignatureType& sig,
                                     std::vector<uint64_t>& out_candidates,
                                     size_t max_probes_per_band = 1) const {
        for (size_t b = 0; b < Bands; ++b) {
            const uint64_t primary_key = compute_band_hash(sig, b);
            auto it = tables_[b].find(primary_key);
            if (it != tables_[b].end()) {
                out_candidates.insert(out_candidates.end(), it->second.begin(), it->second.end());
            }

            // Probe adjacent perturbed buckets by perturbing the signature slice
            if constexpr (std::is_same_v<SignatureType, uint64_t>) {
                for (size_t bit = 0; bit < std::min<size_t>(RowsPerBand, max_probes_per_band);
                     ++bit) {
                    const size_t total_bit = b * RowsPerBand + bit;
                    if (total_bit < 64) {
                        const uint64_t perturbed_sig = sig ^ (1ULL << total_bit);
                        const uint64_t perturbed_key = compute_band_hash(perturbed_sig, b);
                        auto pit = tables_[b].find(perturbed_key);
                        if (pit != tables_[b].end()) {
                            out_candidates.insert(out_candidates.end(), pit->second.begin(),
                                                  pit->second.end());
                        }
                    }
                }
            } else if constexpr (requires { sig.words; }) {
                SignatureType perturbed_sig = sig;
                for (size_t r = 0; r < std::min<size_t>(RowsPerBand, max_probes_per_band); ++r) {
                    const size_t hash_idx = b * RowsPerBand + r;
                    const size_t hashes_per_word =
                        64 / std::remove_cvref_t<decltype(sig)>::kBitMask; // or B
                    // Flip the least significant bit of this hash in the signature
                    const size_t word_idx =
                        hash_idx /
                        (sig.words.size() == 0 ? 1 : (sig.words.size() * 64 / RowsPerBand / Bands));
                    if (word_idx < perturbed_sig.words.size()) {
                        perturbed_sig.words[word_idx] ^= 1ULL;
                        const uint64_t perturbed_key = compute_band_hash(perturbed_sig, b);
                        perturbed_sig.words[word_idx] ^= 1ULL; // restore
                        auto pit = tables_[b].find(perturbed_key);
                        if (pit != tables_[b].end()) {
                            out_candidates.insert(out_candidates.end(), pit->second.begin(),
                                                  pit->second.end());
                        }
                    }
                }
            }
        }

        if (!out_candidates.empty()) {
            std::sort(out_candidates.begin(), out_candidates.end());
            out_candidates.erase(std::unique(out_candidates.begin(), out_candidates.end()),
                                 out_candidates.end());
        }
    }

    [[nodiscard]] std::vector<uint64_t> query_candidates(const SignatureType& sig) const {
        std::vector<uint64_t> candidates;
        query_candidates(sig, candidates);
        return candidates;
    }

    void clear() noexcept {
        for (size_t b = 0; b < Bands; ++b) {
            tables_[b].clear();
        }
    }

private:
    [[nodiscard]] uint64_t compute_band_hash(const SignatureType& sig, size_t band) const noexcept {
        if constexpr (requires { sig.get_hash(size_t{}); }) {
            uint64_t h = band_seeds_[band];
            for (size_t r = 0; r < RowsPerBand; ++r) {
                uint64_t val = sig.get_hash(band * RowsPerBand + r);
                h = detail::wymum_mix128(h ^ val, 0x517cc1b727220a95ULL);
            }
            return h;
        } else if constexpr (std::is_same_v<SignatureType, uint64_t>) {
            static_assert(Bands * RowsPerBand <= 64,
                          "Total indexed bits (Bands * RowsPerBand) cannot exceed 64");
            constexpr uint64_t mask = (RowsPerBand >= 64)
                                          ? ~0ULL
                                          : ((1ULL << (RowsPerBand < 64 ? RowsPerBand : 0)) - 1ULL);
            const size_t shift = band * RowsPerBand;
            const uint64_t slice = (shift >= 64) ? 0ULL : ((sig >> shift) & mask);
            return detail::wymum_mix128(band_seeds_[band] ^ slice, 0x517cc1b727220a95ULL);
        } else {
            const auto* ptr = std::data(sig) + (band * RowsPerBand);
            using ElementType = std::remove_cvref_t<decltype(*ptr)>;
            return detail::wyhash(ptr, RowsPerBand * sizeof(ElementType), band_seeds_[band]);
        }
    }

    std::array<uint64_t, Bands> band_seeds_{};
    std::array<std::unordered_map<uint64_t, std::vector<uint64_t>>, Bands> tables_{};
};

} // namespace algoat::sketches
