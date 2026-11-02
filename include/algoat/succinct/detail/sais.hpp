#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace algoat::succinct::detail {

/// \brief Compact 1-bit per character type vector for L/S suffix classification.
class SaisTypeBitVector {
public:
    explicit SaisTypeBitVector(size_t n) : data_((n + 63) / 64, 0ULL) {}

    [[nodiscard]] inline bool is_s_type(size_t i) const noexcept {
        return (data_[i >> 6] >> (i & 63)) & 1ULL;
    }

    [[nodiscard]] inline bool is_l_type(size_t i) const noexcept {
        return !is_s_type(i);
    }

    inline void set_s_type(size_t i) noexcept {
        data_[i >> 6] |= (1ULL << (i & 63));
    }

    inline void set_l_type(size_t i) noexcept {
        data_[i >> 6] &= ~(1ULL << (i & 63));
    }

    [[nodiscard]] inline bool is_lms(size_t i) const noexcept {
        return i > 0 && is_s_type(i) && is_l_type(i - 1);
    }

private:
    std::vector<uint64_t> data_;
};

/// \brief Computes bucket head (start) or tail (end) boundaries for bucket sorting.
template <typename CharT, typename IndexT>
inline void get_bucket_boundaries(std::span<const CharT> s, std::span<IndexT> bucket_offsets,
                                  IndexT alphabet_size, bool is_tail) {
    std::fill(bucket_offsets.begin(), bucket_offsets.begin() + alphabet_size, 0);
    for (size_t i = 0; i < s.size(); ++i) {
        bucket_offsets[static_cast<size_t>(s[i])]++;
    }
    IndexT sum = 0;
    for (IndexT i = 0; i < alphabet_size; ++i) {
        sum += bucket_offsets[i];
        bucket_offsets[i] = is_tail ? (sum - 1) : (sum - bucket_offsets[i]);
    }
}

/// \brief Induced sort pass for L-type suffixes (forward scan).
template <typename CharT, typename IndexT>
inline void induce_l_type_suffixes(const SaisTypeBitVector& sais_types, std::span<IndexT> sa,
                                   std::span<const CharT> s, std::span<IndexT> bucket_offsets,
                                   IndexT alphabet_size) {
    get_bucket_boundaries(s, bucket_offsets, alphabet_size, false);
    const IndexT n = static_cast<IndexT>(s.size());
    for (IndexT i = 0; i < n; ++i) {
        const IndexT j = sa[i] - 1;
        if (j >= 0 && sais_types.is_l_type(static_cast<size_t>(j))) {
            sa[bucket_offsets[static_cast<size_t>(s[j])]++] = j;
        }
    }
}

/// \brief Induced sort pass for S-type suffixes (backward scan).
template <typename CharT, typename IndexT>
inline void induce_s_type_suffixes(const SaisTypeBitVector& sais_types, std::span<IndexT> sa,
                                   std::span<const CharT> s, std::span<IndexT> bucket_offsets,
                                   IndexT alphabet_size) {
    get_bucket_boundaries(s, bucket_offsets, alphabet_size, true);
    const IndexT n = static_cast<IndexT>(s.size());
    for (IndexT i = n - 1; i >= 0; --i) {
        const IndexT j = sa[i] - 1;
        if (j >= 0 && sais_types.is_s_type(static_cast<size_t>(j))) {
            sa[bucket_offsets[static_cast<size_t>(s[j])]--] = j;
        }
    }
}

/// \brief Core recursive SA-IS routine (Nong, Zhang & Chan, 2009).
/// \details Uses 64-bit signed integers (IndexT = int64_t) to support texts up to 3B+ bases without
/// overflow.
template <typename CharT, typename IndexT = int64_t>
void sais_core(std::span<const CharT> s, std::span<IndexT> sa, IndexT alphabet_size) {
    const IndexT n = static_cast<IndexT>(s.size());
    if (n == 0) {
        return;
    }
    if (n == 1) {
        sa[0] = 0;
        return;
    }

    // 1. Classify suffixes into L-type and S-type
    SaisTypeBitVector sais_types(n);
    sais_types.set_s_type(n - 1); // Sentinel $ is S-type
    for (IndexT i = n - 2; i >= 0; --i) {
        if (s[i] < s[i + 1]) {
            sais_types.set_s_type(i);
        } else if (s[i] > s[i + 1]) {
            sais_types.set_l_type(i);
        } else {
            if (sais_types.is_s_type(i + 1)) {
                sais_types.set_s_type(i);
            } else {
                sais_types.set_l_type(i);
            }
        }
    }

    // 2. Stage 1: Induced sort of LMS substrings
    std::vector<IndexT> bucket_offsets(alphabet_size);
    std::span<IndexT> bkt_span(bucket_offsets);
    get_bucket_boundaries<CharT, IndexT>(s, bkt_span, alphabet_size, true);
    std::fill(sa.begin(), sa.end(), -1);

    for (IndexT i = 1; i < n; ++i) {
        if (sais_types.is_lms(i)) {
            sa[bucket_offsets[static_cast<size_t>(s[i])]--] = i;
        }
    }

    induce_l_type_suffixes<CharT, IndexT>(sais_types, sa, s, bkt_span, alphabet_size);
    induce_s_type_suffixes<CharT, IndexT>(sais_types, sa, s, bkt_span, alphabet_size);

    // 3. Stage 2: Compact sorted LMS suffixes and name them
    IndexT lms_count = 0;
    for (IndexT i = 0; i < n; ++i) {
        if (sais_types.is_lms(sa[i])) {
            sa[lms_count++] = sa[i];
        }
    }
    std::fill(sa.begin() + lms_count, sa.end(), -1);

    IndexT name_id = 0;
    IndexT prev_pos = -1;
    for (IndexT i = 0; i < lms_count; ++i) {
        const IndexT pos = sa[i];
        bool is_different = false;
        if (prev_pos == -1) {
            is_different = true;
        } else {
            for (IndexT d = 0; d < n; ++d) {
                if (s[pos + d] != s[prev_pos + d] ||
                    sais_types.is_s_type(pos + d) != sais_types.is_s_type(prev_pos + d)) {
                    is_different = true;
                    break;
                } else if (d > 0 &&
                           (sais_types.is_lms(pos + d) || sais_types.is_lms(prev_pos + d))) {
                    break;
                }
            }
        }
        if (is_different) {
            name_id++;
            prev_pos = pos;
        }
        sa[lms_count + (pos / 2)] = name_id - 1;
    }

    // Compact names to the upper tail of sa
    for (IndexT i = n - 1, j = n - 1; i >= lms_count; --i) {
        if (sa[i] >= 0) {
            sa[j--] = sa[i];
        }
    }

    std::span<IndexT> reduced_text = sa.subspan(n - lms_count, lms_count);
    std::span<IndexT> summary_sa = sa.subspan(0, lms_count);

    // Solve reduced problem
    if (name_id == lms_count) {
        for (IndexT i = 0; i < lms_count; ++i) {
            summary_sa[reduced_text[i]] = i;
        }
    } else {
        sais_core<IndexT, IndexT>(reduced_text, summary_sa, name_id);
    }

    // 4. Stage 3: Map sorted LMS suffixes back and induce full SA
    IndexT j = 0;
    for (IndexT i = 1; i < n; ++i) {
        if (sais_types.is_lms(i)) {
            reduced_text[j++] = i;
        }
    }
    for (IndexT i = 0; i < lms_count; ++i) {
        summary_sa[i] = reduced_text[summary_sa[i]];
    }
    std::fill(sa.begin() + lms_count, sa.end(), -1);

    get_bucket_boundaries<CharT, IndexT>(s, bkt_span, alphabet_size, true);
    for (IndexT i = lms_count - 1; i >= 0; --i) {
        const IndexT p = sa[i];
        sa[i] = -1;
        sa[bucket_offsets[static_cast<size_t>(s[p])]--] = p;
    }

    induce_l_type_suffixes<CharT, IndexT>(sais_types, sa, s, bkt_span, alphabet_size);
    induce_s_type_suffixes<CharT, IndexT>(sais_types, sa, s, bkt_span, alphabet_size);
}

/// \brief Builds suffix array including the terminal sentinel $ at position n.
/// \return Suffix array of size n + 1 where sa[0] is always n (sentinel).
inline std::vector<size_t> build_suffix_array_with_sentinel(std::span<const uint8_t> text) {
    const size_t n = text.size();
    if (n == 0) {
        return {0};
    }

    std::vector<int64_t> s(n + 1);
    for (size_t i = 0; i < n; ++i) {
        s[i] = static_cast<int64_t>(text[i]) + 1;
    }
    s[n] = 0; // Sentinel $ strictly smaller than all 8-bit characters

    std::vector<int64_t> sa(n + 1);
    sais_core<int64_t, int64_t>(s, sa, 258);

    std::vector<size_t> result(n + 1);
    for (size_t i = 0; i <= n; ++i) {
        result[i] = static_cast<size_t>(sa[i]);
    }
    return result;
}

/// \brief Builds suffix array for input byte span with implicit terminal sentinel $.
/// \return Suffix array of text (indices in [0, n-1]).
inline std::vector<size_t> build_suffix_array(std::span<const uint8_t> text) {
    if (text.empty()) {
        return {};
    }
    auto sa_with_sentinel = build_suffix_array_with_sentinel(text);
    sa_with_sentinel.erase(sa_with_sentinel.begin()); // Drop sentinel suffix sa[0] == n
    return sa_with_sentinel;
}

} // namespace algoat::succinct::detail
