#pragma once

#include "algoat/succinct/detail/sais.hpp"
#include "algoat/succinct/rank_select.hpp"
#include "algoat/succinct/wavelet_matrix.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace algoat::succinct {

/// \brief Configuration options for FM-Index construction.
struct FMIndexConfig {
    size_t sa_sample_rate = 32; ///< Suffix array sample rate (smaller = faster locate, more memory)
    bool build_isa_samples = true; ///< Build inverted suffix array for extract()
};

/// \brief Production-grade compressed FM-Index with Burrows-Wheeler Transform.
/// \details Supports exact pattern matching (count, locate, backward_search) and
///          text substring decompression (extract) in compressed space using Wavelet Matrix Occ
///          table. Supports arbitrary 8-bit byte sequences (including 0x00) and DNA/protein
///          alphabets.
class FMIndex {
public:
    FMIndex() = default;

    /// \brief Builds FM-Index from text string (DNA, ASCII, or arbitrary byte sequences).
    explicit FMIndex(std::string_view text, FMIndexConfig cfg = {}) {
        build_bwt_and_sa(
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(text.data()), text.size()),
            cfg);
    }

    /// \brief Builds from raw byte span.
    explicit FMIndex(std::span<const uint8_t> text, FMIndexConfig cfg = {}) {
        build_bwt_and_sa(text, cfg);
    }

    FMIndex(const FMIndex&) = default;
    FMIndex(FMIndex&&) noexcept = default;
    FMIndex& operator=(const FMIndex&) = default;
    FMIndex& operator=(FMIndex&&) noexcept = default;
    ~FMIndex() = default;

    /// \brief Counts occurrences of pattern P in the indexed text. O(|P| log sigma) time.
    [[nodiscard]] size_t count(std::string_view pattern) const noexcept {
        auto [lo, hi] = backward_search(pattern);
        return (hi > lo) ? (hi - lo) : 0;
    }

    /// \brief Returns all positions where pattern P occurs. O(|P| log sigma + occ * sa_sample_rate)
    /// time.
    [[nodiscard]] std::vector<size_t> locate(std::string_view pattern) const {
        auto [lo, hi] = backward_search(pattern);
        if (lo >= hi || n_ == 0) {
            return {};
        }

        std::vector<size_t> positions;
        positions.reserve(hi - lo);

        for (size_t r = lo; r < hi; ++r) {
            size_t curr = r;
            size_t steps = 0;
            while (!sa_sample_marks_.access(curr)) {
                curr = lf_map(curr);
                ++steps;
            }
            size_t sample_idx = sa_sample_marks_.rank1(curr + 1) - 1;
            size_t pos = (sa_samples_[sample_idx] + steps) % (n_ + 1);
            if (pos < n_) {
                positions.push_back(pos);
            }
        }
        std::sort(positions.begin(), positions.end());
        return positions;
    }

    /// \brief Extracts substring T[pos, pos+len). O(len * sa_sample_rate) time.
    [[nodiscard]] std::string extract(size_t pos, size_t len) const {
        if (len == 0 || n_ == 0 || pos >= n_) {
            return "";
        }
        len = std::min(len, n_ - pos);
        const size_t end = pos + len;

        // Resolve starting row corresponding to suffix T[end..n]
        size_t curr = get_row_for_pos(end);

        std::string result(len, '\0');
        for (size_t i = len; i > 0; --i) {
            // Preceding character in T at suffix position
            assert(curr != sentinel_row_);
            const size_t wm_row = (curr < sentinel_row_) ? curr : (curr - 1);
            uint8_t c = bwt_wm_.access(wm_row);
            result[i - 1] = static_cast<char>(c);
            curr = lf_map(curr);
        }
        return result;
    }

    /// \brief Returns the [lo, hi) half-open BWT interval for pattern P in full suffix range [0,
    /// n+1).
    [[nodiscard]] std::pair<size_t, size_t>
    backward_search(std::string_view pattern) const noexcept {
        if (pattern.empty() || n_ == 0) {
            return {0, 0};
        }

        size_t lo = 0;
        size_t hi = n_ + 1;

        for (size_t k = pattern.size(); k > 0; --k) {
            const uint8_t c = static_cast<uint8_t>(pattern[k - 1]);
            lo = C_[c] + occ(c, lo);
            hi = C_[c] + occ(c, hi);
            if (lo >= hi) {
                return {lo, lo};
            }
        }
        return {lo, hi};
    }

    [[nodiscard]] size_t text_length() const noexcept {
        return n_;
    }

    [[nodiscard]] size_t memory_usage_bytes() const noexcept {
        return sizeof(*this) + bwt_wm_.memory_usage_bytes() + sa_sample_marks_.total_index_bytes() +
               (sa_samples_.size() * sizeof(uint64_t)) + (isa_samples_.size() * sizeof(uint64_t));
    }

    /// \brief Serialize to portable flat binary format.
    [[nodiscard]] std::vector<uint8_t> serialize() const {
        static constexpr uint64_t kMagic = 0x5844494D46544741ULL; // "AGTOFMIDX"
        static constexpr uint32_t kVersion = 2;

        std::vector<uint8_t> out;
        const size_t bwt_len = bwt_wm_.size();
        const size_t header_size = sizeof(uint64_t) * 5 + sizeof(C_) + sizeof(uint64_t) * 2;
        const size_t total_estimate =
            header_size + bwt_len + (sa_samples_.size() + isa_samples_.size()) * sizeof(uint64_t);
        out.reserve(total_estimate);

        auto append_bytes = [&out](const void* src, size_t size) {
            const auto* p = reinterpret_cast<const uint8_t*>(src);
            out.insert(out.end(), p, p + size);
        };

        const uint64_t magic = kMagic;
        const uint64_t ver = kVersion;
        const uint64_t n = n_;
        const uint64_t rate = sa_sample_rate_;
        const uint64_t sentinel_row = sentinel_row_;
        append_bytes(&magic, sizeof(magic));
        append_bytes(&ver, sizeof(ver));
        append_bytes(&n, sizeof(n));
        append_bytes(&rate, sizeof(rate));
        append_bytes(&sentinel_row, sizeof(sentinel_row));
        append_bytes(C_.data(), sizeof(C_));

        // BWT payload
        const uint64_t bwt_size = bwt_len;
        append_bytes(&bwt_size, sizeof(bwt_size));
        for (size_t i = 0; i < bwt_len; ++i) {
            uint8_t c = bwt_wm_.access(i);
            out.push_back(c);
        }

        // sa_samples
        const uint64_t sa_cnt = sa_samples_.size();
        append_bytes(&sa_cnt, sizeof(sa_cnt));
        if (sa_cnt > 0) {
            append_bytes(sa_samples_.data(), sa_cnt * sizeof(uint64_t));
        }

        // sa_sample_marks_ words (for total_rows = n_ + 1)
        const size_t total_rows = n_ + 1;
        const uint64_t num_mark_words = (total_rows + 63) / 64;
        std::vector<uint64_t> mark_words(num_mark_words, 0ULL);
        for (size_t i = 0; i < total_rows; ++i) {
            if (sa_sample_marks_.access(i)) {
                mark_words[i / 64] |= (1ULL << (i % 64));
            }
        }
        append_bytes(&num_mark_words, sizeof(num_mark_words));
        if (num_mark_words > 0) {
            append_bytes(mark_words.data(), num_mark_words * sizeof(uint64_t));
        }

        // isa_samples
        const uint64_t isa_cnt = isa_samples_.size();
        append_bytes(&isa_cnt, sizeof(isa_cnt));
        if (isa_cnt > 0) {
            append_bytes(isa_samples_.data(), isa_cnt * sizeof(uint64_t));
        }

        return out;
    }

    /// \brief Deserialize FMIndex from portable flat binary buffer.
    static FMIndex deserialize(std::span<const uint8_t> data) {
        static constexpr uint64_t kMagic = 0x5844494D46544741ULL;
        static constexpr uint32_t kVersion = 2;

        if (data.size() < sizeof(uint64_t) * 5 + sizeof(std::array<uint64_t, 256>)) {
            throw std::invalid_argument("Serialized FMIndex buffer smaller than header");
        }

        size_t offset = 0;
        auto read_bytes = [&data, &offset](void* dst, size_t size) {
            if (offset + size > data.size()) {
                throw std::invalid_argument("Buffer underflow during FMIndex deserialization");
            }
            std::memcpy(dst, data.data() + offset, size);
            offset += size;
        };

        uint64_t magic = 0;
        uint64_t ver = 0;
        uint64_t n = 0;
        uint64_t rate = 0;
        uint64_t sentinel_row = 0;
        read_bytes(&magic, sizeof(magic));
        read_bytes(&ver, sizeof(ver));
        if (magic != kMagic || ver != kVersion) {
            throw std::invalid_argument("Invalid FMIndex serialization magic or version");
        }
        read_bytes(&n, sizeof(n));
        read_bytes(&rate, sizeof(rate));
        read_bytes(&sentinel_row, sizeof(sentinel_row));

        FMIndex idx;
        idx.n_ = n;
        idx.sa_sample_rate_ = rate;
        idx.sentinel_row_ = sentinel_row;
        read_bytes(idx.C_.data(), sizeof(idx.C_));

        if (n == 0) {
            return idx;
        }

        uint64_t bwt_len = 0;
        read_bytes(&bwt_len, sizeof(bwt_len));
        if (offset + bwt_len > data.size()) {
            throw std::invalid_argument("Truncated BWT payload in FMIndex");
        }
        std::span<const uint8_t> bwt_span{data.data() + offset, static_cast<size_t>(bwt_len)};
        offset += bwt_len;
        idx.bwt_wm_ = WaveletMatrix<uint8_t>(bwt_span);

        uint64_t sa_cnt = 0;
        read_bytes(&sa_cnt, sizeof(sa_cnt));
        if (sa_cnt > 0) {
            idx.sa_samples_.resize(sa_cnt);
            read_bytes(idx.sa_samples_.data(), sa_cnt * sizeof(uint64_t));
        }

        uint64_t num_mark_words = 0;
        read_bytes(&num_mark_words, sizeof(num_mark_words));
        if (num_mark_words > 0) {
            std::vector<uint64_t> mark_words(num_mark_words);
            read_bytes(mark_words.data(), num_mark_words * sizeof(uint64_t));
            idx.sa_sample_marks_ = RankSelectBitVector(mark_words, n + 1);
        }

        uint64_t isa_cnt = 0;
        read_bytes(&isa_cnt, sizeof(isa_cnt));
        if (isa_cnt > 0) {
            idx.isa_samples_.resize(isa_cnt);
            read_bytes(idx.isa_samples_.data(), isa_cnt * sizeof(uint64_t));
        }

        return idx;
    }

private:
    size_t n_{0};
    size_t sa_sample_rate_{32};

    // BWT of text as 8-bit alphabet (omitting sentinel row, exactly n_ entries)
    WaveletMatrix<uint8_t> bwt_wm_;

    // C[c] = number of suffixes in T$ strictly smaller than character c
    std::array<uint64_t, 256> C_{};

    // Sampled suffix array (every sa_sample_rate-th SA value)
    std::vector<uint64_t> sa_samples_;

    // Suffix array sample bitvector: marks sampled positions in full BWT order (size n_ + 1)
    RankSelectBitVector sa_sample_marks_;

    // Inverted sampled SA for fast extract()
    std::vector<uint64_t> isa_samples_;

    // Suffix array index where SA[sentinel_row_] == 0 (preceding symbol is $)
    size_t sentinel_row_{0};

    /// \brief Occ(c, r): number of occurrences of byte c in BWT[0..r) across full [0, n+1) range.
    [[nodiscard]] inline size_t occ(uint8_t c, size_t r) const noexcept {
        const size_t wm_r = (r <= sentinel_row_) ? r : (r - 1);
        return bwt_wm_.rank(c, wm_r);
    }

    /// \brief LF-mapping from row r to row LF(r).
    [[nodiscard]] inline size_t lf_map(size_t r) const noexcept {
        if (r == sentinel_row_) {
            return 0; // Preceding character is $, mapped to row 0
        }
        const size_t wm_r = (r < sentinel_row_) ? r : (r - 1);
        const uint8_t c = bwt_wm_.access(wm_r);
        return C_[c] + occ(c, r);
    }

    /// \brief Resolves BWT row corresponding to suffix starting at text position target_pos.
    [[nodiscard]] size_t get_row_for_pos(size_t target_pos) const {
        if (target_pos == n_) {
            return 0; // Suffix $ is always at row 0
        }
        size_t curr = 0;
        size_t start_pos = n_;

        if (!isa_samples_.empty() && sa_sample_rate_ > 0) {
            const size_t k = (target_pos + sa_sample_rate_ - 1) / sa_sample_rate_;
            if (k < isa_samples_.size()) {
                start_pos = k * sa_sample_rate_;
                curr = isa_samples_[k];
            }
        }

        const size_t gap = start_pos - target_pos;
        for (size_t step = 0; step < gap; ++step) {
            curr = lf_map(curr);
        }
        return curr;
    }

    void build_bwt_and_sa(std::span<const uint8_t> text, FMIndexConfig cfg) {
        n_ = text.size();
        sa_sample_rate_ = std::max<size_t>(1, cfg.sa_sample_rate);
        C_.fill(0);
        sa_samples_.clear();
        isa_samples_.clear();

        if (n_ == 0) {
            bwt_wm_ = WaveletMatrix<uint8_t>();
            sa_sample_marks_ = RankSelectBitVector();
            return;
        }

        const size_t total_rows = n_ + 1;
        std::vector<size_t> sa = detail::build_suffix_array_with_sentinel(text);

        // Populate BWT string of length n_ (skipping sentinel row)
        std::vector<uint8_t> bwt;
        bwt.reserve(n_);
        sentinel_row_ = 0;
        for (size_t i = 0; i < total_rows; ++i) {
            if (sa[i] == 0) {
                sentinel_row_ = i;
            } else {
                bwt.push_back(text[sa[i] - 1]);
            }
        }

        bwt_wm_ = WaveletMatrix<uint8_t>(bwt);

        // Compute C table: C[c] = 1 (sentinel $) + count of all characters in text strictly < c
        std::array<uint64_t, 256> counts{};
        for (size_t i = 0; i < n_; ++i) {
            counts[bwt[i]]++;
        }
        C_[0] = 1; // Suffix $ is strictly smaller than any character
        for (size_t c = 1; c < 256; ++c) {
            C_[c] = C_[c - 1] + counts[c - 1];
        }

        // Build SA samples and marks across all total_rows = n_ + 1 rows
        std::vector<bool> marks(total_rows, false);
        for (size_t i = 0; i < total_rows; ++i) {
            if (sa[i] % sa_sample_rate_ == 0 || sa[i] == n_) {
                marks[i] = true;
                sa_samples_.push_back(sa[i]);
            }
        }
        sa_sample_marks_ = RankSelectBitVector(marks);

        // Build ISA samples if requested
        if (cfg.build_isa_samples && sa_sample_rate_ > 0) {
            const size_t num_isa = (n_ + sa_sample_rate_) / sa_sample_rate_;
            isa_samples_.resize(num_isa, 0);
            for (size_t i = 0; i < total_rows; ++i) {
                if (sa[i] % sa_sample_rate_ == 0) {
                    const size_t k = sa[i] / sa_sample_rate_;
                    if (k < num_isa) {
                        isa_samples_[k] = i;
                    }
                }
            }
        }
    }
};

} // namespace algoat::succinct
