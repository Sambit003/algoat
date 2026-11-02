#pragma once

#include "algoat/succinct/rank_select.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace algoat::succinct {

template <std::unsigned_integral T = uint32_t> class WaveletMatrix {
public:
    static constexpr size_t kLevels = sizeof(T) * 8; // ceil(log2 sigma) for full range

    WaveletMatrix() = default;

    /// \brief Constructs a wavelet matrix from input sequence A[0..n).
    explicit WaveletMatrix(std::span<const T> sequence);

    WaveletMatrix(const WaveletMatrix&) = default;
    WaveletMatrix(WaveletMatrix&&) noexcept = default;
    WaveletMatrix& operator=(const WaveletMatrix&) = default;
    WaveletMatrix& operator=(WaveletMatrix&&) noexcept = default;
    ~WaveletMatrix() = default;

    /// \brief Returns A[i]. O(log sigma) time.
    [[nodiscard]] T access(size_t i) const noexcept;

    /// \brief Returns count of c in A[0..i). O(log sigma) time.
    [[nodiscard]] size_t rank(T c, size_t i) const noexcept;

    /// \brief Returns position of k-th (1-indexed) occurrence of c. O(log sigma) time.
    [[nodiscard]] size_t select(T c, size_t k) const noexcept;

    /// \brief Returns the k-th smallest value in A[l..r). O(log sigma) time.
    [[nodiscard]] T range_quantile(size_t l, size_t r, size_t k) const noexcept;

    /// \brief Counts values in [val_lo, val_hi] within positions [l, r). O(log sigma).
    [[nodiscard]] size_t range_count(size_t l, size_t r, T val_lo, T val_hi) const noexcept;

    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t sigma() const noexcept; // Maximum value + 1
    [[nodiscard]] size_t memory_usage_bytes() const noexcept;

private:
    size_t n_{0};
    size_t sigma_{0};
    std::array<RankSelectBitVector, kLevels> levels_;
    std::array<size_t, kLevels> mid_counts_{}; // midpoints for each level

    [[nodiscard]] std::pair<size_t, size_t> downward_interval(T c, size_t l,
                                                              size_t r) const noexcept;
    /// \brief Returns rank of zeros in level bv at position i.
    [[nodiscard]] static size_t rank0(const RankSelectBitVector& bv, size_t i) noexcept;
    [[nodiscard]] static size_t rank1(const RankSelectBitVector& bv, size_t i) noexcept;
    [[nodiscard]] size_t range_freq_less(size_t l, size_t r, T upper) const noexcept;
};

// ----------------------------------------------------------------------------
// Implementation Details
// ----------------------------------------------------------------------------

template <std::unsigned_integral T>
WaveletMatrix<T>::WaveletMatrix(std::span<const T> sequence) : n_(sequence.size()), sigma_(0) {
    if (n_ == 0) {
        for (size_t d = 0; d < kLevels; ++d) {
            levels_[d] = RankSelectBitVector();
            mid_counts_[d] = 0;
        }
        return;
    }

    T max_val = 0;
    for (T val : sequence) {
        if (val > max_val) {
            max_val = val;
        }
    }
    sigma_ = static_cast<size_t>(max_val) + 1;

    std::vector<T> curr(sequence.begin(), sequence.end());
    std::vector<T> next(n_);
    std::vector<uint64_t> words((n_ + 63) / 64, 0ULL);

    for (size_t d = 0; d < kLevels; ++d) {
        const size_t bit_pos = kLevels - 1 - d;
        std::fill(words.begin(), words.end(), 0ULL);

        size_t zeros = 0;
        for (size_t i = 0; i < n_; ++i) {
            if ((curr[i] >> bit_pos) & 1ULL) {
                words[i / 64] |= (1ULL << (i % 64));
            } else {
                ++zeros;
            }
        }

        mid_counts_[d] = zeros;
        levels_[d] = RankSelectBitVector(words, n_);

        // Stable bit-partition
        size_t zero_idx = 0;
        size_t one_idx = zeros;
        for (size_t i = 0; i < n_; ++i) {
            if (((curr[i] >> bit_pos) & 1ULL) == 0) {
                next[zero_idx++] = curr[i];
            } else {
                next[one_idx++] = curr[i];
            }
        }
        curr.swap(next);
    }
}

template <std::unsigned_integral T> T WaveletMatrix<T>::access(size_t i) const noexcept {
    assert(i < n_);
    T val = 0;
    size_t pos = i;
    for (size_t d = 0; d < kLevels; ++d) {
        val <<= 1;
        const bool bit = levels_[d].access(pos);
        if (bit) {
            val |= 1;
            pos = mid_counts_[d] + rank1(levels_[d], pos);
        } else {
            pos = rank0(levels_[d], pos);
        }
    }
    return val;
}

template <std::unsigned_integral T>
std::pair<size_t, size_t> WaveletMatrix<T>::downward_interval(T c, size_t l,
                                                              size_t r) const noexcept {
    for (size_t d = 0; d < kLevels; ++d) {
        const size_t bit_pos = kLevels - 1 - d;
        const uint64_t bit = (static_cast<uint64_t>(c) >> bit_pos) & 1ULL;
        if (bit) {
            l = mid_counts_[d] + rank1(levels_[d], l);
            r = mid_counts_[d] + rank1(levels_[d], r);
        } else {
            l = rank0(levels_[d], l);
            r = rank0(levels_[d], r);
        }
        if (l >= r) {
            return {r, r};
        }
    }
    return {l, r};
}

template <std::unsigned_integral T> size_t WaveletMatrix<T>::rank(T c, size_t i) const noexcept {
    assert(i <= n_);
    if (n_ == 0 || i == 0) {
        return 0;
    }
    if (sigma_ > 0 && static_cast<size_t>(c) >= sigma_) {
        return 0;
    }

    auto [l, r] = downward_interval(c, 0, i);
    return r - l;
}

template <std::unsigned_integral T> size_t WaveletMatrix<T>::select(T c, size_t k) const noexcept {
    if (k == 0 || n_ == 0) {
        return n_;
    }
    if (sigma_ > 0 && static_cast<size_t>(c) >= sigma_) {
        return n_;
    }

    // Downward pass to find leaf interval [l, r)
    auto [l, r] = downward_interval(c, 0, n_);
    const size_t total_c = r - l;
    if (k > total_c) {
        return n_;
    }

    // Upward pass from leaf position
    size_t pos = l + k - 1;
    for (size_t step = 0; step < kLevels; ++step) {
        const size_t d = kLevels - 1 - step;
        const size_t bit_pos = kLevels - 1 - d;
        const uint64_t bit = (static_cast<uint64_t>(c) >> bit_pos) & 1ULL;
        if (bit) {
            pos = levels_[d].select1(pos - mid_counts_[d] + 1);
        } else {
            pos = levels_[d].select0(pos + 1);
        }
    }
    return pos;
}

template <std::unsigned_integral T>
T WaveletMatrix<T>::range_quantile(size_t l, size_t r, size_t k) const noexcept {
    assert(l < r && r <= n_);
    assert(k < r - l);
    if (l >= r || k >= r - l || l >= n_) [[unlikely]] {
        return 0;
    }

    T res = 0;
    for (size_t d = 0; d < kLevels; ++d) {
        const size_t l0 = rank0(levels_[d], l);
        const size_t r0 = rank0(levels_[d], r);
        const size_t zeros = r0 - l0;
        if (k < zeros) {
            l = l0;
            r = r0;
        } else {
            res |= (static_cast<T>(1) << (kLevels - 1 - d));
            k -= zeros;
            l = mid_counts_[d] + (l - l0);
            r = mid_counts_[d] + (r - r0);
        }
    }
    return res;
}

template <std::unsigned_integral T>
size_t WaveletMatrix<T>::range_freq_less(size_t l, size_t r, T upper) const noexcept {
    if (l >= r || n_ == 0 || upper == 0) {
        return 0;
    }
    if (sigma_ > 0 && static_cast<size_t>(upper) >= sigma_) {
        return r - l;
    }

    size_t count = 0;
    for (size_t d = 0; d < kLevels; ++d) {
        if (l >= r) {
            break;
        }
        const size_t bit_pos = kLevels - 1 - d;
        const uint64_t bit = (static_cast<uint64_t>(upper) >> bit_pos) & 1ULL;
        const size_t l0 = rank0(levels_[d], l);
        const size_t r0 = rank0(levels_[d], r);
        const size_t zeros = r0 - l0;
        if (bit) {
            count += zeros;
            l = mid_counts_[d] + (l - l0);
            r = mid_counts_[d] + (r - r0);
        } else {
            l = l0;
            r = r0;
        }
    }
    return count;
}

template <std::unsigned_integral T>
size_t WaveletMatrix<T>::range_count(size_t l, size_t r, T val_lo, T val_hi) const noexcept {
    if (l >= r || val_lo > val_hi || l >= n_ || n_ == 0) {
        return 0;
    }
    assert(r <= n_);
    r = std::min(r, n_);
    if (sigma_ > 0 && static_cast<size_t>(val_lo) >= sigma_) {
        return 0;
    }

    const size_t hi_cnt = (val_hi == std::numeric_limits<T>::max() ||
                           (sigma_ > 0 && static_cast<size_t>(val_hi) + 1 >= sigma_))
                              ? (r - l)
                              : range_freq_less(l, r, static_cast<T>(val_hi + 1));
    const size_t lo_cnt = (val_lo == 0) ? 0 : range_freq_less(l, r, val_lo);
    return (hi_cnt >= lo_cnt) ? (hi_cnt - lo_cnt) : 0;
}

template <std::unsigned_integral T> bool WaveletMatrix<T>::empty() const noexcept {
    return n_ == 0;
}

template <std::unsigned_integral T> size_t WaveletMatrix<T>::size() const noexcept {
    return n_;
}

template <std::unsigned_integral T> size_t WaveletMatrix<T>::sigma() const noexcept {
    return sigma_;
}

template <std::unsigned_integral T> size_t WaveletMatrix<T>::memory_usage_bytes() const noexcept {
    size_t total = sizeof(*this);
    for (const auto& bv : levels_) {
        total += bv.total_index_bytes() + ((n_ + 63) / 64) * sizeof(uint64_t);
    }
    return total;
}

template <std::unsigned_integral T>
size_t WaveletMatrix<T>::rank0(const RankSelectBitVector& bv, size_t i) noexcept {
    return bv.rank0(i);
}

template <std::unsigned_integral T>
size_t WaveletMatrix<T>::rank1(const RankSelectBitVector& bv, size_t i) noexcept {
    return bv.rank1(i);
}

// Explicit instantiations provided in the library:
extern template class WaveletMatrix<uint8_t>;
extern template class WaveletMatrix<uint16_t>;
extern template class WaveletMatrix<uint32_t>;

} // namespace algoat::succinct
