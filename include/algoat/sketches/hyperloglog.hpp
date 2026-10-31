#pragma once

#include "algoat/sketches/detail/hash.hpp"
#include "algoat/sketches/detail/hyperloglog_tables.hpp"
#include "algoat/sketches/detail/simd_hll.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace algoat::sketches {

/**
 * @brief Industrial-strength HyperLogLog++ Cardinality Estimator (Heule et al. 2013).
 *
 * Implements Google's HyperLogLog++ algorithm with 64-bit hashing, AVX2 SIMD acceleration,
 * dual-representation sparse mode (p' = 25) for high precision at small cardinalities,
 * 6-NN empirical bias correction for small to medium sets, and Redis wire-format serialization.
 *
 * @tparam P Dense precision parameter in the closed interval [4, 18].
 *           Register count m = 2^P.
 */
template <std::size_t P = 14>
    requires(P >= 4 && P <= 18)
class HyperLogLog {
public:
    static constexpr std::size_t precision = P;
    static constexpr std::size_t register_count = 1ULL << P;
    static constexpr double expected_relative_error = 1.04 / (1ULL << (P / 2)); // 1.04 / sqrt(m)

    enum class Mode : uint8_t { Sparse, Dense };

    HyperLogLog() noexcept : mode_(Mode::Sparse) {
        sparse_list_.reserve(128);
        tmp_set_.reserve(64);
    }

    /// Add an item of any hashable type
    template <typename T> void add(const T& item) noexcept {
        add_hash(detail::hash_item(item));
    }

    /// Add a precomputed 64-bit hash
    void add_hash(uint64_t hash_value) noexcept {
        cached_count_.reset();
        if (mode_ == Mode::Dense) {
            add_hash_dense(hash_value);
        } else {
            add_hash_sparse(hash_value);
        }
    }

    /// Compute estimated cardinality
    [[nodiscard]] double estimate() const noexcept {
        if (mode_ == Mode::Sparse) {
            flush_sparse_buffer();
            if (sparse_list_.empty()) {
                return 0.0;
            }
            constexpr double m_prime = static_cast<double>(1ULL << SPARSE_PRECISION);
            const double v_prime = m_prime - static_cast<double>(sparse_list_.size());
            if (v_prime <= 0.0) {
                return static_cast<double>(sparse_list_.size());
            }
            return m_prime * std::log(m_prime / v_prime);
        }

        constexpr std::size_t m = register_count;
        const double harmonic_sum = detail::simd_harmonic_sum(registers_.data(), m);
        const double raw_estimate =
            detail::alpha_m(m) * static_cast<double>(m) * static_cast<double>(m) / harmonic_sum;

        double corrected_estimate = raw_estimate;
        if (raw_estimate <= 5.0 * static_cast<double>(m)) {
            corrected_estimate = raw_estimate - detail::estimate_bias(raw_estimate, P);
            if (corrected_estimate < 0.0) {
                corrected_estimate = 0.0;
            }
        }

        const std::size_t zero_registers = detail::simd_count_zeros(registers_.data(), m);
        double linear_counting_estimate = corrected_estimate;
        if (zero_registers > 0) {
            linear_counting_estimate =
                static_cast<double>(m) *
                std::log(static_cast<double>(m) / static_cast<double>(zero_registers));
        }

        if (zero_registers > 0 && linear_counting_estimate <= detail::THRESHOLDS[P - 4]) {
            return linear_counting_estimate;
        }
        return corrected_estimate;
    }

    /// Return integer cardinality estimate rounded to nearest uint64_t
    [[nodiscard]] uint64_t count() const noexcept {
        if (cached_count_.has_value()) {
            return *cached_count_;
        }
        const double est = estimate();
        if (est <= 0.0) {
            cached_count_ = 0ULL;
            return 0ULL;
        }
        const uint64_t c = static_cast<uint64_t>(std::llround(est));
        cached_count_ = c;
        return c;
    }

    /// Merge another HyperLogLog sketch into this one
    void merge(const HyperLogLog& other) {
        if (this == &other) {
            return;
        }
        cached_count_.reset();

        if (mode_ == Mode::Dense && other.mode_ == Mode::Dense) {
            detail::simd_dense_merge(registers_.data(), other.registers_.data(), register_count);
            return;
        }

        if (mode_ == Mode::Sparse && other.mode_ == Mode::Dense) {
            promote_to_dense();
            detail::simd_dense_merge(registers_.data(), other.registers_.data(), register_count);
            return;
        }

        if (mode_ == Mode::Dense && other.mode_ == Mode::Sparse) {
            other.flush_sparse_buffer();
            for (uint32_t k : other.sparse_list_) {
                auto [idx, r] = decode_sparse_key(k);
                if (idx < register_count && r > registers_[idx]) {
                    registers_[idx] = r;
                }
            }
            return;
        }

        // Both sketches are sparse
        flush_sparse_buffer();
        other.flush_sparse_buffer();

        sparse_list_ = merge_sorted_sparse(sparse_list_, other.sparse_list_);
        check_sparse_promotion();
    }

    /// Serialize sketch to byte vector (Redis wire format compatible when P == 14)
    [[nodiscard]] std::vector<uint8_t> serialize() const {
        constexpr std::size_t dense_payload_size = (register_count * 6 + 7) / 8;
        constexpr std::size_t total_size = 16 + dense_payload_size;

        std::vector<uint8_t> bytes(total_size, 0);

        // Header (struct hllhdr, 16 bytes)
        bytes[0] = 'H';
        bytes[1] = 'Y';
        bytes[2] = 'L';
        bytes[3] = 'L';
        bytes[4] = 0; // HLL_DENSE
        // notused[3]: bytes 5..7 must be 0 for Redis wire compatibility
        bytes[5] = 0;
        bytes[6] = 0;
        bytes[7] = 0;

        // Cached cardinality (LE uint64_t with valid cache bit 0)
        const uint64_t card = count() & 0x7FFFFFFFFFFFFFFFULL;
        for (std::size_t i = 0; i < 8; ++i) {
            bytes[8 + i] = static_cast<uint8_t>((card >> (i * 8)) & 0xFF);
        }

        // Retrieve effective dense registers without mutating const this
        std::span<const uint8_t> reg_span;
        std::vector<uint8_t> temp_regs;
        if (mode_ == Mode::Dense) {
            reg_span = registers_;
        } else {
            flush_sparse_buffer();
            temp_regs.assign(register_count, 0);
            for (uint32_t k : sparse_list_) {
                auto [idx, r] = decode_sparse_key(k);
                if (idx < register_count && r > temp_regs[idx]) {
                    temp_regs[idx] = r;
                }
            }
            reg_span = temp_regs;
        }

        // Dense register packing (6 bits per register, packed LSB-to-MSB)
        const std::size_t num_quads = register_count / 4;
        for (std::size_t i = 0; i < num_quads; ++i) {
            uint8_t r0 = reg_span[4 * i] & 0x3F;
            uint8_t r1 = reg_span[4 * i + 1] & 0x3F;
            uint8_t r2 = reg_span[4 * i + 2] & 0x3F;
            uint8_t r3 = reg_span[4 * i + 3] & 0x3F;

            bytes[16 + 3 * i] = static_cast<uint8_t>(r0 | (r1 << 6));
            bytes[16 + 3 * i + 1] = static_cast<uint8_t>((r1 >> 2) | (r2 << 4));
            bytes[16 + 3 * i + 2] = static_cast<uint8_t>((r2 >> 4) | (r3 << 2));
        }

        return bytes;
    }

    /// Deserialize sketch from byte span
    static HyperLogLog deserialize(std::span<const uint8_t> bytes) {
        if (bytes.size() < 16) {
            throw std::invalid_argument("HyperLogLog deserialize: buffer too small for header");
        }

        if (bytes[0] != 'H' || bytes[1] != 'Y' || bytes[2] != 'L' || bytes[3] != 'L') {
            throw std::invalid_argument("HyperLogLog deserialize: invalid magic bytes");
        }

        uint8_t encoding = bytes[4];
        if (encoding > 1) {
            throw std::invalid_argument("HyperLogLog deserialize: unsupported encoding");
        }

        constexpr std::size_t dense_payload_size = (register_count * 6 + 7) / 8;
        constexpr std::size_t expected_dense_size = 16 + dense_payload_size;

        HyperLogLog hll;
        hll.promote_to_dense();

        if (encoding == 0) { // Dense
            if (bytes.size() != expected_dense_size) {
                throw std::invalid_argument("HyperLogLog deserialize: invalid dense payload size");
            }

            const std::size_t num_quads = register_count / 4;
            for (std::size_t i = 0; i < num_quads; ++i) {
                uint8_t b0 = bytes[16 + 3 * i];
                uint8_t b1 = bytes[16 + 3 * i + 1];
                uint8_t b2 = bytes[16 + 3 * i + 2];

                hll.registers_[4 * i] = b0 & 0x3F;
                hll.registers_[4 * i + 1] = ((b0 >> 6) | (b1 << 2)) & 0x3F;
                hll.registers_[4 * i + 2] = ((b1 >> 4) | (b2 << 4)) & 0x3F;
                hll.registers_[4 * i + 3] = (b2 >> 2) & 0x3F;
            }
        } else { // Sparse RLE
            std::size_t reg_idx = 0;
            std::size_t offset = 16;
            while (offset < bytes.size() && reg_idx < register_count) {
                uint8_t b0 = bytes[offset++];
                if ((b0 & 0xC0) == 0) { // ZERO
                    std::size_t count = (b0 & 0x3F) + 1;
                    reg_idx += count;
                } else if ((b0 & 0xC0) == 0x40) { // XZERO
                    if (offset >= bytes.size()) {
                        throw std::invalid_argument("HyperLogLog deserialize: truncated XZERO");
                    }
                    uint8_t b1 = bytes[offset++];
                    std::size_t count = (((b0 & 0x3F) << 8) | b1) + 1;
                    reg_idx += count;
                } else { // VAL
                    uint8_t val = ((b0 >> 2) & 0x1F) + 1;
                    std::size_t count = (b0 & 0x03) + 1;
                    for (std::size_t c = 0; c < count && reg_idx < register_count; ++c) {
                        hll.registers_[reg_idx++] = val;
                    }
                }
            }
        }

        if ((bytes[15] & 0x80) == 0) {
            uint64_t card = 0;
            for (std::size_t i = 0; i < 8; ++i) {
                card |= (static_cast<uint64_t>(bytes[8 + i]) << (i * 8));
            }
            hll.cached_count_ = card & 0x7FFFFFFFFFFFFFFFULL;
        }

        return hll;
    }

    [[nodiscard]] Mode mode() const noexcept {
        return mode_;
    }
    [[nodiscard]] bool is_sparse() const noexcept {
        return mode_ == Mode::Sparse;
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const noexcept {
        if (mode_ == Mode::Dense) {
            return sizeof(*this) + registers_.capacity() * sizeof(uint8_t);
        }
        return sizeof(*this) + (sparse_list_.capacity() + tmp_set_.capacity()) * sizeof(uint32_t);
    }

    void clear() noexcept {
        cached_count_.reset();
        mode_ = Mode::Sparse;
        sparse_list_.clear();
        tmp_set_.clear();
        registers_.clear();
    }

private:
    static constexpr std::size_t SPARSE_PRECISION = 25; // p' = 25

    Mode mode_{Mode::Sparse};
    mutable std::optional<uint64_t> cached_count_;
    mutable std::vector<uint32_t> sparse_list_;
    mutable std::vector<uint32_t> tmp_set_;
    std::vector<uint8_t> registers_;

    void add_hash_dense(uint64_t hash_value) noexcept {
        const std::size_t idx = static_cast<std::size_t>(hash_value >> (64 - P));
        const uint64_t w = (hash_value << P) | (1ULL << (P - 1));
        const uint8_t rho = static_cast<uint8_t>(std::countl_zero(w) + 1);

        if (rho > registers_[idx]) {
            registers_[idx] = rho;
        }
    }

    void add_hash_sparse(uint64_t hash_value) noexcept {
        uint32_t k = encode_hash(hash_value);
        tmp_set_.push_back(k);

        if (tmp_set_.size() >= 1024) {
            flush_sparse_buffer();
            check_sparse_promotion();
        }
    }

    [[nodiscard]] uint32_t encode_hash(uint64_t x) const noexcept {
        const uint32_t idx_prime = static_cast<uint32_t>(x >> (64 - SPARSE_PRECISION));
        constexpr std::size_t shift_slice = 64 - SPARSE_PRECISION;
        constexpr uint64_t slice_mask = ((1ULL << (SPARSE_PRECISION - P)) - 1);
        const uint64_t slice = (x >> shift_slice) & slice_mask;

        if (slice == 0) {
            const uint64_t w_prime = (x << SPARSE_PRECISION) | (1ULL << (SPARSE_PRECISION - 1));
            const uint8_t rho_prime = static_cast<uint8_t>(std::countl_zero(w_prime) + 1);
            return (idx_prime << 7) | ((rho_prime & 0x3F) << 1) | 1U;
        }
        return (idx_prime << 1) | 0U;
    }

    [[nodiscard]] std::pair<std::size_t, uint8_t> decode_sparse_key(uint32_t k) const noexcept {
        if ((k & 1) == 1) {
            const std::size_t idx = static_cast<std::size_t>((k >> 7) >> (SPARSE_PRECISION - P));
            const uint8_t rho = static_cast<uint8_t>(((k >> 1) & 0x3F) + (SPARSE_PRECISION - P));
            return {idx, rho};
        }
        const std::size_t idx = static_cast<std::size_t>((k >> 1) >> (SPARSE_PRECISION - P));
        constexpr std::size_t slice_bits = SPARSE_PRECISION - P;
        const uint32_t slice = (k >> 1) & ((1U << slice_bits) - 1);
        const int lz = std::countl_zero(slice) - static_cast<int>(32 - slice_bits);
        const uint8_t rho = static_cast<uint8_t>(lz + 1);
        return {idx, rho};
    }

    [[nodiscard]] static uint32_t extract_sparse_index(uint32_t k) noexcept {
        return (k & 1) ? (k >> 7) : (k >> 1);
    }

    [[nodiscard]] static std::vector<uint32_t> merge_sorted_sparse(std::span<const uint32_t> a,
                                                                   std::span<const uint32_t> b) {
        std::vector<uint32_t> merged;
        merged.reserve(a.size() + b.size());

        std::size_t i = 0;
        std::size_t j = 0;
        while (i < a.size() && j < b.size()) {
            uint32_t ki = a[i];
            uint32_t kj = b[j];
            uint32_t idxi = extract_sparse_index(ki);
            uint32_t idxj = extract_sparse_index(kj);

            if (idxi < idxj) {
                merged.push_back(ki);
                ++i;
            } else if (idxj < idxi) {
                merged.push_back(kj);
                ++j;
            } else {
                merged.push_back(std::max(ki, kj));
                ++i;
                ++j;
            }
        }
        while (i < a.size()) {
            merged.push_back(a[i++]);
        }
        while (j < b.size()) {
            merged.push_back(b[j++]);
        }
        return merged;
    }

    void flush_sparse_buffer() const noexcept {
        if (tmp_set_.empty()) {
            return;
        }

        std::ranges::sort(tmp_set_);

        std::vector<uint32_t> dedup_tmp;
        dedup_tmp.reserve(tmp_set_.size());
        for (uint32_t k : tmp_set_) {
            if (!dedup_tmp.empty() &&
                extract_sparse_index(dedup_tmp.back()) == extract_sparse_index(k)) {
                dedup_tmp.back() = std::max(dedup_tmp.back(), k);
            } else {
                dedup_tmp.push_back(k);
            }
        }

        sparse_list_ = merge_sorted_sparse(sparse_list_, dedup_tmp);
        tmp_set_.clear();
    }

    void check_sparse_promotion() noexcept {
        // Threshold: 6 * m bits = (3 * m) / 4 bytes
        constexpr std::size_t memory_limit_bytes = (3 * register_count) / 4;
        if (sparse_list_.size() * sizeof(uint32_t) > memory_limit_bytes) {
            promote_to_dense();
        }
    }

    void promote_to_dense() noexcept {
        flush_sparse_buffer();
        registers_.assign(register_count, 0);

        for (uint32_t k : sparse_list_) {
            auto [idx, r] = decode_sparse_key(k);
            if (idx < register_count && r > registers_[idx]) {
                registers_[idx] = r;
            }
        }

        sparse_list_.clear();
        sparse_list_.shrink_to_fit();
        tmp_set_.clear();
        tmp_set_.shrink_to_fit();
        mode_ = Mode::Dense;
    }
};

} // namespace algoat::sketches
