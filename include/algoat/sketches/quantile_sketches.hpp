/**
 * @file quantile_sketches.hpp
 * @brief Dynamic Streaming Quantile Sketches: DDSketch & t-Digest.
 *
 * Implements:
 * 1. DDSketch (Masson, Rim & Lee, VLDB 2019) with formal relative-error guarantee.
 * 2. t-Digest (Dunning & Ertl, 2019) with scale-function adaptive clustering for extreme tails.
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

namespace algoat::sketches {

/**
 * @brief DDSketch with formal relative-error guarantee.
 *
 * Partitions the positive and negative domains into logarithmically sized buckets
 * such that for any quantile q, the estimated value v_hat satisfies:
 *   |v_hat - v| / v <= alpha
 */
class DDSketch {
public:
    /**
     * @brief Constructs a DDSketch with specified relative error and max storage bins.
     * @param alpha Relative error bound alpha in (0.0, 1.0) (default 0.01 for 1% error).
     * @param max_bins Maximum number of bins before collapsing (default 2048).
     */
    explicit DDSketch(double alpha = 0.01, std::size_t max_bins = 2048)
        : alpha_((alpha > 0.0 && alpha < 1.0) ? alpha : 0.01),
          gamma_((1.0 + alpha_) / (1.0 - alpha_)), inv_ln_gamma_(1.0 / std::log(gamma_)),
          max_bins_(max_bins > 0 ? max_bins : 2048) {
        positive_store_.bins.reserve(std::min(max_bins_, std::size_t{256}));
        negative_store_.bins.reserve(std::min(max_bins_, std::size_t{256}));
    }

    DDSketch(const DDSketch&) = default;
    DDSketch(DDSketch&&) noexcept = default;
    DDSketch& operator=(const DDSketch&) = default;
    DDSketch& operator=(DDSketch&&) noexcept = default;
    ~DDSketch() = default;

    /**
     * @brief Ingests a floating-point sample.
     * @param value The double precision value to add.
     */
    void add(double value) noexcept {
        if (std::isnan(value)) {
            return;
        }

        min_ = std::min(min_, value);
        max_ = std::max(max_, value);
        ++count_;

        constexpr double min_indexable_value = 1e-300;
        if (std::abs(value) <= min_indexable_value) {
            ++zero_count_;
            return;
        }

        if (value > 0.0) {
            const std::int32_t index = value_to_index(value);
            positive_store_.add(index, 1, max_bins_);
        } else {
            const std::int32_t index = value_to_index(-value);
            negative_store_.add(index, 1, max_bins_);
        }
    }

    /**
     * @brief Returns estimated value at quantile q in [0.0, 1.0].
     * @param q Quantile rank in [0.0, 1.0].
     * @return Estimated value at quantile q.
     */
    [[nodiscard]] double get_quantile(double q) const noexcept {
        if (count_ == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (q <= 0.0) {
            return min_;
        }
        if (q >= 1.0) {
            return max_;
        }

        const double target_rank = q * static_cast<double>(count_ - 1);
        double cumulative_count = 0.0;

        // 1. Negative bins: higher index -> larger absolute value -> smaller (more negative)
        if (!negative_store_.bins.empty()) {
            for (std::size_t i = negative_store_.bins.size(); i > 0; --i) {
                const std::uint64_t bucket_count = negative_store_.bins[i - 1];
                if (bucket_count > 0) {
                    cumulative_count += static_cast<double>(bucket_count);
                    if (cumulative_count > target_rank) {
                        const std::int32_t bucket_idx =
                            negative_store_.offset + static_cast<std::int32_t>(i - 1);
                        const double reconstructed = -index_to_value(bucket_idx);
                        return std::clamp(reconstructed, min_, max_);
                    }
                }
            }
        }

        // 2. Zero count
        if (zero_count_ > 0) {
            cumulative_count += static_cast<double>(zero_count_);
            if (cumulative_count > target_rank) {
                return 0.0;
            }
        }

        // 3. Positive bins: lower index -> smaller positive value
        if (!positive_store_.bins.empty()) {
            for (std::size_t i = 0; i < positive_store_.bins.size(); ++i) {
                const std::uint64_t bucket_count = positive_store_.bins[i];
                if (bucket_count > 0) {
                    cumulative_count += static_cast<double>(bucket_count);
                    if (cumulative_count > target_rank) {
                        const std::int32_t bucket_idx =
                            positive_store_.offset + static_cast<std::int32_t>(i);
                        const double reconstructed = index_to_value(bucket_idx);
                        return std::clamp(reconstructed, min_, max_);
                    }
                }
            }
        }

        return max_;
    }

    /**
     * @brief Merges another DDSketch into this instance in linear time.
     * @param other The DDSketch instance to merge from.
     */
    void merge(const DDSketch& other) {
        if (other.count_ == 0) {
            return;
        }
        if (count_ == 0) {
            *this = other;
            return;
        }

        min_ = std::min(min_, other.min_);
        max_ = std::max(max_, other.max_);
        count_ += other.count_;
        zero_count_ += other.zero_count_;

        positive_store_.merge_from(other.positive_store_, max_bins_);
        negative_store_.merge_from(other.negative_store_, max_bins_);
    }

    /**
     * @brief Clears all bins and resets state.
     */
    void clear() noexcept {
        count_ = 0;
        min_ = std::numeric_limits<double>::infinity();
        max_ = -std::numeric_limits<double>::infinity();
        zero_count_ = 0;
        positive_store_.clear();
        negative_store_.clear();
    }

    [[nodiscard]] std::size_t count() const noexcept {
        return count_;
    }
    [[nodiscard]] double min_value() const noexcept {
        return min_;
    }
    [[nodiscard]] double max_value() const noexcept {
        return max_;
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const noexcept {
        return sizeof(DDSketch) +
               (positive_store_.bins.capacity() + negative_store_.bins.capacity()) *
                   sizeof(std::uint64_t);
    }

private:
    struct BucketStore {
        std::vector<std::uint64_t> bins;
        std::int32_t offset{0};

        void clear() noexcept {
            bins.clear();
            offset = 0;
        }

        void add(std::int32_t index, std::uint64_t count, std::size_t max_bins) {
            if (bins.empty()) {
                offset = index;
                bins.push_back(count);
                return;
            }

            if (index >= offset) {
                const std::size_t diff = static_cast<std::size_t>(index - offset);
                if (diff >= bins.size()) {
                    const std::size_t new_size = diff + 1;
                    if (new_size > max_bins) {
                        // Collapse lowest bins to preserve high quantiles
                        const std::int32_t new_offset =
                            index - static_cast<std::int32_t>(max_bins) + 1;
                        const std::int32_t shift = new_offset - offset;
                        if (shift > 0) {
                            std::uint64_t collapsed = 0;
                            const std::int32_t collapse_limit =
                                std::min(shift, static_cast<std::int32_t>(bins.size()));
                            for (std::int32_t i = 0; i < collapse_limit; ++i) {
                                collapsed += bins[static_cast<std::size_t>(i)];
                            }
                            if (static_cast<std::size_t>(shift) < bins.size()) {
                                std::copy(bins.begin() + shift, bins.end(), bins.begin());
                                bins.resize(bins.size() - static_cast<std::size_t>(shift));
                            } else {
                                bins.clear();
                            }
                            offset = new_offset;
                            if (bins.empty()) {
                                bins.push_back(collapsed);
                            } else {
                                bins[0] += collapsed;
                            }
                        }
                    }
                    const std::size_t adjusted_diff = static_cast<std::size_t>(index - offset);
                    if (adjusted_diff >= bins.size()) {
                        bins.resize(adjusted_diff + 1, 0);
                    }
                }
                bins[static_cast<std::size_t>(index - offset)] += count;
            } else {
                // index < offset: expand leftward
                const std::size_t shift = static_cast<std::size_t>(offset - index);
                if (bins.size() + shift > max_bins) {
                    // Span exceeds max_bins; collapse into lowest bin
                    bins[0] += count;
                } else {
                    bins.insert(bins.begin(), shift, 0);
                    offset = index;
                    bins[0] += count;
                }
            }
        }

        void merge_from(const BucketStore& other, std::size_t max_bins) {
            for (std::size_t i = 0; i < other.bins.size(); ++i) {
                if (other.bins[i] > 0) {
                    const std::int32_t idx = other.offset + static_cast<std::int32_t>(i);
                    add(idx, other.bins[i], max_bins);
                }
            }
        }
    };

    double alpha_;
    double gamma_;
    double inv_ln_gamma_;
    std::size_t max_bins_;
    std::size_t count_{0};
    double min_{std::numeric_limits<double>::infinity()};
    double max_{-std::numeric_limits<double>::infinity()};
    std::size_t zero_count_{0};

    BucketStore positive_store_;
    BucketStore negative_store_;

    [[nodiscard]] std::int32_t value_to_index(double value) const noexcept {
        // Exact logarithmic bin mapping using natural log to strictly guarantee relative error <=
        // alpha
        return static_cast<std::int32_t>(std::floor(std::log(value) * inv_ln_gamma_));
    }

    [[nodiscard]] double index_to_value(std::int32_t index) const noexcept {
        // Geometric harmonic midpoint: 2 * gamma^(k+1) / (gamma + 1) == gamma^k * (1 + alpha)
        return std::pow(gamma_, static_cast<double>(index)) * (1.0 + alpha_);
    }
};

/**
 * @brief t-Digest with scale-function adaptive clustering for extreme tails.
 *
 * Employs scale function k_1 to enforce near-zero error at extreme percentiles:
 *   k_1(q) = (delta / (2 * pi)) * arcsin(2 * q - 1)
 */
class TDigest {
public:
    struct alignas(16) Centroid {
        double mean{0.0};
        double weight{0.0};
    };

    /**
     * @brief Constructs a TDigest with specified compression parameter.
     * @param compression Parameter delta controlling number of centroids (default 100.0).
     */
    explicit TDigest(double compression = 100.0)
        : compression_(compression >= 10.0 ? compression : 100.0) {
        const std::size_t buf_cap = static_cast<std::size_t>(compression_ * 8.0);
        const std::size_t cent_cap = static_cast<std::size_t>(compression_ * 2.0);
        buffer_.reserve(buf_cap);
        centroids_.reserve(cent_cap);
        compacted_scratch_.reserve(cent_cap);
    }

    TDigest(const TDigest&) = default;
    TDigest(TDigest&&) noexcept = default;
    TDigest& operator=(const TDigest&) = default;
    TDigest& operator=(TDigest&&) noexcept = default;
    ~TDigest() = default;

    /**
     * @brief Ingests a single floating-point sample.
     */
    void add(double value) noexcept {
        if (std::isnan(value)) {
            return;
        }
        min_ = std::min(min_, value);
        max_ = std::max(max_, value);
        total_weight_ += 1.0;
        buffer_.push_back(value);
        dirty_ = true;

        if (buffer_.size() >= static_cast<std::size_t>(compression_ * 8.0)) {
            compress();
        }
    }

    /**
     * @brief Ingests a weighted sample.
     */
    void add_weighted(double value, double weight) noexcept {
        if (std::isnan(value) || weight <= 0.0) {
            return;
        }
        min_ = std::min(min_, value);
        max_ = std::max(max_, value);
        total_weight_ += weight;

        flush_buffer();
        centroids_.push_back({value, weight});
        dirty_ = true;

        if (centroids_.size() >= static_cast<std::size_t>(compression_ * 8.0)) {
            compress();
        }
    }

    /**
     * @brief Returns estimated value at quantile q in [0.0, 1.0].
     */
    [[nodiscard]] double get_quantile(double q) const noexcept {
        if (dirty_) {
            compress();
        }
        if (centroids_.empty() || total_weight_ <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (q <= 0.0) {
            return min_;
        }
        if (q >= 1.0) {
            return max_;
        }
        if (centroids_.size() == 1) {
            return centroids_[0].mean;
        }

        const double rank = q * total_weight_;

        // Left tail interpolation between min_ and first centroid midpoint
        const double first_midpoint_rank = centroids_[0].weight * 0.5;
        if (rank < first_midpoint_rank) {
            if (first_midpoint_rank <= 0.0) {
                return min_;
            }
            return min_ + (rank / first_midpoint_rank) * (centroids_[0].mean - min_);
        }

        // Right tail interpolation between last centroid midpoint and max_
        const double last_midpoint_rank = total_weight_ - (centroids_.back().weight * 0.5);
        if (rank >= last_midpoint_rank) {
            const double tail_weight_span = total_weight_ - last_midpoint_rank;
            if (tail_weight_span <= 0.0) {
                return max_;
            }
            return centroids_.back().mean + ((rank - last_midpoint_rank) / tail_weight_span) *
                                                (max_ - centroids_.back().mean);
        }

        // Interior linear interpolation between adjacent centroid midpoints
        double prev_midpoint = first_midpoint_rank;
        double cum_weight = centroids_[0].weight;
        for (std::size_t i = 0; i + 1 < centroids_.size(); ++i) {
            const double next_midpoint = cum_weight + (centroids_[i + 1].weight * 0.5);
            if (rank <= next_midpoint) {
                const double t = (rank - prev_midpoint) / (next_midpoint - prev_midpoint);
                return centroids_[i].mean + t * (centroids_[i + 1].mean - centroids_[i].mean);
            }
            cum_weight += centroids_[i + 1].weight;
            prev_midpoint = next_midpoint;
        }

        return max_;
    }

    /**
     * @brief Returns estimated cumulative distribution function (CDF) for value.
     */
    [[nodiscard]] double get_cdf(double value) const noexcept {
        if (dirty_) {
            compress();
        }
        if (centroids_.empty() || total_weight_ <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (value <= min_) {
            return 0.0;
        }
        if (value >= max_) {
            return 1.0;
        }
        if (centroids_.size() == 1) {
            return 0.5;
        }

        // Value before first centroid mean
        const double first_midpoint_rank = centroids_[0].weight * 0.5;
        if (value < centroids_[0].mean) {
            const double span = centroids_[0].mean - min_;
            if (span <= 0.0) {
                return 0.0;
            }
            const double rank = (value - min_) / span * first_midpoint_rank;
            return std::clamp(rank / total_weight_, 0.0, 1.0);
        }

        // Value after last centroid mean
        if (value >= centroids_.back().mean) {
            const double span = max_ - centroids_.back().mean;
            if (span <= 0.0) {
                return 1.0;
            }
            const double last_midpoint_rank = total_weight_ - (centroids_.back().weight * 0.5);
            const double rank = last_midpoint_rank + (value - centroids_.back().mean) / span *
                                                         (total_weight_ - last_midpoint_rank);
            return std::clamp(rank / total_weight_, 0.0, 1.0);
        }

        // Interior search
        double prev_midpoint = first_midpoint_rank;
        double cum_weight = centroids_[0].weight;
        for (std::size_t i = 0; i + 1 < centroids_.size(); ++i) {
            if (value <= centroids_[i + 1].mean) {
                const double next_midpoint = cum_weight + (centroids_[i + 1].weight * 0.5);
                const double span = centroids_[i + 1].mean - centroids_[i].mean;
                const double t = span > 0.0 ? (value - centroids_[i].mean) / span : 0.0;
                const double rank = prev_midpoint + t * (next_midpoint - prev_midpoint);
                return std::clamp(rank / total_weight_, 0.0, 1.0);
            }
            cum_weight += centroids_[i + 1].weight;
            prev_midpoint = cum_weight - (centroids_[i + 1].weight * 0.5);
        }

        return 1.0;
    }

    /**
     * @brief Merges another TDigest into this instance.
     */
    void merge(const TDigest& other) {
        if (other.total_weight_ <= 0.0) {
            return;
        }
        if (total_weight_ <= 0.0) {
            *this = other;
            return;
        }

        flush_buffer();
        other.flush_buffer();

        min_ = std::min(min_, other.min_);
        max_ = std::max(max_, other.max_);
        total_weight_ += other.total_weight_;

        centroids_.insert(centroids_.end(), other.centroids_.begin(), other.centroids_.end());
        dirty_ = true;
        compress();
    }

    /**
     * @brief Compacts buffered points and merges existing centroids according to scale function
     * k_1.
     *
     * Made const noexcept so queries on const TDigest instances trigger zero-allocation
     * compaction safely without const_cast.
     */
    void compress() const noexcept {
        flush_buffer();
        dirty_ = false;

        if (centroids_.size() <= 1) {
            return;
        }

        std::sort(centroids_.begin(), centroids_.end(),
                  [](const Centroid& a, const Centroid& b) noexcept { return a.mean < b.mean; });

        compacted_scratch_.clear();
        compacted_scratch_.reserve(static_cast<std::size_t>(compression_ * 2.0));

        const double w_total = total_weight_;
        const double delta = compression_;

        Centroid curr = centroids_[0];
        double cum_weight = 0.0;

        // Scale function k_1 weight bound: Delta W <= 4 * W * sqrt(q * (1 - q)) / delta
        // Near boundaries (q -> 0 or q -> 1), sqrt(q * (1 - q)) -> 0, enforcing singleton centroids
        // (w <= 1.0).
        for (std::size_t i = 1; i < centroids_.size(); ++i) {
            const Centroid& next = centroids_[i];
            const double proposed_weight = curr.weight + next.weight;
            const double q = std::clamp((cum_weight + (proposed_weight * 0.5)) / w_total, 0.0, 1.0);
            const double tail_factor = std::sqrt(q * (1.0 - q));
            const double max_allowed_weight = std::max(1.0, (4.0 * w_total * tail_factor) / delta);

            // Never merge boundary singleton centroids at extreme tails
            const bool is_boundary = (cum_weight == 0.0) || (i + 1 == centroids_.size());

            if (!is_boundary && (proposed_weight <= max_allowed_weight)) {
                curr.mean = (curr.mean * curr.weight + next.mean * next.weight) / proposed_weight;
                curr.weight = proposed_weight;
            } else {
                compacted_scratch_.push_back(curr);
                cum_weight += curr.weight;
                curr = next;
            }
        }
        compacted_scratch_.push_back(curr);
        centroids_ = compacted_scratch_;
    }

    [[nodiscard]] std::size_t centroid_count() const noexcept {
        if (dirty_) {
            compress();
        }
        return centroids_.size();
    }

    [[nodiscard]] std::size_t memory_usage_bytes() const noexcept {
        return sizeof(TDigest) + (centroids_.capacity() * sizeof(Centroid)) +
               (buffer_.capacity() * sizeof(double)) +
               (compacted_scratch_.capacity() * sizeof(Centroid));
    }

private:
    double compression_;
    mutable std::vector<Centroid> centroids_;
    mutable std::vector<double> buffer_;
    mutable std::vector<Centroid> compacted_scratch_;
    double total_weight_{0.0};
    double min_{std::numeric_limits<double>::infinity()};
    double max_{-std::numeric_limits<double>::infinity()};
    mutable bool dirty_{false};

    void flush_buffer() const noexcept {
        if (!buffer_.empty()) {
            centroids_.reserve(centroids_.size() + buffer_.size());
            for (double v : buffer_) {
                centroids_.push_back({v, 1.0});
            }
            buffer_.clear();
            dirty_ = true;
        }
    }
};

} // namespace algoat::sketches
