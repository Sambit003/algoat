/**
 * @file rstar_tree.hpp
 * @brief R*-Tree spatial index with Sort-Tile-Recursive (STR) bulk-loading and
 *        AVX2 Struct-of-Arrays (SoA) bounding box traversal.
 *
 * References:
 * - Beckmann, N., Kriegel, H. P., Schneider, R., & Seeger, B. (1990).
 *   "The R*-tree: An efficient and robust access method for points and rectangles."
 *   ACM SIGMOD, 322-331.
 * - Leutenegger, S. T., Edgington, J. M., & Lopez, M. A. (1997).
 *   "STR: A simple and efficient algorithm for R-tree packing." ICDE, 497-506.
 * - Guttman, A. (1984). "R-trees: A dynamic index structure for spatial searching."
 *   ACM SIGMOD Record, 14(2), 47-57.
 */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__AVX2__) && !defined(ALGOAT_RSTAR_SCALAR_ONLY) && !defined(ALGOAT_BUILDING_MODULE)
#include <immintrin.h>
#define ALGOAT_RSTAR_HAS_AVX2 1
#endif

namespace algoat::geometry {

/**
 * @struct BoundingBox
 * @brief Axis-aligned bounding box (AABB) in @p Dim dimensions over coordinate type @p T.
 */
template <typename T = double, std::size_t Dim = 2> struct BoundingBox {
    static_assert(Dim > 0, "BoundingBox dimensionality must be positive");
    static_assert(std::is_arithmetic_v<T>, "BoundingBox coordinate type must be arithmetic");

    std::array<T, Dim> min_corner{};
    std::array<T, Dim> max_corner{};

    [[nodiscard]] constexpr bool operator==(const BoundingBox&) const noexcept = default;

    /**
     * @brief Tests closed-interval overlap with @p other across all dimensions.
     *
     * Returns @c false if any coordinate is NaN.
     */
    [[nodiscard]] constexpr bool intersects(const BoundingBox& other) const noexcept {
        for (std::size_t d = 0; d < Dim; ++d) {
            if (!(min_corner[d] <= other.max_corner[d] && max_corner[d] >= other.min_corner[d])) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Tests whether @p other is completely enclosed within this bounding box.
     */
    [[nodiscard]] constexpr bool contains(const BoundingBox& other) const noexcept {
        for (std::size_t d = 0; d < Dim; ++d) {
            if (!(min_corner[d] <= other.min_corner[d] && max_corner[d] >= other.max_corner[d])) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Computes hypervolume (area in 2D, volume in 3D).
     *
     * Returns @c 0.0 for degenerate, inverted, or NaN-containing boxes.
     */
    [[nodiscard]] constexpr double area() const noexcept {
        double prod = 1.0;
        for (std::size_t d = 0; d < Dim; ++d) {
            const double extent =
                static_cast<double>(max_corner[d]) - static_cast<double>(min_corner[d]);
            if (!(extent > 0.0)) {
                return 0.0;
            }
            prod *= extent;
        }
        return prod;
    }

    /**
     * @brief Computes the margin (sum of all edge lengths: 2^(Dim-1) * sum(extents)).
     *
     * Equals perimeter in 2D. Returns @c 0.0 for inverted or NaN-containing boxes.
     */
    [[nodiscard]] double margin() const noexcept {
        double sum = 0.0;
        for (std::size_t d = 0; d < Dim; ++d) {
            const double extent =
                static_cast<double>(max_corner[d]) - static_cast<double>(min_corner[d]);
            if (!(extent >= 0.0)) {
                return 0.0;
            }
            sum += extent;
        }
        return std::ldexp(sum, static_cast<int>(Dim - 1));
    }
};

/**
 * @class RStarTree
 * @brief Dynamic & bulk-loaded R*-Tree with SoA AVX2 SIMD bounding box intersections.
 *
 * @tparam ValueType  Payload stored per spatial entry.
 * @tparam T          Coordinate scalar type (default @c double).
 * @tparam Dim        Spatial dimensionality (default @c 2).
 * @tparam MaxEntries Maximum fanout per node (default @c 16).
 */
template <typename ValueType, typename T = double, std::size_t Dim = 2, std::size_t MaxEntries = 16>
class RStarTree {
    static_assert(Dim > 0, "RStarTree dimensionality must be positive");
    static_assert(MaxEntries >= 4, "RStarTree MaxEntries must be at least 4");
    static_assert(std::is_arithmetic_v<T>, "RStarTree coordinate type must be arithmetic");

public:
    using box_type = BoundingBox<T, Dim>;
    using value_type = ValueType;

    static constexpr std::size_t max_entries = MaxEntries;
    static constexpr std::size_t min_entries =
        std::clamp<std::size_t>((MaxEntries * 2) / 5, 2, MaxEntries / 2);
    static constexpr std::size_t reinsert_count =
        std::clamp<std::size_t>((MaxEntries * 3 + 5) / 10, 1, MaxEntries + 1 - min_entries);

    RStarTree() = default;

    /**
     * @brief Inserts a bounding box and associated value into the R*-tree.
     *
     * Ignores boxes with non-finite or inverted coordinates.
     */
    void insert(const box_type& box, const ValueType& value) {
        if (!is_valid_box(box)) {
            return;
        }

        const uint32_t val_slot = allocate_value(value);
        if (root_idx_ == kInvalidNode) {
            root_idx_ = allocate_node(/*is_leaf=*/true, /*height=*/0);
            set_node_entry(nodes_[root_idx_], 0, box, val_slot);
            nodes_[root_idx_].count = 1;
            total_items_ = 1;
            return;
        }

        std::vector<uint8_t> reinsert_called(static_cast<std::size_t>(nodes_[root_idx_].height) + 2,
                                             0);
        insert_entry(box, val_slot, /*target_height=*/0, reinsert_called);
        ++total_items_;
    }

    /**
     * @brief Removes a single entry matching both @p box and @p value.
     *
     * Uses Guttman's FindLeaf and CondenseTree with R*-tree orphan reinsertion.
     * @return @c true if a matching entry was found and removed, @c false otherwise.
     */
    bool remove(const box_type& box, const ValueType& value)
        requires std::equality_comparable<ValueType>
    {
        if (root_idx_ == kInvalidNode || !is_valid_box(box)) {
            return false;
        }

        uint32_t leaf_idx = kInvalidNode;
        std::size_t slot = 0;
        if (!find_leaf_entry(root_idx_, box, value, leaf_idx, slot)) {
            return false;
        }

        const uint32_t val_slot = nodes_[leaf_idx].children[slot];
        remove_slot(nodes_[leaf_idx], slot);
        values_[val_slot].reset();
        free_values_.push_back(val_slot);
        --total_items_;

        condense_tree(leaf_idx);
        return true;
    }

    /**
     * @brief Bulk-loads items using Sort-Tile-Recursive (STR) packing in O(N log N).
     *
     * Replaces any existing contents in the tree. Expects elements convertible to
     * pair-like `(box_type, ValueType)` via `.first`/`.second` or structured bindings.
     */
    template <std::ranges::contiguous_range R> void bulk_load(R&& items) {
        clear();

        const std::size_t n_raw = std::ranges::size(items);
        if (n_raw == 0) {
            return;
        }

        std::vector<Entry> entries;
        entries.reserve(n_raw);
        values_.reserve(n_raw);

        for (const auto& item : items) {
            const box_type& b = item.first;
            if (!is_valid_box(b)) {
                continue;
            }
            const uint32_t val_slot = static_cast<uint32_t>(values_.size());
            values_.emplace_back(item.second);
            entries.push_back(Entry{b, val_slot});
        }

        if (entries.empty()) {
            values_.clear();
            return;
        }

        total_items_ = entries.size();

        // Estimate total nodes (~1.1 * ceil(N / MaxEntries)) to avoid reallocations
        const std::size_t est_leaves = (entries.size() + MaxEntries - 1) / MaxEntries;
        nodes_.reserve(est_leaves + (est_leaves / std::max<std::size_t>(1, MaxEntries - 1)) + 8);

        std::vector<std::pair<std::size_t, std::size_t>> groups;
        groups.reserve(est_leaves);
        str_pack_ranges(entries, 0, entries.size(), 0, groups);

        std::vector<Entry> current_level;
        current_level.reserve(groups.size());
        for (const auto& [begin, end] : groups) {
            const uint32_t node_idx = allocate_node(/*is_leaf=*/true, /*height=*/0);
            Node& node = nodes_[node_idx];
            const std::size_t cnt = end - begin;
            node.count = static_cast<uint32_t>(cnt);
            box_type mbr = entries[begin].box;
            for (std::size_t i = 0; i < cnt; ++i) {
                const Entry& e = entries[begin + i];
                set_node_entry(node, i, e.box, e.payload);
                mbr = unite_boxes(mbr, e.box);
            }
            current_level.push_back(Entry{mbr, node_idx});
        }

        uint16_t curr_height = 1;
        while (current_level.size() > 1) {
            groups.clear();
            str_pack_ranges(current_level, 0, current_level.size(), 0, groups);

            std::vector<Entry> next_level;
            next_level.reserve(groups.size());
            for (const auto& [begin, end] : groups) {
                const uint32_t node_idx = allocate_node(/*is_leaf=*/false, curr_height);
                const std::size_t cnt = end - begin;
                nodes_[node_idx].count = static_cast<uint32_t>(cnt);
                box_type mbr = current_level[begin].box;
                for (std::size_t i = 0; i < cnt; ++i) {
                    const Entry& e = current_level[begin + i];
                    set_node_entry(nodes_[node_idx], i, e.box, e.payload);
                    nodes_[e.payload].parent = node_idx;
                    mbr = unite_boxes(mbr, e.box);
                }
                next_level.push_back(Entry{mbr, node_idx});
            }
            current_level = std::move(next_level);
            ++curr_height;
        }

        root_idx_ = current_level[0].payload;
    }

    /**
     * @brief Performs a synchronized spatial join returning all pairs `(v1, v2)` whose
     *        bounding boxes intersect.
     */
    template <typename OtherVal, std::size_t OtherMax>
    [[nodiscard]] std::vector<std::pair<ValueType, OtherVal>>
    spatial_join(const RStarTree<OtherVal, T, Dim, OtherMax>& other) const {
        std::vector<std::pair<ValueType, OtherVal>> pairs;
        if (root_idx_ == kInvalidNode || other.root_idx_ == kInvalidNode) {
            return pairs;
        }

        const box_type root_a = compute_node_mbr(nodes_[root_idx_]);
        const box_type root_b =
            RStarTree<OtherVal, T, Dim, OtherMax>::compute_node_mbr(other.nodes_[other.root_idx_]);
        if (!root_a.intersects(root_b)) {
            return pairs;
        }

        struct JoinTask {
            uint32_t idx_a;
            uint32_t idx_b;
            box_type mbr_a;
            box_type mbr_b;
        };

        std::vector<JoinTask> stack;
        stack.reserve(128);
        stack.push_back({root_idx_, other.root_idx_, root_a, root_b});

        while (!stack.empty()) {
            const JoinTask task = stack.back();
            stack.pop_back();

            const Node& na = nodes_[task.idx_a];
            const auto& nb = other.nodes_[task.idx_b];

            if (na.is_leaf && nb.is_leaf) {
                for (std::size_t i = 0; i < na.count; ++i) {
                    const box_type ba = get_entry_box(na, i);
                    if (!ba.intersects(task.mbr_b)) {
                        continue;
                    }
                    const ValueType& va = *values_[na.children[i]];
                    for (std::size_t base = 0; base < nb.count; base += 32) {
                        const std::size_t chunk = std::min<std::size_t>(32, nb.count - base);
                        uint32_t mask =
                            RStarTree<OtherVal, T, Dim, OtherMax>::intersect_chunk_mask_32(
                                nb, base, chunk, ba);
                        while (mask != 0) {
                            const unsigned off = static_cast<unsigned>(std::countr_zero(mask));
                            pairs.emplace_back(va, *other.values_[nb.children[base + off]]);
                            mask &= mask - 1;
                        }
                    }
                }
            } else if (!na.is_leaf && (nb.is_leaf || na.height >= nb.height)) {
                for (std::size_t base = 0; base < na.count; base += 32) {
                    const std::size_t chunk = std::min<std::size_t>(32, na.count - base);
                    uint32_t mask = intersect_chunk_mask_32(na, base, chunk, task.mbr_b);
                    while (mask != 0) {
                        const unsigned off = static_cast<unsigned>(std::countr_zero(mask));
                        const std::size_t slot = base + off;
                        stack.push_back(
                            {na.children[slot], task.idx_b, get_entry_box(na, slot), task.mbr_b});
                        mask &= mask - 1;
                    }
                }
            } else {
                for (std::size_t base = 0; base < nb.count; base += 32) {
                    const std::size_t chunk = std::min<std::size_t>(32, nb.count - base);
                    uint32_t mask = RStarTree<OtherVal, T, Dim, OtherMax>::intersect_chunk_mask_32(
                        nb, base, chunk, task.mbr_a);
                    while (mask != 0) {
                        const unsigned off = static_cast<unsigned>(std::countr_zero(mask));
                        const std::size_t slot = base + off;
                        const auto bb =
                            RStarTree<OtherVal, T, Dim, OtherMax>::get_entry_box(nb, slot);
                        stack.push_back({task.idx_a, nb.children[slot], task.mbr_a, bb});
                        mask &= mask - 1;
                    }
                }
            }
        }

        return pairs;
    }

    /**
     * @brief Searches for all entries whose bounding boxes intersect @p query_box.
     */
    [[nodiscard]] std::vector<ValueType> search_intersecting(const box_type& query_box) const {
        std::vector<ValueType> results;
        if (root_idx_ == kInvalidNode || !is_valid_box(query_box)) {
            return results;
        }

        std::array<uint32_t, 128> fixed_stack{};
        std::vector<uint32_t> overflow_stack;
        std::size_t sp = 0;

        auto push_node = [&](uint32_t idx) {
            if (sp < fixed_stack.size()) {
                fixed_stack[sp++] = idx;
            } else {
                overflow_stack.push_back(idx);
            }
        };
        auto pop_node = [&]() -> uint32_t {
            if (!overflow_stack.empty()) {
                const uint32_t idx = overflow_stack.back();
                overflow_stack.pop_back();
                return idx;
            }
            return fixed_stack[--sp];
        };
        auto has_items = [&]() -> bool { return sp > 0 || !overflow_stack.empty(); };

        push_node(root_idx_);

        while (has_items()) {
            const uint32_t node_idx = pop_node();
            const Node& node = nodes_[node_idx];
            if (node.count == 0) {
                continue;
            }

            for (std::size_t base = 0; base < node.count; base += 32) {
                const std::size_t chunk = std::min<std::size_t>(32, node.count - base);
                uint32_t mask = intersect_chunk_mask_32(node, base, chunk, query_box);
                if (node.is_leaf) {
                    while (mask != 0) {
                        const unsigned off = static_cast<unsigned>(std::countr_zero(mask));
                        results.push_back(*values_[node.children[base + off]]);
                        mask &= mask - 1;
                    }
                } else {
                    while (mask != 0) {
                        const unsigned off = static_cast<unsigned>(std::countr_zero(mask));
                        const uint32_t child_idx = node.children[base + off];
#if defined(__GNUC__) || defined(__clang__)
                        __builtin_prefetch(&nodes_[child_idx], 0, 3);
#endif
                        push_node(child_idx);
                        mask &= mask - 1;
                    }
                }
                if constexpr (MaxEntries <= 32) {
                    break;
                }
            }
        }

        return results;
    }

    /**
     * @brief Performs Best-First Incremental k-Nearest Neighbor (k-NN) search
     *        ordered by non-decreasing Euclidean MINDIST from @p query_point.
     */
    [[nodiscard]] std::vector<std::pair<ValueType, double>>
    knn_search(const std::array<T, Dim>& query_point, std::size_t k) const {
        std::vector<std::pair<ValueType, double>> results;
        if (root_idx_ == kInvalidNode || k == 0) {
            return results;
        }
        if constexpr (std::is_floating_point_v<T>) {
            for (std::size_t d = 0; d < Dim; ++d) {
                if (!std::isfinite(query_point[d])) {
                    return results;
                }
            }
        }

        const std::size_t target_k = std::min(k, total_items_);
        results.reserve(target_k);

        struct HeapItem {
            double dist_sq;
            uint32_t idx;
            bool is_data;

            [[nodiscard]] bool operator>(const HeapItem& other) const noexcept {
                if (dist_sq != other.dist_sq) {
                    return dist_sq > other.dist_sq;
                }
                return is_data < other.is_data;
            }
        };

        std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> pq;
        std::vector<double> best_dists;
        best_dists.reserve(target_k);
        double bound_sq = std::numeric_limits<double>::infinity();

        auto record_candidate_dist = [&](double d2) {
            if (best_dists.size() < target_k) {
                best_dists.push_back(d2);
                std::push_heap(best_dists.begin(), best_dists.end());
                if (best_dists.size() == target_k) {
                    bound_sq = best_dists.front();
                }
            } else if (d2 < bound_sq) {
                std::pop_heap(best_dists.begin(), best_dists.end());
                best_dists.back() = d2;
                std::push_heap(best_dists.begin(), best_dists.end());
                bound_sq = best_dists.front();
            }
        };

        pq.push(HeapItem{0.0, root_idx_, false});
        std::array<double, MaxEntries> child_dists{};

        while (!pq.empty() && results.size() < target_k) {
            const HeapItem top = pq.top();
            pq.pop();

            if (top.dist_sq > bound_sq) {
                break;
            }

            if (top.is_data) {
                results.emplace_back(*values_[top.idx], std::sqrt(top.dist_sq));
                continue;
            }

            const Node& node = nodes_[top.idx];
            if (node.count == 0) {
                continue;
            }

            compute_node_mindist_sq(node, query_point, child_dists.data());
            if (node.is_leaf) {
                for (std::size_t i = 0; i < node.count; ++i) {
                    const double d2 = child_dists[i];
                    if (d2 <= bound_sq) {
                        record_candidate_dist(d2);
                        pq.push(HeapItem{d2, node.children[i], true});
                    }
                }
            } else {
                for (std::size_t i = 0; i < node.count; ++i) {
                    const double d2 = child_dists[i];
                    if (d2 <= bound_sq) {
                        pq.push(HeapItem{d2, node.children[i], false});
                    }
                }
            }
        }

        return results;
    }

    /**
     * @brief Removes all entries and resets the tree to empty.
     */
    void clear() noexcept {
        nodes_.clear();
        free_nodes_.clear();
        values_.clear();
        free_values_.clear();
        root_idx_ = kInvalidNode;
        total_items_ = 0;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return total_items_;
    }
    [[nodiscard]] bool empty() const noexcept {
        return total_items_ == 0;
    }

    /**
     * @brief Returns the height of the tree (0 when empty, 1 for a single leaf root).
     */
    [[nodiscard]] std::size_t height() const noexcept {
        if (root_idx_ == kInvalidNode) {
            return 0;
        }
        return static_cast<std::size_t>(nodes_[root_idx_].height) + 1;
    }

    /**
     * @brief Returns the number of active nodes in the tree.
     */
    [[nodiscard]] std::size_t node_count() const noexcept {
        return nodes_.size() - free_nodes_.size();
    }

    /**
     * @brief Computes average node occupancy ratio in [0.0, 1.0].
     */
    [[nodiscard]] double fill_factor() const noexcept {
        if (root_idx_ == kInvalidNode) {
            return 0.0;
        }
        const std::size_t active = node_count();
        if (active == 0) {
            return 0.0;
        }
        std::size_t total_entries = 0;
        for (const Node& node : nodes_) {
            total_entries += node.count;
        }
        return static_cast<double>(total_entries) / static_cast<double>(active * MaxEntries);
    }

private:
    template <typename OtherVal, typename OtherT, std::size_t OtherDim, std::size_t OtherMax>
    friend class RStarTree;
    static constexpr uint32_t kInvalidNode = 0xFFFFFFFFu;
    static constexpr std::size_t kPaddedMax = ((MaxEntries + 7) / 8) * 8;
    static constexpr std::size_t kOverlapCostThreshold = 32;

    static constexpr T kPosInf = std::numeric_limits<T>::has_infinity
                                     ? std::numeric_limits<T>::infinity()
                                     : std::numeric_limits<T>::max();
    static constexpr T kNegInf = std::numeric_limits<T>::has_infinity
                                     ? -std::numeric_limits<T>::infinity()
                                     : std::numeric_limits<T>::lowest();

    struct alignas(64) Node {
        alignas(32) T min_coords[Dim][kPaddedMax];
        alignas(32) T max_coords[Dim][kPaddedMax];
        uint32_t children[MaxEntries]{};
        uint32_t parent{kInvalidNode};
        uint32_t count{0};
        uint16_t height{0};
        bool is_leaf{true};

        Node() noexcept {
            reset_bounds();
        }

        void reset_bounds() noexcept {
            for (std::size_t d = 0; d < Dim; ++d) {
                for (std::size_t i = 0; i < kPaddedMax; ++i) {
                    min_coords[d][i] = kPosInf;
                    max_coords[d][i] = kNegInf;
                }
            }
        }
    };

    struct Entry {
        box_type box;
        uint32_t payload{0};
    };

    std::vector<Node> nodes_;
    std::vector<uint32_t> free_nodes_;
    std::vector<std::optional<ValueType>> values_;
    std::vector<uint32_t> free_values_;
    uint32_t root_idx_{kInvalidNode};
    std::size_t total_items_{0};

    [[nodiscard]] static std::size_t ceil_sqrt(std::size_t n) noexcept {
        if (n <= 1) {
            return n;
        }
        std::size_t lo = 1;
        std::size_t hi = std::min<std::size_t>(n, std::size_t{1} << 31);
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (mid <= n / mid && (mid * mid < n)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    [[nodiscard]] static std::size_t ceil_pow_ratio(std::size_t p, std::size_t num,
                                                    std::size_t den) noexcept {
        if (p <= 1) {
            return p;
        }
        if (den == 2 && num == 1) {
            return ceil_sqrt(p);
        }
#if defined(__SIZEOF_INT128__)
        // Exact integer binary search for smallest x with x^den >= p^num
        auto pow_ge = [](unsigned __int128 x, std::size_t e1, unsigned __int128 pval,
                         std::size_t e2) -> bool {
            auto sat_pow = [](unsigned __int128 base, std::size_t exp) -> unsigned __int128 {
                unsigned __int128 res = 1;
                constexpr unsigned __int128 kMax = ~static_cast<unsigned __int128>(0);
                while (exp > 0) {
                    if (exp & 1u) {
                        if (base > 0 && res > kMax / base) {
                            return kMax;
                        }
                        res *= base;
                    }
                    exp >>= 1u;
                    if (exp > 0) {
                        if (base > 0 && base > kMax / base) {
                            base = kMax;
                        } else {
                            base *= base;
                        }
                    }
                }
                return res;
            };
            return sat_pow(x, e1) >= sat_pow(pval, e2);
        };
        std::size_t lo = 1;
        std::size_t hi = p;
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (pow_ge(mid, den, p, num)) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
#else
        const long double approx =
            std::pow(static_cast<long double>(p),
                     static_cast<long double>(num) / static_cast<long double>(den));
        return std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(approx)));
#endif
    }

    [[nodiscard]] static double box_center_coord(const box_type& box, std::size_t d) noexcept {
        return 0.5 * static_cast<double>(box.min_corner[d]) +
               0.5 * static_cast<double>(box.max_corner[d]);
    }

    static void str_pack_ranges(std::vector<Entry>& entries, std::size_t begin, std::size_t end,
                                std::size_t dim,
                                std::vector<std::pair<std::size_t, std::size_t>>& out_groups) {
        const std::size_t r = end - begin;
        if (r == 0) {
            return;
        }
        if (r <= MaxEntries) {
            out_groups.emplace_back(begin, end);
            return;
        }

        const std::size_t k = Dim - dim;
        std::sort(entries.begin() + static_cast<std::ptrdiff_t>(begin),
                  entries.begin() + static_cast<std::ptrdiff_t>(end),
                  [dim](const Entry& a, const Entry& b) {
                      return box_center_coord(a.box, dim) < box_center_coord(b.box, dim);
                  });

        if (k <= 1) {
            for (std::size_t i = begin; i < end; i += MaxEntries) {
                out_groups.emplace_back(i, std::min(i + MaxEntries, end));
            }
            return;
        }

        const std::size_t pages = (r + MaxEntries - 1) / MaxEntries;
        const std::size_t slab_pages = ceil_pow_ratio(pages, k - 1, k);
        const std::size_t slab_size = MaxEntries * slab_pages;

        for (std::size_t i = begin; i < end; i += slab_size) {
            str_pack_ranges(entries, i, std::min(i + slab_size, end), dim + 1, out_groups);
        }
    }

    [[nodiscard]] static bool is_valid_box(const box_type& box) noexcept {
        for (std::size_t d = 0; d < Dim; ++d) {
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(box.min_corner[d]) || !std::isfinite(box.max_corner[d])) {
                    return false;
                }
            }
            if (box.min_corner[d] > box.max_corner[d]) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] uint32_t allocate_value(const ValueType& value) {
        if (!free_values_.empty()) {
            const uint32_t slot = free_values_.back();
            free_values_.pop_back();
            values_[slot].emplace(value);
            return slot;
        }
        const uint32_t slot = static_cast<uint32_t>(values_.size());
        values_.emplace_back(value);
        return slot;
    }

    [[nodiscard]] uint32_t allocate_node(bool is_leaf, uint16_t height) {
        if (!free_nodes_.empty()) {
            const uint32_t idx = free_nodes_.back();
            free_nodes_.pop_back();
            nodes_[idx].reset_bounds();
            nodes_[idx].parent = kInvalidNode;
            nodes_[idx].count = 0;
            nodes_[idx].height = height;
            nodes_[idx].is_leaf = is_leaf;
            return idx;
        }
        const uint32_t idx = static_cast<uint32_t>(nodes_.size());
        nodes_.emplace_back();
        nodes_[idx].is_leaf = is_leaf;
        nodes_[idx].height = height;
        return idx;
    }

    void free_node(uint32_t idx) noexcept {
        nodes_[idx].count = 0;
        nodes_[idx].parent = kInvalidNode;
        free_nodes_.push_back(idx);
    }

    static void remove_slot(Node& node, std::size_t slot) noexcept {
        const std::size_t last = --node.count;
        if (slot != last) {
            for (std::size_t d = 0; d < Dim; ++d) {
                node.min_coords[d][slot] = node.min_coords[d][last];
                node.max_coords[d][slot] = node.max_coords[d][last];
            }
            node.children[slot] = node.children[last];
        }
        for (std::size_t d = 0; d < Dim; ++d) {
            node.min_coords[d][last] = kPosInf;
            node.max_coords[d][last] = kNegInf;
        }
        node.children[last] = 0;
    }

    [[nodiscard]] bool find_leaf_entry(uint32_t node_idx, const box_type& box,
                                       const ValueType& value, uint32_t& out_leaf,
                                       std::size_t& out_slot) const noexcept {
        const Node& node = nodes_[node_idx];
        if (node.is_leaf) {
            for (std::size_t i = 0; i < node.count; ++i) {
                if (get_entry_box(node, i) == box && *values_[node.children[i]] == value) {
                    out_leaf = node_idx;
                    out_slot = i;
                    return true;
                }
            }
            return false;
        }

        for (std::size_t i = 0; i < node.count; ++i) {
            if (get_entry_box(node, i).contains(box)) {
                if (find_leaf_entry(node.children[i], box, value, out_leaf, out_slot)) {
                    return true;
                }
            }
        }
        return false;
    }

    void condense_tree(uint32_t leaf_idx) {
        struct EliminatedNode {
            uint32_t node_idx;
            uint16_t height;
        };
        std::vector<EliminatedNode> eliminated;

        uint32_t curr = leaf_idx;
        while (nodes_[curr].parent != kInvalidNode) {
            const uint32_t parent_idx = nodes_[curr].parent;
            const std::size_t slot = find_child_slot(parent_idx, curr);
            if (nodes_[curr].count < min_entries) {
                remove_slot(nodes_[parent_idx], slot);
                nodes_[curr].parent = kInvalidNode;
                eliminated.push_back({curr, nodes_[curr].height});
            } else {
                update_entry_box(parent_idx, slot, compute_node_mbr(nodes_[curr]));
            }
            curr = parent_idx;
        }

        if (!eliminated.empty()) {
            std::vector<uint8_t> reinsert_called(
                static_cast<std::size_t>(nodes_[root_idx_].height) + 2, 0);
            // Pop top-down (higher-level subtrees first, leaves last)
            for (std::size_t idx = eliminated.size(); idx-- > 0;) {
                const uint32_t elim_idx = eliminated[idx].node_idx;
                const uint16_t target_h = eliminated[idx].height;
                const std::size_t cnt = nodes_[elim_idx].count;
                std::vector<Entry> orphans;
                orphans.reserve(cnt);
                for (std::size_t i = 0; i < cnt; ++i) {
                    orphans.push_back(
                        Entry{get_entry_box(nodes_[elim_idx], i), nodes_[elim_idx].children[i]});
                }
                free_node(elim_idx);
                for (const Entry& e : orphans) {
                    reinsert_called.assign(static_cast<std::size_t>(nodes_[root_idx_].height) + 2,
                                           0);
                    insert_entry(e.box, e.payload, target_h, reinsert_called);
                }
            }
        }

        while (root_idx_ != kInvalidNode && !nodes_[root_idx_].is_leaf &&
               nodes_[root_idx_].count == 1) {
            const uint32_t old_root = root_idx_;
            root_idx_ = nodes_[old_root].children[0];
            nodes_[root_idx_].parent = kInvalidNode;
            free_node(old_root);
        }

        if (root_idx_ != kInvalidNode && nodes_[root_idx_].is_leaf &&
            nodes_[root_idx_].count == 0) {
            free_node(root_idx_);
            root_idx_ = kInvalidNode;
        }
    }

    static void set_node_entry(Node& node, std::size_t slot, const box_type& box,
                               uint32_t payload) noexcept {
        for (std::size_t d = 0; d < Dim; ++d) {
            node.min_coords[d][slot] = box.min_corner[d];
            node.max_coords[d][slot] = box.max_corner[d];
        }
        node.children[slot] = payload;
    }

    [[nodiscard]] static box_type get_entry_box(const Node& node, std::size_t slot) noexcept {
        box_type box;
        for (std::size_t d = 0; d < Dim; ++d) {
            box.min_corner[d] = node.min_coords[d][slot];
            box.max_corner[d] = node.max_coords[d][slot];
        }
        return box;
    }

    [[nodiscard]] static box_type compute_node_mbr(const Node& node) noexcept {
        box_type mbr;
        if (node.count == 0) {
            return mbr;
        }
        for (std::size_t d = 0; d < Dim; ++d) {
            T lo = node.min_coords[d][0];
            T hi = node.max_coords[d][0];
            for (std::size_t i = 1; i < node.count; ++i) {
                lo = std::min(lo, node.min_coords[d][i]);
                hi = std::max(hi, node.max_coords[d][i]);
            }
            mbr.min_corner[d] = lo;
            mbr.max_corner[d] = hi;
        }
        return mbr;
    }

    [[nodiscard]] static box_type unite_boxes(const box_type& a, const box_type& b) noexcept {
        box_type u;
        for (std::size_t d = 0; d < Dim; ++d) {
            u.min_corner[d] = std::min(a.min_corner[d], b.min_corner[d]);
            u.max_corner[d] = std::max(a.max_corner[d], b.max_corner[d]);
        }
        return u;
    }

    [[nodiscard]] static double box_overlap(const box_type& a, const box_type& b) noexcept {
        double prod = 1.0;
        for (std::size_t d = 0; d < Dim; ++d) {
            const double hi = std::min(static_cast<double>(a.max_corner[d]),
                                       static_cast<double>(b.max_corner[d]));
            const double lo = std::max(static_cast<double>(a.min_corner[d]),
                                       static_cast<double>(b.min_corner[d]));
            const double extent = hi - lo;
            if (!(extent > 0.0)) {
                return 0.0;
            }
            prod *= extent;
        }
        return prod;
    }

    [[nodiscard]] static uint32_t valid_bitmask(std::size_t count) noexcept {
        if (count == 0) {
            return 0u;
        }
        if (count >= 32) {
            return 0xFFFFFFFFu;
        }
        return (1u << count) - 1u;
    }

    [[nodiscard]] static uint32_t intersect_chunk_mask_32(const Node& node, std::size_t base,
                                                          std::size_t chunk_count,
                                                          const box_type& query_box) noexcept {
        const uint32_t valid = valid_bitmask(chunk_count);
#if defined(ALGOAT_RSTAR_HAS_AVX2)
        if constexpr (std::is_same_v<T, float>) {
            uint32_t total_mask = 0;
            for (std::size_t offset = 0; offset < chunk_count; offset += 8) {
                const std::size_t idx = base + offset;
                __m256 active = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
                for (std::size_t d = 0; d < Dim; ++d) {
                    const __m256 n_min = _mm256_loadu_ps(&node.min_coords[d][idx]);
                    const __m256 n_max = _mm256_loadu_ps(&node.max_coords[d][idx]);
                    const __m256 q_min = _mm256_set1_ps(query_box.min_corner[d]);
                    const __m256 q_max = _mm256_set1_ps(query_box.max_corner[d]);
                    const __m256 le = _mm256_cmp_ps(n_min, q_max, _CMP_LE_OQ);
                    const __m256 ge = _mm256_cmp_ps(n_max, q_min, _CMP_GE_OQ);
                    active = _mm256_and_ps(active, _mm256_and_ps(le, ge));
                }
                const uint32_t m = static_cast<uint32_t>(_mm256_movemask_ps(active));
                total_mask |= (m << offset);
            }
            return total_mask & valid;
        } else if constexpr (std::is_same_v<T, double>) {
            uint32_t total_mask = 0;
            for (std::size_t offset = 0; offset < chunk_count; offset += 4) {
                const std::size_t idx = base + offset;
                __m256d active = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
                for (std::size_t d = 0; d < Dim; ++d) {
                    const __m256d n_min = _mm256_loadu_pd(&node.min_coords[d][idx]);
                    const __m256d n_max = _mm256_loadu_pd(&node.max_coords[d][idx]);
                    const __m256d q_min = _mm256_set1_pd(query_box.min_corner[d]);
                    const __m256d q_max = _mm256_set1_pd(query_box.max_corner[d]);
                    const __m256d le = _mm256_cmp_pd(n_min, q_max, _CMP_LE_OQ);
                    const __m256d ge = _mm256_cmp_pd(n_max, q_min, _CMP_GE_OQ);
                    active = _mm256_and_pd(active, _mm256_and_pd(le, ge));
                }
                const uint32_t m = static_cast<uint32_t>(_mm256_movemask_pd(active));
                total_mask |= (m << offset);
            }
            return total_mask & valid;
        }
#endif
        uint32_t mask = 0;
        for (std::size_t offset = 0; offset < chunk_count; ++offset) {
            const std::size_t idx = base + offset;
            bool hit = true;
            for (std::size_t d = 0; d < Dim; ++d) {
                if (!(node.min_coords[d][idx] <= query_box.max_corner[d] &&
                      node.max_coords[d][idx] >= query_box.min_corner[d])) {
                    hit = false;
                    break;
                }
            }
            if (hit) {
                mask |= (1u << offset);
            }
        }
        return mask & valid;
    }

    static void compute_node_mindist_sq(const Node& node, const std::array<T, Dim>& query_point,
                                        double* out_dist_sq) noexcept {
#if defined(ALGOAT_RSTAR_HAS_AVX2)
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            auto load_4d = [](const T* ptr) noexcept -> __m256d {
                if constexpr (std::is_same_v<T, float>) {
                    return _mm256_cvtps_pd(_mm_loadu_ps(ptr));
                } else {
                    return _mm256_loadu_pd(ptr);
                }
            };
            const __m256d zero = _mm256_setzero_pd();
            for (std::size_t base = 0; base < node.count; base += 4) {
                __m256d acc = zero;
                for (std::size_t d = 0; d < Dim; ++d) {
                    const __m256d q = _mm256_set1_pd(static_cast<double>(query_point[d]));
                    const __m256d n_min = load_4d(&node.min_coords[d][base]);
                    const __m256d n_max = load_4d(&node.max_coords[d][base]);
                    const __m256d lo_gap = _mm256_sub_pd(n_min, q);
                    const __m256d hi_gap = _mm256_sub_pd(q, n_max);
                    const __m256d gap = _mm256_max_pd(zero, _mm256_max_pd(lo_gap, hi_gap));
                    acc = _mm256_add_pd(acc, _mm256_mul_pd(gap, gap));
                }
                alignas(32) double buf[4];
                _mm256_store_pd(buf, acc);
                const std::size_t lim = std::min<std::size_t>(4, node.count - base);
                for (std::size_t j = 0; j < lim; ++j) {
                    out_dist_sq[base + j] = buf[j];
                }
            }
            return;
        }
#endif
        for (std::size_t i = 0; i < node.count; ++i) {
            double sum = 0.0;
            for (std::size_t d = 0; d < Dim; ++d) {
                const double q = static_cast<double>(query_point[d]);
                const double lo = static_cast<double>(node.min_coords[d][i]);
                const double hi = static_cast<double>(node.max_coords[d][i]);
                double gap = 0.0;
                if (q < lo) {
                    gap = lo - q;
                } else if (q > hi) {
                    gap = q - hi;
                }
                sum += gap * gap;
            }
            out_dist_sq[i] = sum;
        }
    }

    [[nodiscard]] std::size_t find_child_slot(uint32_t parent_idx,
                                              uint32_t child_idx) const noexcept {
        const Node& parent = nodes_[parent_idx];
        for (std::size_t i = 0; i < parent.count; ++i) {
            if (parent.children[i] == child_idx) {
                return i;
            }
        }
        return MaxEntries;
    }

    void update_entry_box(uint32_t parent_idx, std::size_t slot, const box_type& box) noexcept {
        Node& parent = nodes_[parent_idx];
        for (std::size_t d = 0; d < Dim; ++d) {
            parent.min_coords[d][slot] = box.min_corner[d];
            parent.max_coords[d][slot] = box.max_corner[d];
        }
    }

    void adjust_ancestors_grow(uint32_t node_idx, const box_type& added_box) noexcept {
        uint32_t curr = node_idx;
        while (nodes_[curr].parent != kInvalidNode) {
            const uint32_t parent_idx = nodes_[curr].parent;
            const std::size_t slot = find_child_slot(parent_idx, curr);
            Node& parent = nodes_[parent_idx];
            bool changed = false;
            for (std::size_t d = 0; d < Dim; ++d) {
                if (added_box.min_corner[d] < parent.min_coords[d][slot]) {
                    parent.min_coords[d][slot] = added_box.min_corner[d];
                    changed = true;
                }
                if (added_box.max_corner[d] > parent.max_coords[d][slot]) {
                    parent.max_coords[d][slot] = added_box.max_corner[d];
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }
            curr = parent_idx;
        }
    }

    void tighten_ancestors(uint32_t node_idx) noexcept {
        uint32_t curr = node_idx;
        while (nodes_[curr].parent != kInvalidNode) {
            const uint32_t parent_idx = nodes_[curr].parent;
            const std::size_t slot = find_child_slot(parent_idx, curr);
            const box_type tight = compute_node_mbr(nodes_[curr]);
            update_entry_box(parent_idx, slot, tight);
            curr = parent_idx;
        }
    }

    [[nodiscard]] uint32_t choose_subtree(const box_type& box, uint16_t target_height) const {
        uint32_t curr = root_idx_;
        while (nodes_[curr].height > target_height) {
            const Node& node = nodes_[curr];
            const std::size_t n = node.count;

            if (node.height == 1) {
                // Child pointers point to leaves: minimize overlap enlargement
                std::array<box_type, MaxEntries> entry_boxes{};
                std::array<double, MaxEntries> areas{};
                std::array<double, MaxEntries> area_enlargements{};
                std::array<std::size_t, MaxEntries> indices{};

                for (std::size_t i = 0; i < n; ++i) {
                    entry_boxes[i] = get_entry_box(node, i);
                    areas[i] = entry_boxes[i].area();
                    area_enlargements[i] = unite_boxes(entry_boxes[i], box).area() - areas[i];
                    indices[i] = i;
                }

                std::size_t candidate_count = n;
                if (n > kOverlapCostThreshold) {
                    std::partial_sort(indices.begin(), indices.begin() + kOverlapCostThreshold,
                                      indices.begin() + static_cast<std::ptrdiff_t>(n),
                                      [&](std::size_t a, std::size_t b) {
                                          if (area_enlargements[a] != area_enlargements[b]) {
                                              return area_enlargements[a] < area_enlargements[b];
                                          }
                                          return areas[a] < areas[b];
                                      });
                    candidate_count = kOverlapCostThreshold;
                }

                std::size_t best_idx = indices[0];
                double best_overlap_enl = std::numeric_limits<double>::infinity();
                double best_area_enl = std::numeric_limits<double>::infinity();
                double best_area = std::numeric_limits<double>::infinity();

                for (std::size_t ci = 0; ci < candidate_count; ++ci) {
                    const std::size_t j = indices[ci];
                    const box_type enlarged = unite_boxes(entry_boxes[j], box);
                    double old_ov = 0.0;
                    double new_ov = 0.0;
                    for (std::size_t i = 0; i < n; ++i) {
                        if (i == j) {
                            continue;
                        }
                        old_ov += box_overlap(entry_boxes[j], entry_boxes[i]);
                        new_ov += box_overlap(enlarged, entry_boxes[i]);
                    }
                    const double ov_enl = new_ov - old_ov;
                    if (ov_enl < best_overlap_enl ||
                        (ov_enl == best_overlap_enl &&
                         (area_enlargements[j] < best_area_enl ||
                          (area_enlargements[j] == best_area_enl && areas[j] < best_area)))) {
                        best_overlap_enl = ov_enl;
                        best_area_enl = area_enlargements[j];
                        best_area = areas[j];
                        best_idx = j;
                    }
                }
                curr = node.children[best_idx];
            } else {
                // Higher directory levels: minimize area enlargement, tie-break by area
                std::size_t best_idx = 0;
                double best_area_enl = std::numeric_limits<double>::infinity();
                double best_area = std::numeric_limits<double>::infinity();

                for (std::size_t i = 0; i < n; ++i) {
                    const box_type eb = get_entry_box(node, i);
                    const double a = eb.area();
                    const double enl = unite_boxes(eb, box).area() - a;
                    if (enl < best_area_enl || (enl == best_area_enl && a < best_area)) {
                        best_area_enl = enl;
                        best_area = a;
                        best_idx = i;
                    }
                }
                curr = node.children[best_idx];
            }
        }
        return curr;
    }

    void insert_entry(const box_type& box, uint32_t payload, uint16_t target_height,
                      std::vector<uint8_t>& reinsert_called) {
        const uint32_t node_idx = choose_subtree(box, target_height);
        if (nodes_[node_idx].count < MaxEntries) {
            const std::size_t slot = nodes_[node_idx].count++;
            set_node_entry(nodes_[node_idx], slot, box, payload);
            if (!nodes_[node_idx].is_leaf) {
                nodes_[payload].parent = node_idx;
            }
            adjust_ancestors_grow(node_idx, box);
            return;
        }

        std::array<Entry, MaxEntries + 1> entries{};
        for (std::size_t i = 0; i < MaxEntries; ++i) {
            entries[i] = Entry{get_entry_box(nodes_[node_idx], i), nodes_[node_idx].children[i]};
        }
        entries[MaxEntries] = Entry{box, payload};

        handle_overflow(node_idx, entries, reinsert_called);
    }

    void handle_overflow(uint32_t node_idx, std::array<Entry, MaxEntries + 1>& entries,
                         std::vector<uint8_t>& reinsert_called) {
        const uint16_t h = nodes_[node_idx].height;
        if (h >= reinsert_called.size()) {
            reinsert_called.resize(static_cast<std::size_t>(h) + 2, 0);
        }

        if (node_idx != root_idx_ && reinsert_called[h] == 0) {
            reinsert_called[h] = 1;
            reinsert_entries(node_idx, entries, reinsert_called);
        } else {
            split_and_propagate(node_idx, entries, reinsert_called);
        }
    }

    void reinsert_entries(uint32_t node_idx, std::array<Entry, MaxEntries + 1>& entries,
                          std::vector<uint8_t>& reinsert_called) {
        box_type full_mbr = entries[0].box;
        for (std::size_t i = 1; i <= MaxEntries; ++i) {
            full_mbr = unite_boxes(full_mbr, entries[i].box);
        }

        std::array<double, Dim> node_center{};
        for (std::size_t d = 0; d < Dim; ++d) {
            node_center[d] = box_center_coord(full_mbr, d);
        }

        std::array<std::pair<double, std::size_t>, MaxEntries + 1> dists{};
        for (std::size_t i = 0; i <= MaxEntries; ++i) {
            double d2 = 0.0;
            for (std::size_t d = 0; d < Dim; ++d) {
                const double diff = box_center_coord(entries[i].box, d) - node_center[d];
                d2 += diff * diff;
            }
            dists[i] = {d2, i};
        }

        // Sort descending: indices [0, reinsert_count) are farthest from node center
        std::sort(dists.begin(), dists.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        std::array<Entry, reinsert_count> to_reinsert{};
        for (std::size_t i = 0; i < reinsert_count; ++i) {
            to_reinsert[i] = entries[dists[i].second];
        }

        const uint16_t node_height = nodes_[node_idx].height;
        const bool is_leaf = nodes_[node_idx].is_leaf;
        nodes_[node_idx].reset_bounds();
        const std::size_t keep_count = (MaxEntries + 1) - reinsert_count;
        nodes_[node_idx].count = static_cast<uint32_t>(keep_count);

        for (std::size_t i = 0; i < keep_count; ++i) {
            const Entry& e = entries[dists[reinsert_count + i].second];
            set_node_entry(nodes_[node_idx], i, e.box, e.payload);
            if (!is_leaf) {
                nodes_[e.payload].parent = node_idx;
            }
        }

        // Tighten node's MBR in ancestors BEFORE reinserting removed entries
        tighten_ancestors(node_idx);

        // Close reinsert: insert closest of the removed entries first
        for (std::size_t i = reinsert_count; i-- > 0;) {
            insert_entry(to_reinsert[i].box, to_reinsert[i].payload, node_height, reinsert_called);
        }
    }

    static void split_entries_rstar(const std::array<Entry, MaxEntries + 1>& entries,
                                    std::array<std::size_t, MaxEntries + 1>& out_order,
                                    std::size_t& out_split_k) {
        constexpr std::size_t N = MaxEntries + 1;
        constexpr std::size_t num_dists = N - 2 * min_entries + 1;

        std::array<std::size_t, N> sorted_lo{};
        std::array<std::size_t, N> sorted_hi{};
        std::array<box_type, N> prefix_mbr{};
        std::array<box_type, N> suffix_mbr{};

        double best_margin_sum = std::numeric_limits<double>::infinity();
        std::array<std::size_t, N> best_axis_lo{};
        std::array<std::size_t, N> best_axis_hi{};

        auto build_prefix_suffix = [&](const std::array<std::size_t, N>& ord) {
            prefix_mbr[0] = entries[ord[0]].box;
            for (std::size_t i = 1; i < N; ++i) {
                prefix_mbr[i] = unite_boxes(prefix_mbr[i - 1], entries[ord[i]].box);
            }
            suffix_mbr[N - 1] = entries[ord[N - 1]].box;
            for (std::size_t i = N - 1; i-- > 0;) {
                suffix_mbr[i] = unite_boxes(entries[ord[i]].box, suffix_mbr[i + 1]);
            }
        };

        auto eval_sort_margins = [&](const std::array<std::size_t, N>& ord) -> double {
            build_prefix_suffix(ord);
            double sum = 0.0;
            for (std::size_t k = 0; k < num_dists; ++k) {
                const std::size_t split = min_entries + k;
                sum += prefix_mbr[split - 1].margin() + suffix_mbr[split].margin();
            }
            return sum;
        };

        for (std::size_t d = 0; d < Dim; ++d) {
            std::iota(sorted_lo.begin(), sorted_lo.end(), std::size_t{0});
            std::sort(sorted_lo.begin(), sorted_lo.end(), [&](std::size_t a, std::size_t b) {
                if (entries[a].box.min_corner[d] != entries[b].box.min_corner[d]) {
                    return entries[a].box.min_corner[d] < entries[b].box.min_corner[d];
                }
                return entries[a].box.max_corner[d] < entries[b].box.max_corner[d];
            });

            std::iota(sorted_hi.begin(), sorted_hi.end(), std::size_t{0});
            std::sort(sorted_hi.begin(), sorted_hi.end(), [&](std::size_t a, std::size_t b) {
                if (entries[a].box.max_corner[d] != entries[b].box.max_corner[d]) {
                    return entries[a].box.max_corner[d] < entries[b].box.max_corner[d];
                }
                return entries[a].box.min_corner[d] < entries[b].box.min_corner[d];
            });

            const double margin_sum = eval_sort_margins(sorted_lo) + eval_sort_margins(sorted_hi);
            if (margin_sum < best_margin_sum) {
                best_margin_sum = margin_sum;
                best_axis_lo = sorted_lo;
                best_axis_hi = sorted_hi;
            }
        }

        double best_overlap = std::numeric_limits<double>::infinity();
        double best_area = std::numeric_limits<double>::infinity();
        bool use_hi_sort = false;
        std::size_t best_split = min_entries;

        auto eval_split_index = [&](const std::array<std::size_t, N>& ord, bool is_hi) {
            build_prefix_suffix(ord);
            for (std::size_t k = 0; k < num_dists; ++k) {
                const std::size_t split = min_entries + k;
                const box_type& b1 = prefix_mbr[split - 1];
                const box_type& b2 = suffix_mbr[split];
                const double ov = box_overlap(b1, b2);
                const double ar = b1.area() + b2.area();
                if (ov < best_overlap || (ov == best_overlap && ar < best_area)) {
                    best_overlap = ov;
                    best_area = ar;
                    use_hi_sort = is_hi;
                    best_split = split;
                }
            }
        };

        eval_split_index(best_axis_lo, false);
        eval_split_index(best_axis_hi, true);

        out_order = use_hi_sort ? best_axis_hi : best_axis_lo;
        out_split_k = best_split;
    }

    void split_and_propagate(uint32_t node_idx, std::array<Entry, MaxEntries + 1>& entries,
                             std::vector<uint8_t>& reinsert_called) {
        std::array<std::size_t, MaxEntries + 1> order{};
        std::size_t split_k = min_entries;
        split_entries_rstar(entries, order, split_k);

        const bool is_leaf = nodes_[node_idx].is_leaf;
        const uint16_t h = nodes_[node_idx].height;
        const uint32_t parent_idx = nodes_[node_idx].parent;

        // Populate group 1 in node_idx
        nodes_[node_idx].reset_bounds();
        nodes_[node_idx].count = static_cast<uint32_t>(split_k);
        for (std::size_t i = 0; i < split_k; ++i) {
            const Entry& e = entries[order[i]];
            set_node_entry(nodes_[node_idx], i, e.box, e.payload);
            if (!is_leaf) {
                nodes_[e.payload].parent = node_idx;
            }
        }
        const box_type mbr1 = compute_node_mbr(nodes_[node_idx]);

        // Allocate sibling for group 2
        const uint32_t sibling_idx = allocate_node(is_leaf, h);
        const std::size_t group2_count = (MaxEntries + 1) - split_k;
        nodes_[sibling_idx].count = static_cast<uint32_t>(group2_count);
        for (std::size_t i = 0; i < group2_count; ++i) {
            const Entry& e = entries[order[split_k + i]];
            set_node_entry(nodes_[sibling_idx], i, e.box, e.payload);
            if (!is_leaf) {
                nodes_[e.payload].parent = sibling_idx;
            }
        }
        const box_type mbr2 = compute_node_mbr(nodes_[sibling_idx]);

        if (parent_idx == kInvalidNode) {
            const uint32_t new_root =
                allocate_node(/*is_leaf=*/false, static_cast<uint16_t>(h + 1));
            set_node_entry(nodes_[new_root], 0, mbr1, node_idx);
            set_node_entry(nodes_[new_root], 1, mbr2, sibling_idx);
            nodes_[new_root].count = 2;
            nodes_[node_idx].parent = new_root;
            nodes_[sibling_idx].parent = new_root;
            root_idx_ = new_root;
            return;
        }

        // Update node_idx's MBR in parent_idx
        const std::size_t slot = find_child_slot(parent_idx, node_idx);
        update_entry_box(parent_idx, slot, mbr1);

        if (nodes_[parent_idx].count < MaxEntries) {
            const std::size_t sib_slot = nodes_[parent_idx].count++;
            set_node_entry(nodes_[parent_idx], sib_slot, mbr2, sibling_idx);
            nodes_[sibling_idx].parent = parent_idx;
            tighten_ancestors(parent_idx);
            return;
        }

        // Parent overflows: update ancestor bounds for mbr1 before overflow treatment
        tighten_ancestors(parent_idx);

        std::array<Entry, MaxEntries + 1> parent_entries{};
        for (std::size_t i = 0; i < MaxEntries; ++i) {
            parent_entries[i] =
                Entry{get_entry_box(nodes_[parent_idx], i), nodes_[parent_idx].children[i]};
        }
        parent_entries[MaxEntries] = Entry{mbr2, sibling_idx};

        handle_overflow(parent_idx, parent_entries, reinsert_called);
    }
};

} // namespace algoat::geometry
