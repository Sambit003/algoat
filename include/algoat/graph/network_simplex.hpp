/**
 * @file network_simplex.hpp
 * @brief Primal Network Simplex solver for Minimum-Cost Flow.
 *
 * Implements Cunningham's strongly feasible spanning tree basis (1976),
 * Ahuja-Magnanti-Orlin Chapter 11 Augmented Threaded Index (ATI) non-recursive
 * tree and dual potential updates, and Block Search pivot pricing (B = 64)
 * following LEMON engineering patterns (Kovacs 2015).
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace algoat::graph {

namespace detail {

/**
 * @struct PortableInt128
 * @brief Minimal signed 128-bit two's-complement integer fallback for compilers without __int128.
 */
struct PortableInt128 {
    uint64_t lo{0};
    int64_t hi{0};

    constexpr PortableInt128() noexcept = default;

    template <std::signed_integral T>
    constexpr explicit PortableInt128(T v) noexcept
        : lo(static_cast<uint64_t>(static_cast<int64_t>(v))),
          hi(static_cast<int64_t>(v) < 0 ? -1 : 0) {}

    constexpr explicit PortableInt128(uint64_t v) noexcept : lo(v), hi(0) {}
    constexpr PortableInt128(uint64_t low, int64_t high) noexcept : lo(low), hi(high) {}

    template <std::signed_integral Target>
    [[nodiscard]] constexpr explicit operator Target() const noexcept {
        return static_cast<Target>(static_cast<int64_t>(lo));
    }

    [[nodiscard]] constexpr PortableInt128 operator-() const noexcept {
        const uint64_t new_lo = ~lo + 1u;
        const uint64_t carry = (new_lo == 0u) ? 1u : 0u;
        const uint64_t new_hi = ~static_cast<uint64_t>(hi) + carry;
        return PortableInt128{new_lo, static_cast<int64_t>(new_hi)};
    }

    [[nodiscard]] constexpr PortableInt128 operator+(const PortableInt128& rhs) const noexcept {
        const uint64_t new_lo = lo + rhs.lo;
        const uint64_t carry = (new_lo < lo) ? 1u : 0u;
        const uint64_t new_hi = static_cast<uint64_t>(hi) + static_cast<uint64_t>(rhs.hi) + carry;
        return PortableInt128{new_lo, static_cast<int64_t>(new_hi)};
    }

    [[nodiscard]] constexpr PortableInt128 operator-(const PortableInt128& rhs) const noexcept {
        const uint64_t new_lo = lo - rhs.lo;
        const uint64_t borrow = (lo < rhs.lo) ? 1u : 0u;
        const uint64_t new_hi = static_cast<uint64_t>(hi) - static_cast<uint64_t>(rhs.hi) - borrow;
        return PortableInt128{new_lo, static_cast<int64_t>(new_hi)};
    }

    [[nodiscard]] constexpr PortableInt128 operator*(const PortableInt128& rhs) const noexcept {
        const uint64_t a0 = lo & 0xFFFFFFFFULL;
        const uint64_t a1 = lo >> 32u;
        const uint64_t b0 = rhs.lo & 0xFFFFFFFFULL;
        const uint64_t b1 = rhs.lo >> 32u;

        const uint64_t p00 = a0 * b0;
        const uint64_t p01 = a0 * b1;
        const uint64_t p10 = a1 * b0;
        const uint64_t p11 = a1 * b1;

        const uint64_t mid = (p00 >> 32u) + (p01 & 0xFFFFFFFFULL) + (p10 & 0xFFFFFFFFULL);
        const uint64_t new_lo = (p00 & 0xFFFFFFFFULL) | (mid << 32u);
        const uint64_t cross_hi = p11 + (p01 >> 32u) + (p10 >> 32u) + (mid >> 32u);
        const uint64_t new_hi =
            cross_hi + static_cast<uint64_t>(hi) * rhs.lo + lo * static_cast<uint64_t>(rhs.hi);
        return PortableInt128{new_lo, static_cast<int64_t>(new_hi)};
    }

    constexpr PortableInt128& operator+=(const PortableInt128& rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }

    constexpr PortableInt128& operator-=(const PortableInt128& rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }

    [[nodiscard]] constexpr bool operator==(const PortableInt128& rhs) const noexcept {
        return hi == rhs.hi && lo == rhs.lo;
    }

    [[nodiscard]] constexpr bool operator!=(const PortableInt128& rhs) const noexcept {
        return !(*this == rhs);
    }

    [[nodiscard]] constexpr bool operator<(const PortableInt128& rhs) const noexcept {
        return (hi < rhs.hi) || (hi == rhs.hi && lo < rhs.lo);
    }

    [[nodiscard]] constexpr bool operator>(const PortableInt128& rhs) const noexcept {
        return rhs < *this;
    }

    [[nodiscard]] constexpr bool operator<=(const PortableInt128& rhs) const noexcept {
        return !(rhs < *this);
    }

    [[nodiscard]] constexpr bool operator>=(const PortableInt128& rhs) const noexcept {
        return !(*this < rhs);
    }
};

#if defined(__SIZEOF_INT128__)
__extension__ using NativeOrPortableInt128 = __int128;
#else
using NativeOrPortableInt128 = PortableInt128;
#endif

template <typename T>
using WideSigned = std::conditional_t<(sizeof(T) < 8), int64_t, NativeOrPortableInt128>;

template <typename T> [[nodiscard]] constexpr T signed_max() noexcept {
    if constexpr (std::is_same_v<T, PortableInt128>) {
        return PortableInt128{0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL};
#if defined(__SIZEOF_INT128__)
    } else if constexpr (std::is_same_v<T, NativeOrPortableInt128>) {
        __extension__ const unsigned __int128 umax = ~static_cast<unsigned __int128>(0) >> 1;
        return static_cast<NativeOrPortableInt128>(umax);
#endif
    } else {
        return std::numeric_limits<T>::max();
    }
}

template <typename T> [[nodiscard]] constexpr uint64_t abs_to_u64(T val) noexcept {
    if (val < 0) {
        return static_cast<uint64_t>(0) - static_cast<uint64_t>(static_cast<int64_t>(val));
    }
    return static_cast<uint64_t>(val);
}

} // namespace detail

/**
 * @class NetworkSimplex
 * @brief Primal Network Simplex solver for directed Minimum-Cost Flow problems.
 *
 * Reduced costs use the AMO convention: rc(u, v) = cost(u, v) - pi[u] + pi[v].
 * Basis trees are maintained in Convention B strongly feasible canonical form
 * (zero-flow basic edges directed toward root; saturated basic edges directed
 * away from root) using a 4-array Augmented Threaded Index representation and
 * Block Search pivot pricing (B = 64).
 */
template <typename CostType = int64_t, typename CapacityType = int64_t> class NetworkSimplex {
    static_assert(std::signed_integral<CostType>, "CostType must be a signed integral type");
    static_assert(std::signed_integral<CapacityType>,
                  "CapacityType must be a signed integral type");

public:
    using cost_type = CostType;
    using capacity_type = CapacityType;

    static constexpr capacity_type kInfiniteCapacity = std::numeric_limits<capacity_type>::max();

    enum class ProblemType : uint8_t { Optimal, Infeasible, Unbounded };

    explicit NetworkSimplex(std::size_t num_vertices)
        : v_(num_vertices), raw_supplies_(num_vertices, 0) {}

    /**
     * @brief Adds a directed edge with lower/upper capacity bounds and unit cost.
     * @throws std::out_of_range if either endpoint >= num_vertices().
     */
    void add_edge(std::size_t from, std::size_t to, capacity_type capacity, cost_type cost,
                  capacity_type lower_bound = 0) {
        if (from >= v_ || to >= v_) {
            throw std::out_of_range("Vertex index out of bounds in NetworkSimplex::add_edge");
        }
        raw_edges_.push_back(RawEdge{from, to, capacity, cost, lower_bound});
        ++e_;
    }

    /**
     * @brief Sets external supply (> 0) or demand (< 0) for a vertex.
     * @throws std::out_of_range if vertex >= num_vertices().
     */
    void set_supply(std::size_t vertex, capacity_type supply) {
        if (vertex >= v_) {
            throw std::out_of_range("Vertex index out of bounds in NetworkSimplex::set_supply");
        }
        raw_supplies_[vertex] = supply;
    }

    /**
     * @brief Solves the Minimum-Cost Flow problem. Safe to invoke repeatedly.
     */
    [[nodiscard]] ProblemType run() {
        flow_out_.assign(e_, 0);
        total_cost_ = 0;
        potential_.assign(v_, 0);

        if (v_ == 0) {
            return ProblemType::Optimal;
        }

        using WideCap = detail::WideSigned<capacity_type>;
        using WideCost = detail::WideSigned<cost_type>;

        WideCap supply_sum{0};
        std::vector<WideCap> shifted_supply_wide(v_, WideCap{0});
        for (std::size_t u = 0; u < v_; ++u) {
            const auto s = static_cast<WideCap>(raw_supplies_[u]);
            supply_sum += s;
            shifted_supply_wide[u] = s;
        }
        if (supply_sum != WideCap{0}) {
            return ProblemType::Infeasible;
        }

        std::size_t active_edges = 0;
        bool has_unbounded_self_loop = false;
        raw_to_internal_.assign(e_, kNullIndex);

        source_.clear();
        target_.clear();
        cost_buf_.clear();
        std::vector<WideCap> active_cap_wide;
        std::vector<uint8_t> active_is_inf;

        WideCap sum_finite_spans{0};
        bool needs_wide_cap = false;
        const WideCap narrow_cap_limit = static_cast<WideCap>(kInfiniteCapacity);

        for (std::size_t i = 0; i < e_; ++i) {
            const RawEdge& edge = raw_edges_[i];
            if (edge.lower_bound > edge.capacity) {
                flow_out_.assign(e_, 0);
                return ProblemType::Infeasible;
            }

            if (edge.from == edge.to) {
                if (edge.cost < 0) {
                    if (edge.capacity >= kInfiniteCapacity) {
                        has_unbounded_self_loop = true;
                        flow_out_[i] = edge.lower_bound;
                    } else {
                        flow_out_[i] = edge.capacity;
                    }
                } else {
                    flow_out_[i] = edge.lower_bound;
                }
                continue;
            }

            const auto lb = static_cast<WideCap>(edge.lower_bound);
            shifted_supply_wide[edge.from] -= lb;
            shifted_supply_wide[edge.to] += lb;

            if (edge.lower_bound == edge.capacity) {
                flow_out_[i] = edge.lower_bound;
                continue;
            }

            const bool is_inf = (edge.capacity >= kInfiniteCapacity);
            const WideCap span =
                is_inf ? detail::signed_max<WideCap>() : (static_cast<WideCap>(edge.capacity) - lb);
            if (!is_inf) {
                if (span >= narrow_cap_limit) {
                    needs_wide_cap = true;
                }
                sum_finite_spans += span;
            }

            raw_to_internal_[i] = active_edges++;
            source_.push_back(edge.from);
            target_.push_back(edge.to);
            active_cap_wide.push_back(span);
            active_is_inf.push_back(static_cast<uint8_t>(is_inf ? 1 : 0));
            cost_buf_.push_back(edge.cost);
        }

        WideCap pos_supply_sum{0};
        for (const WideCap& s : shifted_supply_wide) {
            if (s >= narrow_cap_limit || s <= -narrow_cap_limit) {
                needs_wide_cap = true;
            }
            if (s > WideCap{0}) {
                pos_supply_sum += s;
            }
        }
        if (pos_supply_sum + sum_finite_spans >= narrow_cap_limit) {
            needs_wide_cap = true;
        }

        if (active_edges == 0) {
            for (const WideCap& s : shifted_supply_wide) {
                if (s != WideCap{0}) {
                    flow_out_.assign(e_, 0);
                    return ProblemType::Infeasible;
                }
            }
            if (has_unbounded_self_loop) {
                flow_out_.assign(e_, 0);
                return ProblemType::Unbounded;
            }
            compute_total_cost<WideCost>();
            return ProblemType::Optimal;
        }

        uint64_t max_abs_cost = 0;
        for (const cost_type c : cost_buf_) {
            max_abs_cost = std::max(max_abs_cost, detail::abs_to_u64(c));
        }
        const WideCost art_cost_wide =
            WideCost{1} + static_cast<WideCost>(v_) * static_cast<WideCost>(max_abs_cost);
        const bool use_narrow_cost =
            (art_cost_wide * WideCost{4} + static_cast<WideCost>(max_abs_cost) + WideCost{16} <
             static_cast<WideCost>(std::numeric_limits<cost_type>::max()));

        auto run_with_cap = [&](auto cap_tag) -> ProblemType {
            using WorkCap = decltype(cap_tag);
            std::vector<WorkCap> extracted_flows;

            if (has_unbounded_self_loop) {
                const auto p1 =
                    solve_core<int64_t, WorkCap>(active_edges, shifted_supply_wide, active_cap_wide,
                                                 active_is_inf, int64_t{1}, true, extracted_flows);
                flow_out_.assign(e_, 0);
                potential_.assign(v_, 0);
                return (p1.status == ProblemType::Optimal) ? ProblemType::Unbounded
                                                           : ProblemType::Infeasible;
            }

            SolveOutcome outcome =
                use_narrow_cost
                    ? solve_core<cost_type, WorkCap>(
                          active_edges, shifted_supply_wide, active_cap_wide, active_is_inf,
                          static_cast<cost_type>(art_cost_wide), false, extracted_flows)
                    : solve_core<WideCost, WorkCap>(active_edges, shifted_supply_wide,
                                                    active_cap_wide, active_is_inf, art_cost_wide,
                                                    false, extracted_flows);

            if (outcome.status == ProblemType::Unbounded && outcome.has_nonzero_artificial) {
                const auto p1 =
                    solve_core<int64_t, WorkCap>(active_edges, shifted_supply_wide, active_cap_wide,
                                                 active_is_inf, int64_t{1}, true, extracted_flows);
                if (p1.status != ProblemType::Optimal) {
                    outcome.status = ProblemType::Infeasible;
                }
            }

            if (outcome.status != ProblemType::Optimal) {
                flow_out_.assign(e_, 0);
                potential_.assign(v_, 0);
                return outcome.status;
            }

            for (std::size_t i = 0; i < e_; ++i) {
                if (raw_to_internal_[i] != kNullIndex) {
                    const WideCap shifted =
                        static_cast<WideCap>(extracted_flows[raw_to_internal_[i]]);
                    flow_out_[i] = static_cast<capacity_type>(
                        static_cast<WideCap>(raw_edges_[i].lower_bound) + shifted);
                }
            }
            compute_total_cost<WideCost>();
            return ProblemType::Optimal;
        };

        return needs_wide_cap ? run_with_cap(WideCap{0}) : run_with_cap(capacity_type{0});
    }

    /**
     * @brief Returns the minimum total objective cost from the last Optimal run.
     */
    [[nodiscard]] cost_type total_cost() const noexcept {
        return total_cost_;
    }

    /**
     * @brief Returns the flow assigned to edge @p edge_index (or 0 if out of range).
     */
    [[nodiscard]] capacity_type flow(std::size_t edge_index) const noexcept {
        return edge_index < flow_out_.size() ? flow_out_[edge_index] : capacity_type{0};
    }

    /**
     * @brief Returns the dual potential pi[vertex] from the last Optimal run (or 0 if out of
     * range).
     */
    [[nodiscard]] cost_type potential(std::size_t vertex) const noexcept {
        return vertex < potential_.size() ? potential_[vertex] : cost_type{0};
    }

    /**
     * @brief Returns the number of vertices in the network.
     */
    [[nodiscard]] std::size_t num_vertices() const noexcept {
        return v_;
    }

    /**
     * @brief Returns the number of directed edges added to the network.
     */
    [[nodiscard]] std::size_t num_edges() const noexcept {
        return e_;
    }

private:
    static constexpr std::size_t kNullIndex = std::numeric_limits<std::size_t>::max();
    static constexpr std::size_t kBlockSize = 64;

    enum class ArcState : int8_t { Upper = -1, Tree = 0, Lower = 1 };
    enum class ArcDir : int8_t { Down = -1, Up = 1 };
    enum class LeavingSide : uint8_t { EnteringEdge = 0, FirstPath = 1, SecondPath = 2 };

    struct RawEdge {
        std::size_t from{0};
        std::size_t to{0};
        capacity_type capacity{0};
        cost_type cost{0};
        capacity_type lower_bound{0};
    };

    struct SolveOutcome {
        ProblemType status{ProblemType::Optimal};
        bool has_nonzero_artificial{false};
    };

    std::size_t v_{0};
    std::size_t e_{0};

    // Parallel flat arrays representing the spanning tree basis
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> pred_edge_;
    std::vector<std::size_t> depth_;
    std::vector<std::size_t> thread_;  // Preorder depth-first traversal successor
    std::vector<cost_type> potential_; // Node dual potentials pi[u]

    // Auxiliary basis metadata and flat edge endpoint/cost buffers
    std::vector<std::size_t> rev_thread_;
    std::vector<ArcDir> pred_dir_;
    std::vector<std::size_t> source_;
    std::vector<std::size_t> target_;
    std::vector<ArcState> state_;
    std::vector<cost_type> cost_buf_;

    // Immutable problem specification & extracted outputs
    std::vector<RawEdge> raw_edges_;
    std::vector<capacity_type> raw_supplies_;
    std::vector<std::size_t> raw_to_internal_;
    std::vector<capacity_type> flow_out_;
    cost_type total_cost_{0};

    template <typename WideCost> void compute_total_cost() noexcept {
        WideCost total{0};
        for (std::size_t i = 0; i < e_; ++i) {
            total +=
                static_cast<WideCost>(flow_out_[i]) * static_cast<WideCost>(raw_edges_[i].cost);
        }
        total_cost_ = static_cast<cost_type>(total);
    }

    template <typename WorkCap>
    [[nodiscard]] bool
    has_nonzero_artificial_flow(std::size_t active_edges,
                                const std::vector<WorkCap>& work_flow) const noexcept {
        for (std::size_t e = active_edges; e < active_edges + v_; ++e) {
            if (work_flow[e] != WorkCap{0}) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] std::size_t find_join_node(std::size_t u, std::size_t v) const noexcept {
        while (depth_[u] > depth_[v]) {
            u = parent_[u];
        }
        while (depth_[v] > depth_[u]) {
            v = parent_[v];
        }
        while (u != v) {
            u = parent_[u];
            v = parent_[v];
        }
        return u;
    }

    template <typename WorkCost>
    void update_tree_and_potentials(std::size_t in_edge, std::size_t u_in, std::size_t v_in,
                                    std::size_t u_out, const std::vector<WorkCost>& work_cost,
                                    std::vector<WorkCost>& pi) noexcept {
        const ArcDir in_dir = (source_[in_edge] == u_in) ? ArcDir::Up : ArcDir::Down;
        const WorkCost delta_pi =
            pi[v_in] - pi[u_in] +
            static_cast<WorkCost>(static_cast<int8_t>(in_dir)) * work_cost[in_edge];

        const std::size_t before_out = rev_thread_[u_out];

        auto apply_shift = [](std::size_t& d, WorkCost& p, std::ptrdiff_t depth_diff,
                              WorkCost pot_diff) noexcept {
            d = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(d) + depth_diff);
            p += pot_diff;
        };

        std::size_t stem = u_in;
        std::size_t old_depth = depth_[stem];
        std::size_t new_depth = depth_[v_in] + 1;
        std::ptrdiff_t dd =
            static_cast<std::ptrdiff_t>(new_depth) - static_cast<std::ptrdiff_t>(old_depth);

        std::size_t x = stem;
        apply_shift(depth_[x], pi[x], dd, delta_pi);
        while (depth_[thread_[x]] > old_depth) {
            x = thread_[x];
            apply_shift(depth_[x], pi[x], dd, delta_pi);
        }
        std::size_t last = x;
        std::size_t after = thread_[last];

        std::size_t new_parent = v_in;
        std::size_t new_arc = in_edge;
        ArcDir new_dir = in_dir;

        while (stem != u_out) {
            const std::size_t next_stem = parent_[stem];
            const std::size_t old_arc = pred_edge_[stem];
            const ArcDir old_dir = pred_dir_[stem];

            parent_[stem] = new_parent;
            pred_edge_[stem] = new_arc;
            pred_dir_[stem] = new_dir;

            thread_[last] = next_stem;
            rev_thread_[next_stem] = last;

            old_depth = depth_[next_stem];
            ++new_depth;
            dd = static_cast<std::ptrdiff_t>(new_depth) - static_cast<std::ptrdiff_t>(old_depth);

            x = next_stem;
            apply_shift(depth_[x], pi[x], dd, delta_pi);
            while (thread_[x] != stem) {
                x = thread_[x];
                apply_shift(depth_[x], pi[x], dd, delta_pi);
            }

            thread_[x] = after;
            rev_thread_[after] = x;

            while (depth_[thread_[x]] > old_depth) {
                x = thread_[x];
                apply_shift(depth_[x], pi[x], dd, delta_pi);
            }

            last = x;
            after = thread_[last];

            new_parent = stem;
            new_arc = old_arc;
            new_dir = (old_dir == ArcDir::Up) ? ArcDir::Down : ArcDir::Up;
            stem = next_stem;
        }

        parent_[u_out] = new_parent;
        pred_edge_[u_out] = new_arc;
        pred_dir_[u_out] = new_dir;

        thread_[before_out] = after;
        rev_thread_[after] = before_out;

        const std::size_t v_in_next = thread_[v_in];
        thread_[last] = v_in_next;
        rev_thread_[v_in_next] = last;
        thread_[v_in] = u_in;
        rev_thread_[u_in] = v_in;
    }

    template <typename WorkCost, typename WorkCap>
    [[nodiscard]] bool execute_pivot(std::size_t in_edge, const std::vector<WorkCap>& work_cap,
                                     std::vector<WorkCap>& work_flow,
                                     const std::vector<WorkCost>& work_cost,
                                     std::vector<WorkCost>& pi) noexcept {
        constexpr WorkCap kWorkInf = detail::signed_max<WorkCap>();
        const bool forward_in = (state_[in_edge] == ArcState::Lower);
        const std::size_t first = forward_in ? source_[in_edge] : target_[in_edge];
        const std::size_t second = forward_in ? target_[in_edge] : source_[in_edge];
        const std::size_t join = find_join_node(first, second);

        WorkCap delta = work_cap[in_edge];
        LeavingSide side = LeavingSide::EnteringEdge;
        std::size_t u_out = kNullIndex;

        // Convention B anti-cycling rule: strict '<' along first -> join, '<=' along second -> join
        for (std::size_t u = first; u != join; u = parent_[u]) {
            const std::size_t e = pred_edge_[u];
            const WorkCap d =
                (pred_dir_[u] == ArcDir::Down)
                    ? (work_cap[e] >= kWorkInf ? kWorkInf : work_cap[e] - work_flow[e])
                    : work_flow[e];
            if (d < delta) {
                delta = d;
                u_out = u;
                side = LeavingSide::FirstPath;
            }
        }

        for (std::size_t u = second; u != join; u = parent_[u]) {
            const std::size_t e = pred_edge_[u];
            const WorkCap d =
                (pred_dir_[u] == ArcDir::Up)
                    ? (work_cap[e] >= kWorkInf ? kWorkInf : work_cap[e] - work_flow[e])
                    : work_flow[e];
            if (d <= delta) {
                delta = d;
                u_out = u;
                side = LeavingSide::SecondPath;
            }
        }

        if (delta >= kWorkInf) {
            return false;
        }

        if (delta > WorkCap{0}) {
            work_flow[in_edge] =
                forward_in ? (work_flow[in_edge] + delta) : (work_flow[in_edge] - delta);
            for (std::size_t u = first; u != join; u = parent_[u]) {
                const std::size_t e = pred_edge_[u];
                work_flow[e] = (pred_dir_[u] == ArcDir::Down) ? (work_flow[e] + delta)
                                                              : (work_flow[e] - delta);
            }
            for (std::size_t u = second; u != join; u = parent_[u]) {
                const std::size_t e = pred_edge_[u];
                work_flow[e] =
                    (pred_dir_[u] == ArcDir::Up) ? (work_flow[e] + delta) : (work_flow[e] - delta);
            }
        }

        if (side == LeavingSide::EnteringEdge) {
            state_[in_edge] =
                (state_[in_edge] == ArcState::Lower) ? ArcState::Upper : ArcState::Lower;
            return true;
        }

        const std::size_t e_out = pred_edge_[u_out];
        const bool out_was_forward = (side == LeavingSide::FirstPath)
                                         ? (pred_dir_[u_out] == ArcDir::Down)
                                         : (pred_dir_[u_out] == ArcDir::Up);
        state_[in_edge] = ArcState::Tree;
        state_[e_out] = out_was_forward ? ArcState::Upper : ArcState::Lower;

        const std::size_t u_in = (side == LeavingSide::FirstPath) ? first : second;
        const std::size_t v_in = (side == LeavingSide::FirstPath) ? second : first;
        update_tree_and_potentials(in_edge, u_in, v_in, u_out, work_cost, pi);
        return true;
    }

    template <typename WorkCost, typename WorkCap, typename WideCap>
    [[nodiscard]] SolveOutcome
    solve_core(std::size_t active_edges, const std::vector<WideCap>& shifted_supply_wide,
               const std::vector<WideCap>& active_cap_wide,
               const std::vector<uint8_t>& active_is_inf, WorkCost art_cost, bool phase1_zero_costs,
               std::vector<WorkCap>& out_flows) {
        constexpr WorkCap kWorkInf = detail::signed_max<WorkCap>();
        const std::size_t root = v_;
        const std::size_t num_all_nodes = v_ + 1;
        const std::size_t num_all_edges = active_edges + v_;

        source_.resize(num_all_edges);
        target_.resize(num_all_edges);
        state_.assign(num_all_edges, ArcState::Lower);

        std::vector<WorkCap> work_cap(num_all_edges, kWorkInf);
        std::vector<WorkCap> work_flow(num_all_edges, WorkCap{0});
        for (std::size_t i = 0; i < active_edges; ++i) {
            work_cap[i] = active_is_inf[i] ? kWorkInf : static_cast<WorkCap>(active_cap_wide[i]);
        }

        std::vector<WorkCost> work_cost(num_all_edges, WorkCost{0});
        if (!phase1_zero_costs) {
            for (std::size_t i = 0; i < active_edges; ++i) {
                work_cost[i] = static_cast<WorkCost>(cost_buf_[i]);
            }
        }

        parent_.assign(num_all_nodes, root);
        pred_edge_.assign(num_all_nodes, kNullIndex);
        pred_dir_.assign(num_all_nodes, ArcDir::Up);
        depth_.assign(num_all_nodes, 1);
        thread_.resize(num_all_nodes);
        rev_thread_.resize(num_all_nodes);

        std::vector<WorkCost> pi(num_all_nodes, WorkCost{0});

        parent_[root] = kNullIndex;
        pred_edge_[root] = kNullIndex;
        depth_[root] = 0;

        thread_[root] = 0;
        rev_thread_[0] = root;
        for (std::size_t u = 0; u + 1 < v_; ++u) {
            thread_[u] = u + 1;
            rev_thread_[u + 1] = u;
        }
        thread_[v_ - 1] = root;
        rev_thread_[root] = v_ - 1;

        for (std::size_t u = 0; u < v_; ++u) {
            const std::size_t e = active_edges + u;
            const WorkCap sup = static_cast<WorkCap>(shifted_supply_wide[u]);
            pred_edge_[u] = e;
            state_[e] = ArcState::Tree;
            work_cap[e] = kWorkInf;
            if (sup >= WorkCap{0}) {
                source_[e] = u;
                target_[e] = root;
                pred_dir_[u] = ArcDir::Up;
                work_flow[e] = sup;
                work_cost[e] = WorkCost{0};
                pi[u] = WorkCost{0};
            } else {
                source_[e] = root;
                target_[e] = u;
                pred_dir_[u] = ArcDir::Down;
                work_flow[e] = -sup;
                work_cost[e] = art_cost;
                pi[u] = -art_cost;
            }
        }

        // Heuristic initial pivots for demand nodes
        if (!phase1_zero_costs) {
            std::vector<std::size_t> best_in(v_, kNullIndex);
            for (std::size_t e = 0; e < active_edges; ++e) {
                const std::size_t dst = target_[e];
                if (shifted_supply_wide[dst] < WideCap{0}) {
                    if (best_in[dst] == kNullIndex || work_cost[e] < work_cost[best_in[dst]]) {
                        best_in[dst] = e;
                    }
                }
            }
            for (std::size_t u = 0; u < v_; ++u) {
                const std::size_t e = best_in[u];
                if (e != kNullIndex && state_[e] == ArcState::Lower) {
                    const WorkCost rc = work_cost[e] - pi[source_[e]] + pi[target_[e]];
                    if (rc < WorkCost{0}) {
                        if (!execute_pivot(e, work_cap, work_flow, work_cost, pi)) {
                            return {ProblemType::Unbounded,
                                    has_nonzero_artificial_flow(active_edges, work_flow)};
                        }
                    }
                }
            }
        }

        // Block Search Pivot Pricing (B = 64)
        const std::size_t block_size = std::min(kBlockSize, active_edges);
        std::size_t next_arc = 0;

        auto find_entering_edge = [&]() -> std::size_t {
            WorkCost min_rc{0};
            std::size_t best_edge = kNullIndex;
            std::size_t cnt = block_size;

            auto scan_range = [&](std::size_t begin, std::size_t end,
                                  bool wrap_at_active_edges) -> bool {
                for (std::size_t e = begin; e < end; ++e) {
                    const WorkCost rc = static_cast<WorkCost>(static_cast<int8_t>(state_[e])) *
                                        (work_cost[e] - pi[source_[e]] + pi[target_[e]]);
                    if (rc < min_rc) {
                        min_rc = rc;
                        best_edge = e;
                    }
                    if (--cnt == 0) {
                        if (min_rc < WorkCost{0}) {
                            next_arc =
                                (wrap_at_active_edges && e + 1 == active_edges) ? 0 : (e + 1);
                            return true;
                        }
                        cnt = block_size;
                    }
                }
                return false;
            };

            if (scan_range(next_arc, active_edges, true)) {
                return best_edge;
            }
            if (scan_range(0, next_arc, false)) {
                return best_edge;
            }
            return (min_rc < WorkCost{0}) ? best_edge : kNullIndex;
        };

        while (true) {
            const std::size_t in_edge = find_entering_edge();
            if (in_edge == kNullIndex) {
                break;
            }
            if (!execute_pivot(in_edge, work_cap, work_flow, work_cost, pi)) {
                return {ProblemType::Unbounded,
                        has_nonzero_artificial_flow(active_edges, work_flow)};
            }
        }

        if (has_nonzero_artificial_flow(active_edges, work_flow)) {
            return {ProblemType::Infeasible, true};
        }

        if (!phase1_zero_costs) {
            out_flows.assign(work_flow.begin(),
                             work_flow.begin() + static_cast<std::ptrdiff_t>(active_edges));
            for (std::size_t u = 0; u < v_; ++u) {
                potential_[u] = static_cast<cost_type>(pi[u]);
            }
        }
        return {ProblemType::Optimal, false};
    }
};

} // namespace algoat::graph
