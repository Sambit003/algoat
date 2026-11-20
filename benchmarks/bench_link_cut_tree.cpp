/**
 * @file bench_link_cut_tree.cpp
 * @brief Google Benchmark suite validating Link-Cut Tree O(log V) amortized complexity,
 * adversarial chain stability, path query speedup over BFS, and dynamic MST performance.
 */

#include "algoat/graph/disjoint_set.hpp"
#include "algoat/graph/link_cut_tree.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using algoat::graph::DisjointSetUnion;
using algoat::graph::LinkCutTree;

namespace {

struct BenchQueryPair {
    std::size_t u{0};
    std::size_t v{0};
    int64_t val{0};
};

struct WeightedEdge {
    std::size_t u{0};
    std::size_t v{0};
    int64_t w{0};
};

struct DynamicMstEdgeValue {
    int64_t weight{0};
    std::size_t edge_slot{std::numeric_limits<std::size_t>::max()};
    bool is_edge{false};

    [[nodiscard]] DynamicMstEdgeValue operator+(const DynamicMstEdgeValue& rhs) const noexcept {
        return DynamicMstEdgeValue{weight + rhs.weight, std::numeric_limits<std::size_t>::max(),
                                   false};
    }

    [[nodiscard]] bool operator<(const DynamicMstEdgeValue& rhs) const noexcept {
        if (is_edge != rhs.is_edge) {
            return is_edge;
        }
        if (!is_edge) {
            return false;
        }
        if (weight != rhs.weight) {
            return weight > rhs.weight;
        }
        return edge_slot < rhs.edge_slot;
    }
};

std::vector<std::size_t> build_random_tree_parents(std::size_t n, uint64_t seed) {
    std::vector<std::size_t> parent(n, 0);
    std::mt19937_64 rng(seed);
    for (std::size_t i = 1; i < n; ++i) {
        parent[i] = static_cast<std::size_t>(rng() % i);
    }
    return parent;
}

int64_t bfs_path_sum(const std::vector<std::vector<std::size_t>>& adj,
                     const std::vector<int64_t>& vals, std::size_t src, std::size_t dst,
                     std::vector<uint32_t>& visited, uint32_t& stamp,
                     std::vector<std::size_t>& pred, std::vector<std::size_t>& queue) {
    if (src == dst) {
        return vals[src];
    }
    ++stamp;
    std::size_t head = 0;
    std::size_t tail = 0;
    visited[src] = stamp;
    pred[src] = std::numeric_limits<std::size_t>::max();
    queue[tail++] = src;

    while (head < tail) {
        const std::size_t cur = queue[head++];
        if (cur == dst) {
            int64_t sum = vals[dst];
            for (std::size_t p = pred[dst]; p != std::numeric_limits<std::size_t>::max();
                 p = pred[p]) {
                sum += vals[p];
                if (p == src) {
                    break;
                }
            }
            return sum;
        }
        for (const std::size_t nxt : adj[cur]) {
            if (visited[nxt] != stamp) {
                visited[nxt] = stamp;
                pred[nxt] = cur;
                queue[tail++] = nxt;
            }
        }
    }
    return 0;
}

std::vector<WeightedEdge> generate_edge_stream(std::size_t n, std::size_t m, uint64_t seed) {
    std::vector<WeightedEdge> edges;
    edges.reserve(m);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> v_dist(0, n - 1);
    std::uniform_int_distribution<int64_t> w_dist(1, 1000000);
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t u = v_dist(rng);
        std::size_t v = v_dist(rng);
        if (u == v) {
            v = (u + 1) % n;
        }
        edges.push_back({u, v, w_dist(rng)});
    }
    return edges;
}

int64_t run_dynamic_mst_lct(std::size_t n, const std::vector<WeightedEdge>& edges) {
    LinkCutTree<DynamicMstEdgeValue> lct(2 * n - 1);
    std::vector<std::size_t> edge_u(n - 1, 0);
    std::vector<std::size_t> edge_v(n - 1, 0);
    std::size_t next_slot = 0;
    int64_t total_weight = 0;

    for (const auto& e : edges) {
        if (!lct.connected(e.u, e.v)) {
            const std::size_t slot = next_slot++;
            const std::size_t en = n + slot;
            edge_u[slot] = e.u;
            edge_v[slot] = e.v;
            lct.update_vertex(en, DynamicMstEdgeValue{e.w, slot, true});
            lct.make_root(e.u);
            lct.link(e.u, en);
            lct.link(en, e.v);
            total_weight += e.w;
        } else {
            const auto heaviest = lct.query_path_min(e.u, e.v);
            if (heaviest.is_edge && e.w < heaviest.weight) {
                const std::size_t slot = heaviest.edge_slot;
                const std::size_t en = n + slot;
                lct.cut(edge_u[slot], en);
                lct.cut(edge_v[slot], en);
                edge_u[slot] = e.u;
                edge_v[slot] = e.v;
                lct.update_vertex(en, DynamicMstEdgeValue{e.w, slot, true});
                lct.make_root(e.u);
                lct.link(e.u, en);
                lct.link(en, e.v);
                total_weight += (e.w - heaviest.weight);
            }
        }
    }
    return total_weight;
}

int64_t compute_kruskal_weight(std::size_t n, std::vector<WeightedEdge> edges) {
    std::sort(edges.begin(), edges.end(),
              [](const WeightedEdge& a, const WeightedEdge& b) { return a.w < b.w; });
    DisjointSetUnion dsu(n);
    int64_t mst_weight = 0;
    for (const auto& e : edges) {
        if (dsu.unite(static_cast<algoat::graph::VertexId>(e.u),
                      static_cast<algoat::graph::VertexId>(e.v))) {
            mst_weight += e.w;
        }
    }
    return mst_weight;
}

static void BM_LCT_MixedDynamicOps(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    if (n < 2) {
        state.SkipWithError("Requires at least 2 vertices");
        return;
    }
    const auto parents = build_random_tree_parents(n, 0x17401ULL);
    LinkCutTree<int64_t> lct(n);
    for (std::size_t i = 0; i < n; ++i) {
        lct.update_vertex(i, static_cast<int64_t>(i % 1000) - 500);
    }
    for (std::size_t i = 1; i < n; ++i) {
        lct.link(i, parents[i]);
    }

    constexpr std::size_t kTapeLen = 512;
    std::vector<BenchQueryPair> tape(kTapeLen);
    std::mt19937_64 rng(0x17402ULL);
    for (std::size_t i = 0; i < kTapeLen; ++i) {
        tape[i] = {static_cast<std::size_t>(rng() % (n - 1)) + 1,
                   static_cast<std::size_t>(rng() % n), static_cast<int64_t>(rng() % 2001) - 1000};
    }

    std::size_t cursor = 0;
    for (auto _ : state) {
        const auto& q = tape[cursor];
        cursor = (cursor + 1) & (kTapeLen - 1);

        const std::size_t p = parents[q.u];
        lct.cut(q.u, p);
        lct.make_root(q.u);
        lct.link(q.u, p);
        lct.update_vertex(q.v, q.val);
        int64_t sum = lct.query_path_sum(q.u, q.v);
        int64_t mn = lct.query_path_min(q.u, q.v);
        bool conn = lct.connected(q.u, q.v);
        std::size_t root = lct.find_root(q.u);
        benchmark::DoNotOptimize(sum);
        benchmark::DoNotOptimize(mn);
        benchmark::DoNotOptimize(conn);
        benchmark::DoNotOptimize(root);
    }
    state.SetComplexityN(static_cast<benchmark::IterationCount>(n));
    state.SetItemsProcessed(state.iterations() * 8);
}
BENCHMARK(BM_LCT_MixedDynamicOps)
    ->RangeMultiplier(4)
    ->Range(256, 262144)
    ->Complexity(benchmark::oLogN);

static void BM_LCT_AdversarialChainFindRoot(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    if (n < 2) {
        state.SkipWithError("Requires at least 2 vertices");
        return;
    }
    LinkCutTree<int64_t> lct(n);
    for (std::size_t i = 1; i < n; ++i) {
        lct.link(i, i - 1);
    }

    std::size_t step = 0;
    for (auto _ : state) {
        std::size_t r1 = lct.find_root(n - 1);
        std::size_t r2 = lct.find_root(n / 2);
        benchmark::DoNotOptimize(r1);
        benchmark::DoNotOptimize(r2);
        if ((++step & 15U) == 0U) {
            lct.make_root((step & 16U) != 0U ? (n - 1) : 0);
        }
    }
    state.SetComplexityN(static_cast<benchmark::IterationCount>(n));
    state.SetItemsProcessed(state.iterations() * 2);
}
BENCHMARK(BM_LCT_AdversarialChainFindRoot)
    ->RangeMultiplier(4)
    ->Range(1024, 65536)
    ->Complexity(benchmark::oLogN);

static void BM_LCT_PathQueries(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto parents = build_random_tree_parents(n, 0x17403ULL);
    LinkCutTree<int64_t> lct(n);
    std::vector<std::vector<std::size_t>> adj(n);
    std::vector<int64_t> vals(n);
    for (std::size_t i = 0; i < n; ++i) {
        vals[i] = static_cast<int64_t>(i % 500) - 250;
        lct.update_vertex(i, vals[i]);
    }
    for (std::size_t i = 1; i < n; ++i) {
        lct.link(i, parents[i]);
        adj[i].push_back(parents[i]);
        adj[parents[i]].push_back(i);
    }

    constexpr std::size_t kQueries = 1024;
    std::vector<std::pair<std::size_t, std::size_t>> pairs(kQueries);
    std::mt19937_64 rng(0x17404ULL);
    for (auto& pr : pairs) {
        pr = {static_cast<std::size_t>(rng() % n), static_cast<std::size_t>(rng() % n)};
    }

    // Verify LCT path sum agrees with BFS baseline on initial sample before timing
    std::vector<uint32_t> visited(n, 0);
    std::vector<std::size_t> pred(n, 0);
    std::vector<std::size_t> queue(n, 0);
    uint32_t stamp = 0;
    for (std::size_t i = 0; i < 16; ++i) {
        const auto [u, v] = pairs[i];
        if (lct.query_path_sum(u, v) !=
            bfs_path_sum(adj, vals, u, v, visited, stamp, pred, queue)) {
            state.SkipWithError("LCT path sum diverged from BFS baseline");
            return;
        }
    }

    std::size_t idx = 0;
    for (auto _ : state) {
        const auto [u, v] = pairs[idx];
        idx = (idx + 1) & (kQueries - 1);
        int64_t sum = lct.query_path_sum(u, v);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LCT_PathQueries)->Arg(1024)->Arg(8192)->Arg(65536);

static void BM_NaiveBFS_PathQueries(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto parents = build_random_tree_parents(n, 0x17403ULL);
    std::vector<std::vector<std::size_t>> adj(n);
    std::vector<int64_t> vals(n);
    for (std::size_t i = 0; i < n; ++i) {
        vals[i] = static_cast<int64_t>(i % 500) - 250;
    }
    for (std::size_t i = 1; i < n; ++i) {
        adj[i].push_back(parents[i]);
        adj[parents[i]].push_back(i);
    }

    constexpr std::size_t kQueries = 1024;
    std::vector<std::pair<std::size_t, std::size_t>> pairs(kQueries);
    std::mt19937_64 rng(0x17404ULL);
    for (auto& pr : pairs) {
        pr = {static_cast<std::size_t>(rng() % n), static_cast<std::size_t>(rng() % n)};
    }

    std::vector<uint32_t> visited(n, 0);
    std::vector<std::size_t> pred(n, 0);
    std::vector<std::size_t> queue(n, 0);
    uint32_t stamp = 0;

    std::size_t idx = 0;
    for (auto _ : state) {
        const auto [u, v] = pairs[idx];
        idx = (idx + 1) & (kQueries - 1);
        int64_t sum = bfs_path_sum(adj, vals, u, v, visited, stamp, pred, queue);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_NaiveBFS_PathQueries)->Arg(1024)->Arg(8192)->Arg(65536);

static void BM_DynamicMST_LCT(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    if (n < 2) {
        state.SkipWithError("Requires at least 2 vertices");
        return;
    }
    const std::size_t m = n * 4;
    const auto edges = generate_edge_stream(n, m, 0x17405ULL);

    if (run_dynamic_mst_lct(n, edges) != compute_kruskal_weight(n, edges)) {
        state.SkipWithError("Dynamic MST LCT weight diverged from Kruskal baseline");
        return;
    }

    for (auto _ : state) {
        int64_t total_weight = run_dynamic_mst_lct(n, edges);
        benchmark::DoNotOptimize(total_weight);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(m));
}
BENCHMARK(BM_DynamicMST_LCT)->Arg(256)->Arg(1024)->Arg(4096)->Unit(benchmark::kMillisecond);

static void BM_DynamicMST_KruskalRecompute(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    if (n < 2) {
        state.SkipWithError("Requires at least 2 vertices");
        return;
    }
    const std::size_t m = n * 4;
    const auto edges = generate_edge_stream(n, m, 0x17405ULL);

    for (auto _ : state) {
        int64_t checksum = 0;
        std::vector<WeightedEdge> prefix;
        prefix.reserve(m);
        // Recompute static Kruskal every 64 edge arrivals (amortized offline baseline)
        for (std::size_t i = 0; i < m; ++i) {
            prefix.push_back(edges[i]);
            if ((i & 63U) == 63U || i + 1 == m) {
                checksum ^= compute_kruskal_weight(n, prefix);
            }
        }
        benchmark::DoNotOptimize(checksum);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(m));
}
BENCHMARK(BM_DynamicMST_KruskalRecompute)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4096)
    ->Unit(benchmark::kMillisecond);

} // namespace
