/**
 * @file bench_network_simplex.cpp
 * @brief Google Benchmark suite comparing Primal Network Simplex against Dinic + Successive
 * Shortest Path (with Johnson potentials and 4-ary indexed heap) across DIMACS NETGEN
 * and Bipartite Assignment networks.
 */

#include "algoat/graph/d_ary_heap.hpp"
#include "algoat/graph/max_flow.hpp"
#include "algoat/graph/network_simplex.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using algoat::graph::DAryHeap;
using algoat::graph::MaxFlowEngine;
using algoat::graph::NetworkSimplex;
using algoat::graph::VertexId;

namespace {

struct BenchEdge {
    std::size_t from{0};
    std::size_t to{0};
    int64_t cap{0};
    int64_t cost{0};
};

struct FlowInstance {
    std::size_t num_v{0};
    std::vector<BenchEdge> edges;
    std::vector<int64_t> supplies;
};

struct NetgenParams {
    std::size_t num_v{0};
    std::size_t num_edges{0};
    std::size_t num_sources{0};
    std::size_t num_sinks{0};
    int64_t avg_supply{1000};
    int64_t min_cost{1};
    int64_t max_cost{10000};
    int64_t min_cap{1};
    int64_t max_cap{1000};
    uint64_t seed{42};
};

FlowInstance generate_netgen_instance(const NetgenParams& p) {
    FlowInstance inst;
    inst.num_v = p.num_v;
    inst.supplies.assign(p.num_v, 0);

    std::mt19937_64 rng(p.seed);
    std::uniform_int_distribution<int64_t> cost_dist(p.min_cost, p.max_cost);
    std::uniform_int_distribution<int64_t> cap_dist(p.min_cap, p.max_cap);

    const int64_t total_supply = static_cast<int64_t>(p.num_sources) * p.avg_supply;
    for (std::size_t i = 0; i < p.num_sources; ++i) {
        inst.supplies[i] = p.avg_supply;
    }
    for (std::size_t i = 0; i < p.num_sources; ++i) {
        const std::size_t j = rng() % p.num_sources;
        if (i != j && inst.supplies[i] > 1) {
            const int64_t shift =
                static_cast<int64_t>(rng() % static_cast<uint64_t>(inst.supplies[i]));
            inst.supplies[i] -= shift;
            inst.supplies[j] += shift;
        }
    }

    const std::size_t sink_base = p.num_v - p.num_sinks;
    int64_t rem_demand = total_supply;
    for (std::size_t i = 0; i + 1 < p.num_sinks; ++i) {
        const int64_t share = total_supply / static_cast<int64_t>(p.num_sinks);
        const int64_t jitter =
            static_cast<int64_t>(rng() % static_cast<uint64_t>(std::max<int64_t>(1, share / 2)));
        const int64_t d = std::min(rem_demand, share / 2 + jitter);
        inst.supplies[sink_base + i] = -d;
        rem_demand -= d;
    }
    inst.supplies[sink_base + p.num_sinks - 1] = -rem_demand;

    const std::size_t trans_count = p.num_v - p.num_sources - p.num_sinks;
    std::vector<std::size_t> trans_nodes(trans_count);
    for (std::size_t i = 0; i < trans_count; ++i) {
        trans_nodes[i] = p.num_sources + i;
    }
    std::shuffle(trans_nodes.begin(), trans_nodes.end(), rng);

    std::size_t cursor = 0;
    std::vector<std::size_t> source_tail(p.num_sources);
    for (std::size_t s = 0; s < p.num_sources; ++s) {
        std::size_t prev = s;
        const std::size_t chain_len = (trans_count - cursor) / (p.num_sources - s);
        for (std::size_t k = 0; k < chain_len; ++k) {
            const std::size_t next = trans_nodes[cursor++];
            const int64_t c = (k % 3 == 0) ? p.max_cost : cost_dist(rng);
            inst.edges.push_back({prev, next, std::max(inst.supplies[s], cap_dist(rng)), c});
            prev = next;
        }
        source_tail[s] = prev;
    }

    std::size_t s_idx = 0;
    std::size_t t_idx = 0;
    int64_t rem_s = inst.supplies[0];
    int64_t rem_t = -inst.supplies[sink_base];
    while (s_idx < p.num_sources && t_idx < p.num_sinks) {
        const int64_t flow_needed = std::min(rem_s, rem_t);
        if (flow_needed > 0) {
            inst.edges.push_back({source_tail[s_idx], sink_base + t_idx,
                                  std::max(flow_needed, cap_dist(rng)), p.max_cost});
        }
        rem_s -= flow_needed;
        rem_t -= flow_needed;
        if (rem_s == 0 && ++s_idx < p.num_sources) {
            rem_s = inst.supplies[s_idx];
        }
        if (rem_t == 0 && ++t_idx < p.num_sinks) {
            rem_t = -inst.supplies[sink_base + t_idx];
        }
    }

    while (inst.edges.size() < p.num_edges) {
        const std::size_t u = rng() % sink_base;
        const std::size_t v = p.num_sources + (rng() % (p.num_v - p.num_sources));
        if (u != v) {
            inst.edges.push_back({u, v, cap_dist(rng), cost_dist(rng)});
        }
    }

    return inst;
}

FlowInstance generate_assignment_instance(std::size_t n, uint64_t seed) {
    FlowInstance inst;
    inst.num_v = 2 * n;
    inst.supplies.assign(2 * n, 0);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> cost_dist(1, 10000);

    inst.edges.reserve(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        inst.supplies[i] = 1;
        inst.supplies[n + i] = -1;
        for (std::size_t j = 0; j < n; ++j) {
            inst.edges.push_back({i, n + j, 1, cost_dist(rng)});
        }
    }
    return inst;
}

// Dinic feasibility pre-pass + flat forward-star Successive Shortest Path with Johnson
// potentials and 4-ary indexed heap (DAryHeap<4>). Assumes non-negative costs.
int64_t solve_dinic_ssp_baseline(const FlowInstance& inst) {
    const std::size_t total_v = inst.num_v + 2;
    const std::size_t src = inst.num_v;
    const std::size_t snk = inst.num_v + 1;

    // 1. Dinic feasibility check
    MaxFlowEngine<int64_t> mf(total_v);
    for (const auto& e : inst.edges) {
        mf.add_edge(static_cast<VertexId>(e.from), static_cast<VertexId>(e.to), e.cap);
    }
    int64_t required = 0;
    for (std::size_t u = 0; u < inst.num_v; ++u) {
        if (inst.supplies[u] > 0) {
            mf.add_edge(static_cast<VertexId>(src), static_cast<VertexId>(u), inst.supplies[u]);
            required += inst.supplies[u];
        } else if (inst.supplies[u] < 0) {
            mf.add_edge(static_cast<VertexId>(u), static_cast<VertexId>(snk), -inst.supplies[u]);
        }
    }
    if (mf.dinic(static_cast<VertexId>(src), static_cast<VertexId>(snk)).max_flow < required) {
        throw std::runtime_error("Infeasible instance in Dinic+SSP baseline");
    }

    // 2. Flat forward-star residual graph for SSP
    constexpr std::size_t kNullEdge = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> head(total_v, kNullEdge);
    const std::size_t max_res = (inst.edges.size() + inst.num_v) * 2;
    std::vector<std::size_t> to;
    std::vector<std::size_t> next;
    std::vector<int64_t> cap;
    std::vector<int64_t> cost;
    to.reserve(max_res);
    next.reserve(max_res);
    cap.reserve(max_res);
    cost.reserve(max_res);

    auto add_arc = [&](std::size_t u, std::size_t v, int64_t c, int64_t w) {
        const std::size_t idx = to.size();
        to.push_back(v);
        cap.push_back(c);
        cost.push_back(w);
        next.push_back(head[u]);
        head[u] = idx;

        to.push_back(u);
        cap.push_back(0);
        cost.push_back(-w);
        next.push_back(head[v]);
        head[v] = idx + 1;
    };

    for (const auto& e : inst.edges) {
        add_arc(e.from, e.to, e.cap, e.cost);
    }
    for (std::size_t u = 0; u < inst.num_v; ++u) {
        if (inst.supplies[u] > 0) {
            add_arc(src, u, inst.supplies[u], 0);
        } else if (inst.supplies[u] < 0) {
            add_arc(u, snk, -inst.supplies[u], 0);
        }
    }

    constexpr int64_t kDistInf = std::numeric_limits<int64_t>::max() / 4;
    std::vector<int64_t> pot(total_v, 0);
    std::vector<int64_t> dist(total_v, kDistInf);
    std::vector<std::size_t> parent_node(total_v, 0);
    std::vector<std::size_t> parent_edge(total_v, kNullEdge);
    DAryHeap<4, int64_t, uint32_t> heap(total_v);

    int64_t pushed = 0;
    int64_t total_cost = 0;

    while (pushed < required) {
        std::fill(dist.begin(), dist.end(), kDistInf);
        heap.reset(total_v);
        dist[src] = 0;
        heap.push(static_cast<uint32_t>(src), 0);

        while (!heap.empty()) {
            const auto [u32, d] = heap.pop();
            const std::size_t u = u32;
            if (u == snk) {
                break;
            }
            const int64_t pot_u = pot[u];
            for (std::size_t e = head[u]; e != kNullEdge; e = next[e]) {
                if (cap[e] > 0) {
                    const std::size_t v = to[e];
                    const int64_t nd = d + cost[e] + pot_u - pot[v];
                    if (nd < dist[v]) {
                        dist[v] = nd;
                        parent_node[v] = u;
                        parent_edge[v] = e;
                        heap.push_or_decrease_key(static_cast<uint32_t>(v), nd);
                    }
                }
            }
        }

        const int64_t snk_dist = dist[snk];
        if (snk_dist >= kDistInf) {
            break;
        }
        for (std::size_t u = 0; u < total_v; ++u) {
            pot[u] += (dist[u] < snk_dist) ? dist[u] : snk_dist;
        }

        int64_t aug = required - pushed;
        for (std::size_t v = snk; v != src; v = parent_node[v]) {
            aug = std::min(aug, cap[parent_edge[v]]);
        }
        for (std::size_t v = snk; v != src; v = parent_node[v]) {
            const std::size_t e = parent_edge[v];
            cap[e] -= aug;
            cap[e ^ 1] += aug;
        }
        pushed += aug;
        total_cost += aug * (pot[snk] - pot[src]);
    }

    return total_cost;
}

int64_t solve_network_simplex(const FlowInstance& inst) {
    NetworkSimplex<int64_t, int64_t> ns(inst.num_v);
    for (std::size_t u = 0; u < inst.num_v; ++u) {
        if (inst.supplies[u] != 0) {
            ns.set_supply(u, inst.supplies[u]);
        }
    }
    for (const auto& e : inst.edges) {
        ns.add_edge(e.from, e.to, e.cap, e.cost);
    }
    if (ns.run() != NetworkSimplex<int64_t, int64_t>::ProblemType::Optimal) {
        throw std::runtime_error("Expected Optimal solution in benchmark");
    }
    return ns.total_cost();
}

void verify_benchmark_equivalence(benchmark::State& state, const FlowInstance& inst) {
    if (solve_network_simplex(inst) != solve_dinic_ssp_baseline(inst)) {
        state.SkipWithError("Optimal cost mismatch between NetworkSimplex and Dinic+SSP");
    }
}

} // namespace

// ============================================================================
// 1. DIMACS NETGEN-8 (Sparse: E = 8V)
// ============================================================================

static void BM_MinCostFlow_NETGEN8_NetworkSimplex(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto src_snk = static_cast<std::size_t>(std::sqrt(static_cast<double>(num_v)));
    const auto inst = generate_netgen_instance(
        NetgenParams{num_v, num_v * 8, src_snk, src_snk, 1000, 1, 10000, 1, 1000, 42});
    verify_benchmark_equivalence(state, inst);

    for (auto _ : state) {
        int64_t cost = solve_network_simplex(inst);
        benchmark::DoNotOptimize(cost);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(inst.edges.size()));
}
BENCHMARK(BM_MinCostFlow_NETGEN8_NetworkSimplex)->Arg(512)->Arg(1024)->Arg(2048);

static void BM_MinCostFlow_NETGEN8_DinicSSP(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto src_snk = static_cast<std::size_t>(std::sqrt(static_cast<double>(num_v)));
    const auto inst = generate_netgen_instance(
        NetgenParams{num_v, num_v * 8, src_snk, src_snk, 1000, 1, 10000, 1, 1000, 42});

    for (auto _ : state) {
        int64_t cost = solve_dinic_ssp_baseline(inst);
        benchmark::DoNotOptimize(cost);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(inst.edges.size()));
}
BENCHMARK(BM_MinCostFlow_NETGEN8_DinicSSP)->Arg(512)->Arg(1024)->Arg(2048);

// ============================================================================
// 2. DIMACS NETGEN-SR (Dense: E = V * sqrt(V))
// ============================================================================

static void BM_MinCostFlow_NETGENSR_NetworkSimplex(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto sr = static_cast<std::size_t>(std::sqrt(static_cast<double>(num_v)));
    const auto inst = generate_netgen_instance(
        NetgenParams{num_v, num_v * sr, sr, sr, 1000, 1, 10000, 1, 1000, 99});
    verify_benchmark_equivalence(state, inst);

    for (auto _ : state) {
        int64_t cost = solve_network_simplex(inst);
        benchmark::DoNotOptimize(cost);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(inst.edges.size()));
}
BENCHMARK(BM_MinCostFlow_NETGENSR_NetworkSimplex)->Arg(256)->Arg(512)->Arg(1024);

static void BM_MinCostFlow_NETGENSR_DinicSSP(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto sr = static_cast<std::size_t>(std::sqrt(static_cast<double>(num_v)));
    const auto inst = generate_netgen_instance(
        NetgenParams{num_v, num_v * sr, sr, sr, 1000, 1, 10000, 1, 1000, 99});

    for (auto _ : state) {
        int64_t cost = solve_dinic_ssp_baseline(inst);
        benchmark::DoNotOptimize(cost);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(inst.edges.size()));
}
BENCHMARK(BM_MinCostFlow_NETGENSR_DinicSSP)->Arg(256)->Arg(512)->Arg(1024);

// ============================================================================
// 3. Dense Bipartite Min-Weight Assignment (N x N)
// ============================================================================

static void BM_MinCostFlow_Assignment_NetworkSimplex(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto inst = generate_assignment_instance(n, 777);
    verify_benchmark_equivalence(state, inst);

    for (auto _ : state) {
        int64_t cost = solve_network_simplex(inst);
        benchmark::DoNotOptimize(cost);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(inst.edges.size()));
}
BENCHMARK(BM_MinCostFlow_Assignment_NetworkSimplex)->Arg(128)->Arg(256);

static void BM_MinCostFlow_Assignment_DinicSSP(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto inst = generate_assignment_instance(n, 777);

    for (auto _ : state) {
        int64_t cost = solve_dinic_ssp_baseline(inst);
        benchmark::DoNotOptimize(cost);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(inst.edges.size()));
}
BENCHMARK(BM_MinCostFlow_Assignment_DinicSSP)->Arg(128)->Arg(256);

// ============================================================================
// 4. Network Simplex Empirical Scaling (E = 8V)
// ============================================================================

static void BM_MinCostFlow_Scaling_NetworkSimplex(benchmark::State& state) {
    const auto num_v = static_cast<std::size_t>(state.range(0));
    const auto src_snk = static_cast<std::size_t>(std::sqrt(static_cast<double>(num_v)));
    const auto inst = generate_netgen_instance(
        NetgenParams{num_v, num_v * 8, src_snk, src_snk, 1000, 1, 10000, 1, 1000, 123});

    NetworkSimplex<int64_t, int64_t> ns(inst.num_v);
    for (std::size_t u = 0; u < inst.num_v; ++u) {
        if (inst.supplies[u] != 0) {
            ns.set_supply(u, inst.supplies[u]);
        }
    }
    for (const auto& e : inst.edges) {
        ns.add_edge(e.from, e.to, e.cap, e.cost);
    }

    for (auto _ : state) {
        auto status = ns.run();
        int64_t cost = ns.total_cost();
        benchmark::DoNotOptimize(status);
        benchmark::DoNotOptimize(cost);
    }
    state.SetComplexityN(static_cast<benchmark::IterationCount>(inst.edges.size()));
}
BENCHMARK(BM_MinCostFlow_Scaling_NetworkSimplex)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Complexity();
