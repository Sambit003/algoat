/**
 * @file bench_biconnectivity.cpp
 * @brief Google Benchmark suite for Hopcroft-Tarjan biconnectivity, Bridge-Block Tree, and
 * Block-Cut Tree engines across deep path, sparse Erdos-Renyi, and cactus/barbell topologies.
 */

#include "algoat/graph/biconnectivity.hpp"
#include "algoat/graph/csr_graph.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

using algoat::graph::BiconnectivityResult;
using algoat::graph::BlockCutTree;
using algoat::graph::BridgeBlockTree;
using algoat::graph::build_block_cut_tree;
using algoat::graph::build_bridge_block_tree;
using algoat::graph::CSRGraph;
using algoat::graph::find_biconnectivity;
using algoat::graph::VertexId;

namespace {

CSRGraph<void> build_path_graph(std::size_t n) {
    std::vector<std::pair<VertexId, VertexId>> edges;
    if (n > 1U) {
        edges.reserve(n - 1U);
        for (std::size_t i = 0; i + 1U < n; ++i) {
            edges.emplace_back(static_cast<VertexId>(i), static_cast<VertexId>(i + 1U));
        }
    }
    return CSRGraph<void>::build_from_edges(n, edges, /*directed=*/false);
}

CSRGraph<void> build_sparse_random_graph(std::size_t n, std::size_t avg_deg, uint64_t seed) {
    std::mt19937_64 rng(seed);
    const std::size_t m = (n * avg_deg) / 2U;
    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(m);
    for (std::size_t i = 0; i < m; ++i) {
        edges.emplace_back(static_cast<VertexId>(rng() % n), static_cast<VertexId>(rng() % n));
    }
    return CSRGraph<void>::build_from_edges(n, edges, /*directed=*/false);
}

CSRGraph<void> build_cactus_barbell_graph(std::size_t n) {
    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(n * 2U);
    std::size_t cur = 0;
    while (cur + 1U < n) {
        const std::size_t block_end = std::min(n - 1U, cur + 4U);
        for (std::size_t v = cur; v < block_end; ++v) {
            edges.emplace_back(static_cast<VertexId>(v), static_cast<VertexId>(v + 1U));
        }
        if (block_end > cur + 1U && (cur % 8U) == 0U) {
            edges.emplace_back(static_cast<VertexId>(cur), static_cast<VertexId>(block_end));
        }
        cur = block_end;
    }
    return CSRGraph<void>::build_from_edges(n, edges, /*directed=*/false);
}

void BM_Biconnectivity_Path(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const CSRGraph<void> g = build_path_graph(n);

    for (auto _ : state) {
        BiconnectivityResult res = find_biconnectivity(g);
        benchmark::DoNotOptimize(res);
    }
    state.SetComplexityN(static_cast<benchmark::IterationCount>(n));
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(g.num_edges() / 2U));
}
BENCHMARK(BM_Biconnectivity_Path)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Complexity(benchmark::oN)
    ->Unit(benchmark::kMillisecond);

void BM_BridgeBlockTree_Path(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const CSRGraph<void> g = build_path_graph(n);

    for (auto _ : state) {
        BridgeBlockTree bbt = build_bridge_block_tree(g);
        benchmark::DoNotOptimize(bbt);
    }
    state.SetComplexityN(static_cast<benchmark::IterationCount>(n));
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(g.num_edges() / 2U));
}
BENCHMARK(BM_BridgeBlockTree_Path)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Complexity(benchmark::oN)
    ->Unit(benchmark::kMillisecond);

void BM_Biconnectivity_RandomSparse(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const CSRGraph<void> g = build_sparse_random_graph(n, 6U, 0x17501ULL);

    for (auto _ : state) {
        BiconnectivityResult res = find_biconnectivity(g);
        benchmark::DoNotOptimize(res);
    }
    state.SetComplexityN(static_cast<benchmark::IterationCount>(n));
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(g.num_edges() / 2U));
}
BENCHMARK(BM_Biconnectivity_RandomSparse)
    ->RangeMultiplier(4)
    ->Range(1024, 262144)
    ->Complexity(benchmark::oN)
    ->Unit(benchmark::kMillisecond);

void BM_BridgeBlockTree_RandomSparse(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const CSRGraph<void> g = build_sparse_random_graph(n, 6U, 0x17502ULL);

    for (auto _ : state) {
        BridgeBlockTree bbt = build_bridge_block_tree(g);
        benchmark::DoNotOptimize(bbt);
    }
    state.SetComplexityN(static_cast<benchmark::IterationCount>(n));
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(g.num_edges() / 2U));
}
BENCHMARK(BM_BridgeBlockTree_RandomSparse)
    ->RangeMultiplier(4)
    ->Range(1024, 262144)
    ->Complexity(benchmark::oN)
    ->Unit(benchmark::kMillisecond);

void BM_BlockCutTree_CactusBarbell(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const CSRGraph<void> g = build_cactus_barbell_graph(n);

    for (auto _ : state) {
        BlockCutTree bct = build_block_cut_tree(g);
        benchmark::DoNotOptimize(bct);
    }
    state.SetComplexityN(static_cast<benchmark::IterationCount>(n));
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(g.num_edges() / 2U));
}
BENCHMARK(BM_BlockCutTree_CactusBarbell)
    ->RangeMultiplier(4)
    ->Range(1024, 262144)
    ->Complexity(benchmark::oN)
    ->Unit(benchmark::kMillisecond);

} // namespace
