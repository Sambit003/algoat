#include "algoat/sorting/loser_tree.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <queue>
#include <random>
#include <vector>

namespace {

struct StreamCursor {
    int val;
    std::size_t stream_idx;
    std::size_t elem_idx;

    bool operator>(const StreamCursor& other) const noexcept {
        return val > other.val;
    }
};

std::vector<std::vector<int>> generate_runs(std::size_t k, std::size_t total_elements) {
    std::vector<std::vector<int>> runs(k);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 100000000);

    const std::size_t base_len = total_elements / k;
    const std::size_t rem = total_elements % k;

    for (std::size_t i = 0; i < k; ++i) {
        std::size_t len = base_len + (i < rem ? 1 : 0);
        runs[i].resize(len);
        for (std::size_t j = 0; j < len; ++j) {
            runs[i][j] = dist(gen);
        }
        std::sort(runs[i].begin(), runs[i].end());
    }
    return runs;
}

void BM_StdPriorityQueueStreamMerge(benchmark::State& state) {
    const std::size_t k = static_cast<std::size_t>(state.range(0));
    const std::size_t total_elements = 500000;
    const auto runs = generate_runs(k, total_elements);

    std::vector<int> out(total_elements);

    for (auto _ : state) {
        std::priority_queue<StreamCursor, std::vector<StreamCursor>, std::greater<StreamCursor>> pq;
        for (std::size_t i = 0; i < k; ++i) {
            if (!runs[i].empty()) {
                pq.push(StreamCursor{runs[i][0], i, 0});
            }
        }

        std::size_t out_idx = 0;
        while (!pq.empty()) {
            auto top = pq.top();
            pq.pop();
            out[out_idx++] = top.val;

            if (top.elem_idx + 1 < runs[top.stream_idx].size()) {
                pq.push(StreamCursor{runs[top.stream_idx][top.elem_idx + 1], top.stream_idx,
                                     top.elem_idx + 1});
            }
        }
        benchmark::DoNotOptimize(out.data());
    }

    state.SetItemsProcessed(state.iterations() * total_elements);
}

void BM_LoserTreeDirectStreamMerge(benchmark::State& state) {
    const std::size_t k = static_cast<std::size_t>(state.range(0));
    const std::size_t total_elements = 500000;
    const auto runs = generate_runs(k, total_elements);

    std::vector<int> out(total_elements);
    std::vector<std::size_t> cursors(k, 0);
    std::vector<int> heads(k);

    for (auto _ : state) {
        for (std::size_t i = 0; i < k; ++i) {
            cursors[i] = 0;
            heads[i] = runs[i][0];
        }

        algoat::sorting::LoserTree<int> tree(k);
        tree.initialize(heads);

        std::size_t out_idx = 0;
        while (!tree.empty()) {
            const std::size_t s = tree.top_source();
            out[out_idx++] = tree.top();

            const std::size_t next_idx = ++cursors[s];
            if (next_idx < runs[s].size()) {
                tree.replace_top(runs[s][next_idx]);
            } else {
                tree.pop_stream();
            }
        }
        benchmark::DoNotOptimize(out.data());
    }

    state.SetItemsProcessed(state.iterations() * total_elements);
}

void BM_KWayMergeHelper(benchmark::State& state) {
    const std::size_t k = static_cast<std::size_t>(state.range(0));
    const std::size_t total_elements = 500000;
    const auto runs = generate_runs(k, total_elements);

    std::vector<int> out(total_elements);

    for (auto _ : state) {
        algoat::sorting::k_way_merge(runs, out.begin());
        benchmark::DoNotOptimize(out.data());
    }

    state.SetItemsProcessed(state.iterations() * total_elements);
}

} // namespace

BENCHMARK(BM_StdPriorityQueueStreamMerge)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128);
BENCHMARK(BM_LoserTreeDirectStreamMerge)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128);
BENCHMARK(BM_KWayMergeHelper)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128);

BENCHMARK_MAIN();
