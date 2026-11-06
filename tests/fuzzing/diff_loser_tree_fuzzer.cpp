#include "algoat/sorting/loser_tree.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <queue>
#include <vector>

namespace {

template <typename T> std::vector<T> pq_reference_merge(const std::vector<std::vector<T>>& runs) {
    struct Element {
        T val;
        std::size_t stream_idx;
        std::size_t elem_idx;

        bool operator>(const Element& other) const {
            return val > other.val;
        }
    };

    std::priority_queue<Element, std::vector<Element>, std::greater<Element>> pq;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        if (!runs[i].empty()) {
            pq.push(Element{runs[i][0], i, 0});
        }
    }

    std::vector<T> result;
    while (!pq.empty()) {
        auto top = pq.top();
        pq.pop();
        result.push_back(top.val);

        if (top.elem_idx + 1 < runs[top.stream_idx].size()) {
            pq.push(
                Element{runs[top.stream_idx][top.elem_idx + 1], top.stream_idx, top.elem_idx + 1});
        }
    }
    return result;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4 || size % sizeof(int) != 0) {
        return 0;
    }

    const size_t num_ints = size / sizeof(int);
    if (num_ints < 2) {
        return 0;
    }

    std::vector<int> raw_ints(num_ints);
    std::memcpy(raw_ints.data(), data, size);

    // First integer dictates number of streams K in [1, 32]
    const size_t k = (static_cast<size_t>(std::abs(raw_ints[0])) % 32) + 1;
    const size_t remaining = num_ints - 1;

    std::vector<std::vector<int>> runs(k);
    for (size_t i = 0; i < remaining; ++i) {
        runs[i % k].push_back(raw_ints[1 + i]);
    }

    // Sort each individual run to establish the multiway merge invariant
    for (auto& run : runs) {
        std::sort(run.begin(), run.end());
    }

    std::vector<int> loser_tree_out;
    algoat::sorting::k_way_merge(runs, std::back_inserter(loser_tree_out));

    // Reference merge using std::priority_queue
    std::vector<int> pq_out = pq_reference_merge(runs);

    if (loser_tree_out != pq_out) {
        __builtin_trap();
    }

    if (!std::is_sorted(loser_tree_out.begin(), loser_tree_out.end())) {
        __builtin_trap();
    }

    return 0;
}
