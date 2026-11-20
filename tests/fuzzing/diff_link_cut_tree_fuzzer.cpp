#include "../graph/link_cut_tree_test_helpers.hpp"
#include "algoat/graph/link_cut_tree.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

using algoat::graph::LinkCutTree;
using algoat::graph::testing::ForestOracle;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size == 0) {
        return 0;
    }

    const std::size_t n = static_cast<std::size_t>(data[0]) % 65; // [0, 64]
    LinkCutTree<int64_t> lct(n);
    assert(lct.size() == n);

    if (n == 0) {
        bool threw = false;
        try {
            static_cast<void>(lct.find_root(0));
        } catch (const std::out_of_range&) {
            threw = true;
        }
        assert(threw);
        return 0;
    }

    ForestOracle<int64_t> oracle(n);
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    std::size_t offset = 1;

    while (offset + 3 <= size) {
        const uint8_t op = data[offset] % 10;
        const std::size_t u = static_cast<std::size_t>(data[offset + 1]) % n;
        const std::size_t v = static_cast<std::size_t>(data[offset + 2]) % n;
        offset += 3;

        switch (op) {
        case 0:
        case 1: {
            // Link (or assert invalid_argument on self-loop / cycle / non-root)
            if (u != v && !oracle.connected(u, v) && oracle.is_root(u)) {
                lct.link(u, v);
                oracle.link(u, v);
                edges.emplace_back(u, v);
            } else {
                bool threw = false;
                try {
                    lct.link(u, v);
                } catch (const std::invalid_argument&) {
                    threw = true;
                }
                assert(threw);
            }
            break;
        }
        case 2: {
            // Cut an active edge by index, or probe arbitrary pair
            if (!edges.empty() && (u & 1U) == 0U) {
                const std::size_t idx = v % edges.size();
                auto [eu, ev] = edges[idx];
                if ((data[offset - 1] & 1U) != 0U) {
                    std::swap(eu, ev);
                }
                lct.cut(eu, ev);
                oracle.cut(eu, ev);
                edges[idx] = edges.back();
                edges.pop_back();
            } else if (u != v && oracle.has_edge(u, v)) {
                lct.cut(u, v);
                oracle.cut(u, v);
                for (std::size_t idx = 0; idx < edges.size(); ++idx) {
                    if ((edges[idx].first == u && edges[idx].second == v) ||
                        (edges[idx].first == v && edges[idx].second == u)) {
                        edges[idx] = edges.back();
                        edges.pop_back();
                        break;
                    }
                }
            } else {
                bool threw = false;
                try {
                    lct.cut(u, v);
                } catch (const std::invalid_argument&) {
                    threw = true;
                }
                assert(threw);
            }
            break;
        }
        case 3:
            lct.make_root(u);
            oracle.make_root(u);
            break;
        case 4: {
            int32_t raw_val = 0;
            if (offset + sizeof(int32_t) <= size) {
                std::memcpy(&raw_val, data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
            } else {
                raw_val = static_cast<int32_t>(static_cast<int8_t>(data[offset - 1])) * 1000;
            }
            const auto val = static_cast<int64_t>(raw_val);
            lct.update_vertex(u, val);
            oracle.update_vertex(u, val);
            break;
        }
        case 5:
            assert(lct.connected(u, v) == oracle.connected(u, v));
            break;
        case 6:
            assert(lct.find_root(u) == oracle.find_root(u));
            break;
        case 7:
            if (oracle.connected(u, v)) {
                const auto [exp_sum, exp_min] = oracle.query_path(u, v);
                (void)exp_min;
                assert(lct.query_path_sum(u, v) == exp_sum);
            } else {
                bool threw = false;
                try {
                    static_cast<void>(lct.query_path_sum(u, v));
                } catch (const std::invalid_argument&) {
                    threw = true;
                }
                assert(threw);
            }
            break;
        case 8:
            if (oracle.connected(u, v)) {
                const auto [exp_sum, exp_min] = oracle.query_path(u, v);
                (void)exp_sum;
                assert(lct.query_path_min(u, v) == exp_min);
            } else {
                bool threw = false;
                try {
                    static_cast<void>(lct.query_path_min(u, v));
                } catch (const std::invalid_argument&) {
                    threw = true;
                }
                assert(threw);
            }
            break;
        case 9: {
            // Bounds validation probe
            bool threw = false;
            try {
                static_cast<void>(lct.find_root(n + (u & 3U)));
            } catch (const std::out_of_range&) {
                threw = true;
            }
            assert(threw);
            break;
        }
        default:
            break;
        }
    }

    for (std::size_t u = 0; u < n; ++u) {
        assert(lct.find_root(u) == oracle.find_root(u));
    }

    return 0;
}
