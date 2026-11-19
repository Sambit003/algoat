#include "algoat/graph/max_flow.hpp"
#include "algoat/graph/network_simplex.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

using algoat::graph::MaxFlowEngine;
using algoat::graph::NetworkSimplex;
using algoat::graph::VertexId;

using Solver = NetworkSimplex<int64_t, int64_t>;
using ProblemType = Solver::ProblemType;

namespace {

#if defined(__SIZEOF_INT128__)
__extension__ using WideInt = __int128;
#else
using WideInt = int64_t;
#endif

struct FuzzEdge {
    std::size_t u{0};
    std::size_t v{0};
    int64_t cap{0};
    int64_t cost{0};
    int64_t lower{0};
};

bool dinic_feasible(std::size_t num_v, const std::vector<FuzzEdge>& edges,
                    const std::vector<int64_t>& supplies) {
    int64_t sum = 0;
    std::vector<int64_t> b = supplies;
    for (const int64_t s : supplies) {
        sum += s;
    }
    if (sum != 0) {
        return false;
    }
    for (const auto& e : edges) {
        if (e.lower > e.cap) {
            return false;
        }
        if (e.u != e.v) {
            b[e.u] -= e.lower;
            b[e.v] += e.lower;
        }
    }

    int64_t required = 0;
    for (std::size_t u = 0; u < num_v; ++u) {
        if (b[u] > 0) {
            required += b[u];
        }
    }
    const int64_t safe_inf = required + 1;

    const VertexId src = static_cast<VertexId>(num_v);
    const VertexId snk = static_cast<VertexId>(num_v + 1);
    MaxFlowEngine<int64_t> mf(num_v + 2);
    for (const auto& e : edges) {
        if (e.u != e.v && e.lower < e.cap) {
            const int64_t c = (e.cap == Solver::kInfiniteCapacity) ? safe_inf : (e.cap - e.lower);
            mf.add_edge(static_cast<VertexId>(e.u), static_cast<VertexId>(e.v), c);
        }
    }
    for (std::size_t u = 0; u < num_v; ++u) {
        if (b[u] > 0) {
            mf.add_edge(src, static_cast<VertexId>(u), b[u]);
        } else if (b[u] < 0) {
            mf.add_edge(static_cast<VertexId>(u), snk, -b[u]);
        }
    }
    return mf.dinic(src, snk).max_flow == required;
}

void assert_kkt_optimal(std::size_t num_v, const std::vector<FuzzEdge>& edges,
                        const std::vector<int64_t>& supplies, const Solver& solver) {
    std::vector<WideInt> net_flow(num_v, WideInt{0});
    WideInt computed_cost{0};

    for (std::size_t i = 0; i < edges.size(); ++i) {
        const FuzzEdge& e = edges[i];
        const int64_t f = solver.flow(i);
        assert(f >= e.lower);
        assert(f <= e.cap);

        net_flow[e.u] += static_cast<WideInt>(f);
        net_flow[e.v] -= static_cast<WideInt>(f);
        computed_cost += static_cast<WideInt>(f) * static_cast<WideInt>(e.cost);

        if (e.u != e.v && e.lower < e.cap) {
            const WideInt rc = static_cast<WideInt>(e.cost) -
                               static_cast<WideInt>(solver.potential(e.u)) +
                               static_cast<WideInt>(solver.potential(e.v));
            if (f > e.lower) {
                assert(rc <= WideInt{0});
            }
            if (f < e.cap) {
                assert(rc >= WideInt{0});
            }
        }
    }

    for (std::size_t u = 0; u < num_v; ++u) {
        assert(net_flow[u] == static_cast<WideInt>(supplies[u]));
    }
    assert(static_cast<WideInt>(solver.total_cost()) == computed_cost);
}

std::pair<ProblemType, int64_t> ssp_finite_oracle(std::size_t num_v,
                                                  const std::vector<FuzzEdge>& edges,
                                                  const std::vector<int64_t>& supplies) {
    int64_t sum = 0;
    std::vector<int64_t> b = supplies;
    for (const int64_t s : supplies) {
        sum += s;
    }
    if (sum != 0) {
        return {ProblemType::Infeasible, 0};
    }

    int64_t base_cost = 0;
    struct ResEdge {
        std::size_t to;
        std::size_t rev;
        int64_t cap;
        int64_t cost;
    };
    std::vector<std::vector<ResEdge>> adj(num_v + 2);
    auto add_res = [&](std::size_t u, std::size_t v, int64_t cap, int64_t cost) {
        const std::size_t f = adj[u].size();
        const std::size_t r = adj[v].size();
        adj[u].push_back({v, r, cap, cost});
        adj[v].push_back({u, f, 0, -cost});
    };

    for (const auto& e : edges) {
        if (e.lower > e.cap) {
            return {ProblemType::Infeasible, 0};
        }
        if (e.u == e.v) {
            base_cost += ((e.cost < 0) ? e.cap : e.lower) * e.cost;
            continue;
        }
        base_cost += e.lower * e.cost;
        b[e.u] -= e.lower;
        b[e.v] += e.lower;

        const int64_t rem = e.cap - e.lower;
        if (rem == 0) {
            continue;
        }
        if (e.cost < 0) {
            base_cost += rem * e.cost;
            b[e.u] -= rem;
            b[e.v] += rem;
            add_res(e.v, e.u, rem, -e.cost);
        } else {
            add_res(e.u, e.v, rem, e.cost);
        }
    }

    const std::size_t total_v = num_v + 2;
    const std::size_t src = num_v;
    const std::size_t snk = num_v + 1;
    int64_t required = 0;
    for (std::size_t u = 0; u < num_v; ++u) {
        if (b[u] > 0) {
            add_res(src, u, b[u], 0);
            required += b[u];
        } else if (b[u] < 0) {
            add_res(u, snk, -b[u], 0);
        }
    }

    constexpr int64_t kDistInf = std::numeric_limits<int64_t>::max() / 4;
    std::vector<int64_t> pot(total_v, 0);
    std::vector<int64_t> dist(total_v);
    std::vector<std::size_t> pv(total_v);
    std::vector<std::size_t> pe(total_v);
    using PQItem = std::pair<int64_t, std::size_t>;

    int64_t pushed = 0;
    int64_t total_cost = base_cost;
    while (pushed < required) {
        std::fill(dist.begin(), dist.end(), kDistInf);
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
        dist[src] = 0;
        pq.push({0, src});

        while (!pq.empty()) {
            const auto [d, u] = pq.top();
            pq.pop();
            if (d > dist[u]) {
                continue;
            }
            if (u == snk) {
                break;
            }
            for (std::size_t i = 0; i < adj[u].size(); ++i) {
                const ResEdge& edge = adj[u][i];
                if (edge.cap > 0) {
                    const int64_t nd = d + edge.cost + pot[u] - pot[edge.to];
                    if (nd < dist[edge.to]) {
                        dist[edge.to] = nd;
                        pv[edge.to] = u;
                        pe[edge.to] = i;
                        pq.push({nd, edge.to});
                    }
                }
            }
        }

        if (dist[snk] >= kDistInf) {
            break;
        }
        const int64_t snk_dist = dist[snk];
        for (std::size_t u = 0; u < total_v; ++u) {
            pot[u] += (dist[u] < snk_dist) ? dist[u] : snk_dist;
        }

        int64_t aug = required - pushed;
        for (std::size_t v = snk; v != src; v = pv[v]) {
            aug = std::min(aug, adj[pv[v]][pe[v]].cap);
        }
        for (std::size_t v = snk; v != src; v = pv[v]) {
            ResEdge& fwd = adj[pv[v]][pe[v]];
            fwd.cap -= aug;
            adj[v][fwd.rev].cap += aug;
        }
        pushed += aug;
        total_cost += aug * (pot[snk] - pot[src]);
    }

    if (pushed < required) {
        return {ProblemType::Infeasible, 0};
    }
    return {ProblemType::Optimal, total_cost};
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size < 3) {
        return 0;
    }

    const std::size_t num_v = static_cast<std::size_t>(data[0]) % 25;
    if (num_v == 0) {
        Solver ns(0);
        assert(ns.run() == ProblemType::Optimal);
        assert(ns.total_cost() == 0);
        return 0;
    }

    const std::size_t num_supply_ops = static_cast<std::size_t>(data[1]) % 16;
    const bool inject_unbalance = (data[2] & 0x80u) != 0 && num_v >= 2;
    std::size_t offset = 3;

    std::vector<int64_t> supplies(num_v, 0);
    for (std::size_t k = 0; k < num_supply_ops && offset + 3 <= size; ++k) {
        const std::size_t u = static_cast<std::size_t>(data[offset]) % num_v;
        const std::size_t v = static_cast<std::size_t>(data[offset + 1]) % num_v;
        const int64_t amt = static_cast<int64_t>(data[offset + 2] % 30) + 1;
        supplies[u] += amt;
        supplies[v] -= amt;
        offset += 3;
    }
    if (inject_unbalance && (data[2] & 0x0Fu) == 0) {
        supplies[0] += 1;
    }

    constexpr std::size_t kEdgeBytes = 6;
    std::size_t num_edges = (size - offset) / kEdgeBytes;
    if (num_edges > 96) {
        num_edges = 96;
    }

    std::vector<FuzzEdge> edges;
    edges.reserve(num_edges);
    Solver ns(num_v);
    for (std::size_t u = 0; u < num_v; ++u) {
        ns.set_supply(u, supplies[u]);
    }

    bool all_finite = true;
    for (std::size_t i = 0; i < num_edges; ++i) {
        const uint8_t* p = data + offset + i * kEdgeBytes;
        const std::size_t u = static_cast<std::size_t>(p[0]) % num_v;
        const std::size_t v = static_cast<std::size_t>(p[1]) % num_v;
        int8_t raw_lb = 0;
        std::memcpy(&raw_lb, p + 2, sizeof(int8_t));
        const int64_t lb = static_cast<int64_t>(raw_lb % 16);

        int64_t cap = 0;
        if (p[3] == 255) {
            cap = lb - 1;
        } else if (p[3] % 9 == 0) {
            cap = Solver::kInfiniteCapacity;
            all_finite = false;
        } else {
            cap = lb + static_cast<int64_t>(p[3] % 32);
        }

        int16_t raw_cost = 0;
        std::memcpy(&raw_cost, p + 4, sizeof(int16_t));
        const int64_t cost = static_cast<int64_t>(raw_cost % 1000);

        edges.push_back({u, v, cap, cost, lb});
        ns.add_edge(u, v, cap, cost, lb);
    }

    const ProblemType status = ns.run();
    const bool feasible = dinic_feasible(num_v, edges, supplies);
    assert((status == ProblemType::Infeasible) == !feasible);

    if (status == ProblemType::Optimal) {
        assert_kkt_optimal(num_v, edges, supplies, ns);
    }

    if (all_finite) {
        const auto [ref_status, ref_cost] = ssp_finite_oracle(num_v, edges, supplies);
        assert(status == ref_status);
        if (status == ProblemType::Optimal) {
            assert(ns.total_cost() == ref_cost);
        }
    }

    return 0;
}
