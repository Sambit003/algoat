#include "algoat/graph/max_flow.hpp"
#include "algoat/graph/network_simplex.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <queue>
#include <rapidcheck.h>
#include <tuple>
#include <utility>
#include <vector>

using algoat::graph::MaxFlowEngine;
using algoat::graph::NetworkSimplex;
using algoat::graph::VertexId;

namespace {

using Solver = NetworkSimplex<int64_t, int64_t>;
using ProblemType = Solver::ProblemType;

#if defined(__SIZEOF_INT128__)
__extension__ using WideInt = __int128;
#else
using WideInt = int64_t;
#endif

struct EdgeSpec {
    std::size_t u{0};
    std::size_t v{0};
    int64_t cap{0};
    int64_t cost{0};
    int64_t lower{0};
};

template <typename CostT, typename CapT>
void rc_verify_kkt(std::size_t num_v, const std::vector<EdgeSpec>& edges,
                   const std::vector<int64_t>& supplies,
                   const NetworkSimplex<CostT, CapT>& solver) {
    std::vector<WideInt> net_flow(num_v, WideInt{0});
    WideInt total_cost{0};

    for (std::size_t i = 0; i < edges.size(); ++i) {
        const EdgeSpec& e = edges[i];
        const int64_t f = static_cast<int64_t>(solver.flow(i));
        RC_ASSERT(f >= e.lower);
        RC_ASSERT(f <= e.cap);

        net_flow[e.u] += static_cast<WideInt>(f);
        net_flow[e.v] -= static_cast<WideInt>(f);
        total_cost += static_cast<WideInt>(f) * static_cast<WideInt>(e.cost);

        if (e.u != e.v && e.lower < e.cap) {
            const WideInt rc = static_cast<WideInt>(e.cost) -
                               static_cast<WideInt>(solver.potential(e.u)) +
                               static_cast<WideInt>(solver.potential(e.v));
            if (f > e.lower) {
                RC_ASSERT(rc <= WideInt{0});
            }
            if (f < e.cap) {
                RC_ASSERT(rc >= WideInt{0});
            }
        }
    }

    for (std::size_t u = 0; u < num_v; ++u) {
        RC_ASSERT(net_flow[u] == static_cast<WideInt>(supplies[u]));
    }
    RC_ASSERT(static_cast<WideInt>(solver.total_cost()) == total_cost);
}

bool dinic_is_feasible(std::size_t num_v, const std::vector<EdgeSpec>& edges,
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
            const int64_t cap = (e.cap == Solver::kInfiniteCapacity) ? safe_inf : (e.cap - e.lower);
            mf.add_edge(static_cast<VertexId>(e.u), static_cast<VertexId>(e.v), cap);
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

std::pair<ProblemType, int64_t> ssp_oracle(std::size_t num_v, const std::vector<EdgeSpec>& edges,
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

TEST(NetworkSimplexPBT, DifferentialAgreementWithSSPAndKKT) {
    rc::check("NetworkSimplex matches SSP oracle and satisfies KKT complementary slackness",
              [](std::size_t raw_v,
                 const std::vector<std::tuple<uint8_t, uint8_t, uint16_t, int16_t>>& raw_edges,
                 const std::vector<std::tuple<uint8_t, uint8_t, uint8_t>>& supply_pairs) {
                  const std::size_t num_v = (raw_v % 22) + 2;
                  std::vector<int64_t> supplies(num_v, 0);
                  for (const auto& [u_raw, v_raw, amt_raw] : supply_pairs) {
                      const std::size_t u = u_raw % num_v;
                      const std::size_t v = v_raw % num_v;
                      const int64_t amt = static_cast<int64_t>((amt_raw % 25) + 1);
                      supplies[u] += amt;
                      supplies[v] -= amt;
                  }

                  std::vector<EdgeSpec> edges;
                  Solver ns(num_v);
                  for (std::size_t u = 0; u < num_v; ++u) {
                      ns.set_supply(u, supplies[u]);
                  }
                  const std::size_t edge_limit = std::min<std::size_t>(raw_edges.size(), 64);
                  for (std::size_t i = 0; i < edge_limit; ++i) {
                      const auto [u_raw, v_raw, cap_raw, cost_raw] = raw_edges[i];
                      const std::size_t u = u_raw % num_v;
                      const std::size_t v = v_raw % num_v;
                      const int64_t cap = static_cast<int64_t>(cap_raw % 100);
                      const int64_t cost = static_cast<int64_t>(cost_raw % 500);
                      edges.push_back({u, v, cap, cost, 0});
                      ns.add_edge(u, v, cap, cost, 0);
                  }

                  const ProblemType status = ns.run();
                  const auto [ref_status, ref_cost] = ssp_oracle(num_v, edges, supplies);
                  RC_ASSERT(status == ref_status);
                  if (status == ProblemType::Optimal) {
                      RC_ASSERT(ns.total_cost() == ref_cost);
                      rc_verify_kkt(num_v, edges, supplies, ns);
                  }
              });
}

TEST(NetworkSimplexPBT, ArbitraryLowerBoundsAndNegativeBoundsInvariants) {
    rc::check("Lower/upper bounds including negative lower bounds satisfy KKT and SSP agreement",
              [](std::size_t raw_v,
                 const std::vector<std::tuple<uint8_t, uint8_t, int8_t, uint8_t, int16_t>>& raw_e,
                 const std::vector<std::tuple<uint8_t, uint8_t, uint8_t>>& supply_pairs) {
                  const std::size_t num_v = (raw_v % 16) + 2;
                  std::vector<int64_t> supplies(num_v, 0);
                  for (const auto& [u_raw, v_raw, amt_raw] : supply_pairs) {
                      supplies[u_raw % num_v] += static_cast<int64_t>((amt_raw % 20) + 1);
                      supplies[v_raw % num_v] -= static_cast<int64_t>((amt_raw % 20) + 1);
                  }

                  std::vector<EdgeSpec> edges;
                  Solver ns(num_v);
                  for (std::size_t u = 0; u < num_v; ++u) {
                      ns.set_supply(u, supplies[u]);
                  }
                  const std::size_t edge_limit = std::min<std::size_t>(raw_e.size(), 48);
                  for (std::size_t i = 0; i < edge_limit; ++i) {
                      const auto [u_raw, v_raw, lb_raw, span_raw, cost_raw] = raw_e[i];
                      const std::size_t u = u_raw % num_v;
                      const std::size_t v = v_raw % num_v;
                      const int64_t lb = static_cast<int64_t>(lb_raw % 25);
                      // Occasionally generate lb > cap to exercise invalid-bound rejection
                      const int64_t cap =
                          (span_raw == 255) ? (lb - 1) : (lb + static_cast<int64_t>(span_raw % 40));
                      const int64_t cost = static_cast<int64_t>(cost_raw % 200);
                      edges.push_back({u, v, cap, cost, lb});
                      ns.add_edge(u, v, cap, cost, lb);
                  }

                  const ProblemType status = ns.run();
                  const auto [ref_status, ref_cost] = ssp_oracle(num_v, edges, supplies);
                  const bool feasible = dinic_is_feasible(num_v, edges, supplies);
                  RC_ASSERT(status == ref_status);
                  RC_ASSERT((status == ProblemType::Optimal) == feasible);
                  if (status == ProblemType::Optimal) {
                      RC_ASSERT(ns.total_cost() == ref_cost);
                      rc_verify_kkt(num_v, edges, supplies, ns);
                  }
              });
}

TEST(NetworkSimplexPBT, InfiniteCapacityUnboundedAndInfeasiblePrecedence) {
    rc::check("Infeasible always takes precedence over Unbounded with infinite-capacity edges",
              [](std::size_t raw_v,
                 const std::vector<std::tuple<uint8_t, uint8_t, uint8_t, int16_t, int8_t>>& raw_e,
                 const std::vector<std::tuple<uint8_t, uint8_t, uint8_t>>& supply_pairs) {
                  const std::size_t num_v = (raw_v % 16) + 2;
                  std::vector<int64_t> supplies(num_v, 0);
                  for (const auto& [u_raw, v_raw, amt_raw] : supply_pairs) {
                      supplies[u_raw % num_v] += static_cast<int64_t>((amt_raw % 15) + 1);
                      supplies[v_raw % num_v] -= static_cast<int64_t>((amt_raw % 15) + 1);
                  }

                  std::vector<EdgeSpec> edges;
                  Solver ns(num_v);
                  for (std::size_t u = 0; u < num_v; ++u) {
                      ns.set_supply(u, supplies[u]);
                  }
                  const std::size_t edge_limit = std::min<std::size_t>(raw_e.size(), 40);
                  for (std::size_t i = 0; i < edge_limit; ++i) {
                      const auto [u_raw, v_raw, cap_sel, cost_raw, lb_raw] = raw_e[i];
                      const std::size_t u = u_raw % num_v;
                      const std::size_t v = v_raw % num_v;
                      const int64_t lb = static_cast<int64_t>(lb_raw % 10);
                      const int64_t cap = (cap_sel % 5 == 0)
                                              ? Solver::kInfiniteCapacity
                                              : (lb + static_cast<int64_t>(cap_sel % 30));
                      const int64_t cost = static_cast<int64_t>(cost_raw % 100);
                      edges.push_back({u, v, cap, cost, lb});
                      ns.add_edge(u, v, cap, cost, lb);
                  }

                  const ProblemType status = ns.run();
                  const bool feasible = dinic_is_feasible(num_v, edges, supplies);
                  if (!feasible) {
                      RC_ASSERT(status == ProblemType::Infeasible);
                  } else {
                      RC_ASSERT(status != ProblemType::Infeasible);
                      if (status == ProblemType::Optimal) {
                          rc_verify_kkt(num_v, edges, supplies, ns);
                      }
                  }
              });
}

TEST(NetworkSimplexPBT, Int32InstantiationAndRunIdempotency) {
    rc::check("Int32 NetworkSimplex satisfies KKT and run() is strictly idempotent",
              [](std::size_t raw_v,
                 const std::vector<std::tuple<uint8_t, uint8_t, uint8_t, int16_t>>& raw_e,
                 const std::vector<std::tuple<uint8_t, uint8_t, uint8_t>>& supply_pairs) {
                  const std::size_t num_v = (raw_v % 16) + 2;
                  std::vector<int64_t> supplies(num_v, 0);
                  for (const auto& [u_raw, v_raw, amt_raw] : supply_pairs) {
                      supplies[u_raw % num_v] += static_cast<int64_t>((amt_raw % 10) + 1);
                      supplies[v_raw % num_v] -= static_cast<int64_t>((amt_raw % 10) + 1);
                  }

                  using Solver32 = NetworkSimplex<int32_t, int32_t>;
                  std::vector<EdgeSpec> edges;
                  Solver32 ns(num_v);
                  for (std::size_t u = 0; u < num_v; ++u) {
                      ns.set_supply(u, static_cast<int32_t>(supplies[u]));
                  }
                  const std::size_t edge_limit = std::min<std::size_t>(raw_e.size(), 40);
                  for (std::size_t i = 0; i < edge_limit; ++i) {
                      const auto [u_raw, v_raw, cap_raw, cost_raw] = raw_e[i];
                      const std::size_t u = u_raw % num_v;
                      const std::size_t v = v_raw % num_v;
                      const int32_t cap = static_cast<int32_t>(cap_raw % 30);
                      const int32_t cost = static_cast<int32_t>(cost_raw % 500);
                      edges.push_back({u, v, cap, cost, 0});
                      ns.add_edge(u, v, cap, cost, 0);
                  }

                  const auto s1 = ns.run();
                  const int32_t c1 = ns.total_cost();
                  const auto s2 = ns.run();
                  const int32_t c2 = ns.total_cost();

                  RC_ASSERT(s1 == s2);
                  RC_ASSERT(c1 == c2);
                  if (s1 == Solver32::ProblemType::Optimal) {
                      rc_verify_kkt(num_v, edges, supplies, ns);
                  }
              });
}
