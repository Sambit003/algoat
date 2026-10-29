#include "algoat/graph/disjoint_set.hpp"

#include <algorithm>
#include <atomic>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using namespace algoat::graph;

TEST(DisjointSetTest, SerialBasicOperations) {
    constexpr std::size_t N = 10;
    DisjointSetUnion dsu(N);

    EXPECT_EQ(dsu.size(), N);
    EXPECT_EQ(dsu.num_components(), N);

    for (VertexId i = 0; i < N; ++i) {
        EXPECT_EQ(dsu.find(i), i);
        EXPECT_EQ(dsu.component_size(i), 1u);
    }

    // Unite 0 and 1
    EXPECT_TRUE(dsu.unite(0, 1));
    EXPECT_EQ(dsu.num_components(), N - 1);
    EXPECT_TRUE(dsu.same_set(0, 1));
    EXPECT_FALSE(dsu.same_set(0, 2));
    EXPECT_EQ(dsu.component_size(0), 2u);

    // Redundant unite
    EXPECT_FALSE(dsu.unite(0, 1));
    EXPECT_FALSE(dsu.unite(1, 0));
    EXPECT_EQ(dsu.num_components(), N - 1);

    // Unite 2 and 3
    EXPECT_TRUE(dsu.unite(2, 3));
    EXPECT_EQ(dsu.num_components(), N - 2);

    // Unite components {0, 1} and {2, 3}
    EXPECT_TRUE(dsu.unite(1, 2));
    EXPECT_EQ(dsu.num_components(), N - 3);
    EXPECT_TRUE(dsu.same_set(0, 3));
    EXPECT_EQ(dsu.component_size(3), 4u);
}

TEST(DisjointSetTest, SerialPathHalvingLinearChain) {
    constexpr std::size_t N = 1000;
    DisjointSetUnion dsu(N);

    // Form a long chain: 0-1, 1-2, 2-3, ...
    for (VertexId i = 0; i < N - 1; ++i) {
        EXPECT_TRUE(dsu.unite(i, i + 1));
    }

    EXPECT_EQ(dsu.num_components(), 1u);
    for (VertexId i = 0; i < N; ++i) {
        EXPECT_EQ(dsu.find(i), dsu.find(0));
    }
}

TEST(DisjointSetTest, ConcurrentBasicOperations) {
    constexpr std::size_t N = 100;
    ConcurrentDisjointSet cdsu(N);

    EXPECT_EQ(cdsu.size(), N);
    EXPECT_EQ(cdsu.num_components(), N);

    for (VertexId i = 0; i < N; ++i) {
        EXPECT_EQ(cdsu.find_atomic(i), i);
    }

    EXPECT_TRUE(cdsu.union_atomic(10, 20));
    EXPECT_EQ(cdsu.num_components(), N - 1);
    EXPECT_TRUE(cdsu.same_set(10, 20));
    EXPECT_FALSE(cdsu.same_set(10, 30));

    EXPECT_FALSE(cdsu.union_atomic(10, 20));
    EXPECT_FALSE(cdsu.union_atomic(20, 10));
    EXPECT_EQ(cdsu.num_components(), N - 1);
}

TEST(DisjointSetTest, ConcurrentOpposingUnionsRace) {
    // Stress test: Multiple threads race to unite opposing pairs (u, v) and (v, u)
    constexpr std::size_t N = 1000;
    ConcurrentDisjointSet cdsu(N);

    std::atomic<bool> start_flag{false};
    std::atomic<int> successes{0};

    auto worker = [&](VertexId u, VertexId v) {
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        if (cdsu.union_atomic(u, v)) {
            successes.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 16; ++t) {
        if (t % 2 == 0) {
            threads.emplace_back(worker, 42, 99);
        } else {
            threads.emplace_back(worker, 99, 42);
        }
    }

    start_flag.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }

    // Exactly one thread must have succeeded in uniting 42 and 99
    EXPECT_EQ(successes.load(), 1);
    EXPECT_TRUE(cdsu.same_set(42, 99));
    EXPECT_EQ(cdsu.num_components(), N - 1);
}

TEST(DisjointSetTest, ConcurrentCircularRaceCycleImmunity) {
    // 3 threads race to unite A-B, B-C, C-A concurrently
    // Asymmetric linking (Union-by-ID) guarantees that no cycle forms and no thread hangs.
    for (int iteration = 0; iteration < 100; ++iteration) {
        ConcurrentDisjointSet cdsu(3);
        std::atomic<bool> start{false};
        std::atomic<int> successful_unions{0};

        auto unite_task = [&](VertexId u, VertexId v) {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            if (cdsu.union_atomic(u, v)) {
                successful_unions.fetch_add(1, std::memory_order_relaxed);
            }
        };

        std::thread t1(unite_task, 0, 1);
        std::thread t2(unite_task, 1, 2);
        std::thread t3(unite_task, 2, 0);

        start.store(true, std::memory_order_release);
        t1.join();
        t2.join();
        t3.join();

        // Exactly 2 of the 3 edges should successfully merge sets (since 3 vertices span 2 edges)
        EXPECT_EQ(successful_unions.load(), 2);
        EXPECT_EQ(cdsu.num_components(), 1u);
        EXPECT_TRUE(cdsu.same_set(0, 1));
        EXPECT_TRUE(cdsu.same_set(1, 2));
        EXPECT_TRUE(cdsu.same_set(0, 2));
    }
}

TEST(DisjointSetTest, ConcurrentMultiThreadedStress) {
    constexpr std::size_t N = 10000;
    ConcurrentDisjointSet cdsu(N);

    constexpr int NUM_THREADS = 8;
    constexpr int OPS_PER_THREAD = 20000;

    std::vector<std::thread> threads;
    std::atomic<bool> start_gate{false};

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937_64 rng(1337 + t);
            std::uniform_int_distribution<VertexId> dist(0, static_cast<VertexId>(N - 1));

            while (!start_gate.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                VertexId u = dist(rng);
                VertexId v = dist(rng);
                if (i % 3 == 0) {
                    static_cast<void>(cdsu.find_atomic(u));
                } else if (i % 3 == 1) {
                    static_cast<void>(cdsu.same_set(u, v));
                } else {
                    cdsu.union_atomic(u, v);
                }
            }
        });
    }

    start_gate.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }

    // Graph must remain in a valid state
    EXPECT_GE(cdsu.num_components(), 1u);
    EXPECT_LE(cdsu.num_components(), N);
}
