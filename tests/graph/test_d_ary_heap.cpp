#include "algoat/graph/d_ary_heap.hpp"

#include <gtest/gtest.h>
#include <queue>
#include <random>
#include <vector>

using namespace algoat::graph;

TEST(DAryHeapTest, EmptyHeapProperties) {
    d_ary_heap<4, double, VertexId> heap;
    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0u);
    EXPECT_FALSE(heap.contains(0));
    EXPECT_THROW((void)heap.top_index(), std::underflow_error);
    EXPECT_THROW((void)heap.top_key(), std::underflow_error);
    EXPECT_THROW(heap.pop(), std::underflow_error);
    EXPECT_THROW((void)heap.get_key(0), std::out_of_range);
}

TEST(DAryHeapTest, SingleElementPushPop) {
    d_ary_heap<4, double, VertexId> heap;
    heap.push(42, 3.14);

    EXPECT_FALSE(heap.empty());
    EXPECT_EQ(heap.size(), 1u);
    EXPECT_TRUE(heap.contains(42));
    EXPECT_FALSE(heap.is_settled(42));
    EXPECT_EQ(heap.top_index(), 42u);
    EXPECT_DOUBLE_EQ(heap.top_key(), 3.14);
    EXPECT_DOUBLE_EQ(heap.get_key(42), 3.14);

    auto [val, prio] = heap.pop();
    EXPECT_EQ(val, 42u);
    EXPECT_DOUBLE_EQ(prio, 3.14);
    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0u);
    EXPECT_FALSE(heap.contains(42));
    EXPECT_TRUE(heap.is_settled(42));
}

TEST(DAryHeapTest, MultipleElementsOrderingMinHeap) {
    d_ary_heap<4, double, VertexId> heap;
    heap.push(1, 10.0);
    heap.push(2, 5.0);
    heap.push(3, 20.0);
    heap.push(4, 2.5);
    heap.push(5, 8.0);
    heap.push(6, 1.0);
    heap.push(7, 15.0);

    EXPECT_EQ(heap.size(), 7u);
    EXPECT_EQ(heap.top_index(), 6u);
    EXPECT_DOUBLE_EQ(heap.top_key(), 1.0);

    std::vector<std::pair<VertexId, double>> extracted;
    while (!heap.empty()) {
        extracted.push_back(heap.pop());
    }

    ASSERT_EQ(extracted.size(), 7u);
    EXPECT_EQ(extracted[0].first, 6u);
    EXPECT_DOUBLE_EQ(extracted[0].second, 1.0);
    EXPECT_EQ(extracted[1].first, 4u);
    EXPECT_DOUBLE_EQ(extracted[1].second, 2.5);
    EXPECT_EQ(extracted[2].first, 2u);
    EXPECT_DOUBLE_EQ(extracted[2].second, 5.0);
    EXPECT_EQ(extracted[3].first, 5u);
    EXPECT_DOUBLE_EQ(extracted[3].second, 8.0);
    EXPECT_EQ(extracted[4].first, 1u);
    EXPECT_DOUBLE_EQ(extracted[4].second, 10.0);
    EXPECT_EQ(extracted[5].first, 7u);
    EXPECT_DOUBLE_EQ(extracted[5].second, 15.0);
    EXPECT_EQ(extracted[6].first, 3u);
    EXPECT_DOUBLE_EQ(extracted[6].second, 20.0);
}

TEST(DAryHeapTest, DecreaseKeyOperation) {
    d_ary_heap<4, double, VertexId> heap;
    heap.push(0, 100.0);
    heap.push(1, 80.0);
    heap.push(2, 60.0);
    heap.push(3, 40.0);
    heap.push(4, 20.0);

    EXPECT_EQ(heap.top_index(), 4u);
    EXPECT_DOUBLE_EQ(heap.top_key(), 20.0);

    // Decrease key of vertex 0 from 100.0 to 10.0 -> should become new min
    heap.decrease_key(0, 10.0);
    EXPECT_EQ(heap.top_index(), 0u);
    EXPECT_DOUBLE_EQ(heap.top_key(), 10.0);
    EXPECT_DOUBLE_EQ(heap.get_key(0), 10.0);

    // Decrease key of vertex 2 from 60.0 to 5.0 -> should become new min
    heap.decrease_key(2, 5.0);
    EXPECT_EQ(heap.top_index(), 2u);
    EXPECT_DOUBLE_EQ(heap.top_key(), 5.0);

    // Calling decrease_key with greater value is a no-op
    heap.decrease_key(2, 50.0);
    EXPECT_EQ(heap.top_index(), 2u);
    EXPECT_DOUBLE_EQ(heap.top_key(), 5.0);

    // Push or decrease key
    heap.push_or_decrease_key(5, 1.0); // push new
    EXPECT_EQ(heap.top_index(), 5u);
    EXPECT_DOUBLE_EQ(heap.top_key(), 1.0);

    heap.push_or_decrease_key(3, 0.5); // decrease existing
    EXPECT_EQ(heap.top_index(), 3u);
    EXPECT_DOUBLE_EQ(heap.top_key(), 0.5);
}

TEST(DAryHeapTest, ResetAndClear) {
    d_ary_heap<4, double, VertexId> heap;
    heap.push(10, 5.0);
    heap.push(20, 2.0);
    EXPECT_EQ(heap.size(), 2u);

    heap.clear();
    EXPECT_TRUE(heap.empty());
    EXPECT_FALSE(heap.contains(10));
    EXPECT_FALSE(heap.contains(20));

    heap.reset(50);
    EXPECT_TRUE(heap.empty());
    EXPECT_FALSE(heap.is_settled(10));
}

TEST(DAryHeapTest, FloatPrecisionAndDifferentArities) {
    // Binary heap (Arity = 2)
    d_ary_heap<2, float, uint32_t> heap2;
    heap2.push(1, 4.5f);
    heap2.push(2, 1.2f);
    heap2.push(3, 3.3f);
    EXPECT_EQ(heap2.pop().first, 2u);
    EXPECT_EQ(heap2.pop().first, 3u);
    EXPECT_EQ(heap2.pop().first, 1u);

    // 8-ary heap (Arity = 8)
    d_ary_heap<8, double, VertexId> heap8;
    for (uint32_t i = 0; i < 64; ++i) {
        heap8.push(i, static_cast<double>(100 - i));
    }
    EXPECT_EQ(heap8.size(), 64u);
    EXPECT_EQ(heap8.top_index(), 63u);
    EXPECT_DOUBLE_EQ(heap8.top_key(), 37.0);
}

TEST(DAryHeapTest, StressRandomOperationsVsStdPriorityQueue) {
    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> dist_prio(0.0, 1000.0);
    std::uniform_int_distribution<uint32_t> dist_v(0, 500);

    d_ary_heap<4, double, VertexId> heap;
    std::vector<double> current_prio(501, std::numeric_limits<double>::infinity());

    using Element = std::pair<double, VertexId>;
    std::priority_queue<Element, std::vector<Element>, std::greater<Element>> pq;

    for (int step = 0; step < 5000; ++step) {
        int action = rng() % 3;
        VertexId v = dist_v(rng);
        double p = dist_prio(rng);

        if (action == 0 || !heap.contains(v)) {
            // Push or decrease key
            if (!heap.contains(v)) {
                current_prio[v] = p;
                heap.push(v, p);
                pq.push({p, v});
            } else if (p < current_prio[v]) {
                current_prio[v] = p;
                heap.decrease_key(v, p);
                pq.push({p, v});
            }
        } else if (action == 1) {
            // Decrease key to smaller value
            double new_p = current_prio[v] * 0.5;
            current_prio[v] = new_p;
            heap.decrease_key(v, new_p);
            pq.push({new_p, v});
        } else {
            // Pop min
            if (!heap.empty()) {
                // Advance lazy std::priority_queue to find true valid top
                while (!pq.empty()) {
                    auto [p_top, v_top] = pq.top();
                    if (p_top == current_prio[v_top]) {
                        break;
                    }
                    pq.pop();
                }

                ASSERT_FALSE(pq.empty());
                auto expected_top = pq.top();
                pq.pop();

                auto actual_top = heap.pop();
                EXPECT_EQ(actual_top.first, expected_top.second);
                EXPECT_DOUBLE_EQ(actual_top.second, expected_top.first);

                current_prio[actual_top.first] = std::numeric_limits<double>::infinity();
            }
        }
    }
}
