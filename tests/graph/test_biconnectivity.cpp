#include "algoat/graph/biconnectivity.hpp"
#include "algoat/graph/csr_graph.hpp"
#include "biconnectivity_test_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#ifdef _WIN32
#include <malloc.h>
#endif
#include <new>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if !defined(__SANITIZE_ADDRESS__) && !__has_feature(address_sanitizer)
namespace {

struct AlignedAllocHeader {
    std::size_t requested_bytes;
    std::size_t magic;
};

constexpr std::size_t kAlignedHeaderMagic = 0xA160A7B1C0AAULL;
std::atomic<bool> g_track_aligned_bytes{false};
std::atomic<std::size_t> g_live_aligned_bytes{0};
std::atomic<std::size_t> g_peak_aligned_bytes{0};

class ScopedAlignedByteTracker {
public:
    ScopedAlignedByteTracker() noexcept {
        g_live_aligned_bytes.store(0, std::memory_order_relaxed);
        g_peak_aligned_bytes.store(0, std::memory_order_relaxed);
        g_track_aligned_bytes.store(true, std::memory_order_seq_cst);
    }

    ~ScopedAlignedByteTracker() {
        stop();
    }

    ScopedAlignedByteTracker(const ScopedAlignedByteTracker&) = delete;
    ScopedAlignedByteTracker& operator=(const ScopedAlignedByteTracker&) = delete;

    [[nodiscard]] std::size_t peak_bytes() const noexcept {
        return g_peak_aligned_bytes.load(std::memory_order_seq_cst);
    }

    [[nodiscard]] std::size_t live_bytes() const noexcept {
        return g_live_aligned_bytes.load(std::memory_order_seq_cst);
    }

    void stop() noexcept {
        g_track_aligned_bytes.store(false, std::memory_order_seq_cst);
    }
};

[[nodiscard]] void* aligned_alloc_tracker_impl(std::size_t alignment, std::size_t size) noexcept {
#ifdef _WIN32
    return ::_aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

void aligned_free_tracker_impl(void* ptr) noexcept {
#ifdef _WIN32
    ::_aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace

void* operator new(std::size_t size, std::align_val_t align) {
    const std::size_t raw_align = static_cast<std::size_t>(align);
    if (raw_align == 0U || (raw_align & (raw_align - 1U)) != 0U) {
        throw std::bad_alloc();
    }
    const std::size_t adj_align = std::max<std::size_t>(raw_align, 64U);
    const std::size_t payload = ((size == 0U ? 1U : size) + adj_align - 1U) & ~(adj_align - 1U);
    if (payload > std::numeric_limits<std::size_t>::max() - adj_align) {
        throw std::bad_alloc();
    }
    void* base = aligned_alloc_tracker_impl(adj_align, adj_align + payload);
    if (base == nullptr) {
        throw std::bad_alloc();
    }
    auto* user_ptr = static_cast<char*>(base) + adj_align;
    auto* header = reinterpret_cast<AlignedAllocHeader*>(user_ptr - sizeof(AlignedAllocHeader));
    const bool tracked = g_track_aligned_bytes.load(std::memory_order_relaxed);
    if (tracked) {
        header->requested_bytes = size;
        header->magic = kAlignedHeaderMagic;
        const std::size_t cur =
            g_live_aligned_bytes.fetch_add(size, std::memory_order_relaxed) + size;
        std::size_t prev = g_peak_aligned_bytes.load(std::memory_order_relaxed);
        while (cur > prev &&
               !g_peak_aligned_bytes.compare_exchange_weak(prev, cur, std::memory_order_relaxed)) {
        }
    } else {
        header->requested_bytes = 0;
        header->magic = 0;
    }
    return user_ptr;
}

void* operator new[](std::size_t size, std::align_val_t align) {
    return ::operator new(size, align);
}

void operator delete(void* ptr, std::align_val_t align) noexcept {
    if (ptr == nullptr) {
        return;
    }
    const std::size_t raw_align = static_cast<std::size_t>(align);
    const std::size_t adj_align = std::max<std::size_t>(raw_align, 64U);
    auto* user_ptr = static_cast<char*>(ptr);
    const auto* header =
        reinterpret_cast<const AlignedAllocHeader*>(user_ptr - sizeof(AlignedAllocHeader));
    if (header->magic == kAlignedHeaderMagic) {
        g_live_aligned_bytes.fetch_sub(header->requested_bytes, std::memory_order_relaxed);
    }
    aligned_free_tracker_impl(user_ptr - adj_align);
}

void operator delete(void* ptr, std::size_t /*size*/, std::align_val_t align) noexcept {
    ::operator delete(ptr, align);
}

void operator delete[](void* ptr, std::align_val_t align) noexcept {
    ::operator delete(ptr, align);
}

void operator delete[](void* ptr, std::size_t /*size*/, std::align_val_t align) noexcept {
    ::operator delete(ptr, align);
}
#endif

using algoat::graph::BiconnectivityResult;
using algoat::graph::BlockCutTree;
using algoat::graph::BridgeBlockTree;
using algoat::graph::build_block_cut_tree;
using algoat::graph::build_bridge_block_tree;
using algoat::graph::CSRGraph;
using algoat::graph::find_biconnectivity;
using algoat::graph::VertexId;

// ============================================================================
// Slice 1: Empty, Isolated, Self-Loop, and Directed Rejection
// ============================================================================

TEST(BiconnectivityTest, EmptyGraphReturnsEmptyResults) {
    const auto g = CSRGraph<void>::build_from_edges(0, {}, /*directed=*/false);

    const BiconnectivityResult res = find_biconnectivity(g);
    EXPECT_TRUE(res.articulation_points.empty());
    EXPECT_TRUE(res.bridges.empty());
    EXPECT_TRUE(res.biconnected_components.empty());

    const BridgeBlockTree bbt = build_bridge_block_tree(g);
    EXPECT_EQ(bbt.condensed_tree.num_vertices(), 0U);
    EXPECT_EQ(bbt.condensed_tree.num_edges(), 0U);
    EXPECT_TRUE(bbt.vertex_to_component.empty());

    const BlockCutTree bct = build_block_cut_tree(g);
    EXPECT_EQ(bct.tree.num_vertices(), 0U);
    EXPECT_EQ(bct.tree.num_edges(), 0U);
    EXPECT_TRUE(bct.articulation_points.empty());
    EXPECT_TRUE(bct.blocks.empty());
    EXPECT_EQ(bct.num_articulation_nodes, 0U);
    EXPECT_EQ(bct.num_block_nodes, 0U);
}

TEST(BiconnectivityTest, SingleIsolatedVertexAndSelfLoop) {
    const auto g_iso = CSRGraph<void>::build_from_edges(1, {}, /*directed=*/false);
    const std::vector<std::pair<VertexId, VertexId>> loop_edges = {{0, 0}};
    const auto g_loop = CSRGraph<void>::build_from_edges(1, loop_edges, /*directed=*/false);

    for (const auto& g : {g_iso, g_loop}) {
        const BiconnectivityResult res = find_biconnectivity(g);
        EXPECT_TRUE(res.articulation_points.empty());
        EXPECT_TRUE(res.bridges.empty());
        EXPECT_TRUE(res.biconnected_components.empty());

        const BridgeBlockTree bbt = build_bridge_block_tree(g);
        EXPECT_EQ(bbt.condensed_tree.num_vertices(), 1U);
        EXPECT_EQ(bbt.condensed_tree.num_edges(), 0U);
        ASSERT_EQ(bbt.vertex_to_component.size(), 1U);
        EXPECT_EQ(bbt.vertex_to_component[0], 0U);

        const BlockCutTree bct = build_block_cut_tree(g);
        EXPECT_EQ(bct.tree.num_vertices(), 0U);
        EXPECT_EQ(bct.tree.num_edges(), 0U);
        EXPECT_TRUE(bct.articulation_points.empty());
        EXPECT_TRUE(bct.blocks.empty());
        EXPECT_EQ(bct.num_articulation_nodes, 0U);
        EXPECT_EQ(bct.num_block_nodes, 0U);
    }
}

TEST(BiconnectivityTest, MultipleIsolatedVertices) {
    const auto g = CSRGraph<void>::build_from_edges(4, {}, /*directed=*/false);

    const BiconnectivityResult res = find_biconnectivity(g);
    EXPECT_TRUE(res.articulation_points.empty());
    EXPECT_TRUE(res.bridges.empty());
    EXPECT_TRUE(res.biconnected_components.empty());

    const BridgeBlockTree bbt = build_bridge_block_tree(g);
    EXPECT_EQ(bbt.condensed_tree.num_vertices(), 4U);
    EXPECT_EQ(bbt.condensed_tree.num_edges(), 0U);
    ASSERT_EQ(bbt.vertex_to_component.size(), 4U);
    std::vector<bool> seen(4, false);
    for (std::size_t c : bbt.vertex_to_component) {
        ASSERT_LT(c, 4U);
        EXPECT_FALSE(seen[c]);
        seen[c] = true;
    }

    const BlockCutTree bct = build_block_cut_tree(g);
    EXPECT_EQ(bct.tree.num_vertices(), 0U);
    EXPECT_EQ(bct.tree.num_edges(), 0U);
}

TEST(BiconnectivityTest, DirectedGraphThrowsInvalidArgument) {
    const std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}, {1, 2}};
    const auto dg = CSRGraph<void>::build_from_edges(3, edges, /*directed=*/true);

    EXPECT_THROW(static_cast<void>(find_biconnectivity(dg)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(build_bridge_block_tree(dg)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(build_block_cut_tree(dg)), std::invalid_argument);

    const auto empty_dg = CSRGraph<void>::build_from_edges(0, {}, /*directed=*/true);
    EXPECT_THROW(static_cast<void>(find_biconnectivity(empty_dg)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(build_bridge_block_tree(empty_dg)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(build_block_cut_tree(empty_dg)), std::invalid_argument);
}

// ============================================================================
// Slice 2: Articulation Points and Bridges on Canonical Topologies
// ============================================================================

TEST(BiconnectivityTest, SingleEdgeK2HasOneBridgeAndNoArticulationPoints) {
    const std::vector<std::pair<VertexId, VertexId>> edges = {{0, 1}};
    const auto g = CSRGraph<void>::build_from_edges(2, edges, /*directed=*/false);

    const BiconnectivityResult res = find_biconnectivity(g);
    EXPECT_TRUE(res.articulation_points.empty());
    EXPECT_EQ(res.bridges, (std::vector<std::pair<std::size_t, std::size_t>>{{0, 1}}));
}

TEST(BiconnectivityTest, PathAndStarRootArticulationPointsAndBridges) {
    // Path 0 - 1 - 2
    const auto path = CSRGraph<void>::build_from_edges(
        3, std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}}, false);
    const auto path_res = find_biconnectivity(path);
    EXPECT_EQ(path_res.articulation_points, (std::vector<std::size_t>{1}));
    EXPECT_EQ(path_res.bridges, (std::vector<std::pair<std::size_t, std::size_t>>{{0, 1}, {1, 2}}));

    // Star centered at 0 with 5 leaves
    const auto star = CSRGraph<void>::build_from_edges(
        6, std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}},
        false);
    const auto star_res = find_biconnectivity(star);
    EXPECT_EQ(star_res.articulation_points, (std::vector<std::size_t>{0}));
    EXPECT_EQ(star_res.bridges, (std::vector<std::pair<std::size_t, std::size_t>>{
                                    {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}}));
}

TEST(BiconnectivityTest, CyclesBowtieBarbellWheelAndDisconnectedMixed) {
    // Triangle K3
    const auto tri = CSRGraph<void>::build_from_edges(
        3, std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 0}}, false);
    const auto tri_res = find_biconnectivity(tri);
    EXPECT_TRUE(tri_res.articulation_points.empty());
    EXPECT_TRUE(tri_res.bridges.empty());

    // Bowtie: 0-1-2-0 and 2-3-4-2
    const auto bowtie = CSRGraph<void>::build_from_edges(
        5,
        std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 2}},
        false);
    const auto bowtie_res = find_biconnectivity(bowtie);
    EXPECT_EQ(bowtie_res.articulation_points, (std::vector<std::size_t>{2}));
    EXPECT_TRUE(bowtie_res.bridges.empty());

    // Barbell: triangle 0-1-2, bridge 2-3, triangle 3-4-5
    const auto barbell = CSRGraph<void>::build_from_edges(
        6,
        std::vector<std::pair<VertexId, VertexId>>{
            {0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 5}, {5, 3}},
        false);
    const auto barbell_res = find_biconnectivity(barbell);
    EXPECT_EQ(barbell_res.articulation_points, (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(barbell_res.bridges, (std::vector<std::pair<std::size_t, std::size_t>>{{2, 3}}));

    // Wheel W5: center 0 + 4-cycle 1-2-3-4-1
    const auto wheel = CSRGraph<void>::build_from_edges(
        5,
        std::vector<std::pair<VertexId, VertexId>>{
            {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {2, 3}, {3, 4}, {4, 1}},
        false);
    const auto wheel_res = find_biconnectivity(wheel);
    EXPECT_TRUE(wheel_res.articulation_points.empty());
    EXPECT_TRUE(wheel_res.bridges.empty());

    // Disconnected mixed: K3(0,1,2) + K2(3,4) + iso(5) + P3(6,7,8)
    const auto mixed = CSRGraph<void>::build_from_edges(
        9,
        std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 0}, {3, 4}, {6, 7}, {7, 8}},
        false);
    const auto mixed_res = find_biconnectivity(mixed);
    EXPECT_EQ(mixed_res.articulation_points, (std::vector<std::size_t>{7}));
    EXPECT_EQ(mixed_res.bridges,
              (std::vector<std::pair<std::size_t, std::size_t>>{{3, 4}, {6, 7}, {7, 8}}));
}

TEST(BiconnectivityTest, SelfLoopsOnRootAndNonRootDoNotCreateArticulationOrBridges) {
    const auto k2_loops = CSRGraph<void>::build_from_edges(
        2, std::vector<std::pair<VertexId, VertexId>>{{0, 0}, {0, 1}, {1, 1}}, false);
    const auto res_k2 = find_biconnectivity(k2_loops);
    EXPECT_TRUE(res_k2.articulation_points.empty());
    EXPECT_EQ(res_k2.bridges, (std::vector<std::pair<std::size_t, std::size_t>>{{0, 1}}));

    const auto p3_loops = CSRGraph<void>::build_from_edges(
        3, std::vector<std::pair<VertexId, VertexId>>{{0, 0}, {0, 1}, {1, 1}, {1, 2}, {2, 2}},
        false);
    const auto res_p3 = find_biconnectivity(p3_loops);
    EXPECT_EQ(res_p3.articulation_points, (std::vector<std::size_t>{1}));
    EXPECT_EQ(res_p3.bridges, (std::vector<std::pair<std::size_t, std::size_t>>{{0, 1}, {1, 2}}));
}

// ============================================================================
// Slice 3: Biconnected Component Vertex Lists
// ============================================================================

using algoat::graph::test::canonicalize_components;

TEST(BiconnectivityTest, CanonicalBiconnectedComponentVertexLists) {
    // K2
    const auto k2 = CSRGraph<void>::build_from_edges(
        2, std::vector<std::pair<VertexId, VertexId>>{{0, 1}}, false);
    EXPECT_EQ(canonicalize_components(find_biconnectivity(k2).biconnected_components),
              (std::vector<std::vector<std::size_t>>{{0, 1}}));

    // Path 0-1-2
    const auto path = CSRGraph<void>::build_from_edges(
        3, std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}}, false);
    EXPECT_EQ(canonicalize_components(find_biconnectivity(path).biconnected_components),
              (std::vector<std::vector<std::size_t>>{{0, 1}, {1, 2}}));

    // Star at 0
    const auto star = CSRGraph<void>::build_from_edges(
        5, std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {0, 2}, {0, 3}, {0, 4}}, false);
    EXPECT_EQ(canonicalize_components(find_biconnectivity(star).biconnected_components),
              (std::vector<std::vector<std::size_t>>{{0, 1}, {0, 2}, {0, 3}, {0, 4}}));

    // Bowtie: {0,1,2} and {2,3,4}
    const auto bowtie = CSRGraph<void>::build_from_edges(
        5,
        std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 2}},
        false);
    EXPECT_EQ(canonicalize_components(find_biconnectivity(bowtie).biconnected_components),
              (std::vector<std::vector<std::size_t>>{{0, 1, 2}, {2, 3, 4}}));

    // Barbell: {0,1,2}, {2,3}, {3,4,5}
    const auto barbell = CSRGraph<void>::build_from_edges(
        6,
        std::vector<std::pair<VertexId, VertexId>>{
            {0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 5}, {5, 3}},
        false);
    EXPECT_EQ(canonicalize_components(find_biconnectivity(barbell).biconnected_components),
              (std::vector<std::vector<std::size_t>>{{0, 1, 2}, {2, 3}, {3, 4, 5}}));

    // Diamond + bridge + triangle + tail
    const auto nested = CSRGraph<void>::build_from_edges(
        8,
        std::vector<std::pair<VertexId, VertexId>>{
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}, {2, 4}, {4, 5}, {5, 6}, {6, 4}, {4, 7}},
        false);
    const auto nested_res = find_biconnectivity(nested);
    EXPECT_EQ(nested_res.articulation_points, (std::vector<std::size_t>{2, 4}));
    EXPECT_EQ(nested_res.bridges,
              (std::vector<std::pair<std::size_t, std::size_t>>{{2, 4}, {4, 7}}));
    EXPECT_EQ(canonicalize_components(nested_res.biconnected_components),
              (std::vector<std::vector<std::size_t>>{{0, 1, 2, 3}, {2, 4}, {4, 5, 6}, {4, 7}}));

    // Disconnected mixed (isolated vertex 5 must be omitted from biconnected_components)
    const auto mixed = CSRGraph<void>::build_from_edges(
        9,
        std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 0}, {3, 4}, {6, 7}, {7, 8}},
        false);
    EXPECT_EQ(canonicalize_components(find_biconnectivity(mixed).biconnected_components),
              (std::vector<std::vector<std::size_t>>{{0, 1, 2}, {3, 4}, {6, 7}, {7, 8}}));

    // Self-loops on K2 and P3 do not alter blocks
    const auto k2_loops = CSRGraph<void>::build_from_edges(
        2, std::vector<std::pair<VertexId, VertexId>>{{0, 0}, {0, 1}, {1, 1}}, false);
    EXPECT_EQ(canonicalize_components(find_biconnectivity(k2_loops).biconnected_components),
              (std::vector<std::vector<std::size_t>>{{0, 1}}));
}

// ============================================================================
// Slice 4: Bridge-Block Tree Condensation (2-Edge-Connected Components)
// ============================================================================

TEST(BiconnectivityTest, BridgeBlockTreeCanonicalTopologies) {
    // K2: 2 components joined by 1 undirected bridge edge
    const auto k2 = CSRGraph<void>::build_from_edges(
        2, std::vector<std::pair<VertexId, VertexId>>{{0, 1}}, false);
    const BridgeBlockTree bbt_k2 = build_bridge_block_tree(k2);
    EXPECT_FALSE(bbt_k2.condensed_tree.is_directed());
    EXPECT_EQ(bbt_k2.condensed_tree.num_vertices(), 2U);
    EXPECT_EQ(bbt_k2.condensed_tree.num_edges(), 2U); // 1 undirected edge -> 2 CSR arcs
    ASSERT_EQ(bbt_k2.vertex_to_component.size(), 2U);
    EXPECT_NE(bbt_k2.vertex_to_component[0], bbt_k2.vertex_to_component[1]);

    // Path 0-1-2-3: 4 components, 3 undirected edges
    const auto path = CSRGraph<void>::build_from_edges(
        4, std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 3}}, false);
    const BridgeBlockTree bbt_path = build_bridge_block_tree(path);
    EXPECT_EQ(bbt_path.condensed_tree.num_vertices(), 4U);
    EXPECT_EQ(bbt_path.condensed_tree.num_edges(), 6U);

    // Triangle K3: 1 component, 0 edges
    const auto tri = CSRGraph<void>::build_from_edges(
        3, std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 0}}, false);
    const BridgeBlockTree bbt_tri = build_bridge_block_tree(tri);
    EXPECT_EQ(bbt_tri.condensed_tree.num_vertices(), 1U);
    EXPECT_EQ(bbt_tri.condensed_tree.num_edges(), 0U);
    EXPECT_EQ(bbt_tri.vertex_to_component, (std::vector<std::size_t>{0, 0, 0}));

    // Bowtie: 2 biconnected blocks sharing vertex 2, but 0 bridges -> single 2-ECC
    const auto bowtie = CSRGraph<void>::build_from_edges(
        5,
        std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 2}},
        false);
    const BridgeBlockTree bbt_bowtie = build_bridge_block_tree(bowtie);
    EXPECT_EQ(bbt_bowtie.condensed_tree.num_vertices(), 1U);
    EXPECT_EQ(bbt_bowtie.condensed_tree.num_edges(), 0U);
    EXPECT_EQ(bbt_bowtie.vertex_to_component, (std::vector<std::size_t>{0, 0, 0, 0, 0}));

    // Barbell: 2 triangles joined by bridge 2-3 -> 2 condensed vertices, 1 undirected edge
    const auto barbell = CSRGraph<void>::build_from_edges(
        6,
        std::vector<std::pair<VertexId, VertexId>>{
            {0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 5}, {5, 3}},
        false);
    const BridgeBlockTree bbt_barbell = build_bridge_block_tree(barbell);
    EXPECT_EQ(bbt_barbell.condensed_tree.num_vertices(), 2U);
    EXPECT_EQ(bbt_barbell.condensed_tree.num_edges(), 2U);
    ASSERT_EQ(bbt_barbell.vertex_to_component.size(), 6U);
    const std::size_t c_left = bbt_barbell.vertex_to_component[0];
    const std::size_t c_right = bbt_barbell.vertex_to_component[3];
    EXPECT_NE(c_left, c_right);
    EXPECT_EQ(bbt_barbell.vertex_to_component[1], c_left);
    EXPECT_EQ(bbt_barbell.vertex_to_component[2], c_left);
    EXPECT_EQ(bbt_barbell.vertex_to_component[4], c_right);
    EXPECT_EQ(bbt_barbell.vertex_to_component[5], c_right);
    ASSERT_EQ(bbt_barbell.condensed_tree.out_degree(static_cast<VertexId>(c_left)), 1U);
    EXPECT_EQ(bbt_barbell.condensed_tree.out_neighbors(static_cast<VertexId>(c_left))[0],
              static_cast<VertexId>(c_right));

    // Disconnected mixed: K3(0,1,2) + K2(3,4) + iso(5) + P3(6,7,8) -> 7 2-ECCs, 3 bridges
    const auto mixed = CSRGraph<void>::build_from_edges(
        9,
        std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 0}, {3, 4}, {6, 7}, {7, 8}},
        false);
    const BridgeBlockTree bbt_mixed = build_bridge_block_tree(mixed);
    EXPECT_EQ(bbt_mixed.condensed_tree.num_vertices(), 7U);
    EXPECT_EQ(bbt_mixed.condensed_tree.num_edges(), 6U); // 3 undirected bridges
    EXPECT_EQ(bbt_mixed.vertex_to_component[0], bbt_mixed.vertex_to_component[1]);
    EXPECT_EQ(bbt_mixed.vertex_to_component[1], bbt_mixed.vertex_to_component[2]);
    EXPECT_NE(bbt_mixed.vertex_to_component[3], bbt_mixed.vertex_to_component[4]);
    EXPECT_NE(bbt_mixed.vertex_to_component[5], bbt_mixed.vertex_to_component[0]);
    EXPECT_EQ(bbt_mixed.condensed_tree.out_degree(
                  static_cast<VertexId>(bbt_mixed.vertex_to_component[5])),
              0U);
}

// ============================================================================
// Slice 5: Bipartite Block-Cut Tree
// ============================================================================

TEST(BiconnectivityTest, BlockCutTreeCanonicalTopologiesAndBipartiteInvariants) {
    // Single block (Triangle K3): 0 cut-nodes, 1 block-node, 0 tree edges
    const auto tri = CSRGraph<void>::build_from_edges(
        3, std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 0}}, false);
    const BlockCutTree bct_tri = build_block_cut_tree(tri);
    EXPECT_FALSE(bct_tri.tree.is_directed());
    EXPECT_EQ(bct_tri.num_articulation_nodes, 0U);
    EXPECT_EQ(bct_tri.num_block_nodes, 1U);
    EXPECT_TRUE(bct_tri.articulation_points.empty());
    EXPECT_EQ(bct_tri.blocks, (std::vector<std::vector<std::size_t>>{{0, 1, 2}}));
    EXPECT_EQ(bct_tri.tree.num_vertices(), 1U);
    EXPECT_EQ(bct_tri.tree.num_edges(), 0U);

    // Bowtie: cut-node 0 (vertex 2) connected to both block-nodes (indices 1 and 2)
    const auto bowtie = CSRGraph<void>::build_from_edges(
        5,
        std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 2}},
        false);
    const BlockCutTree bct_bowtie = build_block_cut_tree(bowtie);
    EXPECT_EQ(bct_bowtie.num_articulation_nodes, 1U);
    EXPECT_EQ(bct_bowtie.num_block_nodes, 2U);
    EXPECT_EQ(bct_bowtie.articulation_points, (std::vector<std::size_t>{2}));
    EXPECT_EQ(bct_bowtie.tree.num_vertices(), 3U);
    EXPECT_EQ(bct_bowtie.tree.num_edges(), 4U); // 2 undirected bipartite edges
    EXPECT_EQ(bct_bowtie.tree.out_degree(0), 2U);
    EXPECT_EQ(bct_bowtie.tree.out_degree(1), 1U);
    EXPECT_EQ(bct_bowtie.tree.out_degree(2), 1U);

    // Path 0-1-2-3: 2 cut-nodes {1, 2}, 3 blocks, 4 undirected bipartite edges
    const auto path4 = CSRGraph<void>::build_from_edges(
        4, std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 3}}, false);
    const BlockCutTree bct_path = build_block_cut_tree(path4);
    EXPECT_EQ(bct_path.num_articulation_nodes, 2U);
    EXPECT_EQ(bct_path.num_block_nodes, 3U);
    EXPECT_EQ(bct_path.articulation_points, (std::vector<std::size_t>{1, 2}));
    EXPECT_EQ(bct_path.tree.num_vertices(), 5U);
    EXPECT_EQ(bct_path.tree.num_edges(), 8U);
    for (std::size_t cut = 0; cut < bct_path.num_articulation_nodes; ++cut) {
        EXPECT_GE(bct_path.tree.out_degree(static_cast<VertexId>(cut)), 2U);
        for (const VertexId nb : bct_path.tree.out_neighbors(static_cast<VertexId>(cut))) {
            EXPECT_GE(nb, bct_path.num_articulation_nodes);
            const auto& blk = bct_path.blocks[nb - bct_path.num_articulation_nodes];
            EXPECT_TRUE(
                std::binary_search(blk.begin(), blk.end(), bct_path.articulation_points[cut]));
        }
    }

    // Star with center 0 and 4 leaves
    const auto star = CSRGraph<void>::build_from_edges(
        5, std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {0, 2}, {0, 3}, {0, 4}}, false);
    const BlockCutTree bct_star = build_block_cut_tree(star);
    EXPECT_EQ(bct_star.num_articulation_nodes, 1U);
    EXPECT_EQ(bct_star.num_block_nodes, 4U);
    EXPECT_EQ(bct_star.tree.out_degree(0), 4U);

    // Disconnected mixed: K3(0,1,2) + K2(3,4) + iso(5) + P3(6,7,8)
    const auto mixed = CSRGraph<void>::build_from_edges(
        9,
        std::vector<std::pair<VertexId, VertexId>>{{0, 1}, {1, 2}, {2, 0}, {3, 4}, {6, 7}, {7, 8}},
        false);
    const BlockCutTree bct_mixed = build_block_cut_tree(mixed);
    EXPECT_EQ(bct_mixed.num_articulation_nodes, 1U);
    EXPECT_EQ(bct_mixed.num_block_nodes, 4U);
    EXPECT_EQ(bct_mixed.tree.num_vertices(), 5U);
    EXPECT_EQ(bct_mixed.tree.num_edges(), 4U); // 2 undirected edges for cut vertex 7
}

// ============================================================================
// Slice 6: Weighted CSRGraph Support
// ============================================================================

TEST(BiconnectivityTest, WeightedCSRGraphMatchesUnweightedTopology) {
    const std::vector<std::tuple<VertexId, VertexId, double>> weighted_edges = {
        {0, 1, 2.5}, {1, 2, -1.0}, {2, 0, 4.0}, {2, 3, 7.25},
        {3, 4, 1.5}, {4, 5, 3.0},  {5, 3, 9.0}};
    const auto wg =
        CSRGraph<double>::build_from_weighted_edges(6, weighted_edges, /*directed=*/false);

    const BiconnectivityResult res = find_biconnectivity(wg);
    EXPECT_EQ(res.articulation_points, (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(res.bridges, (std::vector<std::pair<std::size_t, std::size_t>>{{2, 3}}));
    EXPECT_EQ(canonicalize_components(res.biconnected_components),
              (std::vector<std::vector<std::size_t>>{{0, 1, 2}, {2, 3}, {3, 4, 5}}));

    const BridgeBlockTree bbt = build_bridge_block_tree(wg);
    EXPECT_EQ(bbt.condensed_tree.num_vertices(), 2U);
    EXPECT_EQ(bbt.condensed_tree.num_edges(), 2U);

    const BlockCutTree bct = build_block_cut_tree(wg);
    EXPECT_EQ(bct.num_articulation_nodes, 2U);
    EXPECT_EQ(bct.num_block_nodes, 3U);
    EXPECT_EQ(bct.tree.num_vertices(), 5U);
}

// ============================================================================
// Slice 7: Reference Oracles, 10,000 Topologies, 10^6 Path & 3*V*8 Memory Bound
// ============================================================================

using algoat::graph::test::bridge_tree_vertex_groups;
using algoat::graph::test::BruteForceOracle;
using algoat::graph::test::generate_random_topology;
using algoat::graph::test::NetworkXReference;
using algoat::graph::test::verify_all_biconnectivity_invariants;

static_assert(sizeof(algoat::graph::detail::DfsFrame) == 12);

TEST(BiconnectivityTest, BruteForceEquivalenceOnSmallRandomGraphs) {
    std::mt19937_64 rng(0x5EED175ULL);
    for (std::size_t iter = 0; iter < 500U; ++iter) {
        const std::size_t num_v = (rng() % 24U) + 1U;
        const auto edges = generate_random_topology(rng, num_v, static_cast<uint32_t>(iter));
        const auto g = CSRGraph<void>::build_from_edges(num_v, edges, /*directed=*/false);

        const BiconnectivityResult res = find_biconnectivity(g);
        const BridgeBlockTree bbt = build_bridge_block_tree(g);
        const BlockCutTree bct = build_block_cut_tree(g);

        EXPECT_TRUE(verify_all_biconnectivity_invariants(g, res, bbt, bct))
            << "iter=" << iter << " V=" << num_v;

        EXPECT_EQ(res.articulation_points, BruteForceOracle::articulation_points(g))
            << "iter=" << iter << " V=" << num_v;
        EXPECT_EQ(res.bridges, BruteForceOracle::bridges(g)) << "iter=" << iter << " V=" << num_v;
        EXPECT_EQ(bridge_tree_vertex_groups(bbt), BruteForceOracle::bridge_components(g))
            << "iter=" << iter << " V=" << num_v;
    }
}

TEST(BiconnectivityTest, EquivalenceWithNetworkXAcross10000RandomTopologies) {
    std::mt19937_64 rng(175175ULL);
    constexpr std::size_t kTotalGraphs = 10000U;

    for (std::size_t iter = 0; iter < kTotalGraphs; ++iter) {
        std::size_t num_v = 1U;
        if (iter < 9850U) {
            num_v = (rng() % 40U) + 1U;
        } else if (iter < 9970U) {
            num_v = 41U + (rng() % 960U);
        } else if (iter < 9995U) {
            num_v = 1001U + (rng() % 19000U);
        } else if (iter < 9998U) {
            num_v = 50000U + (rng() % 40000U);
        } else {
            num_v = 100000U;
        }

        const auto edges = generate_random_topology(rng, num_v, static_cast<uint32_t>(iter));
        const auto g = CSRGraph<void>::build_from_edges(num_v, edges, /*directed=*/false);

        const BiconnectivityResult res = find_biconnectivity(g);
        const BridgeBlockTree bbt = build_bridge_block_tree(g);
        const auto nx_ref = NetworkXReference::compute(g);

        ASSERT_EQ(res.articulation_points, nx_ref.articulation_points)
            << "Articulation mismatch at iter=" << iter << " V=" << num_v;
        ASSERT_EQ(res.bridges, nx_ref.bridges)
            << "Bridge mismatch at iter=" << iter << " V=" << num_v;
        ASSERT_EQ(canonicalize_components(res.biconnected_components),
                  nx_ref.biconnected_components)
            << "BCC vertex list mismatch at iter=" << iter << " V=" << num_v;
        ASSERT_EQ(bridge_tree_vertex_groups(bbt), nx_ref.bridge_components)
            << "Bridge-component partition mismatch at iter=" << iter << " V=" << num_v;
    }
}

TEST(BiconnectivityTest, NonRecursiveDeepLineGraph1MVertices) {
    constexpr std::size_t kNumVertices = 1000000U;
    std::vector<std::pair<VertexId, VertexId>> edges;
    edges.reserve(kNumVertices - 1U);
    for (VertexId i = 0; i + 1U < static_cast<VertexId>(kNumVertices); ++i) {
        edges.emplace_back(i, i + 1U);
    }

    const auto g = CSRGraph<void>::build_from_edges(kNumVertices, edges, /*directed=*/false);

    const BiconnectivityResult res = find_biconnectivity(g);
    EXPECT_EQ(res.articulation_points.size(), kNumVertices - 2U);
    EXPECT_EQ(res.articulation_points.front(), 1U);
    EXPECT_EQ(res.articulation_points.back(), kNumVertices - 2U);
    EXPECT_EQ(res.bridges.size(), kNumVertices - 1U);
    EXPECT_EQ(res.biconnected_components.size(), kNumVertices - 1U);

    const BridgeBlockTree bbt = build_bridge_block_tree(g);
    EXPECT_EQ(bbt.condensed_tree.num_vertices(), kNumVertices);
    EXPECT_EQ(bbt.condensed_tree.num_edges(), 2U * (kNumVertices - 1U));
    EXPECT_EQ(bbt.vertex_to_component.size(), kNumVertices);
}

#if !defined(__SANITIZE_ADDRESS__) && !__has_feature(address_sanitizer)
TEST(BiconnectivityTest, WorkingMemoryFootprintBoundedBy3xVx8Bytes) {
    constexpr std::size_t kV = 100000U;
    constexpr std::size_t kBudgetBytes = 3U * kV * 8U;

    const auto isolated = CSRGraph<void>::build_from_edges(kV, {}, /*directed=*/false);

    std::vector<std::pair<VertexId, VertexId>> line_edges;
    line_edges.reserve(kV - 1U);
    for (VertexId i = 0; i + 1U < static_cast<VertexId>(kV); ++i) {
        line_edges.emplace_back(i, i + 1U);
    }
    const auto line = CSRGraph<void>::build_from_edges(kV, line_edges, /*directed=*/false);

    std::vector<std::pair<VertexId, VertexId>> cycle_edges = line_edges;
    cycle_edges.emplace_back(static_cast<VertexId>(kV - 1U), 0U);
    const auto cycle = CSRGraph<void>::build_from_edges(kV, cycle_edges, /*directed=*/false);

    for (const auto& g : {isolated, line, cycle}) {
        {
            ScopedAlignedByteTracker tracker;
            const BiconnectivityResult res = find_biconnectivity(g);
            EXPECT_LE(tracker.peak_bytes(), kBudgetBytes);
            EXPECT_EQ(tracker.live_bytes(), 0U);
            EXPECT_LE(res.bridges.size(), kV);
        }
        {
            ScopedAlignedByteTracker tracker;
            {
                const BridgeBlockTree bbt = build_bridge_block_tree(g);
                EXPECT_LE(tracker.peak_bytes(), kBudgetBytes);
                EXPECT_EQ(bbt.vertex_to_component.size(), kV);
            }
            EXPECT_EQ(tracker.live_bytes(), 0U);
        }
        {
            ScopedAlignedByteTracker tracker;
            {
                const BlockCutTree bct = build_block_cut_tree(g);
                const std::size_t tree_nodes = std::max<std::size_t>(kV, bct.tree.num_vertices());
                EXPECT_LE(tracker.peak_bytes(), 3U * tree_nodes * 8U);
                EXPECT_LE(tracker.peak_bytes() - bct.tree.memory_footprint(), kBudgetBytes);
            }
            EXPECT_EQ(tracker.live_bytes(), 0U);
        }
    }
}
#endif
