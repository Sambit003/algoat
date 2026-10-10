#include "algoat/algoat.hpp"
#include "algoat/searching/binary_search.hpp"
#include "algoat/searching/linear_search.hpp"
#include "algoat/sorting/heapsort.hpp"
#include "algoat/sorting/insertionsort.hpp"
#include "algoat/sorting/introsort.hpp"
#include "algoat/sorting/mergesort.hpp"
#include "algoat/sorting/quicksort.hpp"
#include "algoat/sorting/timsort.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <concepts>
#include <deque>
#include <gtest/gtest.h>
#include <iterator>
#include <list>
#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace {

struct Player {
    int id;
    std::string name;
    double score;

    bool operator==(const Player& other) const = default;
};

struct Unsortable {
    int value;
    // No operator< or operator<=>
};

// ============================================================================
// 1. Compile-Time Concept Constraint Diagnostics Verification
// ============================================================================

template <typename R, typename Comp = std::ranges::less, typename Proj = std::identity>
concept CanCallAlgoatSort =
    requires(R&& r, Comp comp, Proj proj) { algoat::sort(std::forward<R>(r), comp, proj); };

template <typename R, typename T, typename Comp = std::ranges::less, typename Proj = std::identity>
concept CanCallAlgoatSearch = requires(R&& r, const T& target, Comp comp, Proj proj) {
    algoat::search(std::forward<R>(r), target, comp, proj);
};

// Random access containers of sortable types must satisfy sort concept
static_assert(CanCallAlgoatSort<std::vector<int>&>);
static_assert(CanCallAlgoatSort<std::array<int, 5>&>);
static_assert(CanCallAlgoatSort<std::deque<int>&>);
static_assert(CanCallAlgoatSort<std::span<int>>);

// Non-random-access ranges (e.g. std::list) must be rejected at compile time
static_assert(!CanCallAlgoatSort<std::list<int>&>);

// Types without comparison operators must be rejected unless a custom comparator is provided
static_assert(!CanCallAlgoatSort<std::vector<Unsortable>&>);

// Projections extracting sortable keys from otherwise unsortable types must be accepted
static_assert(CanCallAlgoatSort<std::vector<Player>&, std::ranges::less, decltype(&Player::score)>);

// Verify borrowed range return semantics for lifetime safety
static_assert(std::same_as<decltype(algoat::sort(std::declval<std::vector<int>&>())),
                           std::vector<int>::iterator>);
static_assert(
    std::same_as<decltype(algoat::sort(std::declval<std::vector<int>>())), std::ranges::dangling>);
static_assert(std::same_as<decltype(algoat::sort(std::declval<std::span<int>>())), void>);
static_assert(
    std::same_as<decltype(algoat::sort(std::declval<std::span<int>&>(), std::ranges::less{})),
                 std::span<int>::iterator>);

// Search concepts
static_assert(CanCallAlgoatSearch<std::vector<int>&, int>);
static_assert(CanCallAlgoatSearch<std::deque<int>&, int>);
static_assert(!CanCallAlgoatSearch<std::list<int>&, int>);
static_assert(
    CanCallAlgoatSearch<std::vector<Player>&, double, std::ranges::less, decltype(&Player::score)>);

// Complex spatial sorting requires contiguous sized ranges; non-contiguous ranges must be rejected
static_assert(algoat::detail::contiguous_sized_range<std::vector<std::complex<float>>&>);
static_assert(!algoat::detail::contiguous_sized_range<std::deque<std::complex<float>>&>);

// ============================================================================
// 2. algoat::sort Standard Container Ergonomics
// ============================================================================

TEST(RangesSortTest, DirectVectorSort) {
    std::vector<int> data = {9, 1, 8, 2, 7, 3, 6, 4, 5};
    auto end_it = algoat::sort(data);

    EXPECT_EQ(end_it, data.end());
    EXPECT_TRUE(std::ranges::is_sorted(data));
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(RangesSortTest, DirectArraySort) {
    std::array<int, 6> data = {5, 2, 6, 3, 1, 4};
    auto end_it = algoat::sort(data);

    EXPECT_EQ(end_it, data.end());
    EXPECT_TRUE(std::ranges::is_sorted(data));
    EXPECT_EQ(data, (std::array<int, 6>{1, 2, 3, 4, 5, 6}));
}

TEST(RangesSortTest, NonContiguousDequeSort) {
    std::deque<int> data = {10, -5, 8, 0, 3, -2, 7, 1};
    auto end_it = algoat::sort(data);

    EXPECT_EQ(end_it, data.end());
    EXPECT_TRUE(std::ranges::is_sorted(data));
    EXPECT_EQ(data, (std::deque<int>{-5, -2, 0, 1, 3, 7, 8, 10}));
}

TEST(RangesSortTest, SubrangeSort) {
    std::vector<int> data = {10, 50, 40, 30, 20, 60};
    auto sub = std::ranges::subrange(data.begin() + 1, data.begin() + 5);
    algoat::sort(sub);

    EXPECT_EQ(data, (std::vector<int>{10, 20, 30, 40, 50, 60}));
}

TEST(RangesSortTest, RvalueVectorLifetimeSafety) {
    std::vector<int> temp = {3, 1, 2};
    auto result = algoat::sort(std::move(temp));
    static_assert(std::same_as<decltype(result), std::ranges::dangling>);
}

// ============================================================================
// 3. Custom Comparators and Projections
// ============================================================================

TEST(RangesSortTest, CustomDescendingComparator) {
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6};
    algoat::sort(data, std::ranges::greater{});

    EXPECT_TRUE(std::ranges::is_sorted(data, std::ranges::greater{}));
    EXPECT_EQ(data, (std::vector<int>{9, 6, 5, 4, 3, 2, 1, 1}));
}

TEST(RangesSortTest, MemberProjectionAscending) {
    std::vector<Player> players = {
        {1, "Alice", 85.5}, {2, "Bob", 92.0}, {3, "Charlie", 74.0}, {4, "Dave", 88.5}};

    algoat::sort(players, {}, &Player::score);

    EXPECT_EQ(players[0].name, "Charlie");
    EXPECT_EQ(players[1].name, "Alice");
    EXPECT_EQ(players[2].name, "Dave");
    EXPECT_EQ(players[3].name, "Bob");
}

TEST(RangesSortTest, MemberProjectionDescending) {
    std::vector<Player> players = {
        {1, "Alice", 85.5}, {2, "Bob", 92.0}, {3, "Charlie", 74.0}, {4, "Dave", 92.0}};

    algoat::sort(players, std::ranges::greater{}, &Player::score);

    EXPECT_GE(players[0].score, players[1].score);
    EXPECT_GE(players[1].score, players[2].score);
    EXPECT_GE(players[2].score, players[3].score);
    EXPECT_EQ(players[3].name, "Charlie");
}

TEST(RangesSortTest, LambdaProjection) {
    std::vector<std::string> words = {"banana", "pie", "apple", "fig"};
    algoat::sort(words, {}, [](const std::string& s) { return s.length(); });

    EXPECT_EQ(words, (std::vector<std::string>{"pie", "fig", "apple", "banana"}));
}

TEST(RangesSortTest, ComplexSpatialVectorSort) {
    std::vector<std::complex<float>> points = {{10.0f, 10.0f}, {0.0f, 0.0f}, {5.0f, 5.0f}};

    algoat::sort(points);
    EXPECT_FLOAT_EQ(points[0].real(), 0.0f);
    EXPECT_FLOAT_EQ(points[0].imag(), 0.0f);
}

// ============================================================================
// 4. algoat::search Standard Container Ergonomics
// ============================================================================

TEST(RangesSearchTest, VectorSearch) {
    std::vector<int> sorted_data = {10, 20, 30, 40, 50, 60};
    auto idx = algoat::search(sorted_data, 30);

    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(*idx, 2);

    auto missing = algoat::search(sorted_data, 35);
    EXPECT_FALSE(missing.has_value());
}

TEST(RangesSearchTest, ArraySearch) {
    std::array<int, 5> sorted_data = {2, 4, 6, 8, 10};
    auto idx = algoat::search(sorted_data, 8);

    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(*idx, 3);
}

TEST(RangesSearchTest, DequeSearch) {
    std::deque<int> sorted_data = {-10, -5, 0, 5, 10, 15};
    auto idx = algoat::search(sorted_data, -5);

    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(*idx, 1);
}

TEST(RangesSearchTest, SearchWithProjection) {
    std::vector<Player> players = {
        {10, "Alice", 100.0}, {20, "Bob", 200.0}, {30, "Charlie", 300.0}};

    auto idx = algoat::search(players, 20, {}, &Player::id);
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(*idx, 1);
    EXPECT_EQ(players[*idx].name, "Bob");

    auto missing = algoat::search(players, 99, {}, &Player::id);
    EXPECT_FALSE(missing.has_value());
}

// ============================================================================
// 5. Direct Free Function Algorithm Overloads
// ============================================================================

TEST(RangesFreeFunctionsTest, QuicksortRange) {
    std::vector<int> v = {7, 2, 9, 1, 5};
    algoat::sorting::quicksort(v);
    EXPECT_EQ(v, (std::vector<int>{1, 2, 5, 7, 9}));

    std::deque<int> dq = {4, 1, 3};
    algoat::sorting::quicksort(dq, std::ranges::greater{});
    EXPECT_EQ(dq, (std::deque<int>{4, 3, 1}));
}

TEST(RangesFreeFunctionsTest, InsertionsortRange) {
    std::vector<int> v = {8, 3, 2, 9};
    algoat::sorting::insertionsort(v);
    EXPECT_EQ(v, (std::vector<int>{2, 3, 8, 9}));

    std::deque<int> dq = {5, 2, 4};
    algoat::sorting::insertionsort(dq, std::ranges::greater{});
    EXPECT_EQ(dq, (std::deque<int>{5, 4, 2}));
}

TEST(RangesFreeFunctionsTest, IntrosortRange) {
    std::vector<int> v = {12, 5, 7, 1, 9, 2, 6};
    algoat::sorting::introsort(v);
    EXPECT_TRUE(std::ranges::is_sorted(v));
}

TEST(RangesFreeFunctionsTest, HeapsortRange) {
    std::array<int, 5> arr = {9, 4, 1, 7, 3};
    algoat::sorting::heapsort(arr);
    EXPECT_TRUE(std::ranges::is_sorted(arr));
}

TEST(RangesFreeFunctionsTest, MergesortRange) {
    std::vector<int> v = {6, 2, 8, 1, 4};
    algoat::sorting::mergesort(v);
    EXPECT_TRUE(std::ranges::is_sorted(v));

    std::deque<int> dq = {9, 3, 7};
    algoat::sorting::mergesort(dq, std::ranges::greater{});
    EXPECT_EQ(dq, (std::deque<int>{9, 7, 3}));
}

TEST(RangesFreeFunctionsTest, TimsortRange) {
    std::vector<int> v = {10, 9, 8, 1, 2, 3, 4, 5, 6, 7};
    algoat::sorting::timsort(v);
    EXPECT_TRUE(std::ranges::is_sorted(v));
}

TEST(RangesFreeFunctionsTest, BinarySearchRange) {
    std::vector<int> v = {1, 3, 5, 7, 9, 11};
    auto idx = algoat::searching::binary_search(v, 7);
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(*idx, 3);
}

TEST(RangesFreeFunctionsTest, LinearSearchRange) {
    std::vector<std::string> words = {"alpha", "beta", "gamma"};
    auto idx = algoat::searching::linear_search(words, std::string("beta"));
    ASSERT_TRUE(idx.has_value());
    EXPECT_EQ(*idx, 1);

    // Can search by projected string length
    auto len_idx = algoat::searching::linear_search(
        words, static_cast<std::size_t>(5), std::ranges::equal_to{},
        [](const std::string& s) { return s.length(); });
    ASSERT_TRUE(len_idx.has_value());
    EXPECT_EQ(*len_idx, 0); // "alpha" has length 5
}

} // namespace
