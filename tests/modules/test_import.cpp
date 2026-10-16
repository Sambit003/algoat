/**
 * @file test_import.cpp
 * @brief Integration tests for C++20 primary module interface (import algoat;).
 *
 * Validates that importing the module exposes the complete public API including
 * free-function sorting and searching, sub-namespaces, policy tags, runtime
 * dispatcher/config, and constexpr execution without any header inclusions of algoat.
 */

#include <algorithm>
#include <array>
#include <concepts>
#include <gtest/gtest.h>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

import algoat;

// Compile-time verification that algoat::sort and algoat::search work in constexpr contexts
constexpr bool verify_constexpr_module_sort() {
    std::array<int, 6> arr{42, 7, 19, -3, 0, 99};
    algoat::sort(arr);
    return arr[0] == -3 && arr[1] == 0 && arr[2] == 7 && arr[3] == 19 && arr[4] == 42 &&
           arr[5] == 99;
}
static_assert(verify_constexpr_module_sort(),
              "algoat::sort must work in constexpr via module import");

constexpr bool verify_constexpr_module_search() {
    std::array<int, 5> sorted_arr{10, 20, 30, 40, 50};
    auto idx = algoat::search(sorted_arr, 30);
    return idx.has_value() && *idx == 2;
}
static_assert(verify_constexpr_module_search(),
              "algoat::search must work in constexpr via module import");

TEST(ModuleImportTest, VectorRangeSort) {
    std::vector<int> data{5, 2, 8, 1, 9, 3, 7, 4, 6};
    algoat::sort(data);
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
    EXPECT_EQ(data.front(), 1);
    EXPECT_EQ(data.back(), 9);
}

TEST(ModuleImportTest, SpanSortWithCustomComparator) {
    std::vector<int> data{1, 5, 2, 4, 3};
    algoat::sort(std::span<int>(data), std::greater<int>{});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end(), std::greater<int>{}));
    EXPECT_EQ(data.front(), 5);
    EXPECT_EQ(data.back(), 1);
}

TEST(ModuleImportTest, SortWithProjection) {
    struct Item {
        int id;
        std::string name;
    };
    std::vector<Item> items{{3, "c"}, {1, "a"}, {2, "b"}};
    algoat::sort(items, {}, &Item::id);
    EXPECT_EQ(items[0].id, 1);
    EXPECT_EQ(items[1].id, 2);
    EXPECT_EQ(items[2].id, 3);
}

TEST(ModuleImportTest, VectorRangeSearch) {
    std::vector<int> sorted_data{2, 4, 6, 8, 10, 12, 14};
    auto found = algoat::search(sorted_data, 8);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, 3);

    auto missing = algoat::search(sorted_data, 7);
    EXPECT_FALSE(missing.has_value());
}

TEST(ModuleImportTest, SpanSearch) {
    std::vector<int> sorted_data{100, 200, 300, 400};
    std::span<const int> s(sorted_data);
    auto found = algoat::search(s, 300);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(*found, 2);
}

TEST(ModuleImportTest, SortingSubnamespaceFunctions) {
    std::vector<int> qdata{9, 1, 8, 2, 7, 3};
    algoat::sorting::quicksort(qdata);
    EXPECT_TRUE(std::is_sorted(qdata.begin(), qdata.end()));

    std::vector<int> idata{4, 2, 5, 1, 3};
    algoat::sorting::introsort(idata);
    EXPECT_TRUE(std::is_sorted(idata.begin(), idata.end()));

    std::vector<int> tdata{6, 5, 4, 3, 2, 1};
    algoat::sorting::timsort(tdata);
    EXPECT_TRUE(std::is_sorted(tdata.begin(), tdata.end()));

    std::vector<int> pdata{10, 3, 7, 2, 8, 1, 9};
    algoat::sorting::pdqsort(pdata);
    EXPECT_TRUE(std::is_sorted(pdata.begin(), pdata.end()));

    std::vector<int> padapter_data{5, 4, 3, 2, 1};
    algoat::sorting::PdqSort{}.sort(std::span{padapter_data});
    EXPECT_TRUE(std::is_sorted(padapter_data.begin(), padapter_data.end()));
}

TEST(ModuleImportTest, SearchingSubnamespaceFunctions) {
    std::vector<int> data{10, 20, 30, 40, 50};
    auto b_res = algoat::searching::binary_search(data, 40);
    ASSERT_TRUE(b_res.has_value());
    EXPECT_EQ(*b_res, 3);

    auto l_res = algoat::searching::linear_search(data, 20);
    ASSERT_TRUE(l_res.has_value());
    EXPECT_EQ(*l_res, 1);

    auto bl_it = algoat::searching::branchless_lower_bound(data.begin(), data.end(), 35);
    EXPECT_EQ(bl_it, data.begin() + 3);
    EXPECT_EQ(*bl_it, 40);

    auto gs_it = algoat::searching::galloping_search(data.begin(), data.end(), 20);
    EXPECT_EQ(gs_it, data.begin() + 1);
    EXPECT_EQ(*gs_it, 20);

    algoat::searching::BranchlessBinarySearch bbs;
    EXPECT_EQ(bbs.name(), "branchlessbinarysearch");
    EXPECT_TRUE(bbs.search(std::span<const int>{data}, 30).has_value());

    algoat::searching::GallopingSearch gs;
    EXPECT_EQ(gs.name(), "gallopingsearch");
    EXPECT_TRUE(gs.search(std::span<const int>{data}, 10).has_value());

    auto hint_res = algoat::search(data, 20, algoat::core::SearchHint::NearStart);
    ASSERT_TRUE(hint_res.has_value());
    EXPECT_EQ(*hint_res, 1);
}

TEST(ModuleImportTest, PolicyChainingAndDescriptor) {
    using namespace algoat::meta::policy;
    constexpr auto chain = stable | in_place;
    constexpr auto desc = chain.to_descriptor();
    static_assert(desc.stable, "Descriptor must have stable flag set");
    static_assert(desc.in_place, "Descriptor must have in_place flag set");
    static_assert(!desc.nearly_sorted, "Descriptor nearly_sorted must be false");
    EXPECT_TRUE(desc.stable);
    EXPECT_TRUE(desc.in_place);
}

TEST(ModuleImportTest, GlobalConfigAndDispatcher) {
    auto config = algoat::get_global_config();
    ASSERT_NE(config, nullptr);
    EXPECT_GT(config->sorting.small_threshold.value_or(32), 0);

    auto dispatcher = algoat::get_dispatcher();
    std::vector<int> data{3, 1, 2};
    dispatcher.sort(std::span<int>(data));
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(ModuleImportTest, UtilityClampAndConcepts) {
    constexpr int clamped = algoat::clamp(15, 0, 10);
    static_assert(clamped == 10, "algoat::clamp must be usable in constexpr via module import");
    EXPECT_EQ(clamped, 10);
    EXPECT_EQ(algoat::clamp(-5, 0, 10), 0);
    EXPECT_EQ(algoat::clamp(5, 0, 10), 5);

    static_assert(algoat::concepts::Size16Bit<uint16_t>,
                  "algoat::concepts::Size16Bit must be exported");
    static_assert(algoat::concepts::Float16OrInt16<uint16_t>,
                  "algoat::concepts::Float16OrInt16 must be exported");
}

TEST(ModuleImportTest, BooleanAndFloat16Sorts) {
    std::array<bool, 5> bool_data{true, false, true, false, true};
    algoat::sorting::boolean_sort(std::span<bool>(bool_data));
    EXPECT_FALSE(bool_data[0]);
    EXPECT_FALSE(bool_data[1]);
    EXPECT_TRUE(bool_data[2]);
    EXPECT_TRUE(bool_data[3]);
    EXPECT_TRUE(bool_data[4]);

    std::vector<uint16_t> f16_data{0x3C00, 0x8000, 0x0000, 0xBC00};
    algoat::numerics::sort_float16(std::span<uint16_t>(f16_data));
    std::vector<uint16_t> expected{0xBC00, 0x8000, 0x0000, 0x3C00};
    EXPECT_EQ(f16_data, expected);
}
