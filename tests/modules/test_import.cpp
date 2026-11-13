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

    std::vector<int> skadata{12, 3, 8, 1, 15, 2, 9, 4};
    algoat::sorting::ska_sort(skadata);
    EXPECT_TRUE(std::is_sorted(skadata.begin(), skadata.end()));

    std::vector<int> ska_adapter_data{9, 8, 7, 6, 5, 4};
    algoat::sorting::SkaSort{}.sort(std::span{ska_adapter_data});
    EXPECT_TRUE(std::is_sorted(ska_adapter_data.begin(), ska_adapter_data.end()));

    std::vector<int> graildata{14, 2, 9, 1, 18, 5, 11, 3};
    algoat::sorting::grailsort(graildata);
    EXPECT_TRUE(std::is_sorted(graildata.begin(), graildata.end()));

    std::vector<int> grail_adapter_data{8, 7, 6, 5, 4, 3, 2, 1};
    algoat::sorting::GrailSort{}.sort(std::span{grail_adapter_data});
    EXPECT_TRUE(std::is_sorted(grail_adapter_data.begin(), grail_adapter_data.end()));
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

TEST(ModuleImportTest, StringsSuiteBurstsortAndMultikeyQuicksort) {
    std::vector<std::string> words1 = {"cherry", "banana", "apple", "date"};
    algoat::strings::burstsort(std::span<std::string>{words1});
    EXPECT_TRUE(std::is_sorted(words1.begin(), words1.end()));

    std::vector<std::string_view> words2 = {"grape", "elderberry", "fig", "apricot"};
    algoat::strings::multikey_quicksort(std::span<std::string_view>{words2});
    EXPECT_TRUE(std::is_sorted(words2.begin(), words2.end()));
}

TEST(ModuleImportTest, StringsSuiteSubstringSearch) {
    std::string text = "the quick brown fox jumps over the lazy dog";
    EXPECT_EQ(algoat::find_substring(text, "quick"), 4);
    EXPECT_EQ(algoat::strings::find_substring(text, "fox"), 16);

    auto all_the = algoat::strings::find_all_substrings(text, "the");
    std::vector<std::string_view::size_type> expected_the = {0, 31};
    EXPECT_EQ(all_the, expected_the);

    auto it = algoat::search_pattern(text, "lazy");
    ASSERT_NE(it, text.end());
    EXPECT_EQ(std::distance(text.begin(), it), 35);
}

TEST(ModuleImportTest, SuccinctSuiteRoaringBitmap) {
    algoat::succinct::RoaringBitmap rb;
    rb.add(42);
    rb.add(100000);
    EXPECT_EQ(rb.cardinality(), 2);
    EXPECT_TRUE(rb.contains(42));
    EXPECT_TRUE(rb.contains(100000));
    EXPECT_FALSE(rb.contains(99));

    algoat::RoaringBitmap rb_alias;
    rb_alias.add(7);
    EXPECT_TRUE(rb_alias.contains(7));
}

TEST(ModuleImportTest, SuccinctSuiteRankSelectBitVector) {
    std::vector<bool> bits = {true, false, true, true, false, true};
    algoat::succinct::RankSelectBitVector bv(bits);
    EXPECT_EQ(bv.num_bits(), 6ULL);
    EXPECT_EQ(bv.total_ones(), 4ULL);
    EXPECT_EQ(bv.rank1(3), 2ULL);
    EXPECT_EQ(bv.select1(1), 0ULL);
    EXPECT_EQ(bv.select1(2), 2ULL);

    algoat::RankSelectBitVector bv_alias(bits);
    EXPECT_EQ(bv_alias.rank1(3), 2ULL);
}

TEST(ModuleImportTest, StaticSearchAndSelectPolicyDispatch) {
    namespace p = algoat::meta::policy;

    std::vector<int> sorted = {2, 5, 8, 12, 16, 23, 38, 56};
    EXPECT_EQ(algoat::meta::static_search(sorted, 16, p::branchless | p::prefetch), 4);
    EXPECT_EQ(algoat::meta::static_search(sorted, 23, p::eytzinger | p::prefetch), 5);
    EXPECT_EQ(algoat::meta::static_search(sorted, 38, p::static_btree), 6);
    EXPECT_EQ(algoat::meta::static_search(sorted, 8, p::galloping), 2);
    EXPECT_EQ(algoat::meta::static_search(sorted, 99, p::branchless), std::nullopt);

    std::vector<int> data = {42, 7, 19, 3, 25, 11, 17};
    auto it1 = algoat::meta::static_select(data, 3, p::introselect | p::in_place);
    EXPECT_EQ(*it1, 17);

    std::vector<int> data2 = {42, 7, 19, 3, 25, 11, 17};
    auto it2 = algoat::meta::static_select(data2, 4, p::floyd_rivest);
    EXPECT_EQ(*it2, 19);

    std::vector<int> data3 = {42, 7, 19, 3, 25, 11, 17};
    auto it3 = algoat::meta::static_select(data3, 1, p::branchless);
    EXPECT_EQ(*it3, 7);
}

TEST(ModuleImportTest, SimdSortingNetworkExport) {
    std::array<std::int32_t, 8> data8 = {8, 3, 7, 1, 5, 2, 6, 4};
    algoat::meta::SIMDSortingNetwork<8, std::int32_t>::sort(std::span<std::int32_t, 8>{data8});
    EXPECT_TRUE(std::is_sorted(data8.begin(), data8.end()));

    std::array<double, 16> data16 = {16.0, 2.0, 14.0, 4.0, 12.0, 6.0, 10.0, 8.0,
                                     15.0, 1.0, 13.0, 3.0, 11.0, 5.0, 9.0,  7.0};
    algoat::meta::sort_simd_network(std::span<double, 16>{data16});
    EXPECT_TRUE(std::is_sorted(data16.begin(), data16.end()));
}

TEST(ModuleImportTest, HardwareTopologyExport) {
    const auto& topo = algoat::core::HardwareTopology::instance();
    EXPECT_GT(topo.l1d_cache_size(), 0U);
    EXPECT_GT(topo.cache_line_size(), 0U);
    EXPECT_GT(topo.optimal_l1_block_elements<int>(0.5), 0U);

    const algoat::core::CacheProperties& props = topo.l1d_properties();
    EXPECT_EQ(props.level, 1U);
    EXPECT_EQ(props.size_bytes, topo.l1d_cache_size());
}

TEST(ModuleImportTest, StackArenaAndScratchBufferPmrExport) {
    algoat::core::StackArena<4096> arena;
    EXPECT_EQ(arena.capacity(), 4096U);

    algoat::core::ScratchBuffer<int> scratch(8, arena.resource());
    EXPECT_EQ(scratch.size(), 8U);

    std::vector<int> merge_vec = {9, 3, 7, 1, 5, 2, 8, 4, 6};
    algoat::sorting::mergesort(std::span{merge_vec}, arena.resource());
    EXPECT_TRUE(std::is_sorted(merge_vec.begin(), merge_vec.end()));

    std::vector<int> lsd_vec = {90, -10, 40, 20, -30, 0};
    algoat::sorting::radixsort_lsd(std::span{lsd_vec}, arena.resource());
    EXPECT_TRUE(std::is_sorted(lsd_vec.begin(), lsd_vec.end()));

    std::vector<int> msd_vec = {90, -10, 40, 20, -30, 0};
    algoat::sorting::radixsort_msd(std::span{msd_vec}, arena.resource());
    EXPECT_TRUE(std::is_sorted(msd_vec.begin(), msd_vec.end()));

    arena.release();
    std::vector<int> radix_vec = {90, -10, 40, 20, -30, 0};
    algoat::sorting::radixsort(std::span{radix_vec}, arena.resource());
    EXPECT_TRUE(std::is_sorted(radix_vec.begin(), radix_vec.end()));
}

TEST(ModuleImportTest, GeometryPolygonBooleanExport) {
    algoat::geometry::BooleanPolygon<> a;
    a.outer_ring = {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}};

    algoat::geometry::boolean::Polygon<> b;
    b.outer_ring = {{5.0, 5.0}, {15.0, 5.0}, {15.0, 15.0}, {5.0, 15.0}};

    algoat::MultiPolygon<> inter = algoat::polygon_intersection(a, b);
    ASSERT_EQ(inter.size(), 1u);
    EXPECT_NEAR(inter[0].area(), 25.0, 1e-12);

    auto uni = algoat::geometry::polygon_boolean(a, b, algoat::BooleanOp::Union);
    ASSERT_EQ(uni.size(), 1u);
    EXPECT_NEAR(uni[0].area(), 175.0, 1e-12);

    auto uni_helper = algoat::polygon_union(a, b);
    ASSERT_EQ(uni_helper.size(), 1u);
    EXPECT_NEAR(uni_helper[0].area(), 175.0, 1e-12);

    auto diff = algoat::polygon_difference(a, b);
    ASSERT_EQ(diff.size(), 1u);
    EXPECT_NEAR(diff[0].area(), 75.0, 1e-12);

    auto sym = algoat::polygon_xor(a, b);
    ASSERT_EQ(sym.size(), 2u);
    EXPECT_NEAR(sym[0].area() + sym[1].area(), 150.0, 1e-12);
}
