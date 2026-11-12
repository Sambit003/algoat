#include "algoat/meta/static_search.hpp"
#include "algoat/searching/eytzinger_search_tree.hpp"
#include "algoat/searching/searching.hpp"

#include <algorithm>
#include <deque>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <vector>

using namespace algoat::searching;

template <typename Algo> class SearchingPBT : public ::testing::Test {
protected:
    Algo algo;
};

using SearchAlgos = ::testing::Types<LinearSearch, BinarySearch, InterpolationSearch,
                                     AdaptiveBinarySearch, HybridInterpolationSearch,
                                     BranchlessBinarySearch, GallopingSearch, SIMDKarySearch>;

TYPED_TEST_SUITE(SearchingPBT, SearchAlgos);

TYPED_TEST(SearchingPBT, Invariants) {
    rc::check("Searching invariants", [this](std::vector<int> data, int target) {
        if (this->algo.requires_sorted()) {
            std::sort(data.begin(), data.end());
        }

        auto result = this->algo.search(std::span<const int>{data}, target);

        if (result.has_value()) {
            RC_ASSERT(data[result.value()] == target);
        } else {
            bool found = false;
            for (auto x : data) {
                if (x == target)
                    found = true;
            }
            RC_ASSERT(!found);
        }
    });
}

TEST(BranchlessBinarySearchPBT, LowerBoundMatchesStdLowerBound) {
    rc::check("BranchlessBinarySearch lower_bound invariants",
              [](std::vector<int> data, int target) {
                  std::sort(data.begin(), data.end());
                  auto std_it = std::lower_bound(data.begin(), data.end(), target);
                  auto actual_it = branchless_lower_bound(data.begin(), data.end(), target);
                  RC_ASSERT(actual_it == std_it);
              });
}

TEST(GallopingSearchPBT, LowerBoundMatchesStdLowerBound) {
    rc::check("GallopingSearch lower_bound invariants", [](std::vector<int> data, int target) {
        std::sort(data.begin(), data.end());
        auto std_it = std::lower_bound(data.begin(), data.end(), target);
        auto actual_it = galloping_search(data.begin(), data.end(), target);
        RC_ASSERT(actual_it == std_it);
    });
}

TEST(EytzingerSearchPBT, LowerBoundMatchesStdLowerBound) {
    rc::check("EytzingerSearchTree lower_bound invariants", [](std::vector<int> data, int target) {
        std::sort(data.begin(), data.end());
        EytzingerSearchTree<int> tree(std::span<const int>{data});

        auto std_it = std::lower_bound(data.begin(), data.end(), target);
        std::size_t expected_idx = static_cast<std::size_t>(std_it - data.begin());

        RC_ASSERT(tree.size() == data.size());
        RC_ASSERT(tree.lower_bound(target) == expected_idx);
    });
}

TEST(PGMIndexPBT, LowerBoundMatchesStdLowerBound) {
    rc::check("PGMIndex lower_bound matches std::lower_bound",
              [](std::vector<int> data, int target) {
                  std::sort(data.begin(), data.end());
                  PGMIndex<int, 16, 4> index(data);

                  auto std_it = std::lower_bound(data.begin(), data.end(), target);
                  std::size_t expected_idx = static_cast<std::size_t>(std_it - data.begin());
                  RC_ASSERT(index.lower_bound(data, target) == expected_idx);
              });
}

TEST(StaticBTreePBT, LowerBoundMatchesStdLowerBound) {
    rc::check("StaticBTree lower_bound matches std::lower_bound",
              [](std::vector<int> data, int target) {
                  std::sort(data.begin(), data.end());
                  StaticBTree<int> tree(std::span<const int>{data});

                  auto std_it = std::lower_bound(data.begin(), data.end(), target);
                  std::size_t expected_idx = static_cast<std::size_t>(std_it - data.begin());

                  RC_ASSERT(tree.size() == data.size());
                  RC_ASSERT(tree.lower_bound(target) == expected_idx);
              });
}

TEST(StaticBTreePBT, UnsignedSIMDBiasMatchesStdLowerBound) {
    rc::check("StaticBTree unsigned lower_bound matches std::lower_bound",
              [](std::vector<std::uint32_t> data, std::uint32_t target) {
                  std::sort(data.begin(), data.end());
                  StaticBTree<std::uint32_t> tree(std::span<const std::uint32_t>{data});

                  auto std_it = std::lower_bound(data.begin(), data.end(), target);
                  std::size_t expected_idx = static_cast<std::size_t>(std_it - data.begin());

                  RC_ASSERT(tree.size() == data.size());
                  RC_ASSERT(tree.lower_bound(target) == expected_idx);
              });
}

TEST(StaticBTreePBT, BatchedLowerBoundMatchesSingleQuery) {
    rc::check("StaticBTree lower_bound_batch matches std::lower_bound",
              [](std::vector<int> data, std::vector<int> targets) {
                  std::sort(data.begin(), data.end());
                  StaticBTree<int> tree(std::span<const int>{data});

                  std::vector<std::size_t> results(targets.size());
                  tree.lower_bound_batch(targets, results);

                  for (std::size_t i = 0; i < targets.size(); ++i) {
                      auto std_it = std::lower_bound(data.begin(), data.end(), targets[i]);
                      std::size_t expected = static_cast<std::size_t>(std_it - data.begin());
                      RC_ASSERT(results[i] == expected);
                      RC_ASSERT(results[i] == tree.lower_bound(targets[i]));
                  }
              });
}

TEST(SIMDKarySearchPBT, LowerBoundMatchesStdLowerBoundInt) {
    rc::check("simd_kary_search matches std::lower_bound for int",
              [](std::vector<int> data, int target) {
                  std::sort(data.begin(), data.end());
                  auto std_it = std::lower_bound(data.begin(), data.end(), target);
                  std::size_t expected = static_cast<std::size_t>(std_it - data.begin());
                  RC_ASSERT(simd_kary_search(std::span<const int>{data}, target) == expected);
              });
}

TEST(SIMDKarySearchPBT, LowerBoundMatchesStdLowerBoundUint32) {
    rc::check("simd_kary_search matches std::lower_bound for uint32",
              [](std::vector<std::uint32_t> data, std::uint32_t target) {
                  std::sort(data.begin(), data.end());
                  auto std_it = std::lower_bound(data.begin(), data.end(), target);
                  std::size_t expected = static_cast<std::size_t>(std_it - data.begin());
                  RC_ASSERT(simd_kary_search(std::span<const std::uint32_t>{data}, target) ==
                            expected);
              });
}

TEST(SIMDKarySearchPBT, LowerBoundMatchesStdLowerBoundFloat) {
    rc::check("simd_kary_search matches std::lower_bound for float", [](std::vector<float> data,
                                                                        float target) {
        // Filter NaNs for strict ordering
        data.erase(std::remove_if(data.begin(), data.end(), [](float f) { return std::isnan(f); }),
                   data.end());
        if (std::isnan(target))
            return;
        std::sort(data.begin(), data.end());
        auto std_it = std::lower_bound(data.begin(), data.end(), target);
        std::size_t expected = static_cast<std::size_t>(std_it - data.begin());
        RC_ASSERT(simd_kary_search(std::span<const float>{data}, target) == expected);
    });
}

TEST(SIMDKarySearchTreePBT, TreeMatchesStdLowerBoundAndFind) {
    rc::check("SIMDKarySearchTree matches std::lower_bound and find",
              [](std::vector<int> data, int target) {
                  std::sort(data.begin(), data.end());
                  SIMDKarySearchTree<int, 8> tree8(data);
                  SIMDKarySearchTree<int, 16> tree16(data);

                  auto std_it = std::lower_bound(data.begin(), data.end(), target);
                  std::size_t expected = static_cast<std::size_t>(std_it - data.begin());

                  RC_ASSERT(tree8.size() == data.size());
                  RC_ASSERT(tree16.size() == data.size());
                  RC_ASSERT(tree8.lower_bound(target) == expected);
                  RC_ASSERT(tree16.lower_bound(target) == expected);

                  bool std_found = std::binary_search(data.begin(), data.end(), target);
                  RC_ASSERT(tree8.find(target).has_value() == std_found);
                  RC_ASSERT(tree16.find(target).has_value() == std_found);
                  if (std_found) {
                      RC_ASSERT(data[tree8.find(target).value()] == target);
                      RC_ASSERT(data[tree16.find(target).value()] == target);
                  }
              });
}

TEST(FractionalCascadingPBT, LowerBoundMatchesStdLowerBoundAcrossArbitraryLists) {
    rc::check("FractionalCascading search_all matches std::lower_bound across lists",
              [](std::vector<std::vector<int>> lists, int target) {
                  for (auto& l : lists) {
                      std::sort(l.begin(), l.end());
                  }

                  FractionalCascading<int> fc(lists);
                  RC_ASSERT(fc.list_count() == lists.size());

                  auto actual = fc.search_all(target);
                  RC_ASSERT(actual.size() == lists.size());

                  std::vector<std::size_t> span_out(lists.size());
                  fc.search_all_to_span(target, span_out);
                  RC_ASSERT(span_out == actual);

                  for (std::size_t i = 0; i < lists.size(); ++i) {
                      auto std_it = std::lower_bound(lists[i].begin(), lists[i].end(), target);
                      std::size_t expected = static_cast<std::size_t>(std_it - lists[i].begin());
                      RC_ASSERT(actual[i] == expected);
                  }
              });
}

TEST(SearchingInvariantsPBT, StaticSearchPolicyChainsMatchOracle) {
    namespace p = algoat::meta::policy;

    rc::check("static_search policy chains match binary search oracle on sorted ranges",
              [](std::vector<int> data, int target) {
                  std::ranges::sort(data);
                  std::deque<int> dq(data.begin(), data.end());

                  auto verify = [&](std::optional<std::size_t> actual) {
                      bool expected_found = std::ranges::binary_search(data, target);
                      RC_ASSERT(actual.has_value() == expected_found);
                      if (actual.has_value()) {
                          RC_ASSERT(*actual < data.size());
                          RC_ASSERT(data[*actual] == target);
                      }
                  };

                  verify(algoat::meta::static_search(data, target));
                  verify(algoat::meta::static_search(data, target, p::branchless | p::prefetch));
                  verify(algoat::meta::static_search(data, target, p::branchless | p::no_prefetch));
                  verify(algoat::meta::static_search(data, target, p::eytzinger | p::prefetch));
                  verify(algoat::meta::static_search(data, target, p::eytzinger | p::no_prefetch));
                  verify(algoat::meta::static_search(data, target, p::static_btree | p::prefetch));
                  verify(
                      algoat::meta::static_search(data, target, p::static_btree | p::no_prefetch));
                  verify(algoat::meta::static_search(data, target, p::galloping));
                  verify(algoat::meta::static_search(dq, target, p::eytzinger));
                  verify(algoat::meta::static_search(dq, target, p::static_btree));
              });
}
