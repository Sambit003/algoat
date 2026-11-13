#include "../sorting/test_stable_item.hpp"
#include "algoat/core/scratch_buffer.hpp"
#include "algoat/sorting/mergesort.hpp"
#include "algoat/sorting/radixsort.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory_resource>
#include <new>
#include <rapidcheck.h>
#include <span>
#include <utility>
#include <vector>

namespace {

using algoat::core::ScratchBuffer;
using algoat::core::StackArena;
using algoat::sorting::testing::StableItem;

struct alignas(32) Aligned32Item {
    std::uint64_t words[4]{};
};

TEST(ScratchBufferPBT, StackArenaMonotonicAndDisjointAllocations) {
    rc::check("StackArena serves aligned disjoint allocations until capacity exhaustion",
              [](std::vector<std::pair<std::uint16_t, std::uint8_t>> requests) {
                  StackArena<4096> arena;
                  std::vector<std::pair<std::uintptr_t, std::size_t>> ranges;
                  std::size_t allocated_sum = 0;

                  for (auto [raw_bytes, raw_align_pow] : requests) {
                      const std::size_t bytes = (raw_bytes % 512U) + 1U;
                      const std::size_t align = std::size_t{1} << (raw_align_pow % 7U); // 1..64

                      try {
                          void* p = arena.resource()->allocate(bytes, align);
                          const auto addr = reinterpret_cast<std::uintptr_t>(p);
                          RC_ASSERT(addr % align == 0U);
                          for (const auto& [prev_addr, prev_len] : ranges) {
                              const bool disjoint =
                                  (addr + bytes <= prev_addr) || (prev_addr + prev_len <= addr);
                              RC_ASSERT(disjoint);
                          }
                          ranges.emplace_back(addr, bytes);
                          allocated_sum += bytes;
                          RC_ASSERT(allocated_sum <= arena.capacity());
                      } catch (const std::bad_alloc&) {
                          // Expected once remaining aligned capacity < bytes
                          RC_ASSERT(allocated_sum + bytes > arena.capacity() || ranges.size() > 0U);
                          break;
                      }
                  }
              });
}

TEST(ScratchBufferPBT, ScratchBufferMoveAndZeroCountInvariants) {
    rc::check("ScratchBuffer preserves size, alignment, and move ownership",
              [](std::uint16_t raw_count) {
                  const std::size_t count = raw_count % 256U;
                  StackArena<16384> arena;

                  ScratchBuffer<Aligned32Item> buf1(count, arena.resource());
                  RC_ASSERT(buf1.size() == count);
                  RC_ASSERT(buf1.empty() == (count == 0U));
                  if (count > 0U) {
                      RC_ASSERT(reinterpret_cast<std::uintptr_t>(buf1.data()) % 32U == 0U);
                  } else {
                      RC_ASSERT(buf1.data() == nullptr);
                  }

                  Aligned32Item* orig_ptr = buf1.data();
                  ScratchBuffer<Aligned32Item> buf2(std::move(buf1));
                  RC_ASSERT(buf1.data() == nullptr);
                  RC_ASSERT(buf1.size() == 0U);
                  RC_ASSERT(buf2.data() == orig_ptr);
                  RC_ASSERT(buf2.size() == count);
              });
}

TEST(ScratchBufferPBT, MergeSortWithStackArenaMatchesStableSort) {
    rc::check("mergesort with StackArena preserves sortedness and stability",
              [](std::vector<int> vals) {
                  if (vals.size() > 2000U) {
                      vals.resize(2000U);
                  }
                  std::vector<StableItem> data;
                  data.reserve(vals.size());
                  for (std::size_t i = 0; i < vals.size(); ++i) {
                      data.push_back({vals[i], static_cast<int>(i)});
                  }

                  auto expected = data;
                  std::stable_sort(
                      expected.begin(), expected.end(),
                      [](const StableItem& a, const StableItem& b) { return a.key < b.key; });

                  StackArena<65536> arena;
                  algoat::sorting::mergesort(std::span{data}, arena.resource());

                  RC_ASSERT(data.size() == expected.size());
                  for (std::size_t i = 0; i < data.size(); ++i) {
                      RC_ASSERT(data[i].key == expected[i].key);
                      RC_ASSERT(data[i].original_index == expected[i].original_index);
                  }
              });
}

TEST(ScratchBufferPBT, RadixSortWithStackArenaMatchesStdSort) {
    rc::check("radixsort_lsd and radixsort_msd with StackArena match std::sort",
              [](std::vector<std::int32_t> vals, std::vector<std::int8_t> small_vals) {
                  if (vals.size() > 4000U) {
                      vals.resize(4000U);
                  }
                  if (small_vals.size() > 4000U) {
                      small_vals.resize(4000U);
                  }

                  auto expected32 = vals;
                  std::sort(expected32.begin(), expected32.end());

                  StackArena<65536> arena;
                  auto lsd32 = vals;
                  algoat::sorting::radixsort_lsd(std::span{lsd32}, arena.resource());
                  RC_ASSERT(lsd32 == expected32);

                  arena.release();
                  auto msd32 = vals;
                  algoat::sorting::radixsort_msd(std::span{msd32}, arena.resource());
                  RC_ASSERT(msd32 == expected32);

                  auto expected8 = small_vals;
                  std::sort(expected8.begin(), expected8.end());

                  arena.release();
                  auto lsd8 = small_vals;
                  algoat::sorting::radixsort_lsd(std::span{lsd8}, arena.resource());
                  RC_ASSERT(lsd8 == expected8);

                  arena.release();
                  auto msd8 = small_vals;
                  algoat::sorting::radixsort_msd(std::span{msd8}, arena.resource());
                  RC_ASSERT(msd8 == expected8);
              });
}

} // namespace
