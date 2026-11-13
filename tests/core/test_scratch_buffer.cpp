#include "algoat/core/scratch_buffer.hpp"
#include "pmr_test_helpers.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory_resource>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

ALGOAT_INSTALL_GLOBAL_ALLOC_COUNTER()

namespace algoat::core::testing {
namespace {

using ::algoat::testing::CountingMemoryResource;
using ::algoat::testing::ScopedGlobalAllocTracker;

// Compile-time ownership guarantees for StackArena
static_assert(!std::is_copy_constructible_v<StackArena<64>>);
static_assert(!std::is_copy_assignable_v<StackArena<64>>);
static_assert(!std::is_move_constructible_v<StackArena<64>>);
static_assert(!std::is_move_assignable_v<StackArena<64>>);

TEST(StackArenaTest, DefaultCapacityAndResource) {
    StackArena<> arena;
    EXPECT_EQ(arena.capacity(), 65536U);
    ASSERT_NE(arena.resource(), nullptr);

    void* ptr = arena.resource()->allocate(1024, alignof(std::max_align_t));
    EXPECT_NE(ptr, nullptr);
    arena.resource()->deallocate(ptr, 1024, alignof(std::max_align_t));
}

TEST(StackArenaTest, ThrowsBadAllocOnOverflowWithDefaultNullUpstream) {
    StackArena<128> arena;
    void* first = arena.resource()->allocate(64, 16);
    EXPECT_NE(first, nullptr);

    EXPECT_THROW((void)arena.resource()->allocate(128, 16), std::bad_alloc);
}

TEST(StackArenaTest, FallsBackToCustomUpstreamOnOverflow) {
    CountingMemoryResource upstream;
    {
        StackArena<64> arena(&upstream);
        void* on_stack = arena.resource()->allocate(32, 8);
        EXPECT_NE(on_stack, nullptr);
        EXPECT_EQ(upstream.allocate_calls(), 0U);

        void* spilled = arena.resource()->allocate(128, 16);
        EXPECT_NE(spilled, nullptr);
        EXPECT_GE(upstream.allocate_calls(), 1U);
    }
    EXPECT_EQ(upstream.deallocate_calls(), upstream.allocate_calls());
}

TEST(StackArenaTest, Supports64ByteAlignmentWithoutInitialWaste) {
    StackArena<64> arena;
    void* ptr = arena.resource()->allocate(64, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64U, 0U);
}

TEST(StackArenaTest, ReleaseReclaimsUpstreamAndResetsStackBuffer) {
    CountingMemoryResource upstream;
    StackArena<64> arena(&upstream);

    (void)arena.resource()->allocate(32, 8);
    (void)arena.resource()->allocate(128, 8);
    const std::size_t allocs_before = upstream.allocate_calls();
    EXPECT_GE(allocs_before, 1U);

    arena.release();
    EXPECT_EQ(upstream.deallocate_calls(), allocs_before);

    void* reused = arena.resource()->allocate(64, 8);
    EXPECT_NE(reused, nullptr);
    EXPECT_EQ(upstream.allocate_calls(), allocs_before);
}

TEST(StackArenaTest, ZeroCapacityAndNullUpstreamDoNotCrash) {
    StackArena<0> zero_arena(nullptr);
    EXPECT_EQ(zero_arena.capacity(), 0U);
    EXPECT_THROW((void)zero_arena.resource()->allocate(1, 1), std::bad_alloc);
}

struct TrackedObject {
    inline static int live_count = 0;
    inline static int throw_after = -1;
    int value{7};

    TrackedObject() {
        if (throw_after >= 0 && live_count >= throw_after) {
            throw std::runtime_error("constructed throw");
        }
        ++live_count;
    }
    TrackedObject(const TrackedObject& other) : value(other.value) {
        ++live_count;
    }
    TrackedObject(TrackedObject&& other) noexcept : value(other.value) {
        ++live_count;
    }
    TrackedObject& operator=(const TrackedObject&) = default;
    TrackedObject& operator=(TrackedObject&&) noexcept = default;
    ~TrackedObject() {
        --live_count;
    }
};

struct alignas(64) OverAlignedItem {
    std::uint64_t payload[8]{};
};

static_assert(std::is_same_v<ScratchBuffer<int>::value_type, int>);
static_assert(!std::is_copy_constructible_v<ScratchBuffer<int>>);
static_assert(!std::is_copy_assignable_v<ScratchBuffer<int>>);
static_assert(std::is_move_constructible_v<ScratchBuffer<int>>);
static_assert(std::is_move_assignable_v<ScratchBuffer<int>>);

TEST(ScratchBufferTest, AllocatesContiguousSpanFromStackArena) {
    StackArena<4096> arena;
    ScratchBuffer<int> buf(16, arena.resource());
    EXPECT_EQ(buf.size(), 16U);
    EXPECT_FALSE(buf.empty());
    ASSERT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.span().data(), buf.data());
    EXPECT_EQ(buf.span().size(), 16U);

    for (int i = 0; i < 16; ++i) {
        buf.span()[i] = i * 3;
    }
    EXPECT_EQ(buf.data()[15], 45);
}

TEST(ScratchBufferTest, ZeroCountPerformsZeroAllocations) {
    CountingMemoryResource mr;
    {
        ScratchBuffer<int> buf(0, &mr);
        EXPECT_EQ(buf.size(), 0U);
        EXPECT_TRUE(buf.empty());
        EXPECT_EQ(buf.data(), nullptr);
        EXPECT_TRUE(buf.span().empty());
        EXPECT_EQ(mr.allocate_calls(), 0U);
    }
    EXPECT_EQ(mr.deallocate_calls(), 0U);
}

TEST(ScratchBufferTest, NullMemoryResourceFallsBackToDefault) {
    ScratchBuffer<int> buf(8, nullptr);
    EXPECT_EQ(buf.size(), 8U);
    ASSERT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.memory_resource(), std::pmr::get_default_resource());
}

TEST(ScratchBufferTest, ManagesNonTrivialLifecyclesCleanly) {
    TrackedObject::live_count = 0;
    TrackedObject::throw_after = -1;
    CountingMemoryResource mr;

    {
        ScratchBuffer<TrackedObject> buf(10, &mr);
        EXPECT_EQ(TrackedObject::live_count, 10);
        EXPECT_EQ(mr.allocate_calls(), 1U);
        for (auto& obj : buf.span()) {
            obj.value = 99;
        }
        EXPECT_EQ(buf.data()[9].value, 99);
    }

    EXPECT_EQ(TrackedObject::live_count, 0);
    EXPECT_EQ(mr.deallocate_calls(), 1U);
}

TEST(ScratchBufferTest, ExceptionDuringConstructionCleansUpWithoutLeak) {
    TrackedObject::live_count = 0;
    TrackedObject::throw_after = 4;
    CountingMemoryResource mr;

    EXPECT_THROW((void)ScratchBuffer<TrackedObject>(10, &mr), std::runtime_error);
    EXPECT_EQ(TrackedObject::live_count, 0);
    EXPECT_EQ(mr.allocate_calls(), 1U);
    EXPECT_EQ(mr.deallocate_calls(), 1U);

    TrackedObject::throw_after = -1;
}

TEST(ScratchBufferTest, RespectsOverAlignedTypes) {
    CountingMemoryResource mr;
    {
        ScratchBuffer<OverAlignedItem> buf(4, &mr);
        ASSERT_NE(buf.data(), nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buf.data()) % 64U, 0U);
        EXPECT_EQ(mr.last_alignment(), 64U);
    }
    EXPECT_EQ(mr.deallocate_calls(), 1U);
}

TEST(ScratchBufferTest, MoveConstructAndMoveAssignTransferOwnership) {
    TrackedObject::live_count = 0;
    TrackedObject::throw_after = -1;
    CountingMemoryResource mr1;
    CountingMemoryResource mr2;

    {
        ScratchBuffer<TrackedObject> a(5, &mr1);
        TrackedObject* a_ptr = a.data();
        EXPECT_EQ(TrackedObject::live_count, 5);

        ScratchBuffer<TrackedObject> b(std::move(a));
        EXPECT_EQ(a.data(), nullptr);
        EXPECT_EQ(a.size(), 0U);
        EXPECT_EQ(b.data(), a_ptr);
        EXPECT_EQ(b.size(), 5U);
        EXPECT_EQ(TrackedObject::live_count, 5);

        ScratchBuffer<TrackedObject> c(3, &mr2);
        EXPECT_EQ(TrackedObject::live_count, 8);

        c = std::move(b);
        EXPECT_EQ(TrackedObject::live_count, 5);
        EXPECT_EQ(mr2.deallocate_calls(), 1U);
        EXPECT_EQ(c.data(), a_ptr);
        EXPECT_EQ(c.size(), 5U);
        EXPECT_EQ(b.data(), nullptr);
        EXPECT_EQ(b.size(), 0U);

        // Self-move guard (via reference alias to avoid compiler self-move warning)
        ScratchBuffer<TrackedObject>& c_ref = c;
        c = std::move(c_ref);
        EXPECT_EQ(c.data(), a_ptr);
        EXPECT_EQ(c.size(), 5U);
        EXPECT_EQ(TrackedObject::live_count, 5);
    }

    EXPECT_EQ(TrackedObject::live_count, 0);
    EXPECT_EQ(mr1.deallocate_calls(), 1U);
}

TEST(ScratchBufferTest, StackArenaZeroGlobalHeapAllocations) {
    ScopedGlobalAllocTracker tracker;
    {
        StackArena<65536> arena;
        ScratchBuffer<std::int64_t> buf(4096, arena.resource());
        for (std::size_t i = 0; i < buf.size(); ++i) {
            buf.data()[i] = static_cast<std::int64_t>(i);
        }
    }
    EXPECT_EQ(tracker.count(), 0U);
}

} // namespace
} // namespace algoat::core::testing
