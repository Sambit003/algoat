#include "algoat/core/scratch_buffer.hpp"
#include "algoat/sorting/mergesort.hpp"
#include "algoat/sorting/radixsort.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory_resource>
#include <new>
#include <span>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < 4U) {
        return 0;
    }

    // 1. Fuzz StackArena + ScratchBuffer with bounded and unbounded upstream
    {
        const std::size_t count = (static_cast<std::size_t>(data[0]) << 8U) | data[1];
        const std::size_t capped_count = count % 4096U;
        algoat::core::StackArena<1024> strict_arena;
        try {
            algoat::core::ScratchBuffer<std::uint32_t> buf(capped_count, strict_arena.resource());
            if (buf.size() != capped_count) {
                __builtin_trap();
            }
            if (capped_count * sizeof(std::uint32_t) > strict_arena.capacity()) {
                __builtin_trap();
            }
            for (std::size_t i = 0; i < buf.size(); ++i) {
                buf.data()[i] = static_cast<std::uint32_t>(i);
            }
        } catch (const std::bad_alloc&) {
            if (capped_count * sizeof(std::uint32_t) <= strict_arena.capacity()) {
                __builtin_trap();
            }
        }

        algoat::core::StackArena<256> fallback_arena(std::pmr::new_delete_resource());
        algoat::core::ScratchBuffer<std::uint64_t> buf_fb(capped_count % 512U,
                                                          fallback_arena.resource());
        if (buf_fb.size() != (capped_count % 512U)) {
            __builtin_trap();
        }
    }

    // 2. Fuzz PMR-backed mergesort, radixsort_lsd, radixsort_msd
    if (size >= sizeof(int)) {
        const std::size_t num_elems = std::min<std::size_t>(size / sizeof(int), 4096U);
        std::vector<int> original(num_elems);
        std::memcpy(original.data(), data, num_elems * sizeof(int));

        auto expected = original;
        std::sort(expected.begin(), expected.end());

        algoat::core::StackArena<65536> arena;

        {
            auto v = original;
            algoat::sorting::mergesort(std::span{v}, arena.resource());
            if (v != expected) {
                __builtin_trap();
            }
        }
        arena.release();
        {
            auto v = original;
            algoat::sorting::radixsort_lsd(std::span{v}, arena.resource());
            if (v != expected) {
                __builtin_trap();
            }
        }
        arena.release();
        {
            auto v = original;
            algoat::sorting::radixsort_msd(std::span{v}, arena.resource());
            if (v != expected) {
                __builtin_trap();
            }
        }
    }

    return 0;
}
