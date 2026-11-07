#include "algoat/searching/veb_tree.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>

using namespace algoat::searching;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 3)
        return 0;

    VanEmdeBoasTree<16> veb;
    std::set<uint32_t> ref_set;

    size_t offset = 0;
    while (offset + 3 <= size) {
        uint8_t op = data[offset++];
        uint16_t val = 0;
        std::memcpy(&val, data + offset, sizeof(uint16_t));
        offset += sizeof(uint16_t);

        switch (op % 5) {
        case 0: { // insert
            veb.insert(val);
            ref_set.insert(val);
            break;
        }
        case 1: { // erase
            veb.erase(val);
            ref_set.erase(val);
            break;
        }
        case 2: { // contains
            bool actual = veb.contains(val);
            bool expected = (ref_set.count(val) > 0);
            if (actual != expected)
                __builtin_trap();
            break;
        }
        case 3: { // successor
            auto actual = veb.successor(val);
            auto it = ref_set.upper_bound(val);
            std::optional<uint32_t> expected =
                (it != ref_set.end()) ? std::make_optional(*it) : std::nullopt;
            if (actual != expected)
                __builtin_trap();
            break;
        }
        case 4: { // predecessor
            auto actual = veb.predecessor(val);
            auto it = ref_set.lower_bound(val);
            std::optional<uint32_t> expected = std::nullopt;
            if (it != ref_set.begin()) {
                --it;
                expected = *it;
            }
            if (actual != expected)
                __builtin_trap();
            break;
        }
        }

        // Extremum checks
        if (veb.empty() != ref_set.empty())
            __builtin_trap();

        if (!ref_set.empty()) {
            if (!veb.min().has_value() || *veb.min() != *ref_set.begin())
                __builtin_trap();
            if (!veb.max().has_value() || *veb.max() != *ref_set.rbegin())
                __builtin_trap();
        } else {
            if (veb.min().has_value() || veb.max().has_value())
                __builtin_trap();
        }
    }

    return 0;
}
