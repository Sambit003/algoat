#pragma once

#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <vector>

namespace algoat::sorting::testing {

struct StableItem {
    int key;
    int original_index;

    auto operator<=>(const StableItem& other) const {
        return key <=> other.key;
    }
    bool operator==(const StableItem& other) const {
        return key == other.key;
    }
};

inline void verify_stability(const std::vector<StableItem>& data) {
    for (std::size_t i = 1; i < data.size(); ++i) {
        EXPECT_LE(data[i - 1].key, data[i].key);
        if (data[i - 1].key == data[i].key) {
            EXPECT_LT(data[i - 1].original_index, data[i].original_index);
        }
    }
}

inline std::vector<StableItem> make_standard_stable_dataset() {
    return {{3, 0}, {1, 1}, {3, 2}, {2, 3}, {1, 4}, {3, 5}};
}

template <typename Algo> inline void test_sort_stability(const Algo& algo) {
    auto data = make_standard_stable_dataset();
    algo.sort(std::span{data});
    verify_stability(data);
}

} // namespace algoat::sorting::testing
