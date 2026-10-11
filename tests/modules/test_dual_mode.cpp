/**
 * @file test_dual_mode.cpp
 * @brief Dual-mode interop test between #include <algoat/algoat.hpp> and import algoat;.
 */

#include "helper_header.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

import algoat;

TEST(DualModeInteropTest, SeamlessDataExchangeAndSorting) {
    std::vector<int> data1{9, 3, 5, 1, 8, 2, 7, 4, 6};
    std::vector<int> data2 = data1;

    // Sort data1 using import algoat;
    algoat::sort(data1);

    // Sort data2 using helper compiled with #include <algoat/algoat.hpp>
    dual_mode_test::sort_via_header(data2);

    EXPECT_TRUE(std::is_sorted(data1.begin(), data1.end()));
    EXPECT_TRUE(std::is_sorted(data2.begin(), data2.end()));
    EXPECT_EQ(data1, data2);
}

TEST(DualModeInteropTest, SeamlessSearchInteroperability) {
    std::vector<int> sorted_data{1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Search using import algoat;
    auto mod_found = algoat::search(sorted_data, 5);
    ASSERT_TRUE(mod_found.has_value());
    EXPECT_EQ(*mod_found, 4);

    // Search using header-compiled helper
    bool header_found = dual_mode_test::search_via_header(sorted_data, 5);
    EXPECT_TRUE(header_found);

    auto mod_missing = algoat::search(sorted_data, 99);
    EXPECT_FALSE(mod_missing.has_value());
    bool header_missing = dual_mode_test::search_via_header(sorted_data, 99);
    EXPECT_FALSE(header_missing);
}
