#pragma once

#include <vector>

namespace dual_mode_test {

// Function compiled in a translation unit using traditional #include <algoat/algoat.hpp>
void sort_via_header(std::vector<int>& data);
bool search_via_header(const std::vector<int>& data, int target);

} // namespace dual_mode_test
