#include "helper_header.hpp"

#include <algoat/algoat.hpp>

namespace dual_mode_test {

void sort_via_header(std::vector<int>& data) {
    algoat::sort(data);
}

bool search_via_header(const std::vector<int>& data, int target) {
    auto res = algoat::search(data, target);
    return res.has_value();
}

} // namespace dual_mode_test
