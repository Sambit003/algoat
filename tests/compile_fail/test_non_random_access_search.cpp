#include <algoat/algoat.hpp>
#include <list>

namespace p = algoat::meta::policy;

int main() {
    std::list<int> data{1, 2, 3, 4, 5};
    // MUST FAIL TO COMPILE: static_search requires std::ranges::random_access_range.
    [[maybe_unused]] auto idx = algoat::meta::static_search(data, 3, p::eytzinger);
    return 0;
}
