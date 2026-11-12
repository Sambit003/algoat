#include <algoat/algoat.hpp>
#include <vector>

namespace p = algoat::meta::policy;

int main() {
    std::vector<int> data{1, 2, 3, 4, 5};
    // MUST FAIL TO COMPILE: eytzinger and static_btree are mutually exclusive memory layouts.
    [[maybe_unused]] auto idx =
        algoat::meta::static_search(data, 3, p::eytzinger | p::static_btree);
    return 0;
}
