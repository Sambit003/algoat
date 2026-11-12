#include <algoat/algoat.hpp>
#include <vector>

namespace p = algoat::meta::policy;

int main() {
    std::vector<int> data{5, 3, 1, 4, 2};
    // MUST FAIL TO COMPILE: floyd_rivest and branchless (pdqselect) are mutually exclusive kernels.
    [[maybe_unused]] auto it =
        algoat::meta::static_select(data, 2, p::floyd_rivest | p::branchless);
    return 0;
}
