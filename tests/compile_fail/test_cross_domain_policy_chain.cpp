#include <algoat/algoat.hpp>
#include <vector>

namespace p = algoat::meta::policy;

int main() {
    std::vector<int> data{1, 2, 3, 4, 5};
    // MUST FAIL TO COMPILE: stable is a sorting policy tag, incompatible with
    // SearchPolicyDescriptor.
    [[maybe_unused]] auto idx = algoat::meta::static_search(data, 3, p::stable);
    return 0;
}
