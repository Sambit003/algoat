/**
 * @file aho_corasick.cpp
 * @brief Explicit template instantiations for AhoCorasick engine.
 */

#include "algoat/dp/aho_corasick.hpp"

namespace algoat::dp {

template class AhoCorasick<char>;
template class AhoCorasick<uint8_t>;

} // namespace algoat::dp
