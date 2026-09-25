/**
 * @file dispatcher.cpp
 * @brief Implementation of the Dispatcher constructor and algorithm registration.
 */

#include "algoat/core/dispatcher.hpp"

namespace algoat::core {

Dispatcher::Dispatcher(AlgoConfig& config)
    : sort_registry_(Registry<sorting::SortVariant>::global("sorting")),
      search_registry_(Registry<searching::SearchVariant>::global("searching")), config_(config) {}

} // namespace algoat::core
