#include "algoat/sketches/bloom_cuckoo_filter.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <unordered_set>
#include <vector>

using namespace algoat::sketches;

// ============================================================================
// 1. BlockedBloomFilter: Zero False Negatives Invariant
// ============================================================================

RC_GTEST_PROP(BloomCuckooFilterPBT, BlockedBloomFilterZeroFalseNegatives,
              (const std::vector<uint64_t>& keys)) {
    const size_t expected_items = std::max<size_t>(100, keys.size() * 2);
    BlockedBloomFilter<uint64_t> filter(expected_items, 0.01);

    for (uint64_t k : keys) {
        filter.insert(k);
    }

    for (uint64_t k : keys) {
        RC_ASSERT(filter.contains(k));
    }
}

// ============================================================================
// 2. BlockedBloomFilter: Batch Query Equivalence
// ============================================================================

RC_GTEST_PROP(BloomCuckooFilterPBT, BlockedBloomFilterBatchEquivalence,
              (const std::vector<uint64_t>& insert_keys, const std::vector<uint64_t>& query_keys)) {
    const size_t expected_items = std::max<size_t>(100, insert_keys.size() * 2);
    BlockedBloomFilter<uint64_t> filter(expected_items, 0.01);

    for (uint64_t k : insert_keys) {
        filter.insert(k);
    }

    std::vector<bool> expected_results(query_keys.size());
    for (size_t i = 0; i < query_keys.size(); ++i) {
        expected_results[i] = filter.contains(query_keys[i]);
    }

    if (!query_keys.empty()) {
        std::unique_ptr<bool[]> dynamic_bools = std::make_unique<bool[]>(query_keys.size());
        std::span<bool> dynamic_span(dynamic_bools.get(), query_keys.size());
        filter.contains_batch(query_keys, dynamic_span);

        for (size_t i = 0; i < query_keys.size(); ++i) {
            RC_ASSERT(dynamic_span[i] == expected_results[i]);
        }
    }
}

// ============================================================================
// 3. CuckooFilter: Zero False Negatives for Successfully Inserted Items
// ============================================================================

RC_GTEST_PROP(BloomCuckooFilterPBT, CuckooFilterZeroFalseNegatives,
              (const std::vector<uint64_t>& keys)) {
    // Sized with comfortable headroom
    const size_t capacity = std::max<size_t>(128, keys.size() * 4);
    CuckooFilter<uint64_t, 16, 4> cf(capacity);

    std::vector<uint64_t> inserted;
    for (uint64_t k : keys) {
        if (cf.insert(k)) {
            inserted.push_back(k);
        }
    }

    for (uint64_t k : inserted) {
        RC_ASSERT(cf.contains(k));
    }
    RC_ASSERT(cf.size() == inserted.size());
}

// ============================================================================
// 4. CuckooFilter: Remove Invariant (Full Erase & Size Preservation)
// ============================================================================

RC_GTEST_PROP(BloomCuckooFilterPBT, CuckooFilterRemoveInvariant,
              (const std::vector<uint64_t>& raw_keys)) {
    std::unordered_set<uint64_t> unique_set(raw_keys.begin(), raw_keys.end());
    std::vector<uint64_t> keys(unique_set.begin(), unique_set.end());

    const size_t capacity = std::max<size_t>(128, keys.size() * 4);
    CuckooFilter<uint64_t, 16, 4> cf(capacity);

    std::vector<uint64_t> successfully_inserted;
    for (uint64_t k : keys) {
        if (cf.insert(k)) {
            successfully_inserted.push_back(k);
        }
    }

    size_t expected_size = successfully_inserted.size();
    RC_ASSERT(cf.size() == expected_size);

    for (uint64_t k : successfully_inserted) {
        RC_ASSERT(cf.remove(k));
        --expected_size;
        RC_ASSERT(cf.size() == expected_size);
    }

    RC_ASSERT(cf.size() == 0);
    // After complete removal of all keys, table must be empty
    for (uint64_t k : successfully_inserted) {
        RC_ASSERT(!cf.contains(k));
    }
}
