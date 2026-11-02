#include <algoat/succinct/fm_index.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <string>
#include <string_view>
#include <vector>

using algoat::succinct::FMIndex;
using algoat::succinct::FMIndexConfig;

namespace {

size_t naive_count(std::string_view text, std::string_view pattern) {
    if (pattern.empty() || text.empty()) {
        return 0;
    }
    size_t count = 0;
    size_t pos = text.find(pattern, 0);
    while (pos != std::string_view::npos) {
        count++;
        pos = text.find(pattern, pos + 1);
    }
    return count;
}

} // namespace

// ============================================================================
// 1. Invariant: Count matches naive brute-force search
// ============================================================================

RC_GTEST_PROP(FMIndexPBT, CountFindEquivalence, (const std::string& text)) {
    RC_PRE(!text.empty());

    FMIndex idx{text};
    RC_ASSERT(idx.text_length() == text.size());

    // Test a valid substring
    size_t start = *rc::gen::inRange<size_t>(0, text.size());
    size_t len = *rc::gen::inRange<size_t>(1, std::min<size_t>(10, text.size() - start + 1));
    std::string pattern = text.substr(start, len);

    size_t expected = naive_count(text, pattern);
    size_t actual = idx.count(pattern);
    RC_ASSERT(actual == expected);
}

// ============================================================================
// 2. Invariant: Locate accuracy and correspondence with count
// ============================================================================

RC_GTEST_PROP(FMIndexPBT, LocateAccuracyAndCorrespondence, (const std::string& text)) {
    RC_PRE(!text.empty());

    FMIndex idx{text};

    size_t start = *rc::gen::inRange<size_t>(0, text.size());
    size_t len = *rc::gen::inRange<size_t>(1, std::min<size_t>(8, text.size() - start + 1));
    std::string pattern = text.substr(start, len);

    std::vector<size_t> locs = idx.locate(pattern);
    RC_ASSERT(locs.size() == idx.count(pattern));

    for (size_t pos : locs) {
        RC_ASSERT(pos + len <= text.size());
        RC_ASSERT(text.substr(pos, len) == pattern);
    }

    // Verify positions are sorted and distinct
    for (size_t i = 1; i < locs.size(); ++i) {
        RC_ASSERT(locs[i - 1] < locs[i]);
    }
}

// ============================================================================
// 3. Invariant: Extract recovers original text substrings
// ============================================================================

RC_GTEST_PROP(FMIndexPBT, ExtractRoundTrip, (const std::string& text)) {
    RC_PRE(!text.empty());

    FMIndex idx{text};

    size_t pos = *rc::gen::inRange<size_t>(0, text.size());
    size_t len = *rc::gen::inRange<size_t>(0, text.size() - pos + 1);

    std::string extracted = idx.extract(pos, len);
    RC_ASSERT(extracted == text.substr(pos, len));
}

// ============================================================================
// 4. Invariant: Binary serialization round-trip fidelity
// ============================================================================

RC_GTEST_PROP(FMIndexPBT, SerializationRoundTrip, (const std::string& text)) {
    RC_PRE(!text.empty());

    FMIndex orig{text};
    std::vector<uint8_t> bytes = orig.serialize();
    RC_ASSERT(!bytes.empty());

    FMIndex restored = FMIndex::deserialize(bytes);
    RC_ASSERT(restored.text_length() == orig.text_length());

    size_t pos = *rc::gen::inRange<size_t>(0, text.size());
    size_t len = *rc::gen::inRange<size_t>(1, std::min<size_t>(6, text.size() - pos + 1));
    std::string pattern = text.substr(pos, len);

    RC_ASSERT(restored.count(pattern) == orig.count(pattern));
    RC_ASSERT(restored.locate(pattern) == orig.locate(pattern));
    RC_ASSERT(restored.extract(pos, len) == orig.extract(pos, len));
}
