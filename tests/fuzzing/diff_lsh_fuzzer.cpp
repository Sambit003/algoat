#include "algoat/sketches/lsh.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace algoat::sketches;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size < 16) {
        return 0;
    }

    const std::size_t num_u64 = size / sizeof(uint64_t);
    std::vector<uint64_t> tokens(num_u64);
    std::memcpy(tokens.data(), data, num_u64 * sizeof(uint64_t));

    // 1. Fuzz BBitMinHash
    BBitMinHash<64, 2> minhash(0x12345678ULL);
    auto sig = minhash.compute_signature(tokens);

    if (sig.hamming_distance(sig) != 0U) {
        __builtin_trap();
    }
    double self_sim = sig.jaccard_similarity(sig);
    if (self_sim < 0.999 || self_sim > 1.001) {
        __builtin_trap();
    }

    // 2. Fuzz SimHash
    uint64_t fp = SimHash::compute_64(tokens);
    if (SimHash::hamming_distance(fp, fp) != 0U) {
        __builtin_trap();
    }
    double cos_self = SimHash::cosine_similarity(fp, fp);
    if (cos_self < 0.999 || cos_self > 1.001) {
        __builtin_trap();
    }

    // Scalar reference verification for SimHash
    std::array<float, 64> scalar_acc{};
    for (uint64_t h : tokens) {
        for (size_t j = 0; j < 64; ++j) {
            scalar_acc[j] += ((h >> j) & 1ULL) ? 1.0f : -1.0f;
        }
    }
    uint64_t expected_fp = 0;
    for (size_t j = 0; j < 64; ++j) {
        if (scalar_acc[j] > 0.0f) {
            expected_fp |= (1ULL << j);
        }
    }
    if (fp != expected_fp) {
        __builtin_trap(); // Differential equivalence failure between SIMD and scalar
    }

    // 3. Fuzz LSHIndex
    LSHIndex<BBitMinHash<64, 2>::Signature, 4, 16> index;
    index.insert(42, sig);
    std::vector<uint64_t> candidates;
    index.query_candidates(sig, candidates);

    bool found = false;
    for (uint64_t c : candidates) {
        if (c == 42) {
            found = true;
            break;
        }
    }
    if (!found) {
        __builtin_trap(); // LSH candidate recall invariant violation
    }

    return 0;
}
