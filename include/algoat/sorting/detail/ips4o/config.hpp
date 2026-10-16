/******************************************************************************
 * algoat/sorting/detail/ips4o/config.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "thread_pool.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace algoat::sorting::detail::ips4o {

template <bool AllowEqualBuckets_ = true, std::ptrdiff_t BaseCase_ = 16,
          std::ptrdiff_t BaseCaseM_ = 16, std::ptrdiff_t BlockSize_ = (2 << 10),
          class BucketT_ = std::ptrdiff_t, std::size_t DataAlign_ = (4 << 10),
          std::ptrdiff_t EqualBuckTh_ = 5, int LogBuckets_ = 8, std::ptrdiff_t MinParBlks_ = 4,
          int OversampleF_ = 20, int UnrollClass_ = 7>
struct Config {
    using bucket_type = BucketT_;

    static constexpr const bool kIs64Bit = sizeof(std::uintptr_t) == 8;
    static_assert(kIs64Bit || sizeof(std::uintptr_t) == 4, "Architecture must be 32 or 64 bit");

    static constexpr const bool kAllowEqualBuckets = AllowEqualBuckets_;
    static constexpr const std::ptrdiff_t kBaseCaseSize = BaseCase_;
    static constexpr const int kBaseCaseMultiplier = BaseCaseM_;
    static constexpr const std::ptrdiff_t kBlockSizeInBytes = BlockSize_;
    static constexpr const std::size_t kDataAlignment = DataAlign_;
    static constexpr const std::ptrdiff_t kEqualBucketsThreshold = EqualBuckTh_;
    static constexpr const int kLogBuckets = LogBuckets_;
    static constexpr const std::ptrdiff_t kMinParallelBlocksPerThread = MinParBlks_;
    static_assert(kMinParallelBlocksPerThread > 0, "Min. blocks per thread must be at least 1.");

    static constexpr const int kUnrollClassifier = UnrollClass_;

    static constexpr const std::ptrdiff_t kSingleLevelThreshold =
        kBaseCaseSize * (1ul << kLogBuckets);
    static constexpr const std::ptrdiff_t kTwoLevelThreshold =
        kSingleLevelThreshold * (1ul << kLogBuckets);

    static constexpr double oversamplingFactor(std::ptrdiff_t n) {
        const double f = OversampleF_ / 100.0 * detail::log2(n);
        return f < 1.0 ? 1.0 : f;
    }

    static int logBuckets(const std::ptrdiff_t n) {
        if (n <= kSingleLevelThreshold) {
            return std::max(1ul, detail::log2(n / kBaseCaseSize));
        } else if (n <= kTwoLevelThreshold) {
            return std::max(1ul, (detail::log2(n / kBaseCaseSize) + 1) / 2);
        } else {
            return kLogBuckets;
        }
    }

    template <class It>
    static constexpr int numThreadsFor(const It& begin, const It& end, int max_threads) {
        const std::ptrdiff_t blocks = (end - begin) * sizeof(decltype(*begin)) / kBlockSizeInBytes;
        return (blocks < (kMinParallelBlocksPerThread * max_threads)) ? 1 : max_threads;
    }
};

template <class It_, class Comp_, class Cfg = Config<>, class ThreadPool_ = DefaultThreadPool>
struct ExtendedConfig : public Cfg {
    using BaseConfig = Cfg;
    using iterator = It_;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using value_type = typename std::iterator_traits<iterator>::value_type;
    using less = Comp_;
    using ThreadPool = ThreadPool_;
    using Sync = decltype(std::declval<ThreadPool&>().sync());

    static constexpr const int kMaxBuckets = 1ul << (Cfg::kLogBuckets + Cfg::kAllowEqualBuckets);

    static constexpr const difference_type kBlockSize =
        1ul << (detail::log2(Cfg::kBlockSizeInBytes < sizeof(value_type)
                                 ? 1
                                 : (Cfg::kBlockSizeInBytes / sizeof(value_type))));

    static constexpr const difference_type kBaseCaseSize = Cfg::kBaseCaseSize;
    static constexpr const difference_type kEqualBucketsThreshold = Cfg::kEqualBucketsThreshold;

    static_assert(Cfg::kLogBuckets <= 8, "Max. bucket count must be <= 512.");
    static_assert((kBlockSize & (kBlockSize - 1)) == 0, "Block size must be a power of two.");
    static_assert(Cfg::kUnrollClassifier <= kBaseCaseSize,
                  "Base case size must be larger than unroll factor.");

    static constexpr difference_type alignToNextBlock(difference_type p) {
        return (p + kBlockSize - 1) & ~(kBlockSize - 1);
    }
};

} // namespace algoat::sorting::detail::ips4o
