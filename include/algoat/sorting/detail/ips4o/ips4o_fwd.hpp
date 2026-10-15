/******************************************************************************
 * algoat/sorting/detail/ips4o/ips4o_fwd.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include <iterator>
#include <utility>

namespace algoat::sorting::detail::ips4o {
namespace detail {

template <class It, class Comp> inline void baseCaseSort(It begin, It end, Comp&& comp);

template <class It, class RandomGen>
inline void selectSample(It begin, It end,
                         typename std::iterator_traits<It>::difference_type num_samples,
                         RandomGen&& gen);

template <class Cfg> class Sorter {
public:
    using iterator = typename Cfg::iterator;
    using diff_t = typename Cfg::difference_type;
    using value_type = typename Cfg::value_type;

    class BufferStorage;
    class Block;
    class Buffers;
    class BucketPointers;
    class Classifier;
    struct LocalData;
    struct SharedData;
    explicit Sorter(LocalData& local) : local_(local) {}

    void sequential(iterator begin, iterator end);

    template <class TaskSorter>
    void parallelPrimary(iterator begin, iterator end, SharedData& shared, int num_threads,
                         TaskSorter&& task_sorter);

    void parallelSecondary(SharedData& shared, int id, int num_threads);

private:
    LocalData& local_;
    SharedData* shared_{nullptr};
    Classifier* classifier_{nullptr};

    diff_t* bucket_start_{nullptr};
    BucketPointers* bucket_pointers_{nullptr};
    Block* overflow_{nullptr};

    iterator begin_{};
    iterator end_{};
    int num_buckets_{0};
    int my_id_{0};
    int num_threads_{1};

    static inline int computeLogBuckets(diff_t n);

    std::pair<int, bool> buildClassifier(iterator begin, iterator end, Classifier& classifier);

    template <bool kEqualBuckets> diff_t classifyLocally(iterator my_begin, iterator my_end);

    inline void parallelClassification(bool use_equal_buckets);

    inline void sequentialClassification(bool use_equal_buckets);

    void moveEmptyBlocks(diff_t my_begin, diff_t my_end, diff_t my_first_empty_block);

    inline int computeOverflowBucket();

    template <bool kEqualBuckets, bool kIsParallel>
    inline int classifyAndReadBlock(int read_bucket);

    template <bool kEqualBuckets, bool kIsParallel>
    inline int swapBlock(diff_t max_off, int dest_bucket, bool current_swap);

    template <bool kEqualBuckets, bool kIsParallel> void permuteBlocks();

    inline std::pair<int, diff_t> saveMargins(int last_bucket);

    void writeMargins(int first_bucket, int last_bucket, int overflow_bucket, int swap_bucket,
                      diff_t in_swap_buffer);

    template <bool kIsParallel>
    std::pair<int, bool> partition(iterator begin, iterator end, diff_t* bucket_start,
                                   SharedData* shared, int my_id, int num_threads);

    inline void processSmallTasks(iterator begin, SharedData& shared);
};

} // namespace detail

template <class Cfg> class SequentialSorter;

template <class Cfg> class ParallelSorter;

} // namespace algoat::sorting::detail::ips4o
