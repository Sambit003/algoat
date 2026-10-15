/******************************************************************************
 * algoat/sorting/detail/ips4o/memory.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "bucket_pointers.hpp"
#include "buffers.hpp"
#include "classifier.hpp"
#include "ips4o_fwd.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace algoat::sorting::detail::ips4o {
namespace detail {

template <class T> inline T* alignPointer(T* ptr, std::size_t alignment) {
    auto v = reinterpret_cast<std::uintptr_t>(ptr);
    v = (v - 1 + alignment) & ~(alignment - 1);
    return reinterpret_cast<T*>(v);
}

template <class T> class AlignedPtr {
public:
    AlignedPtr() = default;

    template <class... Args>
    explicit AlignedPtr(std::size_t alignment, Args&&... args)
        : alloc_(new char[sizeof(T) + alignment]),
          value_(new (alignPointer(alloc_, alignment)) T(std::forward<Args>(args)...)) {}

    AlignedPtr(const AlignedPtr&) = delete;
    AlignedPtr& operator=(const AlignedPtr&) = delete;

    AlignedPtr(AlignedPtr&& rhs) noexcept : alloc_(rhs.alloc_), value_(rhs.value_) {
        rhs.alloc_ = nullptr;
        rhs.value_ = nullptr;
    }

    AlignedPtr& operator=(AlignedPtr&& rhs) noexcept {
        if (this != &rhs) {
            if (alloc_) {
                value_->~T();
                delete[] alloc_;
            }
            alloc_ = rhs.alloc_;
            value_ = rhs.value_;
            rhs.alloc_ = nullptr;
            rhs.value_ = nullptr;
        }
        return *this;
    }

    ~AlignedPtr() {
        if (alloc_) {
            value_->~T();
            delete[] alloc_;
        }
    }

    T& get() {
        return *value_;
    }
    const T& get() const {
        return *value_;
    }

private:
    char* alloc_{nullptr};
    T* value_{nullptr};
};

template <> class AlignedPtr<void> {
public:
    AlignedPtr() = default;

    explicit AlignedPtr(std::size_t alignment, std::size_t size)
        : alloc_(new char[size + alignment]), value_(alignPointer(alloc_, alignment)) {}

    AlignedPtr(const AlignedPtr&) = delete;
    AlignedPtr& operator=(const AlignedPtr&) = delete;

    AlignedPtr(AlignedPtr&& rhs) noexcept : alloc_(rhs.alloc_), value_(rhs.value_) {
        rhs.alloc_ = nullptr;
        rhs.value_ = nullptr;
    }

    AlignedPtr& operator=(AlignedPtr&& rhs) noexcept {
        if (this != &rhs) {
            delete[] alloc_;
            alloc_ = rhs.alloc_;
            value_ = rhs.value_;
            rhs.alloc_ = nullptr;
            rhs.value_ = nullptr;
        }
        return *this;
    }

    ~AlignedPtr() {
        delete[] alloc_;
    }

    char* get() {
        return value_;
    }
    const char* get() const {
        return value_;
    }

private:
    char* alloc_{nullptr};
    char* value_{nullptr};
};

template <class Cfg> class Sorter<Cfg>::BufferStorage : public AlignedPtr<void> {
    static constexpr const auto kPerThread = Cfg::kBlockSizeInBytes * Cfg::kMaxBuckets;

public:
    BufferStorage() = default;

    explicit BufferStorage(int num_threads)
        : AlignedPtr<void>(Cfg::kDataAlignment, num_threads * kPerThread) {}

    char* forThread(int id) {
        return this->get() + id * kPerThread;
    }
};

template <class Cfg> struct Sorter<Cfg>::LocalData {
    using diff_t = typename Cfg::difference_type;

    diff_t bucket_size[Cfg::kMaxBuckets];
    Buffers buffers;
    Block swap[2];
    Block overflow;

    BucketPointers bucket_pointers[Cfg::kMaxBuckets];

    Classifier classifier;

    diff_t first_block{0};
    diff_t first_empty_block{0};

    std::linear_congruential_engine<std::uintptr_t, Cfg::kIs64Bit ? 6364136223846793005u : 1664525u,
                                    Cfg::kIs64Bit ? 1442695040888963407u : 1013904223u, 0u>
        random_generator;

    LocalData(typename Cfg::less comp, char* buffer_storage)
        : buffers(buffer_storage), classifier(std::move(comp)) {
        std::random_device rdev;
        std::uintptr_t seed = rdev();
        if constexpr (Cfg::kIs64Bit)
            seed = (seed << 32) | rdev();
        random_generator.seed(seed);
        reset();
    }

    void reset() {
        classifier.reset();
        std::fill_n(bucket_size, Cfg::kMaxBuckets, 0);
    }
};

struct ParallelTask {
    std::ptrdiff_t begin;
    std::ptrdiff_t end;
    int level;

    bool operator==(const ParallelTask& rhs) const = default;
    bool operator<(const ParallelTask& rhs) const {
        return end - begin < rhs.end - rhs.begin;
    }
    bool operator<=(const ParallelTask& rhs) const {
        return end - begin <= rhs.end - rhs.begin;
    }
    bool operator>(const ParallelTask& rhs) const {
        return end - begin > rhs.end - rhs.begin;
    }
    bool operator>=(const ParallelTask& rhs) const {
        return end - begin >= rhs.end - rhs.begin;
    }
};

template <class Cfg> struct Sorter<Cfg>::SharedData {
    typename Cfg::difference_type bucket_start[Cfg::kMaxBuckets + 1];
    BucketPointers bucket_pointers[Cfg::kMaxBuckets];
    Block* overflow{nullptr};
    int num_buckets{0};
    bool use_equal_buckets{false};

    Classifier classifier;

    typename Cfg::Sync sync;

    std::vector<LocalData*> local;

    typename Cfg::iterator begin_{};
    std::vector<ParallelTask> big_tasks;
    std::vector<ParallelTask> small_tasks;
    std::atomic_size_t small_task_index{0};

    SharedData(typename Cfg::less comp, typename Cfg::Sync s, std::size_t num_threads)
        : classifier(std::move(comp)), sync(s), local(num_threads) {
        reset();
    }

    void reset() {
        classifier.reset();
        std::fill_n(bucket_start, Cfg::kMaxBuckets + 1, 0);
        overflow = nullptr;
    }
};

} // namespace detail
} // namespace algoat::sorting::detail::ips4o
