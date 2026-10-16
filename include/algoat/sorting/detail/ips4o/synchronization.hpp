/******************************************************************************
 * algoat/sorting/detail/ips4o/synchronization.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include <condition_variable>
#include <mutex>
#include <utility>

namespace algoat::sorting::detail::ips4o {
namespace detail {

/**
 * Thread barrier, also supports single() execution.
 */
class Barrier {
public:
    explicit Barrier(int num_threads)
        : init_count_(num_threads), hit_count_(num_threads), flag_(false) {}

    void barrier() {
        std::unique_lock<std::mutex> lk(mutex_);
        if (--hit_count_ == 0) {
            notify_all(lk);
        } else {
            cv_.wait(lk, [this, f = flag_] { return f != flag_; });
        }
    }

    template <class F> void single(F&& func) {
        std::unique_lock<std::mutex> lk(mutex_);
        if (hit_count_-- == init_count_) {
            lk.unlock();
            func();
            lk.lock();
            --hit_count_;
        }
        if (hit_count_ < 0) {
            notify_all(lk);
        } else {
            cv_.wait(lk, [this, f = flag_] { return f != flag_; });
        }
    }

    void setNumThreads(int num_threads) {
        std::unique_lock<std::mutex> lk(mutex_);
        hit_count_ = init_count_ = num_threads;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int init_count_;
    int hit_count_;
    bool flag_;

    void notify_all(std::unique_lock<std::mutex>& lk) {
        hit_count_ = init_count_;
        flag_ = !flag_;
        lk.unlock();
        cv_.notify_all();
    }
};

/**
 * General synchronization support.
 */
class Sync : public Barrier {
public:
    explicit Sync(int num_threads) : Barrier(num_threads) {}

    template <class F> void critical(F&& func) {
        std::unique_lock<std::mutex> lk(mutex_critical_);
        func();
    }

private:
    std::mutex mutex_critical_;
};

} // namespace detail
} // namespace algoat::sorting::detail::ips4o
