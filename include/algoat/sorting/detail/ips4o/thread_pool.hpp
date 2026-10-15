/******************************************************************************
 * algoat/sorting/detail/ips4o/thread_pool.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 * BSD 2-Clause License
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 *****************************************************************************/

#pragma once

#include "synchronization.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace algoat::sorting::detail::ips4o {

/**
 * A thread pool using std::thread.
 */
class StdThreadPool {
public:
    using Sync = detail::Sync;

    explicit StdThreadPool(int num_threads = StdThreadPool::maxNumThreads())
        : impl_(std::make_unique<Impl>(num_threads)) {}

    template <class F>
    void operator()(F&& func, int num_threads = std::numeric_limits<int>::max()) {
        num_threads = std::min(num_threads, numThreads());
        if (num_threads > 1) {
            impl_->run(std::forward<F>(func), num_threads);
        } else {
            func(0, 1);
        }
    }

    Sync& sync() {
        return impl_->sync_;
    }

    int numThreads() const {
        return static_cast<int>(impl_->threads_.size()) + 1;
    }

    static int maxNumThreads() {
        const unsigned int hw = std::thread::hardware_concurrency();
        return hw > 0 ? static_cast<int>(hw) : 1;
    }

private:
    struct Impl {
        Sync sync_;
        detail::Barrier pool_barrier_;
        std::vector<std::thread> threads_;
        std::function<void(int, int)> func_;
        int num_threads_;
        bool done_ = false;

        Impl(int num_threads)
            : sync_(std::max(1, num_threads)), pool_barrier_(std::max(1, num_threads)),
              num_threads_(std::max(1, num_threads)) {
            threads_.reserve(static_cast<std::size_t>(num_threads_ - 1));
            for (int i = 1; i < num_threads_; ++i) {
                threads_.emplace_back(&Impl::main, this, i);
            }
        }

        ~Impl() {
            done_ = true;
            pool_barrier_.barrier();
            for (auto& t : threads_) {
                if (t.joinable()) {
                    t.join();
                }
            }
        }

        template <class F> void run(F&& func, const int num_threads) {
            func_ = func;
            num_threads_ = num_threads;
            sync_.setNumThreads(num_threads);

            pool_barrier_.barrier();
            func_(0, num_threads);
            pool_barrier_.barrier();
        }

        void main(const int my_id) {
            for (;;) {
                pool_barrier_.barrier();
                if (done_)
                    break;
                if (my_id < num_threads_) {
                    func_(my_id, num_threads_);
                }
                pool_barrier_.barrier();
            }
        }
    };

    std::unique_ptr<Impl> impl_;
};

using DefaultThreadPool = StdThreadPool;

} // namespace algoat::sorting::detail::ips4o
