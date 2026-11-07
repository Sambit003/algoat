#include "algoat/algoat.hpp"
#include "algoat/selection/selection.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < sizeof(int) * 2) {
        return 0;
    }

    size_t num_elements = size / sizeof(int);
    std::vector<int> original(num_elements);
    std::memcpy(original.data(), data, num_elements * sizeof(int));

    // Choose target k deterministically from first element modulo size
    size_t k = static_cast<size_t>(
        (static_cast<int64_t>(original[0]) % static_cast<int64_t>(num_elements) +
         static_cast<int64_t>(num_elements)) %
        static_cast<int64_t>(num_elements));

    // Ground truth via std::nth_element
    std::vector<int> expected = original;
    std::nth_element(expected.begin(), expected.begin() + k, expected.end());
    const int expected_kth = expected[k];

    // 1. Differential test algoat::selection::nth_element (Musser Introselect + BFPRT)
    {
        std::vector<int> test_data = original;
        algoat::selection::nth_element(test_data.begin(), test_data.begin() + k, test_data.end());

        if (test_data[k] != expected_kth) {
            __builtin_trap();
        }
        for (size_t i = 0; i < k; ++i) {
            if (test_data[i] > test_data[k]) {
                __builtin_trap();
            }
        }
        for (size_t j = k + 1; j < num_elements; ++j) {
            if (test_data[j] < test_data[k]) {
                __builtin_trap();
            }
        }
    }

    // 2. Differential test algoat::selection::select_optimal (Floyd-Rivest Algorithm 489)
    {
        std::vector<int> test_data = original;
        algoat::selection::select_optimal(test_data.begin(), test_data.begin() + k,
                                          test_data.end());

        if (test_data[k] != expected_kth) {
            __builtin_trap();
        }
        for (size_t i = 0; i < k; ++i) {
            if (test_data[i] > test_data[k]) {
                __builtin_trap();
            }
        }
        for (size_t j = k + 1; j < num_elements; ++j) {
            if (test_data[j] < test_data[k]) {
                __builtin_trap();
            }
        }
    }

    // 3. Differential test algoat::selection::top_k_unsorted
    {
        std::vector<int> test_data = original;
        algoat::selection::top_k_unsorted(test_data, k);

        if (k > 0 && k < num_elements) {
            auto max_prefix = *std::max_element(test_data.begin(), test_data.begin() + k);
            auto min_suffix = *std::min_element(test_data.begin() + k, test_data.end());
            if (max_prefix > min_suffix) {
                __builtin_trap();
            }
        }
    }

    // 4. Differential test algoat::selection::top_k_sorted
    {
        std::vector<int> test_data = original;
        algoat::selection::top_k_sorted(test_data, k);

        if (k > 0 && k < num_elements) {
            if (!std::is_sorted(test_data.begin(), test_data.begin() + k)) {
                __builtin_trap();
            }
            if (test_data[k - 1] > *std::min_element(test_data.begin() + k, test_data.end())) {
                __builtin_trap();
            }
        } else if (k >= num_elements && !test_data.empty()) {
            if (!std::is_sorted(test_data.begin(), test_data.end())) {
                __builtin_trap();
            }
        }
    }

    // 5. Differential test covering adversarial median-of-three killer sequences
    if (num_elements >= 4) {
        std::vector<int> sorted = original;
        std::sort(sorted.begin(), sorted.end());

        // Construct organ-pipe adversarial killer sequence
        std::vector<int> organ_pipe(num_elements);
        size_t left = 0;
        size_t right = num_elements - 1;
        for (size_t i = 0; i < num_elements; ++i) {
            if (i % 2 == 0) {
                organ_pipe[left++] = sorted[i];
            } else {
                organ_pipe[right--] = sorted[i];
            }
        }

        const int expected_val = sorted[k];

        // Verify Musser Introselect with BFPRT handles adversarial pattern
        std::vector<int> test_intro = organ_pipe;
        algoat::selection::nth_element(test_intro.begin(), test_intro.begin() + k,
                                       test_intro.end());
        if (test_intro[k] != expected_val) {
            __builtin_trap();
        }

        // Verify Floyd-Rivest handles adversarial pattern
        std::vector<int> test_opt = organ_pipe;
        algoat::selection::select_optimal(test_opt.begin(), test_opt.begin() + k, test_opt.end());
        if (test_opt[k] != expected_val) {
            __builtin_trap();
        }
    }

    // 6. Differential test algoat::selection::multi_select vs std::sort
    if (num_elements >= 4) {
        std::vector<int> sorted = original;
        std::sort(sorted.begin(), sorted.end());

        std::vector<size_t> ranks = {k / 2, k, k + (num_elements - 1 - k) / 2};
        std::sort(ranks.begin(), ranks.end());

        std::vector<int> test_multi = original;
        algoat::selection::multi_select(test_multi, ranks);

        for (size_t r : ranks) {
            if (test_multi[r] != sorted[r]) {
                __builtin_trap();
            }
        }

        // Check bucket boundaries
        for (size_t i = 0; i < ranks[0]; ++i) {
            if (test_multi[i] > test_multi[ranks[0]]) {
                __builtin_trap();
            }
        }
        for (size_t r = 0; r + 1 < ranks.size(); ++r) {
            for (size_t i = ranks[r] + 1; i < ranks[r + 1]; ++i) {
                if (test_multi[i] < test_multi[ranks[r]] ||
                    test_multi[i] > test_multi[ranks[r + 1]]) {
                    __builtin_trap();
                }
            }
        }
        for (size_t i = ranks.back() + 1; i < num_elements; ++i) {
            if (test_multi[i] < test_multi[ranks.back()]) {
                __builtin_trap();
            }
        }
    }

    // 7. Differential test algoat::selection::quantiles
    if (num_elements >= 4) {
        std::vector<int> sorted = original;
        std::sort(sorted.begin(), sorted.end());

        std::vector<int> test_q = original;
        auto q_res = algoat::selection::quantiles(std::span<int>{test_q}, 4);
        if (q_res.size() != 3) {
            __builtin_trap();
        }
        for (size_t i = 0; i < q_res.size(); ++i) {
            size_t target_r = std::clamp(((i + 1) * num_elements) / 4, size_t{0}, num_elements - 1);
            if (q_res[i] != sorted[target_r]) {
                __builtin_trap();
            }
        }
    }

    return 0;
}
