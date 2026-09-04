/**
 * @file dispatcher.cpp
 * @brief Implementation of the Dispatcher constructor and algorithm registration.
 */

#include "algoat/core/dispatcher.hpp"

#include "algoat/searching/adaptive_binary_search.hpp"
#include "algoat/searching/binary_search.hpp"
#include "algoat/searching/interpolation_search.hpp"
#include "algoat/searching/linear_search.hpp"
#include "algoat/sorting/bitonicsort.hpp"
#include "algoat/sorting/blocksort.hpp"
#include "algoat/sorting/bubblesort.hpp"
#include "algoat/sorting/bucketsort.hpp"
#include "algoat/sorting/combsort.hpp"
#include "algoat/sorting/countingsort.hpp"
#include "algoat/sorting/cyclesort.hpp"
#include "algoat/sorting/gnomesort.hpp"
#include "algoat/sorting/heapsort.hpp"
#include "algoat/sorting/insertionsort.hpp"
#include "algoat/sorting/introsort.hpp"
#include "algoat/sorting/mergesort.hpp"
#include "algoat/sorting/pigeonholesort.hpp"
#include "algoat/sorting/quicksort.hpp"
#include "algoat/sorting/radixsort.hpp"
#include "algoat/sorting/selectionsort.hpp"
#include "algoat/sorting/shellsort.hpp"
#include "algoat/sorting/timsort.hpp"

namespace algoat::core {

Dispatcher::Dispatcher(AlgoConfig& config) : config_(config) {
    // Register all supported sorting algorithms into the sorting registry
    // Each algorithm maps to a variant constructor lambda.
    sort_registry_.register_algo("insertionsort",
                                 []() -> sorting::SortVariant { return sorting::InsertionSort{}; });
    sort_registry_.register_algo("quicksort",
                                 []() -> sorting::SortVariant { return sorting::QuickSort{}; });
    sort_registry_.register_algo("mergesort",
                                 []() -> sorting::SortVariant { return sorting::MergeSort{}; });
    sort_registry_.register_algo("heapsort",
                                 []() -> sorting::SortVariant { return sorting::HeapSort{}; });
    sort_registry_.register_algo("selectionsort",
                                 []() -> sorting::SortVariant { return sorting::SelectionSort{}; });
    sort_registry_.register_algo("bubblesort",
                                 []() -> sorting::SortVariant { return sorting::BubbleSort{}; });
    sort_registry_.register_algo("shellsort",
                                 []() -> sorting::SortVariant { return sorting::ShellSort{}; });
    sort_registry_.register_algo("combsort",
                                 []() -> sorting::SortVariant { return sorting::CombSort{}; });
    sort_registry_.register_algo("gnomesort",
                                 []() -> sorting::SortVariant { return sorting::GnomeSort{}; });
    sort_registry_.register_algo("cyclesort",
                                 []() -> sorting::SortVariant { return sorting::CycleSort{}; });
    sort_registry_.register_algo("bitonicsort",
                                 []() -> sorting::SortVariant { return sorting::BitonicSort{}; });
    sort_registry_.register_algo("countingsort",
                                 []() -> sorting::SortVariant { return sorting::CountingSort{}; });
    sort_registry_.register_algo(
        "pigeonholesort", []() -> sorting::SortVariant { return sorting::PigeonholeSort{}; });
    sort_registry_.register_algo("radixsortlsd",
                                 []() -> sorting::SortVariant { return sorting::RadixSortLSD{}; });
    sort_registry_.register_algo("radixsortmsd",
                                 []() -> sorting::SortVariant { return sorting::RadixSortMSD{}; });
    sort_registry_.register_algo("bucketsort",
                                 []() -> sorting::SortVariant { return sorting::BucketSort{}; });
    sort_registry_.register_algo("introsort",
                                 []() -> sorting::SortVariant { return sorting::IntroSort{}; });
    sort_registry_.register_algo("timsort",
                                 []() -> sorting::SortVariant { return sorting::TimSort{}; });
    sort_registry_.register_algo("blocksort",
                                 []() -> sorting::SortVariant { return sorting::BlockSort{}; });

    // Register searching
    search_registry_.register_algo(
        "linearsearch", []() -> searching::SearchVariant { return searching::LinearSearch{}; });
    search_registry_.register_algo(
        "binarysearch", []() -> searching::SearchVariant { return searching::BinarySearch{}; });
    search_registry_.register_algo("interpolationsearch", []() -> searching::SearchVariant {
        return searching::InterpolationSearch{};
    });
    search_registry_.register_algo("adaptivebinarysearch", []() -> searching::SearchVariant {
        return searching::AdaptiveBinarySearch{};
    });
}

} // namespace algoat::core
