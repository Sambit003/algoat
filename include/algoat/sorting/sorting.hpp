/**
 * @file sorting.hpp
 * @brief Umbrella header for all sorting algorithms and C++20 concepts in Algoat.
 * 
 * Aggregates all 19 sorting algorithm implementations and defines the @c SortAlgorithm
 * concept along with the @c SortVariant type definition for dynamic registry storage.
 */

#pragma once

#include <concepts>
#include <string_view>
#include <span>
#include <cstddef>
#include <variant>

#include "algoat/sorting/insertionsort.hpp"
#include "algoat/sorting/quicksort.hpp"
#include "algoat/sorting/mergesort.hpp"
#include "algoat/sorting/heapsort.hpp"
#include "algoat/sorting/selectionsort.hpp"
#include "algoat/sorting/bubblesort.hpp"
#include "algoat/sorting/shellsort.hpp"
#include "algoat/sorting/combsort.hpp"
#include "algoat/sorting/gnomesort.hpp"
#include "algoat/sorting/cyclesort.hpp"
#include "algoat/sorting/bitonicsort.hpp"
#include "algoat/sorting/countingsort.hpp"
#include "algoat/sorting/pigeonholesort.hpp"
#include "algoat/sorting/radixsort.hpp"
#include "algoat/sorting/bucketsort.hpp"
#include "algoat/sorting/introsort.hpp"
#include "algoat/sorting/timsort.hpp"
#include "algoat/sorting/blocksort.hpp"

namespace algoat::sorting {

/**
 * @concept SortAlgorithm
 * @brief Specifies the compile-time requirements for any sorting algorithm in Algoat.
 * 
 * A conforming sorting algorithm struct must provide:
 * 1. <tt>name()</tt>: Returns a string identifier convertible to @c std::string_view.
 * 2. <tt>sort(std::span<T>)</tt>: Sorts the given span in-place with void return.
 * 3. <tt>preferred_min_size()</tt>: Returns minimum recommended element threshold.
 * 
 * @tparam Algo The algorithm struct type.
 * @tparam T The element type in the span.
 */
template<typename Algo, typename T>
concept SortAlgorithm = requires(Algo algo, std::span<T> data) {
    { algo.name() } -> std::convertible_to<std::string_view>;
    { algo.sort(data) } -> std::same_as<void>;
    { algo.preferred_min_size() } -> std::convertible_to<std::size_t>;
};

/**
 * @brief Variant type encapsulating all supported sorting algorithm implementations.
 * 
 * Used by <tt>algoat::core::Registry<SortVariant></tt> for type-safe static dispatch
 * via @c std::visit without virtual function table overhead.
 */
using SortVariant = std::variant<
    InsertionSort, 
    QuickSort, 
    MergeSort, 
    HeapSort, 
    SelectionSort, 
    BubbleSort, 
    ShellSort, 
    CombSort, 
    GnomeSort, 
    CycleSort, 
    BitonicSort, 
    CountingSort, 
    PigeonholeSort, 
    RadixSortLSD, 
    RadixSortMSD, 
    BucketSort, 
    IntroSort, 
    TimSort, 
    BlockSort
>;

} // namespace algoat::sorting
