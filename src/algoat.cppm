module;

// Enforce scalar profiler fallback inside module BMI compilation across all compilers (including
// Clang 18) to prevent Highway static inline template exposures and LoadU overload resolution
// failures.
#define ALGOAT_SIMD_PROFILER_SCALAR_ONLY 1

// Global Module Fragment (GMF): only preprocessor directives and non-modular includes
#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// Include library internal implementations within the GMF to preserve global external linkage
#include <algoat/algoat.hpp>
#include <algoat/core/config.hpp>
#include <algoat/core/config_manager.hpp>
#include <algoat/core/dispatcher.hpp>
#include <algoat/core/registry.hpp>
#include <algoat/core/traits.hpp>
#include <algoat/meta/policy.hpp>
#include <algoat/meta/sorting_network.hpp>
#include <algoat/meta/static_sort.hpp>
#include <algoat/numerics/float16_sort.hpp>
#include <algoat/numerics/hilbert.hpp>
#include <algoat/numerics/hybrid_sfc.hpp>
#include <algoat/numerics/morton.hpp>
#include <algoat/searching/searching.hpp>
#include <algoat/selection/selection.hpp>
#include <algoat/sorting/boolean_sort.hpp>
#include <algoat/sorting/sorting.hpp>
#include <algoat/strings/burstsort.hpp>
#include <algoat/strings/multikey_quicksort.hpp>
#include <algoat/strings/substring_search.hpp>
#include <algoat/utils.hpp>

export module algoat;

// ISO C++20 Clause 10: "export using namespace algoat;" is ill-formed.
// Must export declarations explicitly via exported namespace blocks:
export namespace algoat {
// Primary public API
using algoat::branchless_lower_bound;
using algoat::BranchlessBinarySearch;
using algoat::clamp;
using algoat::find_all_substrings;
using algoat::find_substring;
using algoat::galloping_search;
using algoat::GallopingSearch;
using algoat::get_dispatcher;
using algoat::get_global_config;
using algoat::load_global_config;
using algoat::nth_element;
using algoat::search;
using algoat::search_pattern;
using algoat::select_optimal;
using algoat::sort;
using algoat::SpaceFillingCurve;
using algoat::top_k_sorted;
using algoat::top_k_unsorted;
using algoat::core::SearchHint;

namespace selection {
using algoat::selection::nth_element;
using algoat::selection::select_optimal;
using algoat::selection::top_k_sorted;
using algoat::selection::top_k_unsorted;
} // namespace selection

namespace concepts {
using algoat::concepts::Float16;
using algoat::concepts::Float16OrInt16;
using algoat::concepts::Int16;
using algoat::concepts::Size16Bit;
} // namespace concepts

namespace core {
using algoat::core::AlgoConfig;
using algoat::core::ConfigManager;
using algoat::core::DataTraits;
using algoat::core::Dispatcher;
using algoat::core::load_config;
using algoat::core::Registry;
using algoat::core::SearchHint;
} // namespace core

namespace meta {
using algoat::meta::IsPolicyChain;
using algoat::meta::IsPolicyTag;
using algoat::meta::kernel_facet;
using algoat::meta::memory_facet;
using algoat::meta::order_facet;
using algoat::meta::PolicyChain;
using algoat::meta::PolicyDescriptor;
using algoat::meta::SortingNetworkTable;
using algoat::meta::stability_facet;
using algoat::meta::static_sort;

namespace policy {
using algoat::meta::policy::force_radix;
using algoat::meta::policy::force_radix_t;
using algoat::meta::policy::in_place;
using algoat::meta::policy::in_place_t;
using algoat::meta::policy::nearly_sorted;
using algoat::meta::policy::nearly_sorted_t;
using algoat::meta::policy::stable;
using algoat::meta::policy::stable_t;
using algoat::meta::policy::operator|;
} // namespace policy
} // namespace meta

namespace numerics {
using algoat::numerics::Float16Compare;
using algoat::numerics::float_to_hilbert2d;
using algoat::numerics::float_to_morton;
using algoat::numerics::morton_interleave;
using algoat::numerics::MortonCompare;
using algoat::numerics::sort_complex_hybrid;
using algoat::numerics::sort_complex_morton;
using algoat::numerics::sort_float16;
} // namespace numerics

namespace sorting {
using algoat::sorting::boolean_sort;
using algoat::sorting::sort_boolean;
using algoat::sorting::SortAlgorithm;
using algoat::sorting::SortVariant;

// Free-function sorting templates
using algoat::sorting::bitonicsort;
using algoat::sorting::blocksort;
using algoat::sorting::bubblesort;
using algoat::sorting::bucketsort;
using algoat::sorting::combsort;
using algoat::sorting::countingsort;
using algoat::sorting::crumsort;
using algoat::sorting::cyclesort;
using algoat::sorting::fluxsort;
using algoat::sorting::gnomesort;
using algoat::sorting::heapsort;
using algoat::sorting::hilbert_sort;
using algoat::sorting::insertionsort;
using algoat::sorting::introsort;
using algoat::sorting::ips4o;
using algoat::sorting::ips4o_parallel_sort;
using algoat::sorting::mergesort;
using algoat::sorting::pdqsort;
using algoat::sorting::pigeonholesort;
using algoat::sorting::quicksort;
using algoat::sorting::radixsort_inplace_msd;
using algoat::sorting::radixsort_lsd;
using algoat::sorting::radixsort_msd;
using algoat::sorting::selectionsort;
using algoat::sorting::shellsort;
using algoat::sorting::timsort;

// Struct algorithm types
using algoat::sorting::BitonicSort;
using algoat::sorting::BlockSort;
using algoat::sorting::BubbleSort;
using algoat::sorting::BucketSort;
using algoat::sorting::CombSort;
using algoat::sorting::CountingSort;
using algoat::sorting::CrumSort;
using algoat::sorting::CycleSort;
using algoat::sorting::FluxSort;
using algoat::sorting::GnomeSort;
using algoat::sorting::HeapSort;
using algoat::sorting::HilbertSort;
using algoat::sorting::InsertionSort;
using algoat::sorting::IntroSort;
using algoat::sorting::Ips4oSort;
using algoat::sorting::IPS4OSort;
using algoat::sorting::MergeSort;
using algoat::sorting::PdqSort;
using algoat::sorting::PDQSort;
using algoat::sorting::PigeonholeSort;
using algoat::sorting::QuickSort;
using algoat::sorting::RadixSortInPlaceMSD;
using algoat::sorting::RadixSortLSD;
using algoat::sorting::RadixSortMSD;
using algoat::sorting::SelectionSort;
using algoat::sorting::ShellSort;
using algoat::sorting::TimSort;
} // namespace sorting

namespace searching {
using algoat::searching::SearchAlgorithm;
using algoat::searching::SearchVariant;

// Free-function searching templates
using algoat::searching::adaptive_binary_search;
using algoat::searching::binary_search;
using algoat::searching::branchless_lower_bound;
using algoat::searching::galloping_search;
using algoat::searching::hybrid_interpolation_search;
using algoat::searching::interpolation_search;
using algoat::searching::linear_search;

// Struct algorithm types & trees
using algoat::searching::AdaptiveBinarySearch;
using algoat::searching::BinarySearch;
using algoat::searching::BranchlessBinarySearch;
using algoat::searching::EytzingerSearchTree;
using algoat::searching::GallopingSearch;
using algoat::searching::HybridInterpolationSearch;
using algoat::searching::InterpolationSearch;
using algoat::searching::LinearSearch;
using algoat::searching::StaticBTree;
} // namespace searching

namespace strings {
using algoat::strings::burstsort;
using algoat::strings::find_all_substrings;
using algoat::strings::find_substring;
using algoat::strings::multikey_quicksort;
using algoat::strings::search_pattern;
} // namespace strings
} // namespace algoat
