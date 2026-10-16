/**
 * @file prefetch.hpp
 * @brief Portable software memory prefetch instruction wrapper.
 */

#pragma once

#if defined(_MSC_VER)
#include <xmmintrin.h>
#endif

namespace algoat::searching::detail {

/**
 * @brief Software memory prefetch instruction wrapper for GCC, Clang, and MSVC.
 *
 * Emits non-faulting prefetch instructions targeting the L1/L2/L3 cache hierarchy.
 * On GCC/Clang, maps to __builtin_prefetch with read intent (0) and low temporal locality (0).
 * On MSVC, maps to _mm_prefetch(_MM_HINT_T0).
 *
 * @tparam Ptr Pointer element type.
 * @param ptr Address to prefetch.
 */
template <typename Ptr> inline void prefetch_read(const Ptr* ptr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(static_cast<const void*>(ptr), 0, 0);
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(static_cast<const void*>(ptr)), _MM_HINT_T0);
#else
    (void)ptr;
#endif
}

} // namespace algoat::searching::detail
