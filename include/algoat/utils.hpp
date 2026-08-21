#pragma once

#include <cassert>

namespace algoat {

template <class T> constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
    assert(!(hi < lo) && "clamp requires lo <= hi");
    return (v < lo) ? lo : ((hi < v) ? hi : v);
}

} // namespace algoat
