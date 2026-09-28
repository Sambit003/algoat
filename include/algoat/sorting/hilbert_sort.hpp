#pragma once

#include "algoat/numerics/hilbert.hpp"
#include "algoat/sorting/introsort.hpp"

#include <span>
#include <string_view>

namespace algoat::sorting {

struct HilbertSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "hilbertsort";
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 256;
    }

    template <typename T> void sort(std::span<T> data) const {
        if (data.size() <= 1) {
            return;
        }

        if constexpr (requires {
                          data[0].real();
                          data[0].imag();
                      }) {
            numerics::sort_complex_hilbert(data);
        } else {
            IntroSort{}.sort(data);
        }
    }
};

} // namespace algoat::sorting

ALGOAT_REGISTER_ALGORITHM("sorting", "hilbertsort", ::algoat::sorting::HilbertSort)
