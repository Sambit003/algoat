#include "algoat/simd/isa.hpp"

#include <hwy/highway.h>

namespace algoat::simd {

[[nodiscard]] InstructionSet detect_host_isa() noexcept {
    // Highway caches this on first call, handling all OS and hardware checks safely
    const int64_t targets = hwy::SupportedTargets();

    if (targets & HWY_AVX3) {
        return InstructionSet::AVX512;
    }
    if (targets & HWY_AVX2) {
        return InstructionSet::AVX2;
    }
    if (targets & HWY_NEON) {
        return InstructionSet::Neon;
    }

    return InstructionSet::Scalar;
}

} // namespace algoat::simd
