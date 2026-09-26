#include "algoat/simd/isa.hpp"

#include <hwy/highway.h>

namespace algoat::simd {

[[nodiscard]] InstructionSet detect_host_isa() noexcept {
    // Thread-safe static initialization guarantees zero runtime penalty on repeated dispatches.
    static const InstructionSet cached_isa = []() noexcept {
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
    }();

    return cached_isa;
}

} // namespace algoat::simd
