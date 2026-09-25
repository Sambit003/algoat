#pragma once

namespace algoat::simd {

enum class InstructionSet { Scalar, AVX2, AVX512, Neon };

[[nodiscard]] InstructionSet detect_host_isa() noexcept;

} // namespace algoat::simd
