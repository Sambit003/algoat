/**
 * @file hardware_topology.cpp
 * @brief Cross-platform runtime detection of CPU cache topology and SIMD extensions.
 */

#include "algoat/core/hardware_topology.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <vector>
#include <windows.h>
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) &&                                                 \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <cpuid.h>
#include <immintrin.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__) || defined(__unix__)
#include <unistd.h>
#if defined(__has_include)
#if __has_include(<sys/auxv.h>)
#include <sys/auxv.h>
#define ALGOAT_HAS_SYS_AUXV 1
#endif
#if __has_include(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif
#endif
#endif

namespace algoat::core {

namespace {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

void cpuid_query(std::uint32_t leaf, std::uint32_t subleaf,
                 std::array<std::uint32_t, 4>& regs) noexcept {
#if defined(_MSC_VER)
    int raw[4]{0, 0, 0, 0};
    __cpuidex(raw, static_cast<int>(leaf), static_cast<int>(subleaf));
    for (std::size_t i = 0; i < 4; ++i) {
        regs[i] = static_cast<std::uint32_t>(raw[i]);
    }
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#else
    regs.fill(0);
    (void)leaf;
    (void)subleaf;
#endif
}

#if defined(_MSC_VER)
std::uint64_t read_xcr0() noexcept {
    return static_cast<std::uint64_t>(_xgetbv(0));
}
#elif defined(__GNUC__) || defined(__clang__)
[[gnu::target("xsave")]] std::uint64_t read_xcr0() noexcept {
    return static_cast<std::uint64_t>(_xgetbv(0));
}
#endif

#endif // x86 / x86-64

#if defined(__APPLE__)

template <typename T> bool query_sysctl(const char* name, T& out) noexcept {
    std::size_t size = sizeof(T);
    return ::sysctlbyname(name, &out, &size, nullptr, 0) == 0 && size == sizeof(T);
}

#endif // __APPLE__

#if defined(__linux__) || defined(__unix__)

std::size_t query_sysconf_value(int name) noexcept {
    if (name < 0) {
        return 0U;
    }
    const long val = ::sysconf(name);
    return (val > 0) ? static_cast<std::size_t>(val) : 0U;
}

#endif // __linux__ || __unix__

} // namespace

const HardwareTopology& HardwareTopology::instance() noexcept {
    static const HardwareTopology topology;
    return topology;
}

HardwareTopology::HardwareTopology() noexcept {
    detect_x86_cpuid();
    detect_posix_sysconf();
    if (detected_any_cache_ && !detected_l3_cache_) {
        l3_props_.size_bytes = 0U;
    }
}

void HardwareTopology::record_cache(std::size_t level, std::size_t size_bytes,
                                    std::size_t line_bytes, std::size_t assoc) noexcept {
    CacheProperties* target = nullptr;
    switch (level) {
    case 1U:
        target = &l1d_props_;
        break;
    case 2U:
        target = &l2_props_;
        break;
    case 3U:
        target = &l3_props_;
        if (size_bytes > 0U) {
            detected_l3_cache_ = true;
        }
        break;
    default:
        return;
    }

    if (size_bytes > 0U) {
        target->level = level;
        target->size_bytes = size_bytes;
        detected_any_cache_ = true;
    }
    if (line_bytes > 0U) {
        target->line_size_bytes = line_bytes;
        if (level == 1U || cache_line_size_ == 0U) {
            cache_line_size_ = line_bytes;
        }
    }
    if (assoc > 0U) {
        target->associativity = assoc;
    }
}

void HardwareTopology::detect_x86_cpuid() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    std::array<std::uint32_t, 4> regs{};
    cpuid_query(0U, 0U, regs);
    const std::uint32_t max_basic_leaf = regs[0];
    if (max_basic_leaf < 1U) {
        return;
    }

    // Leaf 01H: CLFLUSH line size fallback, XSAVE/OSXSAVE, and hardware AVX
    cpuid_query(1U, 0U, regs);
    const std::uint32_t ebx1 = regs[1];
    const std::uint32_t ecx1 = regs[2];
    const std::uint32_t edx1 = regs[3];

    if ((edx1 & (1U << 19)) != 0U) {
        const std::size_t clflush_bytes = static_cast<std::size_t>((ebx1 >> 8) & 0xFFU) * 8U;
        if (clflush_bytes >= 32U && clflush_bytes <= 256U) {
            cache_line_size_ = clflush_bytes;
            l1d_props_.line_size_bytes = clflush_bytes;
        }
    }

    const bool has_xsave = (ecx1 & (1U << 26)) != 0U;
    const bool has_osxsave = (ecx1 & (1U << 27)) != 0U;
    const bool hw_avx = (ecx1 & (1U << 28)) != 0U;

    bool os_supports_avx = false;
    bool os_supports_avx512 = false;
    if (has_xsave && has_osxsave) {
        const std::uint64_t xcr0 = read_xcr0();
        constexpr std::uint64_t kXcr0AvxMask = 0x06ULL;    // SSE (bit 1) + AVX (bit 2)
        constexpr std::uint64_t kXcr0Avx512Mask = 0xE6ULL; // SSE + AVX + opmask/ZMM (bits 5..7)
        os_supports_avx = (xcr0 & kXcr0AvxMask) == kXcr0AvxMask;
        os_supports_avx512 = (xcr0 & kXcr0Avx512Mask) == kXcr0Avx512Mask;
    }

    // Leaf 07H Subleaf 0: Extended vector & bit-manipulation flags
    if (max_basic_leaf >= 7U) {
        cpuid_query(7U, 0U, regs);
        const std::uint32_t ebx7 = regs[1];
        has_bmi2_ = (ebx7 & (1U << 8)) != 0U;
        has_avx2_ = hw_avx && os_supports_avx && ((ebx7 & (1U << 5)) != 0U);
        has_avx512f_ = hw_avx && os_supports_avx512 && ((ebx7 & (1U << 16)) != 0U);
    }

    // Deterministic Cache Parameters helper shared by Intel Leaf 04H and AMD Leaf 8000001DH
    auto probe_deterministic_caches = [this](std::uint32_t leaf) noexcept -> bool {
        bool found_any = false;
        constexpr std::uint32_t kMaxCacheSubleaves = 32U;
        for (std::uint32_t subleaf = 0; subleaf < kMaxCacheSubleaves; ++subleaf) {
            std::array<std::uint32_t, 4> c{};
            cpuid_query(leaf, subleaf, c);
            const std::uint32_t cache_type = c[0] & 0x1FU;
            if (cache_type == 0U) {
                break;
            }
            // Skip Instruction-only caches (type 2); inspect Data (1) or Unified (3)
            if (cache_type != 1U && cache_type != 3U) {
                continue;
            }

            const std::uint32_t level = (c[0] >> 5) & 0x7U;
            if (level == 1U && cache_type != 1U) {
                continue;
            }

            const std::uint64_t line_size = static_cast<std::uint64_t>(c[1] & 0xFFFU) + 1ULL;
            const std::uint64_t partitions =
                static_cast<std::uint64_t>((c[1] >> 12) & 0x3FFU) + 1ULL;
            const std::uint64_t ways = static_cast<std::uint64_t>((c[1] >> 22) & 0x3FFU) + 1ULL;
            const std::uint64_t sets = static_cast<std::uint64_t>(c[2]) + 1ULL;
            const std::uint64_t total_bytes = ways * partitions * line_size * sets;

            if (total_bytes == 0ULL) {
                continue;
            }

            record_cache(level, static_cast<std::size_t>(total_bytes),
                         static_cast<std::size_t>(line_size), static_cast<std::size_t>(ways));
            found_any = true;
        }
        return found_any;
    };

    bool cpuid_cache_found = false;
    if (max_basic_leaf >= 4U) {
        cpuid_cache_found = probe_deterministic_caches(4U);
    }

    // AMD CPUs reserve Leaf 04H (all zeros) and expose cache topology via Leaf 8000001DH or
    // legacy extended leaves 80000005H / 80000006H.
    if (!cpuid_cache_found) {
        cpuid_query(0x80000000U, 0U, regs);
        const std::uint32_t max_ext_leaf = regs[0];
        if (max_ext_leaf >= 0x80000001U) {
            cpuid_query(0x80000001U, 0U, regs);
            const bool has_topoext = (regs[2] & (1U << 22)) != 0U;
            if (has_topoext && max_ext_leaf >= 0x8000001DU) {
                cpuid_cache_found = probe_deterministic_caches(0x8000001DU);
            }
        }
        if (!cpuid_cache_found && max_ext_leaf >= 0x80000005U) {
            cpuid_query(0x80000005U, 0U, regs);
            const std::uint32_t l1d_kb = (regs[2] >> 24) & 0xFFU;
            const std::uint32_t l1d_assoc = (regs[2] >> 16) & 0xFFU;
            const std::uint32_t l1d_line = regs[2] & 0xFFU;
            if (l1d_kb > 0U) {
                record_cache(1U, static_cast<std::size_t>(l1d_kb) * 1024U,
                             static_cast<std::size_t>(l1d_line),
                             static_cast<std::size_t>(l1d_assoc));
            }
        }
        if (!cpuid_cache_found && max_ext_leaf >= 0x80000006U) {
            cpuid_query(0x80000006U, 0U, regs);
            const std::uint32_t l2_kb = (regs[2] >> 16) & 0xFFFFU;
            const std::uint32_t l2_line = regs[2] & 0xFFU;
            if (l2_kb > 0U) {
                record_cache(2U, static_cast<std::size_t>(l2_kb) * 1024U,
                             static_cast<std::size_t>(l2_line), 0U);
            }
            const std::uint32_t l3_512kb = (regs[3] >> 18) & 0x3FFFU;
            const std::uint32_t l3_line = regs[3] & 0xFFU;
            if (l3_512kb > 0U) {
                record_cache(3U, static_cast<std::size_t>(l3_512kb) * 512U * 1024U,
                             static_cast<std::size_t>(l3_line), 0U);
            }
        }
    }
#endif
}

void HardwareTopology::detect_posix_sysconf() noexcept {
#if defined(__APPLE__)
    std::uint64_t line_size = 0;
    if (query_sysctl("hw.cachelinesize", line_size) && line_size > 0) {
        const std::size_t parsed_line = static_cast<std::size_t>(line_size);
        cache_line_size_ = parsed_line;
        l1d_props_.line_size_bytes = parsed_line;
    }

    // Apple Silicon uses heterogeneous perflevels where hw.l1dcachesize fails with ENOENT.
    std::uint32_t nperflevels = 0;
    bool found_perflevel_l1 = false;
    bool found_perflevel_l2 = false;
    if (query_sysctl("hw.nperflevels", nperflevels) && nperflevels > 0) {
        std::size_t min_l1d = std::numeric_limits<std::size_t>::max();
        std::size_t min_l2 = std::numeric_limits<std::size_t>::max();
        for (std::uint32_t i = 0; i < nperflevels && i < 8U; ++i) {
            std::uint64_t val = 0;
            char key[64]{};
            std::snprintf(key, sizeof(key), "hw.perflevel%u.l1dcachesize", i);
            if (query_sysctl(key, val) && val > 0) {
                min_l1d = std::min(min_l1d, static_cast<std::size_t>(val));
                found_perflevel_l1 = true;
            }
            std::snprintf(key, sizeof(key), "hw.perflevel%u.l2cachesize", i);
            if (query_sysctl(key, val) && val > 0) {
                min_l2 = std::min(min_l2, static_cast<std::size_t>(val));
                found_perflevel_l2 = true;
            }
        }
        if (found_perflevel_l1) {
            record_cache(1U, min_l1d, cache_line_size_, 0U);
        }
        if (found_perflevel_l2) {
            record_cache(2U, min_l2, cache_line_size_, 0U);
        }
    }

    if (!found_perflevel_l1) {
        std::uint64_t l1d = 0;
        if (query_sysctl("hw.l1dcachesize", l1d) && l1d > 0) {
            record_cache(1U, static_cast<std::size_t>(l1d), cache_line_size_, 0U);
        }
    }
    if (!found_perflevel_l2) {
        std::uint64_t l2 = 0;
        if (query_sysctl("hw.l2cachesize", l2) && l2 > 0) {
            record_cache(2U, static_cast<std::size_t>(l2), cache_line_size_, 0U);
        }
    }

    std::uint64_t l3 = 0;
    if (query_sysctl("hw.l3cachesize", l3) && l3 > 0) {
        record_cache(3U, static_cast<std::size_t>(l3), cache_line_size_, 0U);
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    int flag = 0;
    if (query_sysctl("hw.optional.AdvSIMD", flag)) {
        has_neon_ = (flag != 0);
    } else if (query_sysctl("hw.optional.neon", flag)) {
        has_neon_ = (flag != 0);
    } else {
        has_neon_ = true; // Mandatory in AArch64
    }
#elif defined(__x86_64__) || defined(_M_X64)
    int flag = 0;
    if (query_sysctl("hw.optional.avx2_0", flag)) {
        has_avx2_ = (flag != 0);
    }
    if (query_sysctl("hw.optional.avx512f", flag)) {
        has_avx512f_ = (flag != 0);
    }
    if (query_sysctl("hw.optional.bmi2", flag)) {
        has_bmi2_ = (flag != 0);
    }
#endif

#elif defined(__linux__) || defined(__unix__)
#ifdef _SC_LEVEL1_DCACHE_LINESIZE
    const std::size_t sc_line = query_sysconf_value(_SC_LEVEL1_DCACHE_LINESIZE);
    if (sc_line > 0U) {
        cache_line_size_ = sc_line;
        l1d_props_.line_size_bytes = sc_line;
    }
#endif

    struct SysconfLevelQuery {
        std::size_t level;
        int size_opt;
        int assoc_opt;
    };

    constexpr std::array<SysconfLevelQuery, 3> kSysconfLevels = {{
        {1U,
#ifdef _SC_LEVEL1_DCACHE_SIZE
         _SC_LEVEL1_DCACHE_SIZE,
#else
         -1,
#endif
#ifdef _SC_LEVEL1_DCACHE_ASSOC
         _SC_LEVEL1_DCACHE_ASSOC
#else
         -1
#endif
        },
        {2U,
#ifdef _SC_LEVEL2_CACHE_SIZE
         _SC_LEVEL2_CACHE_SIZE,
#else
         -1,
#endif
#ifdef _SC_LEVEL2_CACHE_ASSOC
         _SC_LEVEL2_CACHE_ASSOC
#else
         -1
#endif
        },
        {3U,
#ifdef _SC_LEVEL3_CACHE_SIZE
         _SC_LEVEL3_CACHE_SIZE,
#else
         -1,
#endif
#ifdef _SC_LEVEL3_CACHE_ASSOC
         _SC_LEVEL3_CACHE_ASSOC
#else
         -1
#endif
        },
    }};

    for (const auto& q : kSysconfLevels) {
        const std::size_t sc_size = query_sysconf_value(q.size_opt);
        const std::size_t sc_assoc = query_sysconf_value(q.assoc_opt);
        if (sc_size > 0U) {
            record_cache(q.level, sc_size, 0U, sc_assoc);
        }
    }

#if defined(__linux__)
#if defined(__aarch64__)
#if defined(ALGOAT_HAS_SYS_AUXV) && defined(HWCAP_ASIMD)
    has_neon_ = (::getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0UL;
#else
    has_neon_ = true; // ASIMD is mandatory in AArch64
#endif
#elif defined(__arm__)
#if defined(ALGOAT_HAS_SYS_AUXV) && defined(HWCAP_NEON)
    has_neon_ = (::getauxval(AT_HWCAP) & HWCAP_NEON) != 0UL;
#endif
#endif
#endif // __linux__

#elif defined(_WIN32)
    DWORD buffer_bytes = 0;
    if (!::GetLogicalProcessorInformationEx(RelationCache, nullptr, &buffer_bytes) &&
        ::GetLastError() == ERROR_INSUFFICIENT_BUFFER && buffer_bytes > 0) {
        alignas(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) std::uint8_t stack_buf[4096];
        std::vector<std::uint8_t> heap_buf;
        std::uint8_t* raw_bytes = stack_buf;
        if (buffer_bytes > sizeof(stack_buf)) {
            heap_buf.resize(buffer_bytes);
            raw_bytes = heap_buf.data();
        }
        auto* raw_ptr = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(raw_bytes);
        if (::GetLogicalProcessorInformationEx(RelationCache, raw_ptr, &buffer_bytes)) {
            std::size_t offset = 0;
            while (offset < buffer_bytes) {
                const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                    raw_bytes + offset);
                if (info->Size == 0) {
                    break;
                }
                if (info->Relationship == RelationCache) {
                    const CACHE_RELATIONSHIP& cache = info->Cache;
                    const bool is_data_or_unified =
                        (cache.Type == CacheData || cache.Type == CacheUnified);
                    const bool valid_l1d = (cache.Level != 1 || cache.Type == CacheData);
                    if (is_data_or_unified && valid_l1d) {
                        const std::size_t assoc =
                            (cache.Associativity == CACHE_FULLY_ASSOCIATIVE)
                                ? 0U
                                : static_cast<std::size_t>(cache.Associativity);
                        record_cache(static_cast<std::size_t>(cache.Level),
                                     static_cast<std::size_t>(cache.CacheSize),
                                     static_cast<std::size_t>(cache.LineSize), assoc);
                    }
                }
                offset += info->Size;
            }
        }
    }

#if defined(_M_ARM64)
    has_neon_ = ::IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != 0;
#endif
#endif
}

} // namespace algoat::core
