#!/usr/bin/env bash
set -euo pipefail

# Script to verify branchless binary search CMOV emission and zero-spill in-register
# SIMDSortingNetwork codegen on GCC and Clang.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

TEST_CPP="${TMP_DIR}/test_cmov.cpp"
SIMD_CPP="${TMP_DIR}/test_simd_net.cpp"

cat << 'CPP' > "${TEST_CPP}"
#include "algoat/searching/branchless_binary_search.hpp"

extern "C" const int* algoat_verify_branchless(const int* first, const int* last, int target) {
    return algoat::searching::branchless_lower_bound(first, last, target);
}
CPP

cat << 'CPP' > "${SIMD_CPP}"
#include "algoat/meta/simd_sorting_network.hpp"
#include <cstdint>
#include <span>

#define DEFINE_SORT_FN(T, TAG, N)                                                                  \
    extern "C" __attribute__((noinline)) void algoat_verify_simd_sort_##TAG##_##N(T* p) noexcept { \
        algoat::meta::SIMDSortingNetwork<N, T>::sort(std::span<T, N>{p, N});                       \
    }

#define DEFINE_ALL_SIZES(T, TAG) \
    DEFINE_SORT_FN(T, TAG, 8)    \
    DEFINE_SORT_FN(T, TAG, 16)   \
    DEFINE_SORT_FN(T, TAG, 32)   \
    DEFINE_SORT_FN(T, TAG, 64)

DEFINE_ALL_SIZES(std::int32_t, i32)
DEFINE_ALL_SIZES(std::uint32_t, u32)
DEFINE_ALL_SIZES(std::int64_t, i64)
DEFINE_ALL_SIZES(std::uint64_t, u64)
DEFINE_ALL_SIZES(float, f32)
DEFINE_ALL_SIZES(double, f64)
CPP

verify_compiler() {
    local CXX="$1"
    local NAME="$2"

    if ! command -v "${CXX}" >/dev/null 2>&1; then
        echo "Compiler ${CXX} not found; skipping ${NAME} verification."
        return 0
    fi

    echo "==> Verifying assembly with ${NAME} (${CXX})..."
    local OBJ="${TMP_DIR}/${NAME}.o"
    "${CXX}" -O3 -std=c++20 -I"${REPO_ROOT}/include" -c "${TEST_CPP}" -o "${OBJ}"

    local DISASM
    DISASM="$(objdump -d -M intel "${OBJ}")"

    # Verify that cmov instruction exists in the disassembled function
    if ! echo "${DISASM}" | grep -E "cmov(ne|nz|l|b|a|e|g)" >/dev/null; then
        echo "ERROR: No CMOV instructions detected in ${NAME} disassembly!"
        echo "${DISASM}"
        exit 1
    fi

    echo "SUCCESS: ${NAME} emitted conditional move (CMOV) instructions."

    echo "==> Verifying zero-spill SIMDSortingNetwork assembly with ${NAME} (${CXX})..."
    local SIMD_OBJ="${TMP_DIR}/${NAME}_simd.o"
    "${CXX}" -O3 -std=c++20 -mavx2 -mavx512f -mavx512vl -mavx512dq -mavx512bw -mfma \
        -I"${REPO_ROOT}/include" -c "${SIMD_CPP}" -o "${SIMD_OBJ}"

    local SIMD_DISASM
    SIMD_DISASM="$(objdump -dr -M intel "${SIMD_OBJ}")"

    python3 - "${NAME}" <<PY
import re, sys

name = sys.argv[1]
disasm = """${SIMD_DISASM}"""

all_funcs = {}
func_relocs = {}
cur = None
for line in disasm.splitlines():
    m = re.match(r"^[0-9a-f]+ <([^>]+)>:", line)
    if m:
        cur = m.group(1)
        all_funcs[cur] = []
        func_relocs[cur] = []
        continue
    if cur is not None:
        if not line.strip():
            cur = None
            continue
        if "R_X86_64_" in line:
            rm = re.search(r"R_X86_64_\S+\s+([^\s+-]+)", line)
            if rm:
                func_relocs[cur].append(rm.group(1))
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            insn = parts[2].split("#")[0].strip()
            if insn and "nop" not in insn.split():
                all_funcs[cur].append(insn)

def resolve_target(fname):
    insns = all_funcs[fname]
    if len(insns) <= 2 and insns[-1].startswith("jmp"):
        for target in func_relocs.get(fname, []):
            if target in all_funcs:
                return all_funcs[target]
    return insns

wrappers = [k for k in sorted(all_funcs) if k.startswith("algoat_verify_simd_sort_")]
assert len(wrappers) == 24, f"Expected 24 functions for {name}, got {len(wrappers)}"

compute_prefixes = ("vpmin", "vpmax", "vminp", "vmaxp", "vperm", "vshuf", "vpblend", "vblend", "valign")
failures = []
for wname in wrappers:
    insns = resolve_target(wname)

    # 1. No stack frame allocations or stack spills anywhere
    stack_refs = [i for i in insns if re.search(r"\b(rsp|rbp)\b", i)]
    if stack_refs:
        failures.append((wname, f"stack refs: {stack_refs[:3]}"))
        continue

    # 2. Must contain SIMD compute instructions and no unexpected calls
    compute_idx = [idx for idx, i in enumerate(insns) if i.startswith(compute_prefixes)]
    if not compute_idx:
        failures.append((wname, "no SIMD compute instructions emitted"))
        continue
    calls = [i for i in insns if i.startswith(("call", "jmp")) and "endbr" not in i]
    if calls:
        failures.append((wname, f"unexpected control flow: {calls[:3]}"))
        continue

    first_c, last_c = compute_idx[0], compute_idx[-1]

    # 3. Classify every non-RIP memory access ([...]) as load or store:
    #    - Every non-RIP access must target [rdi...] (never stack or scratch memory).
    #    - Exact vector count M loaded once and stored once (1 load + 1 store per SIMD lane vector).
    #    - Every load occurs strictly before every store (no read-after-write spill to [rdi]).
    tag, n_str = wname.removeprefix("algoat_verify_simd_sort_").split("_")
    n = int(n_str)
    elem_bytes = 4 if tag in ("i32", "u32", "f32") else 8
    total_bytes = n * elem_bytes
    expected_vecs = 1 if total_bytes <= 32 else (total_bytes + 63) // 64

    loads = []
    stores = []
    for idx, insn in enumerate(insns):
        if "[" not in insn or "[rip" in insn:
            continue
        ops_part = insn.split(None, 1)[1] if " " in insn else ""
        first_op = ops_part.split(",")[0]
        if "[" in first_op:
            if "[rdi" not in first_op:
                failures.append((wname, f"non-input store at #{idx}: {insn}"))
                break
            stores.append((idx, insn))
        else:
            if "[rdi" not in ops_part:
                failures.append((wname, f"non-input load at #{idx}: {insn}"))
                break
            loads.append((idx, insn))
    else:
        if len(loads) != expected_vecs or len(stores) != expected_vecs:
            failures.append(
                (wname, f"expected {expected_vecs} loads/stores, got {len(loads)} loads, {len(stores)} stores")
            )
        elif max(idx for idx, _ in loads) > min(idx for idx, _ in stores):
            failures.append((wname, "memory load occurred after memory store"))

if failures:
    for f, reason in failures:
        print(f"ERROR [{name}] {f}: {reason}", file=sys.stderr)
    sys.exit(1)

print(f"SUCCESS: {name} compiled all 24 SIMDSortingNetwork kernels with 0 stack spills and 0 mid-network reloads.")
PY
}

verify_compiler g++ "GCC"
verify_compiler clang++ "Clang"

echo "==> All compiler assembly verifications passed successfully!"
