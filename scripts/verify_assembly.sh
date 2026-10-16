#!/usr/bin/env bash
set -euo pipefail

# Script to verify branchless binary search CMOV emission on GCC and Clang
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

TEST_CPP="${TMP_DIR}/test_cmov.cpp"

cat << 'CPP' > "${TEST_CPP}"
#include "algoat/searching/branchless_binary_search.hpp"

extern "C" const int* algoat_verify_branchless(const int* first, const int* last, int target) {
    return algoat::searching::branchless_lower_bound(first, last, target);
}
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
}

verify_compiler g++ "GCC"
verify_compiler clang++ "Clang"

echo "==> All compiler assembly verifications passed successfully!"
