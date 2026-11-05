"""
Differential verification of algoat::numerics::BigInt against Python's native arbitrary-precision int.
Verifies all 4 operations (+, -, *, /) across 10,000 random test cases up to 4096 bits.
"""

import sys
import os
import random
import time
import pytest

# Ensure python build directory is in sys.path
sys.path.insert(0, os.path.abspath("build/debug/python"))
sys.path.insert(0, os.path.abspath("build/release/python"))
sys.path.insert(0, os.path.abspath("python"))

try:
    import algoat
    if hasattr(algoat, "BigInt"):
        BigInt = algoat.BigInt
    elif hasattr(algoat, "_algoat_impl") and hasattr(algoat._algoat_impl, "BigInt"):
        BigInt = algoat._algoat_impl.BigInt
    else:
        import _algoat_impl
        BigInt = _algoat_impl.BigInt
except ImportError:
    try:
        import _algoat_impl
        BigInt = _algoat_impl.BigInt
    except ImportError:
        BigInt = None


def c_div(a: int, b: int) -> int:
    """C++20 truncation towards zero integer division."""
    sign = -1 if (a < 0) ^ (b < 0) else 1
    return sign * (abs(a) // abs(b))


def test_differential_10k_cases():
    if BigInt is None:
        pytest.skip("algoat.BigInt is not available in the current environment")
    print("Starting 10,000 differential test cases up to 4096 bits...")
    random.seed(42)

    total_cases = 10000
    start_time = time.time()

    for idx in range(total_cases):
        # Varying bit-widths: small (1-64), medium (65-512), large (513-2048), huge (2049-4096)
        bit_choice = random.choice([32, 64, 128, 256, 512, 1024, 2048, 4096])
        bits_a = random.randint(1, bit_choice)
        bits_b = random.randint(1, bit_choice)

        val_a = random.getrandbits(bits_a)
        val_b = random.getrandbits(bits_b)

        if random.random() < 0.5:
            val_a = -val_a
        if random.random() < 0.5:
            val_b = -val_b

        # Divisor cannot be 0
        if val_b == 0:
            val_b = 1

        big_a = BigInt(str(val_a))
        big_b = BigInt(str(val_b))

        # 1. Addition (+)
        sum_expected = val_a + val_b
        sum_actual = int(str(big_a + big_b))
        assert sum_actual == sum_expected, f"Add failed at case {idx}: {val_a} + {val_b}"

        # 2. Subtraction (-)
        sub_expected = val_a - val_b
        sub_actual = int(str(big_a - big_b))
        assert sub_actual == sub_expected, f"Sub failed at case {idx}: {val_a} - {val_b}"

        # 3. Multiplication (*)
        mul_expected = val_a * val_b
        mul_actual = int(str(big_a * big_b))
        assert mul_actual == mul_expected, f"Mul failed at case {idx}: {val_a} * {val_b}"

        # 4. Division (/)
        div_expected = c_div(val_a, val_b)
        div_actual = int(str(big_a // big_b))
        assert div_actual == div_expected, f"Div failed at case {idx}: {val_a} / {val_b} expected {div_expected} got {div_actual}"

        # 5. Bitwise AND (&), OR (|), XOR (^), Invert (~)
        and_expected = val_a & val_b
        and_actual = int(str(big_a & big_b))
        assert and_actual == and_expected, f"AND failed at case {idx}: {val_a} & {val_b} expected {and_expected} got {and_actual}"

        or_expected = val_a | val_b
        or_actual = int(str(big_a | big_b))
        assert or_actual == or_expected, f"OR failed at case {idx}: {val_a} | {val_b} expected {or_expected} got {or_actual}"

        xor_expected = val_a ^ val_b
        xor_actual = int(str(big_a ^ big_b))
        assert xor_actual == xor_expected, f"XOR failed at case {idx}: {val_a} ^ {val_b} expected {xor_expected} got {xor_actual}"

        inv_expected = ~val_a
        inv_actual = int(str(~big_a))
        assert inv_actual == inv_expected, f"INV failed at case {idx}: ~{val_a} expected {inv_expected} got {inv_actual}"

        # 6. Right shift (>>) by random amount in [0, 128]
        shift = idx % 129
        shift_expected = val_a >> shift
        shift_actual = int(str(big_a >> shift))
        assert shift_actual == shift_expected, f"RSHIFT failed at case {idx}: {val_a} >> {shift} expected {shift_expected} got {shift_actual}"

        if (idx + 1) % 2000 == 0:
            print(f"  Verified {idx + 1}/{total_cases} cases...")

    elapsed = time.time() - start_time
    print(f"SUCCESS: 10,000 cases verified across (+, -, *, /, &, |, ^, ~, >>) in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    test_differential_10k_cases()
