from contextlib import contextmanager
import gc
import statistics
import time
from typing import Any, Callable

import algoat
import numpy as np


@contextmanager
def disabled_gc():
    """Context manager to collect and disable GC during benchmark measurements."""
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def bench_sort(
    sort_fn: Callable[[np.ndarray], Any],
    data: np.ndarray,
    number: int = 10,
    warmup: int = 3,
) -> dict[str, float]:
    """Run sort_fn on pre-allocated copies of data with isolated GC."""
    # Warmup phase
    for _ in range(warmup):
        sort_fn(data.copy())

    # Pre-allocate copies for measurement phase
    copies = [data.copy() for _ in range(number)]

    times_sec: list[float] = []
    with disabled_gc():
        for copy in copies:
            t0 = time.perf_counter_ns()
            sort_fn(copy)
            t1 = time.perf_counter_ns()
            times_sec.append((t1 - t0) / 1e9)

    min_val = min(times_sec)
    median_val = statistics.median(times_sec)
    mad_val = statistics.median([abs(x - median_val) for x in times_sec])

    return {"min": min_val, "median": median_val, "mad": mad_val}


def benchmark_sorting():
    sizes = [10_000, 100_000, 1_000_000]

    for size in sizes:
        print(f"\n--- Array Size: {size} ---")

        # 1. Float16 sorting
        arr_f16 = np.random.rand(size).astype(np.float16)
        # require 'A' so it's memory aligned, just in case
        arr_f16_aligned = np.require(arr_f16, requirements=['A'])

        stats_np_f16 = bench_sort(np.sort, arr_f16_aligned, number=10)
        stats_algoat_f16 = bench_sort(algoat.sort, arr_f16_aligned, number=10)

        print(
            f"Float16 np.sort:     min={stats_np_f16['min']:.6f}s, "
            f"median={stats_np_f16['median']:.6f}s, mad={stats_np_f16['mad']:.6f}s"
        )
        print(
            f"Float16 algoat.sort: min={stats_algoat_f16['min']:.6f}s, "
            f"median={stats_algoat_f16['median']:.6f}s, mad={stats_algoat_f16['mad']:.6f}s"
        )
        print(f"Speedup (median):    {stats_np_f16['median'] / stats_algoat_f16['median']:.2f}x")

        # 2. Complex64 sorting (Morton Z-order)
        # numpy doesn't naturally sort complex like we do (we use Morton), but we can still benchmark speed.
        # np.sort on complex arrays sorts by real part, then imaginary part.
        arr_c64 = (np.random.rand(size) + 1j * np.random.rand(size)).astype(np.complex64)

        stats_np_c64 = bench_sort(np.sort, arr_c64, number=10)
        stats_algoat_c64 = bench_sort(algoat.sort, arr_c64, number=10)

        print(
            f"Complex64 np.sort:     min={stats_np_c64['min']:.6f}s, "
            f"median={stats_np_c64['median']:.6f}s, mad={stats_np_c64['mad']:.6f}s"
        )
        print(
            f"Complex64 algoat.sort: min={stats_algoat_c64['min']:.6f}s, "
            f"median={stats_algoat_c64['median']:.6f}s, mad={stats_algoat_c64['mad']:.6f}s"
        )
        print(f"Speedup (median):      {stats_np_c64['median'] / stats_algoat_c64['median']:.2f}x")

        # 3. Boolean sorting
        arr_bool = np.random.choice([False, True], size=size)

        stats_np_bool = bench_sort(np.sort, arr_bool, number=10)
        stats_algoat_bool = bench_sort(algoat.sort, arr_bool, number=10)

        print(
            f"Bool np.sort:     min={stats_np_bool['min']:.6f}s, "
            f"median={stats_np_bool['median']:.6f}s, mad={stats_np_bool['mad']:.6f}s"
        )
        print(
            f"Bool algoat.sort: min={stats_algoat_bool['min']:.6f}s, "
            f"median={stats_algoat_bool['median']:.6f}s, mad={stats_algoat_bool['mad']:.6f}s"
        )
        print(f"Speedup (median): {stats_np_bool['median'] / stats_algoat_bool['median']:.2f}x")


if __name__ == '__main__':
    benchmark_sorting()
