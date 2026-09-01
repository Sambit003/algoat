#!/usr/bin/env python3
"""Algoat vs Python Native — Performance Benchmark

Compares algoat's C++ backed sort/search against Python builtins
across varying dataset sizes.
"""

import bisect
from contextlib import contextmanager
import functools
import gc
import operator
import random
import statistics
import sys
import time
from typing import Any, Callable

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import algoat


@contextmanager
def disabled_gc():
    """Context manager to collect and disable GC during benchmark measurements."""
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def bench(fn: Callable[[], Any], *, warmup: int = 3, runs: int = 50) -> dict[str, float]:
    """Run zero-argument fn() `runs` times after warmup with isolated GC, returning min/median/MAD latencies in µs."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    with disabled_gc():
        for _ in range(runs):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1_000)  # ns → µs

    min_val = min(times)
    median_val = statistics.median(times)
    mad_val = statistics.median([abs(x - median_val) for x in times])

    return {"min": min_val, "median": median_val, "mad": mad_val}


def bench_sort(
    sort_fn: Callable[[list[Any]], Any],
    data: list[Any],
    *,
    warmup: int = 3,
    runs: int = 50,
) -> dict[str, float]:
    """Run sort_fn on pre-allocated copies of data with isolated GC, returning min/median/MAD latencies in µs."""
    if data:
        for _ in range(warmup):
            sample = data.copy()
            sort_fn(sample)

    copies = [data.copy() for _ in range(runs)]

    times: list[float] = []
    with disabled_gc():
        for copy in copies:
            t0 = time.perf_counter_ns()
            sort_fn(copy)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1_000)  # ns → µs

    min_val = min(times)
    median_val = statistics.median(times)
    mad_val = statistics.median([abs(x - median_val) for x in times])

    return {"min": min_val, "median": median_val, "mad": mad_val}


def generate_data(n, seed=42):
    rng = random.Random(seed)
    return [rng.randint(0, n * 10) for _ in range(n)]


def run_sort_benchmarks(sizes):
    print("=" * 86)
    print("SORTING BENCHMARK — algoat.sort() vs algoat.sort_inplace() vs native")
    print("=" * 86)
    print(f"{'N':>10} │ {'algoat (µs)':>14} │ {'inplace (µs)':>14} │ {'sorted()':>12} │ {'list.sort()':>13} │ {'Ratio':>8}")
    print("─" * 10 + "─┼─" + "─" * 14 + "─┼─" + "─" * 14 + "─┼─" + "─" * 12 + "─┼─" + "─" * 13 + "─┼─" + "─" * 8)

    results = []
    for n in sizes:
        data = generate_data(n)

        stats_algoat = bench_sort(algoat.sort, data)
        stats_inplace = bench_sort(algoat.sort_inplace, data)
        stats_sorted = bench_sort(sorted, data)
        stats_listsort = bench_sort(operator.methodcaller("sort"), data)

        t_algoat = stats_algoat["median"]
        t_inplace = stats_inplace["median"]
        t_sorted = stats_sorted["median"]
        t_listsort = stats_listsort["median"]

        # Compare inplace to list.sort()
        ratio = t_inplace / t_listsort if t_listsort > 0 else float('inf')
        results.append((n, t_algoat, t_inplace, t_sorted, t_listsort, ratio))

        print(f"{n:>10,} │ {t_algoat:>14.2f} │ {t_inplace:>14.2f} │ {t_sorted:>12.2f} │ {t_listsort:>13.2f} │ {ratio:>7.2f}x")

    return results


def run_search_benchmarks(sizes):
    print()
    print("=" * 72)
    print("SEARCHING BENCHMARK — algoat.search() vs bisect vs list.index()")
    print("=" * 72)
    print(f"{'N':>10} │ {'algoat (µs)':>14} │ {'bisect (µs)':>14} │ {'index() (µs)':>14} │ {'Ratio':>8}")
    print("─" * 10 + "─┼─" + "─" * 14 + "─┼─" + "─" * 14 + "─┼─" + "─" * 14 + "─┼─" + "─" * 8)

    results = []
    for n in sizes:
        data = generate_data(n)
        sorted_data = sorted(data)
        # Pick a target we know exists
        target = sorted_data[n // 3]

        # algoat.search on sorted data via functools.partial to eliminate closure overhead
        stats_algoat = bench(functools.partial(algoat.search, sorted_data, target))

        # bisect (binary search) on sorted data
        def bisect_search():
            idx = bisect.bisect_left(sorted_data, target)
            return idx if idx < len(sorted_data) and sorted_data[idx] == target else None

        stats_bisect = bench(bisect_search)

        # list.index (linear scan) via functools.partial to eliminate closure overhead
        stats_index = bench(functools.partial(sorted_data.index, target))

        t_algoat = stats_algoat["median"]
        t_bisect = stats_bisect["median"]
        t_index = stats_index["median"]

        ratio = t_algoat / t_bisect if t_bisect > 0 else float('inf')
        results.append((n, t_algoat, t_bisect, t_index, ratio))

        print(f"{n:>10,} │ {t_algoat:>14.2f} │ {t_bisect:>14.2f} │ {t_index:>14.2f} │ {ratio:>7.2f}x")

    return results


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         Algoat vs Python Native — Performance Benchmark              ║")
    print("║         Median of 50 runs per size (3 warmup iterations)             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    sizes = [100, 1_000, 10_000, 50_000, 100_000]

    sort_results = run_sort_benchmarks(sizes)
    search_results = run_search_benchmarks(sizes)

    print()
    print("=" * 72)
    print("ANALYSIS")
    print("=" * 72)
    print()
    print("Ratio = algoat / native (< 1.0 means algoat is faster)")
    print()
    print("Note: algoat pays a fixed cost for list→vector copy across the")
    print("Python/C++ boundary. Python's sorted()/bisect are implemented in")
    print("highly optimized C (Timsort / bisect module). The overhead of")
    print("nanobind marshalling dominates at small N, but algoat's dispatch")
    print("engine demonstrates competitive throughput at larger scales.")


if __name__ == "__main__":
    main()
