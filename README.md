# Algoat

Algoat is a high-performance C++20 and Python algorithm library featuring smart fallback and dynamic algorithm dispatch based on dataset traits and runtime configuration. It is designed to provide zero-overhead algorithmic primitives for high-performance computing, data processing, and scientific applications.

---

## Features

- **Comprehensive Sorting Suite (20 Algorithms)**:
  - **Hybrid & High-Performance**: Intro Sort, Tim Sort, Block Sort.
  - **Divide & Conquer / Comparison**: Quick Sort, Merge Sort, Heap Sort, Insertion Sort, Shell Sort (Ciura gap sequence), Comb Sort, Cycle Sort, Bitonic Sort, Bubble Sort, Selection Sort, Gnome Sort.
  - **Distribution & Non-Comparative**: Radix Sort (LSD / MSD), Counting Sort, Bucket Sort, Pigeonhole Sort, Boolean Sort.
- **Searching Suite**: Binary Search, Interpolation Search (with arithmetic probe and fallback), Linear Search.
- **Smart Dynamic Dispatch**: Automatically analyzes data profiles in a single $O(N)$ pass to select the optimal algorithm (e.g., `InsertionSort` for $N < 32$, `TimSort` for partially ordered data, `RadixSortLSD` for large integers, `IntroSort` as default).
- **Runtime JSON Configuration**: Dynamically override algorithms or fallback behavior via JSON configuration (`nlohmann/json`).
- **Python nanobind Integration**: Zero-copy Python bindings with support for native Python lists and direct contiguous NumPy `ndarray` views (`float16`, `float32`, `float64`, `int32`, `int64`, `uint32`, `uint64`, `complex64`, `complex128`, `bool_`).

---

## Domain-Specific Numerics & Hardware Acceleration

Algoat implements specialized, non-comparative sorting backends to bypass the standard $O(N \log N)$ comparison constraints for specific numeric domains, significantly outperforming standard libraries:

- **Boolean Arrays (`bool`)**: Utilizes a branchless counting pass and hardware memory sets for $O(N)$ execution (**~23.9x speedup** vs `numpy.sort`).
- **Float16 (`numpy.float16`)**: Employs an order-preserving bit-flipping technique to maintain strict weak ordering without floating-point comparisons, coupled with a cache-aligned 16-bit Counting Sort (**~2.1x speedup** vs `numpy.sort`).
- **Complex Numbers (`complex64`, `complex128`)**: Replaces 1D lexicographical sorting with 2D spatial locality. Computes 64-bit Morton Z-order interleaved keys in $O(N)$ using hardware BMI2 `_pdep_u64` instructions, followed by a tuned 4-pass 16-bit Radix Sort (**~2.1x speedup** vs `numpy.sort`).

---

## Installation

### Python (via pip)

Cross-platform binary wheels are available for Python 3.10+:
```bash
pip install algoat
```

For editable local development:
```bash
CXX=g++ CC=gcc pip install --no-build-isolation -e .
```

### C++ (via CMake / Source)

Algoat requires a C++20 compliant compiler (`g++-12+`, `clang-14+`, or MSVC 2022+) and CMake 3.21+:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cmake --install build
```

### Windows

For Windows users with Visual Studio 2022 or another C++20-compatible compiler:

```powershell
cmake -S . -B build
cmake --build build --config Release
cmake --install build --config Release
```

For development, testing, and contribution instructions, see [CONTRIBUTING.md](CONTRIBUTING.md).

### C++ (via Package Managers)

> [!NOTE]
> **Coming Soon**: Official packages on Conan Center and vcpkg registries are currently in preparation for the upcoming major release.

**Conan 2.x:**
```bash
conan install --requires algoat/0.1.1
```

**vcpkg:**
```bash
vcpkg install algoat
```

---

## Usage

### C++ Usage

```cpp
#include <algoat/algoat.hpp>
#include <iostream>
#include <vector>
#include <span>

int main() {
    std::vector<int> data = {42, 7, 19, 100, 3, 55, 21};

    // Automatically selects the best sorting algorithm based on data traits
    algoat::sort(std::span{data});

    // Automatically selects the best searching algorithm (BinarySearch for sorted data)
    auto res = algoat::search(std::span{data}, 19);
    if (res) {
        std::cout << "Found 19 at index " << *res << "\n";
    }

    return 0;
}
```

### Python Usage

```python
import algoat
import numpy as np

# Smart dispatch — automatically routes to the O(N) Float16 hybrid sort
arr = np.random.rand(1_000_000).astype(np.float16)
sorted_arr = algoat.sort(arr)

# Sort standard Python lists out-of-place or in-place
data = [42, 7, 19, 100, 3, 55, 21]
algoat.sort_inplace(data) # [3, 7, 19, 21, 42, 55, 100]

# Search in sorted data
index = algoat.search(data, 19)
print(index)  # 2
```

---

## Examples & Benchmarks

For complete, runnable standalone examples and benchmark suites, see:
- **[C++ Examples](examples/cpp/README.md)**: Contains `basic_usage.cpp` and `physics_simulation_2d.cpp`.
- **[Python Examples](examples/python/README.md)**: Contains `basic_usage.py`, `iot_sensor_analysis.py`, and the `benchmark.py` NumPy speedup suite.

---

## Architecture & Contributing

Algoat uses a **Registry and Dispatcher** pattern. Algorithms are registered using a `std::variant` to eliminate virtual table (`vtable`) overhead. The `Dispatcher` dynamically analyzes incoming data using `DataTraits` (size, sortedness ratio, duplicates) and selects an optimal algorithm if one is not explicitly configured via `AlgoConfig`.

- For an in-depth guide on the internal architecture, static dispatch mechanics, and domain numerics, see [ARCHITECTURE.md](ARCHITECTURE.md).
- For local environment setup, build and test workflows, code style guidelines, and contribution recipes, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Algoat is released under the [GNU Lesser General Public License v3.0 (LGPL-3.0)](LICENSE).
