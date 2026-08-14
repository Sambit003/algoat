# Algoat

Algoat is a high-performance C++20 algorithm library featuring smart fallback and dynamic algorithm dispatch based on data profiles and runtime configuration. It is designed to provide zero-overhead algorithmic primitives for both C++ and Python applications.

## Features

- **Sorting Algorithms**: Insertion Sort, Quick Sort, Merge Sort, Heap Sort, Tim Sort, Intro Sort, Block Sort, Radix Sort (LSD/MSD), Counting Sort, Bucket Sort, Shell Sort, Comb Sort, Cycle Sort, and Bitonic Sort.
- **Searching Algorithms**: Linear Search, Binary Search, Interpolation Search.
- **Smart Dispatching**: Automatically analyzes array profiles ($O(n)$) to select the optimal algorithm (e.g., Insertion Sort for small arrays, Merge Sort for nearly sorted sequences).
- **Runtime Configuration**: Dynamic algorithm overrides via JSON configuration (`nlohmann/json`).
- **Python Integration**: Zero-copy Python bindings via `nanobind`, allowing seamless execution over native Python lists and NumPy arrays.

## Domain-Specific Numerics

Algoat implements highly specialized, non-comparative sorting algorithms to bypass the standard $O(N \log N)$ constraints of `std::sort` for specific numeric domains, significantly outperforming standard libraries:

- **Boolean Arrays (`bool`)**: Utilizes a branchless counting pass for $O(N)$ execution. (~23.0x speedup vs `numpy.sort`)
- **Float16 (`numpy.float16`)**: Employs a custom bit-flipping technique to maintain strict weak ordering without floating-point comparisons, coupled with an L2-aligned 16-bit Counting Sort. (~2.2x speedup vs `numpy.sort`)
- **Complex Numbers (`complex64`, `complex128`)**: Replaces lexicographical sorting with spatial ordering. Computes 64-bit Morton Z-order interleaved keys in $O(N)$, followed by a highly tuned 4-pass 16-bit Radix Sort to maximize 2D spatial locality. (~2.0x speedup vs `numpy.sort`)

## Installation

### Python (via pip)

Cross-platform binary wheels are available for Python 3.10+.
```bash
pip install algoat
```

### C++ (via CMake / Source)

Algoat requires a C++20 compliant compiler and CMake 3.21+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install
```

### C++ (via Package Managers)

**Conan 2.x:**
```bash
conan install --requires algoat/0.1.0
```

**vcpkg:**
```bash
vcpkg install algoat
```

## Usage

### C++ Usage

```cpp
#include <algoat/algoat.hpp>
#include <iostream>
#include <vector>
#include <span>

int main() {
    std::vector<int> data = {42, 7, 19, 100, 3, 55, 21};
    
    // Automatically selects the best sorting algorithm
    algoat::sort(std::span{data});
    
    // Automatically selects the best searching algorithm
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

# Sort standard Python lists in-place
data = [42, 7, 19, 100, 3, 55, 21]
algoat.sort_inplace(data) # [3, 7, 19, 21, 42, 55, 100]

# Search in sorted data
index = algoat.search(data, 19)
print(index)  # 2

### Usage Examples Repository

For complete, runnable standalone examples of these APIs (including the NumPy benchmarking scripts), please see the dedicated example directories:
- **[C++ Examples](examples/cpp/README.md)**: Contains `basic_usage.cpp`.
- **[Python Examples](examples/python/README.md)**: Contains `basic_usage.py` and the `benchmark.py` NumPy speedup suite.

## Architecture

Algoat uses a Registry and Dispatcher pattern. Algorithms are registered using a `std::variant` to avoid vtable overhead. The `Dispatcher` dynamically analyzes incoming data using `DataTraits` (size, sortedness ratio) and selects an optimal algorithm if one is not explicitly configured via `AlgoConfig`.
