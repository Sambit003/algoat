# Contributing to Algoat

Thank you for your interest in contributing to **Algoat**! 🎉

Algoat is a high-performance C++20 and Python algorithm library featuring dynamic heuristic dispatch, static polymorphism, domain-specific hardware acceleration, and zero-copy Python bindings. We welcome contributions ranging from new sorting algorithms and hardware optimizations to documentation improvements, performance benchmarks, and bug fixes.

This document outlines the development workflow, coding standards, and step-by-step recipes for contributing to the project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Git Workflow & Guidelines](#git-workflow--guidelines)
  - [Forking & Cloning](#forking--cloning)
  - [Branch Naming Conventions](#branch-naming-conventions)
  - [Commit Message Guidelines](#commit-message-guidelines)
  - [Pull Request Title Format](#pull-request-title-format)
  - [Pull Request Checklist](#pull-request-checklist)
- [Development Environment Setup](#development-environment-setup)
  - [Prerequisites](#prerequisites)
  - [Python Virtual Environment](#python-virtual-environment)
- [Building and Testing](#building-and-testing)
  - [C++ Workflows (CMake & Presets)](#c-workflows-cmake--presets)
  - [Memory Sanitizers (ASan / UBSan)](#memory-sanitizers-asan--ubsan)
  - [Python nanobind Workflows](#python-nanobind-workflows)
  - [Running Benchmarks](#running-benchmarks)
- [Code Style & Formatting](#code-style--formatting)
  - [C++ Formatting (`clang-format`)](#c-formatting-clang-format)
  - [Static Analysis (`clang-tidy`)](#static-analysis-clang-tidy)
  - [Python Style](#python-style)
- [Step-by-Step Recipes](#step-by-step-recipes)
  - [Adding a New Sorting Algorithm](#adding-a-new-sorting-algorithm)
  - [Extending Python nanobind Bindings](#extending-python-nanobind-bindings)
- [Questions and Support](#questions-and-support)

---

## Code of Conduct

This project and everyone participating in it is governed by the [Algoat Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report any unacceptable behavior to [smbtchakraborty@gmail.com](mailto:smbtchakraborty@gmail.com).

---

## Git Workflow & Guidelines

### Forking & Cloning

1. **Fork** the repository on GitHub by clicking the **Fork** button at [Sambit003/algoat](https://github.com/Sambit003/algoat).
2. **Clone** your personal fork locally:
   ```bash
   git clone https://github.com/<your-username>/algoat.git
   cd algoat
   ```
3. **Add the upstream remote** to stay synchronized with the main repository:
   ```bash
   git remote add upstream https://github.com/Sambit003/algoat.git
   git fetch upstream
   ```
4. **Keep your `main` branch up to date**:
   ```bash
   git checkout main
   git pull upstream main
   ```

---

### Branch Naming Conventions

Always create a dedicated feature or fix branch from the latest `main`. Branch names **must** use one of the approved prefixes:

| Prefix | Use Case | Example |
| :--- | :--- | :--- |
| `feat/` | New features or sorting capabilities | `feat/ciura-shell-sort` |
| `fix/` | Bug fixes | `fix/nanobind-gil-release` |
| `refactor/` | Code refactoring without changing behavior | `refactor/dispatcher-heuristics` |
| `perf/` | Performance optimizations | `perf/float16-bitflip-vectorize` |
| `docs/` | Documentation additions or corrections | `docs/update-contributing-guide` |
| `chore/` | Tooling, dependencies, or repository maintenance | `chore/update-googletest-tag` |
| `patch/` | Minor corrections and small fixes | `patch/cmake-install-target` |
| `hotfix/` | Critical urgent fixes | `hotfix/segfault-on-empty-span` |
| `vuln/` | Security vulnerability mitigations | `vuln/buffer-overflow-sanitizer` |
| `enhance/` | Incremental enhancements to existing features | `enhance/morton-bmi2-fallback` |

Create your branch with:
```bash
git checkout -b feat/my-new-feature
```

---

### Commit Message Guidelines

All commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```text
<type>(<optional scope>): <description>

[optional body]

[optional footer(s)]
```

Common types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`.

**Examples**:
- `feat(sorting): add block sort algorithm with O(1) auxiliary space`
- `fix(core): guard against zero-length span in data traits analyzer`
- `perf(numerics): accelerate complex Morton key generation with BMI2`
- `docs(readme): add benchmark reproduction steps`

---

### Pull Request Title Format

Pull Request titles must follow the structured format:

```text
<Type>(<scope>): <Precise PR deliverable>
```

**Examples**:
- `Feat(sorting): Add Ciura ShellSort algorithm`
- `Fix(python): Resolve GIL race condition in PyGenericWrapper`
- `Perf(numerics): Vectorize Float16 bit-flip counting sort`
- `Docs(contributing): Add step-by-step algorithm recipe`
- `Refactor(core): Simplify Registry std::variant instantiation`

---

### Pull Request Checklist

Before submitting your PR, ensure that:

- [ ] All C++ unit tests pass (`ctest --preset debug` or `ctest --test-dir build --output-on-failure`).
- [ ] All Python tests pass (`pytest tests/python/ -v`).
- [ ] Memory safety sanitizers report no errors (`cmake --preset asan`).
- [ ] Code is formatted with `clang-format`.
- [ ] New features include corresponding unit tests in `tests/`.
- [ ] Performance-critical changes include benchmark numbers before and after the change.
- [ ] Documentation and comments are updated where appropriate.

---

### Issue & Pull Request Labeling

Algoat uses an automated and standardized labeling taxonomy to streamline triage and reviews:

- **PR Sizing (`size:*`)**: Automatically computed by CI based on net lines changed:
  - `size: XS` (1–9 lines) | `size: S` (10–49 lines) | `size: M` (50–249 lines)
  - `size: L` (250–499 lines) | `size: XL` (500–999 lines) | `size: XXL` (1000+ lines, split recommended)
- **Review Effort (`review:*`)**: `review: quick`, `review: standard`, `review: deep`, `review: security-critical`.
- **Area (`area:*`)**: Scoped to the subsystem (e.g. `area: sorting`, `area: numerics`, `area: python`, `area: core`).
- **Issue Priority (`priority:*`)**: `priority: critical (P0)` to `priority: low (P3)`.
- **Difficulty (`difficulty:*`)**: `difficulty: easy`, `difficulty: medium`, `difficulty: hard` for onboarding contributors.

---

## Development Environment Setup

### Prerequisites

To build and develop Algoat, install the following tools:

- **C++ Compiler**: A modern C++20 compliant compiler:
  - GCC 12+ (`g++`)
  - Clang 14+ (`clang++`)
  - MSVC 2022+ (Windows)
- **Build System**:
  - `CMake 3.21+`
  - `Ninja` (recommended for fast incremental builds)
- **Formatters & Linters**:
  - `clang-format` (version 14+)
  - `clang-tidy`
- **Python**:
  - `Python 3.10+` with headers (`python3-dev`)

---

### Python Virtual Environment

Set up an isolated virtual environment for Python bindings and test dependencies:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Upgrade pip and install development dependencies
pip install --upgrade pip
pip install numpy pytest scikit-build-core nanobind pre-commit
```

---

### Pre-Commit Hooks

Algoat uses [`pre-commit`](https://pre-commit.com/) to automatically run `clang-format`, file hygiene checks (trailing whitespace, YAML/JSON validation), and Conventional Commit message checks before changes are committed:

```bash
# Install git hook scripts
pre-commit install
pre-commit install --hook-type commit-msg

# (Optional) Run against all files manually
pre-commit run --all-files
```

---

## Building and Testing

### C++ Workflows (CMake & Presets)

Algoat provides first-class support for CMake Presets configured in `CMakePresets.json`:

#### 1. Debug Build (Tests Enabled)
```bash
# Configure, build, and test using the 'debug' preset
cmake --preset debug
cmake --build --preset debug -j$(nproc)
ctest --preset debug
```

#### 2. Release Build (Benchmarks Enabled)
```bash
cmake --preset release
cmake --build --preset release -j$(nproc)
```

#### 3. Manual CMake Configuration
If you prefer configuring CMake manually:
```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DALGOAT_BUILD_TESTS=ON -DALGOAT_BUILD_EXAMPLES=ON

# Build
cmake --build build -j$(nproc)

# Run GoogleTest suite
ctest --test-dir build --output-on-failure
```

---

### Memory Sanitizers (ASan / UBSan)

We strongly encourage running the AddressSanitizer and UndefinedBehaviorSanitizer to verify memory safety, out-of-bounds safety, and absence of undefined behavior:

```bash
# Configure and build with AddressSanitizer + UBSan
cmake --preset asan
cmake --build --preset asan -j$(nproc)

# Run tests under sanitizer
ctest --test-dir build/asan --output-on-failure
```

---

### Python nanobind Workflows

Algoat uses [nanobind](https://github.com/wjakob/nanobind) and [scikit-build-core](https://github.com/scikit-build/scikit-build-core) to provide zero-copy Python bindings for native lists and contiguous NumPy `ndarray` views.

#### 1. Editable Local Installation
Compile the C++ extension in-place into your active virtual environment:

```bash
CXX=g++ CC=gcc pip install --no-build-isolation -e .
```

> [!TIP]
> Whenever you modify C++ source files under `src/`, `include/`, or `python/`, re-run the command above to recompile the native extension.

#### 2. Running Python Tests
Execute the pytest suite:
```bash
pytest tests/python/ -v
```

---

### Running Benchmarks

#### C++ Sorting Benchmarks
Build with the `release` preset and run the benchmark binary:
```bash
cmake --preset release
cmake --build --preset release -j$(nproc)
./build/release/benchmarks/bench_sorting
```

#### Python / NumPy Speedup Benchmarks
Run the Python benchmark suite comparing Algoat against `numpy.sort`:
```bash
python benchmarks/bench_python.py
```

---

## Code Style & Formatting

### C++ Formatting (`clang-format`)

All C++ headers and source files must adhere to the repository `.clang-format` configuration (LLVM-based, C++20, 4-space indentation, 100 column limit, left pointer alignment).

Check and apply formatting across all files:
```bash
clang-format -i include/algoat/**/*.hpp src/**/*.cpp tests/**/*.cpp benchmarks/**/*.cpp examples/**/*.cpp
```

To verify without modifying files:
```bash
clang-format --dry-run --Werror include/algoat/**/*.hpp src/**/*.cpp tests/**/*.cpp
```

### Static Analysis (`clang-tidy`)

Run `clang-tidy` to check for modernization, bugprone patterns, and C++ Core Guidelines:
```bash
clang-tidy -p build/debug src/**/*.cpp
```

### Python Style

- Follow PEP 8 conventions.
- Use explicit type hints for public helper functions.
- Keep nanobind wrappers lean, delegating computation to C++ core algorithms.

---

## Step-by-Step Recipes

### Adding a New Sorting Algorithm

To contribute a new sorting algorithm to Algoat, follow these 5 steps:

#### Step 1: Implement the Algorithm Header
Create a new header file `include/algoat/sorting/my_new_sort.hpp`. Your struct must satisfy the `algoat::sorting::SortAlgorithm` concept:

```cpp
#pragma once

#include <concepts>
#include <span>
#include <string_view>

namespace algoat::sorting {

struct MyNewSort {
    [[nodiscard]] constexpr std::string_view name() const noexcept {
        return "mynewsort";
    }

    [[nodiscard]] constexpr std::size_t preferred_min_size() const noexcept {
        return 0; // Minimum dataset size where this algorithm is effective
    }

    template<std::totally_ordered T>
    void sort(std::span<T> data) const {
        if (data.size() <= 1) {
            return;
        }
        // Implement algorithm logic here...
    }
};

} // namespace algoat::sorting
```

#### Step 2: Register in `SortVariant`
In [`include/algoat/sorting/sorting.hpp`](include/algoat/sorting/sorting.hpp):
1. Add the include:
   ```cpp
   #include "algoat/sorting/my_new_sort.hpp"
   ```
2. Add `MyNewSort` to the `SortVariant` type alias:
   ```cpp
   using SortVariant = std::variant<
       IntroSort,
       TimSort,
       // ... existing algorithms ...
       MyNewSort
   >;
   ```

#### Step 3: Register the Algorithm
At the bottom of your header file (outside the namespace), register your algorithm with the decentralized factory:

```cpp
// include/algoat/sorting/my_new_sort.hpp
// ... algorithm implementation ...
} // namespace algoat::sorting

ALGOAT_REGISTER_ALGORITHM("sorting", "mynewsort", ::algoat::sorting::MyNewSort)
```

#### Step 4: Add Unit Tests
Create [`tests/sorting/test_my_new_sort.cpp`](tests/sorting/):

```cpp
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "algoat/sorting/my_new_sort.hpp"

TEST(MyNewSortTest, HandlesRandomVector) {
    std::vector<int> data = {9, 3, 1, 5, 2, 8, 4, 7, 6};
    algoat::sorting::MyNewSort{}.sort(std::span{data});
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(MyNewSortTest, HandlesEmptyAndSingleElement) {
    std::vector<int> empty;
    algoat::sorting::MyNewSort{}.sort(std::span{empty});
    EXPECT_TRUE(empty.empty());

    std::vector<int> single = {42};
    algoat::sorting::MyNewSort{}.sort(std::span{single});
    EXPECT_EQ(single.front(), 42);
}
```

Register the test executable in [`tests/CMakeLists.txt`](tests/CMakeLists.txt):
```cmake
algoat_add_test(test_my_new_sort sorting/test_my_new_sort.cpp)
```

#### Step 5: (Optional) Add to Benchmarks
Add your sorting algorithm to the benchmark suite in [`benchmarks/bench_sorting.cpp`](benchmarks/bench_sorting.cpp) to evaluate performance characteristics against standard benchmarks.

---

### Extending Python nanobind Bindings

When exposing new algorithms or data types to Python:

1. **Update nanobind bindings**: Add module definitions and function wrappers in [`python/bindings.cpp`](python/bindings.cpp).
2. **Handle GIL Release**: Ensure CPU-heavy C++ sort operations release the Global Interpreter Lock using `nb::gil_scoped_release`.
3. **Add pytest tests**: Add corresponding unit tests in [`tests/python/test_algoat.py`](tests/python/test_algoat.py) verifying both standard Python lists and NumPy arrays.

---

## Questions and Support

If you have questions, encounter an issue, or want to discuss a new idea before writing code, please open an issue on GitHub at [Sambit003/algoat/issues](https://github.com/Sambit003/algoat/issues).

Thank you for helping make Algoat faster and better! 🚀
