cmake_minimum_required(VERSION 3.21)
project(algoat VERSION 0.2.0 LANGUAGES CXX C)

# Options
option(ALGOAT_BUILD_TESTS      "Build unit tests"      ON)
option(ALGOAT_BUILD_BENCHMARKS "Build benchmarks"      OFF)
option(ALGOAT_BUILD_EXAMPLES   "Build example programs" ON)
option(ALGOAT_BUILD_PYTHON     "Build Python bindings" OFF)

# Core library
add_subdirectory(src)

# Optional targets
if(ALGOAT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ALGOAT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(ALGOAT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ALGOAT_BUILD_PYTHON)
    add_subdirectory(python)
endif()
