cmake_minimum_required(VERSION 3.21)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28.0")
    cmake_policy(VERSION 3.28)
endif()

project(algoat VERSION 0.2.0 LANGUAGES CXX C)

# Options
option(ALGOAT_BUILD_TESTS      "Build unit tests"      ON)
option(ALGOAT_BUILD_BENCHMARKS "Build benchmarks"      OFF)
option(ALGOAT_BUILD_EXAMPLES   "Build example programs" ON)
option(ALGOAT_BUILD_PYTHON     "Build Python bindings" OFF)

# C++20 Modules capability detection (CMake >= 3.28, Ninja/VS generator, supporting compiler)
set(_algoat_modules_supported FALSE)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28.0" AND
   (CMAKE_GENERATOR MATCHES "Ninja" OR CMAKE_GENERATOR MATCHES "Visual Studio"))
    if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "16.0") OR
       (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "14.0") OR
       (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.34"))
        set(_algoat_modules_supported TRUE)
    endif()
endif()

option(ALGOAT_BUILD_MODULES "Build C++20 primary module interface (algoat.cppm)" OFF)
if(ALGOAT_BUILD_MODULES AND NOT _algoat_modules_supported)
    message(FATAL_ERROR "ALGOAT_BUILD_MODULES=ON requires CMake >= 3.28, Ninja or Visual Studio generator, and Clang >= 16, GCC >= 14, or MSVC >= 19.34.")
endif()

# Hardware FMA support detection
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-ffp-contract=off" ALGOAT_SUPPORTS_FP_CONTRACT_OFF)
if(ALGOAT_SUPPORTS_FP_CONTRACT_OFF)
    add_compile_options("-ffp-contract=off")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    check_cxx_compiler_flag("-mfma" ALGOAT_SUPPORTS_MFMA)
    if(ALGOAT_SUPPORTS_MFMA)
        include(CheckCXXSourceRuns)
        set(CMAKE_REQUIRED_FLAGS "-mfma")
        check_cxx_source_runs("
            #include <immintrin.h>
            int main() {
                __m128d a = _mm_set1_pd(1.0);
                __m128d b = _mm_set1_pd(2.0);
                __m128d c = _mm_set1_pd(3.0);
                __m128d r = _mm_fmadd_sd(a, b, c);
                return ((int)_mm_cvtsd_f64(r) == 5) ? 0 : 1;
            }
        " ALGOAT_HOST_HAS_FMA)
        unset(CMAKE_REQUIRED_FLAGS)
        if(ALGOAT_HOST_HAS_FMA)
            add_compile_options("-mfma")
        endif()
    endif()
endif()

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
