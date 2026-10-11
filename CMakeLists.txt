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

option(ALGOAT_BUILD_MODULES "Build C++20 primary module interface (algoat.cppm)" ${_algoat_modules_supported})
if(ALGOAT_BUILD_MODULES AND NOT _algoat_modules_supported)
    message(FATAL_ERROR "ALGOAT_BUILD_MODULES=ON requires CMake >= 3.28, Ninja or Visual Studio generator, and Clang >= 16, GCC >= 14, or MSVC >= 19.34.")
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
