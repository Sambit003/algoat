include(FetchContent)
FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.0
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(bench_sorting bench_sorting.cpp)
target_link_libraries(bench_sorting PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_profiler bench_profiler.cpp)
target_link_libraries(bench_profiler PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)
