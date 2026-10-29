include(FetchContent)
FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.0
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

find_package(OpenMP)

add_executable(bench_sorting bench_sorting.cpp)
target_link_libraries(bench_sorting PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_sorting PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(bench_profiler bench_profiler.cpp)
target_link_libraries(bench_profiler PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_eytzinger_search bench_eytzinger_search.cpp)
target_link_libraries(bench_eytzinger_search PRIVATE algoat benchmark::benchmark)

add_executable(bench_config_dispatch bench_config_dispatch.cpp)
target_link_libraries(bench_config_dispatch PRIVATE algoat benchmark::benchmark)

add_executable(bench_branchless_galloping_search bench_branchless_galloping_search.cpp)
target_link_libraries(bench_branchless_galloping_search PRIVATE algoat benchmark::benchmark)

add_executable(bench_selection bench_selection.cpp)
target_link_libraries(bench_selection PRIVATE algoat benchmark::benchmark)

add_executable(bench_strings bench_strings.cpp)
target_link_libraries(bench_strings PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_kdtree bench_kdtree.cpp)
target_link_libraries(bench_kdtree PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_lbvh bench_lbvh.cpp)
target_link_libraries(bench_lbvh PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_lbvh PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(bench_predicates bench_predicates.cpp)
target_link_libraries(bench_predicates PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_delaunay bench_delaunay.cpp)
target_link_libraries(bench_delaunay PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_polygon bench_polygon.cpp)
target_link_libraries(bench_polygon PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_graph bench_graph.cpp)
target_link_libraries(bench_graph PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_graph PRIVATE OpenMP::OpenMP_CXX)
endif()
