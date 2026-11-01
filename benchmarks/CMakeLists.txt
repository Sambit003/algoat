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

add_executable(bench_shortest_path bench_shortest_path.cpp)
target_link_libraries(bench_shortest_path PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_shortest_path PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(bench_max_flow bench_max_flow.cpp)
target_link_libraries(bench_max_flow PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_max_flow PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(bench_bipartite_matching bench_bipartite_matching.cpp)
target_link_libraries(bench_bipartite_matching PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_bipartite_matching PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(bench_centrality bench_centrality.cpp)
target_link_libraries(bench_centrality PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_centrality PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(bench_mst bench_mst.cpp)
target_link_libraries(bench_mst PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_mst PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(bench_hyperloglog bench_hyperloglog.cpp)
target_link_libraries(bench_hyperloglog PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)

add_executable(bench_count_min bench_count_min.cpp)
target_link_libraries(bench_count_min PRIVATE algoat benchmark::benchmark benchmark::benchmark_main)
