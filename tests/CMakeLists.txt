include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.16.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(RC_ENABLE_GTEST ON CACHE BOOL "Enable GTest support for RapidCheck" FORCE)
set(RC_ENABLE_GMOCK ON CACHE BOOL "Enable GMock support for RapidCheck" FORCE)
FetchContent_Declare(rapidcheck
    GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(rapidcheck)

include(GoogleTest)

function(algoat_add_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE algoat GTest::gtest_main)
    target_compile_definitions(${TEST_NAME} PRIVATE ALGOAT_TEST_FIXTURES_DIR="${PROJECT_SOURCE_DIR}/tests/core/fixtures")
    gtest_discover_tests(${TEST_NAME} DISCOVERY_MODE PRE_TEST)
endfunction()

# Core tests
algoat_add_test(test_traits core/test_traits.cpp)
algoat_add_test(test_config core/test_config.cpp)
algoat_add_test(test_registry core/test_registry.cpp)
algoat_add_test(test_dispatcher core/test_dispatcher.cpp)
algoat_add_test(test_ranges_api core/test_ranges_api.cpp)

# Sorting tests
algoat_add_test(test_hilbert_sort sorting/test_hilbert_sort.cpp)
algoat_add_test(test_hybrid_sfc_sort sorting/test_hybrid_sfc_sort.cpp)
algoat_add_test(test_comparative_sorts sorting/test_comparative_sorts.cpp)
algoat_add_test(test_insertionsort sorting/test_insertionsort.cpp)
algoat_add_test(test_quicksort sorting/test_quicksort.cpp)
algoat_add_test(test_mergesort sorting/test_mergesort.cpp)
algoat_add_test(test_heapsort sorting/test_heapsort.cpp)
algoat_add_test(test_classical_sorts sorting/test_classical_sorts.cpp)
algoat_add_test(test_linear_sorts sorting/test_linear_sorts.cpp)
algoat_add_test(test_hybrid_sorts sorting/test_hybrid_sorts.cpp)
algoat_add_test(test_sorting_dispatch sorting/test_sorting_dispatch.cpp)
algoat_add_test(test_boolean_sort sorting/test_boolean_sort.cpp)
algoat_add_test(test_inplace_radixsort sorting/test_inplace_radixsort.cpp)

# Searching tests
algoat_add_test(test_linear_search searching/test_linear_search.cpp)
algoat_add_test(test_binary_search searching/test_binary_search.cpp)
algoat_add_test(test_interpolation_search searching/test_interpolation_search.cpp)
algoat_add_test(test_hybrid_interpolation_search searching/test_hybrid_interpolation_search.cpp)
algoat_add_test(test_adaptive_binary_search searching/test_adaptive_binary_search.cpp)
algoat_add_test(test_eytzinger_search_tree searching/test_eytzinger_search_tree.cpp)
algoat_add_test(test_searching_dispatch searching/test_searching_dispatch.cpp)

# SIMD tests
algoat_add_test(test_simd simd/test_simd.cpp)

# Property-Based Tests
add_executable(test_sorting_invariants pbt/test_sorting_invariants.cpp)
add_executable(test_searching_invariants pbt/test_searching_invariants.cpp)
target_link_libraries(test_searching_invariants PRIVATE algoat GTest::gtest_main rapidcheck rapidcheck_gtest)
gtest_discover_tests(test_searching_invariants DISCOVERY_MODE PRE_TEST)
target_link_libraries(test_sorting_invariants PRIVATE algoat GTest::gtest_main rapidcheck rapidcheck_gtest)
gtest_discover_tests(test_sorting_invariants DISCOVERY_MODE PRE_TEST)

# Fuzzing Tests
if(ALGOAT_ENABLE_FUZZING)
    add_executable(diff_search_fuzzer fuzzing/diff_search_fuzzer.cpp)
    target_link_libraries(diff_search_fuzzer PRIVATE algoat)
    target_compile_options(diff_search_fuzzer PRIVATE -fsanitize=address,undefined,fuzzer)
    target_link_options(diff_search_fuzzer PRIVATE -fsanitize=address,undefined,fuzzer)
    add_executable(diff_sort_fuzzer fuzzing/diff_sort_fuzzer.cpp)
    target_link_libraries(diff_sort_fuzzer PRIVATE algoat)
    target_compile_options(diff_sort_fuzzer PRIVATE -fsanitize=address,undefined,fuzzer)
    target_link_options(diff_sort_fuzzer PRIVATE -fsanitize=address,undefined,fuzzer)
endif()
