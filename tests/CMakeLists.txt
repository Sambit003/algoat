include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.16.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(algoat_add_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE algoat GTest::gtest_main)
    target_compile_definitions(${TEST_NAME} PRIVATE ALGOAT_TEST_FIXTURES_DIR="${PROJECT_SOURCE_DIR}/tests/core/fixtures")
    gtest_discover_tests(${TEST_NAME})
endfunction()

# Core tests
algoat_add_test(test_traits core/test_traits.cpp)
algoat_add_test(test_config core/test_config.cpp)
algoat_add_test(test_registry core/test_registry.cpp)
algoat_add_test(test_dispatcher core/test_dispatcher.cpp)

# Sorting tests
algoat_add_test(test_comparative_sorts sorting/test_comparative_sorts.cpp)
algoat_add_test(test_insertionsort sorting/test_insertionsort.cpp)
algoat_add_test(test_quicksort sorting/test_quicksort.cpp)
algoat_add_test(test_mergesort sorting/test_mergesort.cpp)
algoat_add_test(test_heapsort sorting/test_heapsort.cpp)
algoat_add_test(test_classical_sorts sorting/test_classical_sorts.cpp)
algoat_add_test(test_linear_sorts sorting/test_linear_sorts.cpp)
algoat_add_test(test_hybrid_sorts sorting/test_hybrid_sorts.cpp)
algoat_add_test(test_sorting_dispatch sorting/test_sorting_dispatch.cpp)

# Searching tests
algoat_add_test(test_linear_search searching/test_linear_search.cpp)
algoat_add_test(test_binary_search searching/test_binary_search.cpp)
algoat_add_test(test_interpolation_search searching/test_interpolation_search.cpp)
algoat_add_test(test_searching_dispatch searching/test_searching_dispatch.cpp)
