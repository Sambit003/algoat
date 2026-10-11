add_executable(test_module_import test_import.cpp)
target_link_libraries(test_module_import PRIVATE algoat GTest::gtest_main)
gtest_discover_tests(test_module_import DISCOVERY_MODE PRE_TEST)

add_executable(test_dual_mode test_dual_mode.cpp helper_header.cpp)
target_link_libraries(test_dual_mode PRIVATE algoat GTest::gtest_main)
gtest_discover_tests(test_dual_mode DISCOVERY_MODE PRE_TEST)

add_executable(test_module_invariants test_module_invariants.cpp)
target_link_libraries(test_module_invariants PRIVATE algoat GTest::gtest_main rapidcheck rapidcheck_gtest)
gtest_discover_tests(test_module_invariants DISCOVERY_MODE PRE_TEST)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    foreach(tgt test_module_import test_dual_mode test_module_invariants)
        target_compile_options(${tgt} PRIVATE -Wno-template-names-tu-local)
    endforeach()
endif()
