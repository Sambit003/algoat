add_library(algoat
    algoat.cpp
    c_api/algoat_c.cpp
    core/config.cpp
    core/dispatcher.cpp
    simd/isa.cpp
    simd/partition.cpp
    sorting/quicksort.cpp
    sorting/mergesort.cpp
    sorting/heapsort.cpp
    searching/binary_search.cpp
    searching/interpolation_search.cpp
    searching/hybrid_interpolation_search.cpp
    strings/substring_search.cpp
)

target_include_directories(algoat
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(algoat PUBLIC cxx_std_20)
set_target_properties(algoat PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ALGOAT_BUILD_MODULES)
    target_sources(algoat
        PUBLIC
            FILE_SET CXX_MODULES
                BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                FILES algoat.cppm
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(algoat PRIVATE -Wno-template-names-tu-local)
    endif()
endif()

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.12.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(json)
endif()

include(FetchContent)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG        1.2.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(highway)

find_package(Threads REQUIRED)
find_package(OpenMP)

include(CheckLibraryExists)
check_library_exists(atomic __atomic_fetch_add_16 "" ALGOAT_HAS_LIBATOMIC)

target_link_libraries(algoat
    PUBLIC
        $<BUILD_INTERFACE:hwy>
        Threads::Threads
    PRIVATE
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
)

if(ALGOAT_HAS_LIBATOMIC)
    target_link_libraries(algoat PUBLIC atomic)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(algoat PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(algoat_c SHARED c_api/algoat_c.cpp)
target_include_directories(algoat_c
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(algoat_c PUBLIC cxx_std_20)
target_compile_definitions(algoat_c PRIVATE ALGOAT_C_BUILD_SHARED)
target_link_libraries(algoat_c PRIVATE algoat)
set_target_properties(algoat_c PROPERTIES POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

set(_algoat_module_install_args "")
if(ALGOAT_BUILD_MODULES)
    set(_algoat_module_install_args FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/algoat/modules)
endif()

install(TARGETS algoat algoat_c
    EXPORT AlgoatTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    ${_algoat_module_install_args}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/algoat
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

set(_algoat_export_module_args "")
if(ALGOAT_BUILD_MODULES)
    set(_algoat_export_module_args CXX_MODULES_DIRECTORY cxx-modules)
endif()

install(EXPORT AlgoatTargets
    FILE AlgoatTargets.cmake
    NAMESPACE Algoat::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Algoat
    ${_algoat_export_module_args}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/AlgoatConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/AlgoatConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Algoat
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/AlgoatConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/AlgoatConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/AlgoatConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Algoat
)

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/algoat.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/algoat.pc"
    @ONLY
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/algoat.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)
