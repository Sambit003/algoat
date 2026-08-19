add_library(algoat
    algoat.cpp
    core/config.cpp
    core/dispatcher.cpp
    sorting/quicksort.cpp
    sorting/mergesort.cpp
    sorting/heapsort.cpp
    searching/binary_search.cpp
    searching/interpolation_search.cpp
)

target_include_directories(algoat
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(algoat PUBLIC cxx_std_20)
set_target_properties(algoat PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.12.0
    )
    FetchContent_MakeAvailable(json)
endif()
target_link_libraries(algoat PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)

include(GNUInstallDirs)

install(TARGETS algoat
    EXPORT AlgoatTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/algoat
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

install(EXPORT AlgoatTargets
    FILE AlgoatTargets.cmake
    NAMESPACE Algoat::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Algoat
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
