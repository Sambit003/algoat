find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_algoat_impl bindings.cpp)
target_link_libraries(_algoat_impl PRIVATE algoat)

# Install into the Python package directory
install(TARGETS _algoat_impl DESTINATION algoat)
