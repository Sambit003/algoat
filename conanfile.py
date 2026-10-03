from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy
import os

class AlgoatConan(ConanFile):
    name = "algoat"
    version = "0.2.0"
    license = "LGPL-3.0-only"
    author = "Sambit Chakraborty <smbtchakraborty@gmail.com>"
    url = "https://github.com/user/algoat"
    description = "A high-performance algorithmic dispatch library for sorting and searching"
    topics = ("algorithms", "sorting", "searching", "dispatch")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    exports_sources = "CMakeLists.txt", "include/*", "src/*", "cmake/*", "LICENSE"

    def requirements(self):
        self.requires("nlohmann_json/3.11.3")

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["ALGOAT_BUILD_TESTS"] = "OFF"
        tc.variables["ALGOAT_BUILD_EXAMPLES"] = "OFF"
        tc.variables["ALGOAT_BUILD_BENCHMARKS"] = "OFF"
        tc.variables["ALGOAT_BUILD_PYTHON"] = "OFF"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["algoat"]
