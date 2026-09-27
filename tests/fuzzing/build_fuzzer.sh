#!/bin/bash
CC=clang CXX=clang++ cmake -B build-fuzz -S . -DALGOAT_ENABLE_FUZZING=ON
cmake --build build-fuzz --target diff_sort_fuzzer
