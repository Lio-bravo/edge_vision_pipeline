# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 edge_vision_pipeline contributors
cmake_minimum_required(VERSION 3.10)
project(evp_runtime CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Point ncnn_DIR at <ncnn-install>/lib/cmake/ncnn (host build or cross-compiled armv7 build).
find_package(ncnn REQUIRED)

add_library(evp_runtime STATIC
    src/meta.cpp
    src/image.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/model.cpp)
target_include_directories(evp_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(evp_runtime PUBLIC ncnn)
if(NOT MSVC)
    target_compile_options(evp_runtime PRIVATE -Wall -Wextra)
endif()

add_executable(evp_demo demo/evp_demo.cpp)
target_link_libraries(evp_demo PRIVATE evp_runtime)

option(EVP_BUILD_TESTS "build C++ unit tests" ON)
if(EVP_BUILD_TESTS)
    add_executable(evp_unit_tests tests/unit_tests.cpp)
    target_link_libraries(evp_unit_tests PRIVATE evp_runtime)
endif()
