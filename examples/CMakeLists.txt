cmake_minimum_required(VERSION 3.15)

# Simple inference example
add_executable(simple_inference simple_inference.cpp)
target_link_libraries(simple_inference PRIVATE inference)

# Model loading example
add_executable(model_loading model_loading.cpp)
target_link_libraries(model_loading PRIVATE inference)

# Performance benchmark example
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE inference)

# Custom operator example
add_executable(custom_operator custom_operator.cpp)
target_link_libraries(custom_operator PRIVATE inference)

message(STATUS "Examples configured")

