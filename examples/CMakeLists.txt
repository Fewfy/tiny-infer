cmake_minimum_required(VERSION 3.15)

# 简单推理示例
add_executable(simple_inference simple_inference.cpp)
target_link_libraries(simple_inference PRIVATE inference)

# 模型加载示例
add_executable(model_loading model_loading.cpp)
target_link_libraries(model_loading PRIVATE inference)

# 性能测试示例
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE inference)

# 自定义算子示例
add_executable(custom_operator custom_operator.cpp)
target_link_libraries(custom_operator PRIVATE inference)

message(STATUS "Examples configured")

