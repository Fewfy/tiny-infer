cmake_minimum_required(VERSION 3.15)

# 简单的测试框架（如果需要更完整的测试，可以集成GoogleTest）

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE inference)

add_executable(test_operators test_operators.cpp)
target_link_libraries(test_operators PRIVATE inference)

# 添加测试
add_test(NAME TensorTest COMMAND test_tensor)
add_test(NAME OperatorsTest COMMAND test_operators)

message(STATUS "Tests configured")

