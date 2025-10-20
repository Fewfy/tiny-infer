cmake_minimum_required(VERSION 3.15)

# Simple test framework (GoogleTest can be integrated for more complete testing)

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE inference)

add_executable(test_operators test_operators.cpp)
target_link_libraries(test_operators PRIVATE inference)

# Add tests
add_test(NAME TensorTest COMMAND test_tensor)
add_test(NAME OperatorsTest COMMAND test_operators)

message(STATUS "Tests configured")

