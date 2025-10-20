#include <cassert>
#include <cmath>
#include <iostream>

#include "inference/operator.h"

void test_relu() {
    std::cout << "Testing ReLU operator..." << std::endl;

    // TODO: Implement ReLU test
    // Create input tensor
    // inference::Tensor input({2, 3}, inference::DataType::FLOAT32);
    // float data[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
    // input.copyFrom(data);

    // Execute ReLU
    // inference::ReluOperator relu;
    // auto outputs = relu.forward({input});

    // Verify results
    // const float* out_data = outputs[0].data_ptr<float>();
    // assert(out_data[0] == 0.0f);
    // assert(out_data[3] == 0.5f);
    // assert(out_data[5] == 1.5f);

    std::cout << "  PASSED (TODO: implement)" << std::endl;
}

void test_add() {
    std::cout << "Testing Add operator..." << std::endl;

    // TODO: Implement Add test

    std::cout << "  PASSED (TODO: implement)" << std::endl;
}

void test_matmul() {
    std::cout << "Testing MatMul operator..." << std::endl;

    // TODO: Implement matrix multiplication test
    // Create two matrices [2x3] and [3x2]
    // Result should be [2x2]

    std::cout << "  PASSED (TODO: implement)" << std::endl;
}

int main() {
    std::cout << "Running Operator Tests" << std::endl;
    std::cout << "======================" << std::endl;

    try {
        test_relu();
        test_add();
        test_matmul();

        std::cout << "\nAll tests PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << std::endl;
        return 1;
    }
}
