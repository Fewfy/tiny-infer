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
    {
        inference::Tensor input1({2, 3}, inference::DataType::FLOAT32);
        inference::Tensor input2({2, 3}, inference::DataType::FLOAT32);
        input1.fill(1.0f);
        input2.fill(2.0f);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        assert(out_data[0] == 3.0f);
        assert(out_data[3] == 5.0f);
    }
    // Additional Add operator unit test cases

    // Test: Adding negative numbers
    {
        inference::Tensor input1({2, 3}, inference::DataType::FLOAT32);
        inference::Tensor input2({2, 3}, inference::DataType::FLOAT32);
        float val1[] = {-1.0f, -2.0f, 0.0f, 1.0f, 2.0f, 3.0f};
        float val2[] = {1.0f, 2.0f, -1.0f, -1.0f, -2.0f, 4.0f};
        input1.copyFrom(val1);
        input2.copyFrom(val2);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        assert(out_data[0] == 0.0f);
        assert(out_data[1] == 0.0f);
        assert(out_data[2] == -1.0f);
        assert(out_data[3] == 0.0f);
        assert(out_data[4] == 0.0f);
        assert(out_data[5] == 7.0f);
    }

    // Test: Adding zeros
    {
        inference::Tensor input1({2, 2}, inference::DataType::FLOAT32);
        inference::Tensor input2({2, 2}, inference::DataType::FLOAT32);
        input1.fill(0.0f);
        input2.fill(0.0f);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        for (int i = 0; i < 4; ++i) {
            assert(out_data[i] == 0.0f);
        }
    }

    // Test: Adding INT32 tensors
    {
        inference::Tensor input1({3}, inference::DataType::INT32);
        inference::Tensor input2({3}, inference::DataType::INT32);
        int32_t val1[] = {1, -2, 3};
        int32_t val2[] = {4, 5, -3};
        input1.copyFrom(val1);
        input2.copyFrom(val2);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const int32_t* out_data = outputs[0].data_ptr<int32_t>();
        assert(out_data[0] == 5);
        assert(out_data[1] == 3);
        assert(out_data[2] == 0);
    }

    // Test: Simple test for UINT8
    {
        inference::Tensor input1({2}, inference::DataType::UINT8);
        inference::Tensor input2({2}, inference::DataType::UINT8);
        uint8_t arr1[] = {255, 1};
        uint8_t arr2[] = {1, 255};
        input1.copyFrom(arr1);
        input2.copyFrom(arr2);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const uint8_t* out_data = outputs[0].data_ptr<uint8_t>();
        // Expect wraparound (if using uint8_t arithmetic)
        assert(out_data[0] == 0);  // 255 + 1 == 0 (mod 256)
        assert(out_data[1] == 0);  // 1 + 255 == 0 (mod 256)
    }

    // Test: Identical tensors
    {
        inference::Tensor input1({4}, inference::DataType::FLOAT32);
        float arr[] = {1.5f, 2.5f, -3.5f, 0.0f};
        input1.copyFrom(arr);
        inference::Tensor input2 = input1;
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        assert(out_data[0] == 3.0f);
        assert(out_data[1] == 5.0f);
        assert(out_data[2] == -7.0f);
        assert(out_data[3] == 0.0f);
    }

    std::cout << "  PASSED add operator" << std::endl;
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
