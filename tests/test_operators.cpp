#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

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
        assert(out_data[3] == 3.0f);
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

void test_add_broadcasting() {
    std::cout << "Testing Add operator with broadcasting..." << std::endl;

    // Test: Scalar broadcasting (1D tensor + scalar)
    {
        inference::Tensor input1({3}, inference::DataType::FLOAT32);
        inference::Tensor input2({1}, inference::DataType::FLOAT32);
        float val1[] = {1.0f, 2.0f, 3.0f};
        float val2[] = {10.0f};
        input1.copyFrom(val1);
        input2.copyFrom(val2);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        assert(out_data[0] == 11.0f);
        assert(out_data[1] == 12.0f);
        assert(out_data[2] == 13.0f);
    }

    // Test: Row broadcasting (2D tensor + 1D tensor)
    {
        inference::Tensor input1({2, 3}, inference::DataType::FLOAT32);
        inference::Tensor input2({3}, inference::DataType::FLOAT32);
        float val1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float val2[] = {10.0f, 20.0f, 30.0f};
        input1.copyFrom(val1);
        input2.copyFrom(val2);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        // First row: [1,2,3] + [10,20,30] = [11,22,33]
        assert(out_data[0] == 11.0f);
        assert(out_data[1] == 22.0f);
        assert(out_data[2] == 33.0f);
        // Second row: [4,5,6] + [10,20,30] = [14,25,36]
        assert(out_data[3] == 14.0f);
        assert(out_data[4] == 25.0f);
        assert(out_data[5] == 36.0f);
    }

    // Test: Column broadcasting (2D tensor + column vector)
    {
        inference::Tensor input1({2, 3}, inference::DataType::FLOAT32);
        inference::Tensor input2({2, 1}, inference::DataType::FLOAT32);
        float val1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float val2[] = {10.0f, 20.0f};
        input1.copyFrom(val1);
        input2.copyFrom(val2);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        // First row: [1,2,3] + [10,10,10] = [11,12,13]
        assert(out_data[0] == 11.0f);
        assert(out_data[1] == 12.0f);
        assert(out_data[2] == 13.0f);
        // Second row: [4,5,6] + [20,20,20] = [24,25,26]
        assert(out_data[3] == 24.0f);
        assert(out_data[4] == 25.0f);
        assert(out_data[5] == 26.0f);
    }

    // Test: Complex broadcasting (3D tensor + 2D tensor)
    {
        inference::Tensor input1({2, 1, 3}, inference::DataType::FLOAT32);
        inference::Tensor input2({3, 3}, inference::DataType::FLOAT32);
        float val1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float val2[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f};
        input1.copyFrom(val1);
        input2.copyFrom(val2);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        // Shape should be [2, 3, 3]
        // First slice: [1,2,3] + [10,20,30] = [11,22,33]
        assert(out_data[0] == 11.0f);
        assert(out_data[1] == 22.0f);
        assert(out_data[2] == 33.0f);
        // Second slice: [1,2,3] + [40,50,60] = [41,52,63]
        assert(out_data[3] == 41.0f);
        assert(out_data[4] == 52.0f);
        assert(out_data[5] == 63.0f);
        // Third slice: [1,2,3] + [70,80,90] = [71,82,93]
        assert(out_data[6] == 71.0f);
        assert(out_data[7] == 82.0f);
        assert(out_data[8] == 93.0f);
        // Second batch: [4,5,6] + [10,20,30] = [14,25,36]
        assert(out_data[9] == 14.0f);
        assert(out_data[10] == 25.0f);
        assert(out_data[11] == 36.0f);
    }

    // Test: Scalar broadcasting with different data types
    {
        inference::Tensor input1({2, 2}, inference::DataType::INT32);
        inference::Tensor input2({1}, inference::DataType::INT32);
        int32_t val1[] = {1, 2, 3, 4};
        int32_t val2[] = {10};
        input1.copyFrom(val1);
        input2.copyFrom(val2);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const int32_t* out_data = outputs[0].data_ptr<int32_t>();
        assert(out_data[0] == 11);
        assert(out_data[1] == 12);
        assert(out_data[2] == 13);
        assert(out_data[3] == 14);
    }

    // Test: Edge case - both tensors have same shape (no broadcasting)
    {
        inference::Tensor input1({2, 2}, inference::DataType::FLOAT32);
        inference::Tensor input2({2, 2}, inference::DataType::FLOAT32);
        input1.fill(1.0f);
        input2.fill(2.0f);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        for (int i = 0; i < 4; ++i) {
            assert(out_data[i] == 3.0f);
        }
    }

    // Test: Edge case - single element tensor
    {
        inference::Tensor input1({1}, inference::DataType::FLOAT32);
        inference::Tensor input2({1}, inference::DataType::FLOAT32);
        input1.fill(5.0f);
        input2.fill(3.0f);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        assert(out_data[0] == 8.0f);
    }

    std::cout << "  PASSED add operator broadcasting" << std::endl;
}

void test_add_error_cases() {
    std::cout << "Testing Add operator error cases..." << std::endl;

    // Test: Incompatible shapes (should throw)
    {
        inference::Tensor input1({2, 3}, inference::DataType::FLOAT32);
        inference::Tensor input2({2, 4}, inference::DataType::FLOAT32);  // Incompatible shape
        input1.fill(1.0f);
        input2.fill(2.0f);
        inference::AddOperator add;

        bool exception_thrown = false;
        try {
            auto outputs = add.forward({input1, input2});
        } catch (const std::runtime_error& e) {
            exception_thrown = true;
            // Check that the error message contains "broadcasting"
            std::string error_msg = e.what();
            assert(error_msg.find("broadcasting") != std::string::npos);
        }
        assert(exception_thrown);
    }

    // Test: Different data types (should throw)
    {
        inference::Tensor input1({2, 3}, inference::DataType::FLOAT32);
        inference::Tensor input2({2, 3}, inference::DataType::INT32);  // Different data type
        input1.fill(1.0f);
        input2.fill(2);
        inference::AddOperator add;

        bool exception_thrown = false;
        try {
            auto outputs = add.forward({input1, input2});
        } catch (const std::runtime_error& e) {
            exception_thrown = true;
            // Check that the error message contains "data type"
            std::string error_msg = e.what();
            assert(error_msg.find("data type") != std::string::npos);
        }
        assert(exception_thrown);
    }

    // Test: Wrong number of inputs (should throw)
    {
        inference::Tensor input1({2, 3}, inference::DataType::FLOAT32);
        inference::Tensor input2({2, 3}, inference::DataType::FLOAT32);
        inference::Tensor input3({2, 3}, inference::DataType::FLOAT32);
        input1.fill(1.0f);
        input2.fill(2.0f);
        input3.fill(3.0f);
        inference::AddOperator add;

        bool exception_thrown = false;
        try {
            auto outputs = add.forward({input1, input2, input3});  // 3 inputs instead of 2
        } catch (const std::runtime_error& e) {
            exception_thrown = true;
            // Check that the error message contains "exactly 2 inputs"
            std::string error_msg = e.what();
            assert(error_msg.find("exactly 2 inputs") != std::string::npos);
        }
        assert(exception_thrown);
    }

    std::cout << "  PASSED add operator error cases" << std::endl;
}

void test_matmul() {
    std::cout << "Testing MatMul operator..." << std::endl;

    // TODO: Implement matrix multiplication test
    // Create two matrices [2x3] and [3x2]
    // Result should be [2x2]

    std::cout << "  PASSED (TODO: implement)" << std::endl;
}

void test_operator_edge_cases() {
    std::cout << "Testing operator edge cases..." << std::endl;

    // Test: Empty tensor addition
    {
        inference::Tensor input1({0}, inference::DataType::FLOAT32);
        inference::Tensor input2({0}, inference::DataType::FLOAT32);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        assert(outputs[0].numel() == 0);
    }

    // Test: Single element tensor addition
    {
        inference::Tensor input1({1}, inference::DataType::FLOAT32);
        inference::Tensor input2({1}, inference::DataType::FLOAT32);
        input1.fill(1.5f);
        input2.fill(2.5f);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        assert(out_data[0] == 4.0f);
    }

    // Test: Large tensor addition (performance test)
    {
        const int size = 1000;
        inference::Tensor input1({size}, inference::DataType::FLOAT32);
        inference::Tensor input2({size}, inference::DataType::FLOAT32);
        input1.fill(1.0f);
        input2.fill(2.0f);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const float* out_data = outputs[0].data_ptr<float>();
        for (int i = 0; i < size; ++i) {
            assert(out_data[i] == 3.0f);
        }
    }

    // Test: Different data types edge cases
    {
        // Test INT8 overflow
        inference::Tensor input1({2}, inference::DataType::INT8);
        inference::Tensor input2({2}, inference::DataType::INT8);
        int8_t val1[] = {100, -100};
        int8_t val2[] = {50, -50};
        input1.copyFrom(val1);
        input2.copyFrom(val2);
        inference::AddOperator add;
        auto outputs = add.forward({input1, input2});
        const int8_t* out_data = outputs[0].data_ptr<int8_t>();
        assert(out_data[0] == 150);
        assert(out_data[1] == -150);
    }

    std::cout << "  PASSED operator edge cases" << std::endl;
}

void test_operator_performance() {
    std::cout << "Testing operator performance..." << std::endl;

    // Test: Performance with different tensor sizes
    const std::vector<int> sizes = {10, 100, 1000, 10000};

    for (int size : sizes) {
        inference::Tensor input1({size}, inference::DataType::FLOAT32);
        inference::Tensor input2({size}, inference::DataType::FLOAT32);
        input1.fill(1.0f);
        input2.fill(2.0f);

        inference::AddOperator add;
        auto start = std::chrono::high_resolution_clock::now();
        auto outputs = add.forward({input1, input2});
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  Size " << size << ": " << duration.count() << " microseconds" << std::endl;

        // Verify correctness
        const float* out_data = outputs[0].data_ptr<float>();
        for (int i = 0; i < size; ++i) {
            assert(out_data[i] == 3.0f);
        }
    }

    std::cout << "  PASSED operator performance" << std::endl;
}

int main() {
    std::cout << "Running Operator Tests" << std::endl;
    std::cout << "======================" << std::endl;

    try {
        test_relu();
        test_add();
        test_add_broadcasting();
        test_add_error_cases();
        test_operator_edge_cases();
        test_operator_performance();
        test_matmul();

        std::cout << "\nAll tests PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << std::endl;
        return 1;
    }
}
