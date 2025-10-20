#include <cassert>
#include <iostream>

#include "inference/tensor.h"

void test_tensor_creation() {
    std::cout << "Testing tensor creation..." << std::endl;

    std::vector<int64_t> shape = {2, 3, 4};
    inference::Tensor tensor(shape, inference::DataType::FLOAT32);

    assert(tensor.ndim() == 3);
    assert(tensor.numel() == 24);
    assert(tensor.shape()[0] == 2);
    assert(tensor.shape()[1] == 3);
    assert(tensor.shape()[2] == 4);

    std::cout << "  PASSED" << std::endl;
}

void test_tensor_reshape() {
    std::cout << "Testing tensor reshape..." << std::endl;

    inference::Tensor tensor({2, 3, 4}, inference::DataType::FLOAT32);
    tensor.reshape({4, 6});

    assert(tensor.shape()[0] == 4);
    assert(tensor.shape()[1] == 6);
    assert(tensor.numel() == 24);

    std::cout << "  PASSED" << std::endl;
}

void test_tensor_fill() {
    std::cout << "Testing tensor fill..." << std::endl;

    inference::Tensor tensor({3, 3}, inference::DataType::FLOAT32);
    tensor.fill(3.14f);

    const float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        assert(data[i] == 3.14f);
    }

    std::cout << "  PASSED" << std::endl;
}

void test_tensor_copy() {
    std::cout << "Testing tensor copy..." << std::endl;

    inference::Tensor tensor1({2, 2}, inference::DataType::FLOAT32);
    tensor1.fill(1.0f);

    inference::Tensor tensor2 = tensor1;

    assert(tensor2.numel() == tensor1.numel());
    assert(tensor2.shape() == tensor1.shape());

    std::cout << "  PASSED" << std::endl;
}

void test_tensor_move() {}

int main() {
    std::cout << "Running Tensor Tests" << std::endl;
    std::cout << "====================" << std::endl;

    try {
        test_tensor_creation();
        test_tensor_reshape();
        test_tensor_fill();
        test_tensor_copy();

        std::cout << "\nAll tests PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << std::endl;
        return 1;
    }
}
