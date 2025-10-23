#include "inference/operator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "inference/tensor.h"

namespace inference {

// ==================== Conv2dOperator ====================
std::vector<Tensor> Conv2dOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: Implement 2D convolution operation
    // inputs[0]: Input feature map [N, C_in, H, W]
    // inputs[1]: Convolution kernel weights [C_out, C_in, K_H, K_W]
    // inputs[2]: Bias [C_out] (optional)

    // Get parameters
    // int stride = getAttribute<int>("stride", 1);
    // int padding = getAttribute<int>("padding", 0);

    std::vector<Tensor> outputs;
    // TODO: Create output tensor and perform convolution computation

    return outputs;
}

// ==================== MaxPool2dOperator ====================
std::vector<Tensor> MaxPool2dOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: Implement max pooling operation
    // inputs[0]: Input feature map [N, C, H, W]

    // Get parameters
    // int kernel_size = getAttribute<int>("kernel_size", 2);
    // int stride = getAttribute<int>("stride", 2);

    std::vector<Tensor> outputs;
    // TODO: Create output tensor and perform pooling computation

    return outputs;
}

// ==================== LinearOperator ====================
std::vector<Tensor> LinearOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: Implement fully connected layer operation
    // inputs[0]: Input [N, in_features]
    // inputs[1]: Weights [out_features, in_features]
    // inputs[2]: Bias [out_features] (optional)

    std::vector<Tensor> outputs;
    // TODO: Create output tensor and perform matrix multiplication and add bias
    // output = input @ weight.T + bias

    return outputs;
}

// ==================== ReluOperator ====================
std::vector<Tensor> ReluOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: Implement ReLU activation function
    // output = max(0, input)

    std::vector<Tensor> outputs;
    // TODO: Create output tensor and apply ReLU

    return outputs;
}

// ==================== SoftmaxOperator ====================
std::vector<Tensor> SoftmaxOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: Implement Softmax operation
    // softmax(x_i) = exp(x_i) / sum(exp(x_j))

    // Get parameters
    // int axis = getAttribute<int>("axis", -1);

    std::vector<Tensor> outputs;
    // TODO: Create output tensor and compute Softmax

    return outputs;
}

// ==================== MatMulOperator ====================
std::vector<Tensor> MatMulOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: Implement matrix multiplication
    // inputs[0]: Left matrix [M, K]
    // inputs[1]: Right matrix [K, N]
    // output: Result matrix [M, N]

    std::vector<Tensor> outputs;
    // TODO: Create output tensor and perform matrix multiplication

    return outputs;
}

// ==================== AddOperator ====================
std::vector<Tensor> AddOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: Implement element-wise addition (with broadcasting support)
    // inputs[0]: First input tensor
    // inputs[1]: Second input tensor
    assert(inputs.size() == 2);
    assert(inputs[0].shape() == inputs[1].shape());
    assert(inputs[0].dtype() == inputs[1].dtype());
    assert(inputs[0].ndim() == inputs[1].ndim());

    Tensor first = inputs[0];
    Tensor second = inputs[1];

    std::vector<Tensor> outputs;
    Tensor output(first.shape(), first.dtype());

    auto add_func = []<typename T>(const T* first_data, const T* second_data, T* output_data,
                                   int64_t numel) {
        for (int64_t i = 0; i < numel; ++i) {
            output_data[i] = first_data[i] + second_data[i];
        }
    };
    switch (first.dtype()) {
        case DataType::FLOAT32: {
            const float* first_data = first.data_ptr<float>();
            const float* second_data = second.data_ptr<float>();
            auto* output_data = output.data_ptr<float>();
            add_func(first_data, second_data, output_data, first.numel());
            break;
        }
        case DataType::FLOAT16: {
            const uint16_t* first_data = first.data_ptr<uint16_t>();
            const uint16_t* second_data = second.data_ptr<uint16_t>();
            auto* output_data = output.data_ptr<uint16_t>();
            add_func(first_data, second_data, output_data, first.numel());
        } break;
        case DataType::INT32: {
            const int32_t* first_data = first.data_ptr<int32_t>();
            const int32_t* second_data = second.data_ptr<int32_t>();
            auto* output_data = output.data_ptr<int32_t>();
            add_func(first_data, second_data, output_data, first.numel());
            break;
        }
        case DataType::INT8: {
            const int8_t* first_data = first.data_ptr<int8_t>();
            const int8_t* second_data = second.data_ptr<int8_t>();
            auto* output_data = output.data_ptr<int8_t>();
            add_func(first_data, second_data, output_data, first.numel());
            break;
        }
        case DataType::UINT8: {
            const uint8_t* first_data = first.data_ptr<uint8_t>();
            const uint8_t* second_data = second.data_ptr<uint8_t>();
            auto* output_data = output.data_ptr<uint8_t>();
            add_func(first_data, second_data, output_data, first.numel());
            break;
        } break;
        default:
            throw std::runtime_error("Unsupported data type");
    }

    outputs.push_back(std::move(output));

    return outputs;
}

}  // namespace inference
