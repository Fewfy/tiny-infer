#include "inference/operator.h"

#include <algorithm>
#include <cmath>

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

    std::vector<Tensor> outputs;
    // TODO: Create output tensor and perform addition

    return outputs;
}

}  // namespace inference
