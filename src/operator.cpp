#include "inference/operator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "inference/tensor.h"

namespace inference {

// ==================== Broadcast Helper Functions ====================

// Compute the broadcast shape of two tensors
std::vector<int64_t> compute_broadcast_shape(const std::vector<int64_t>& shape1,
                                             const std::vector<int64_t>& shape2) {
    // Handle empty tensors
    if (shape1.empty() && shape2.empty()) {
        return {};
    }
    if (shape1.empty()) {
        return shape2;
    }
    if (shape2.empty()) {
        return shape1;
    }

    size_t max_dims = std::max(shape1.size(), shape2.size());
    std::vector<int64_t> result(max_dims);

    // Align dimensions from right to left
    for (size_t i = 0; i < max_dims; ++i) {
        int64_t dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        int64_t dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;

        // here we check if the dimensions are compatible for broadcasting
        if (dim1 == dim2 || dim1 == 1 || dim2 == 1) {
            result[max_dims - 1 - i] = std::max(dim1, dim2);
        } else {
            throw std::runtime_error("Incompatible shapes for broadcasting: [" +
                                     std::to_string(dim1) + "] and [" + std::to_string(dim2) + "]");
        }
    }

    return result;
}

// Convert linear index to multi-dimensional index
std::vector<int64_t> linear_to_multi_index(int64_t linear_idx, const std::vector<int64_t>& shape) {
    std::vector<int64_t> indices(shape.size());
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        indices[i] = linear_idx % shape[i];
        linear_idx /= shape[i];
    }
    return indices;
}

// Convert multi-dimensional index to linear index
int64_t multi_to_linear_index(const std::vector<int64_t>& indices,
                              const std::vector<int64_t>& shape) {
    int64_t linear_idx = 0;
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        linear_idx += indices[i] * stride;
        stride *= shape[i];
    }
    return linear_idx;
}

// Get the linear index in the original tensor given the broadcasted index
int64_t get_broadcasted_index(int64_t output_idx, const std::vector<int64_t>& output_shape,
                              const std::vector<int64_t>& input_shape) {
    std::vector<int64_t> multi_idx = linear_to_multi_index(output_idx, output_shape);

    // Adjust indices for broadcasting (dimension size 1 maps to index 0)
    std::vector<int64_t> input_multi_idx(input_shape.size());
    int64_t offset =
        static_cast<int64_t>(output_shape.size()) - static_cast<int64_t>(input_shape.size());

    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] == 1) {
            input_multi_idx[i] = 0;  // Broadcast dimension
        } else {
            input_multi_idx[i] = multi_idx[offset + i];
        }
    }

    return multi_to_linear_index(input_multi_idx, input_shape);
}

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

template <typename T>
void matmul_impl(const Tensor& tensor_1, const Tensor& tensor_2, Tensor& result) {
    int m = tensor_1.shape()[0];
    int n = tensor_1.shape()[1];
    int p = tensor_2.shape()[1];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            T sum = 0;
            for (int k = 0; k < n; k++) {
                sum += tensor_1.data_ptr<T>()[i * n + k] * tensor_2.data_ptr<T>()[k * p + j];
            }
            result.data_ptr<T>()[i * p + j] = sum;
        }
    }
}

// ==================== MatMulOperator ====================
std::vector<Tensor> MatMulOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: Implement matrix multiplication
    // inputs[0]: Left matrix [M, K]
    // inputs[1]: Right matrix [K, N]
    // output: Result matrix [M, N]
    if (inputs.size() != 2) {
        throw std::runtime_error("MatMulOperator requires inputs with same data type");
    }

    const Tensor& tensor_1 = inputs[0];
    const Tensor& tensor_2 = inputs[1];

    if (tensor_1.dtype() != tensor_2.dtype()) {
        throw std::runtime_error("MatMulOperator: inputs must have same data type");
    }

    int row_1 = tensor_1.shape()[0];
    int col_1 = tensor_1.shape()[1];

    int row_2 = tensor_2.shape()[0];
    int col_2 = tensor_2.shape()[1];

    if (col_1 != row_2) {
        throw std::runtime_error("MatMulOperator: inner dimensions do not match");
    }

    Tensor out({row_1, col_2}, tensor_1.dtype());
    switch (tensor_1.dtype()) {
        case DataType::FLOAT16: {
            matmul_impl<uint16_t>(tensor_1, tensor_2, out);
            break;
        }
        case DataType::FLOAT32: {
            matmul_impl<float>(tensor_1, tensor_2, out);
            break;
        }
        case DataType::INT32: {
            matmul_impl<int32_t>(tensor_1, tensor_2, out);
            break;
        }
        case DataType::INT8: {
            matmul_impl<int8_t>(tensor_1, tensor_2, out);
            break;
        }
        case DataType::UINT8: {
            matmul_impl<uint8_t>(tensor_1, tensor_2, out);
            break;
        }
        default: {
            throw std::runtime_error("MatMulOperator: unsupported data type");
        }
    }
    std::vector<Tensor> outputs;
    outputs.emplace_back(std::move(out));
    // TODO: Create output tensor and perform matrix multiplication

    return outputs;
}

// ==================== AddOperator ====================
std::vector<Tensor> AddOperator::forward(const std::vector<Tensor>& inputs) {
    // Implement element-wise addition with broadcasting support
    // inputs[0]: First input tensor
    // inputs[1]: Second input tensor
    if (inputs.size() != 2) {
        throw std::runtime_error("AddOperator requires exactly 2 inputs");
    }

    const Tensor& first = inputs[0];
    const Tensor& second = inputs[1];

    // Check data type compatibility
    if (first.dtype() != second.dtype()) {
        throw std::runtime_error("AddOperator requires inputs with same data type");
    }

    // Compute broadcast shape
    std::vector<int64_t> output_shape = compute_broadcast_shape(first.shape(), second.shape());

    std::vector<Tensor> outputs;
    Tensor output(output_shape, first.dtype());

    // Broadcast addition function
    auto broadcast_add_func = [&](auto* first_data, auto* second_data, auto* output_data) {
        int64_t output_numel = output.numel();
        for (int64_t i = 0; i < output_numel; ++i) {
            int64_t first_idx = get_broadcasted_index(i, output_shape, first.shape());
            int64_t second_idx = get_broadcasted_index(i, output_shape, second.shape());
            output_data[i] = first_data[first_idx] + second_data[second_idx];
        }
    };

    switch (first.dtype()) {
        case DataType::FLOAT32: {
            const float* first_data = first.data_ptr<float>();
            const float* second_data = second.data_ptr<float>();
            auto* output_data = output.data_ptr<float>();
            broadcast_add_func(first_data, second_data, output_data);
            break;
        }
        case DataType::FLOAT16: {
            const uint16_t* first_data = first.data_ptr<uint16_t>();
            const uint16_t* second_data = second.data_ptr<uint16_t>();
            auto* output_data = output.data_ptr<uint16_t>();
            broadcast_add_func(first_data, second_data, output_data);
            break;
        }
        case DataType::INT32: {
            const int32_t* first_data = first.data_ptr<int32_t>();
            const int32_t* second_data = second.data_ptr<int32_t>();
            auto* output_data = output.data_ptr<int32_t>();
            broadcast_add_func(first_data, second_data, output_data);
            break;
        }
        case DataType::INT8: {
            const int8_t* first_data = first.data_ptr<int8_t>();
            const int8_t* second_data = second.data_ptr<int8_t>();
            auto* output_data = output.data_ptr<int8_t>();
            broadcast_add_func(first_data, second_data, output_data);
            break;
        }
        case DataType::UINT8: {
            const uint8_t* first_data = first.data_ptr<uint8_t>();
            const uint8_t* second_data = second.data_ptr<uint8_t>();
            auto* output_data = output.data_ptr<uint8_t>();
            broadcast_add_func(first_data, second_data, output_data);
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type");
    }

    outputs.push_back(std::move(output));

    return outputs;
}

}  // namespace inference
