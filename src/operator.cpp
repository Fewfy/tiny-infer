#include "inference/operator.h"
#include <cmath>
#include <algorithm>

namespace inference {

// ==================== Conv2dOperator ====================
std::vector<Tensor> Conv2dOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: 实现2D卷积操作
    // inputs[0]: 输入特征图 [N, C_in, H, W]
    // inputs[1]: 卷积核权重 [C_out, C_in, K_H, K_W]
    // inputs[2]: 偏置 [C_out] (可选)
    
    // 获取参数
    // int stride = getAttribute<int>("stride", 1);
    // int padding = getAttribute<int>("padding", 0);
    
    std::vector<Tensor> outputs;
    // TODO: 创建输出张量并进行卷积计算
    
    return outputs;
}

// ==================== MaxPool2dOperator ====================
std::vector<Tensor> MaxPool2dOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: 实现最大池化操作
    // inputs[0]: 输入特征图 [N, C, H, W]
    
    // 获取参数
    // int kernel_size = getAttribute<int>("kernel_size", 2);
    // int stride = getAttribute<int>("stride", 2);
    
    std::vector<Tensor> outputs;
    // TODO: 创建输出张量并进行池化计算
    
    return outputs;
}

// ==================== LinearOperator ====================
std::vector<Tensor> LinearOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: 实现全连接层操作
    // inputs[0]: 输入 [N, in_features]
    // inputs[1]: 权重 [out_features, in_features]
    // inputs[2]: 偏置 [out_features] (可选)
    
    std::vector<Tensor> outputs;
    // TODO: 创建输出张量并进行矩阵乘法和加偏置
    // output = input @ weight.T + bias
    
    return outputs;
}

// ==================== ReluOperator ====================
std::vector<Tensor> ReluOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: 实现ReLU激活函数
    // output = max(0, input)
    
    std::vector<Tensor> outputs;
    // TODO: 创建输出张量并应用ReLU
    
    return outputs;
}

// ==================== SoftmaxOperator ====================
std::vector<Tensor> SoftmaxOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: 实现Softmax操作
    // softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    // 获取参数
    // int axis = getAttribute<int>("axis", -1);
    
    std::vector<Tensor> outputs;
    // TODO: 创建输出张量并计算Softmax
    
    return outputs;
}

// ==================== MatMulOperator ====================
std::vector<Tensor> MatMulOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: 实现矩阵乘法
    // inputs[0]: 左矩阵 [M, K]
    // inputs[1]: 右矩阵 [K, N]
    // output: 结果矩阵 [M, N]
    
    std::vector<Tensor> outputs;
    // TODO: 创建输出张量并进行矩阵乘法
    
    return outputs;
}

// ==================== AddOperator ====================
std::vector<Tensor> AddOperator::forward(const std::vector<Tensor>& inputs) {
    // TODO: 实现元素级加法（支持广播）
    // inputs[0]: 第一个输入张量
    // inputs[1]: 第二个输入张量
    
    std::vector<Tensor> outputs;
    // TODO: 创建输出张量并进行加法运算
    
    return outputs;
}

} // namespace inference

