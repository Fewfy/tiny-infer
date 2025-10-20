#pragma once

#include "tensor.h"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <variant>

namespace inference {

// 算子参数类型
using Attribute = std::variant<int, float, std::string, std::vector<int>>;
using AttributeMap = std::map<std::string, Attribute>;

// 算子基类
class Operator {
public:
    Operator(const std::string& name) : name_(name) {}
    virtual ~Operator() = default;
    
    // 前向推理
    virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs) = 0;
    
    // 获取算子名称
    const std::string& name() const { return name_; }
    
    // 设置属性
    void setAttribute(const std::string& key, const Attribute& value) {
        attributes_[key] = value;
    }
    
    // 获取属性
    template<typename T>
    T getAttribute(const std::string& key, const T& default_value) const {
        auto it = attributes_.find(key);
        if (it != attributes_.end()) {
            return std::get<T>(it->second);
        }
        return default_value;
    }
    
protected:
    std::string name_;
    AttributeMap attributes_;
};

// 卷积算子
class Conv2dOperator : public Operator {
public:
    Conv2dOperator() : Operator("Conv2d") {}
    
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

// 池化算子
class MaxPool2dOperator : public Operator {
public:
    MaxPool2dOperator() : Operator("MaxPool2d") {}
    
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

// 全连接算子
class LinearOperator : public Operator {
public:
    LinearOperator() : Operator("Linear") {}
    
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

// ReLU激活
class ReluOperator : public Operator {
public:
    ReluOperator() : Operator("ReLU") {}
    
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

// Softmax
class SoftmaxOperator : public Operator {
public:
    SoftmaxOperator() : Operator("Softmax") {}
    
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

// 矩阵乘法
class MatMulOperator : public Operator {
public:
    MatMulOperator() : Operator("MatMul") {}
    
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

// 加法
class AddOperator : public Operator {
public:
    AddOperator() : Operator("Add") {}
    
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

} // namespace inference

