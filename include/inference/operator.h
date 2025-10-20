#pragma once

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "tensor.h"

namespace inference {

// Operator attribute types
using Attribute = std::variant<int, float, std::string, std::vector<int>>;
using AttributeMap = std::map<std::string, Attribute>;

// Operator base class
class Operator {
  public:
    Operator(const std::string& name) : name_(name) {}
    virtual ~Operator() = default;

    // Forward inference
    virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs) = 0;

    // Get operator name
    const std::string& name() const { return name_; }

    // Set attribute
    void setAttribute(const std::string& key, const Attribute& value) { attributes_[key] = value; }

    // Get attribute
    template <typename T>
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

// Convolution operator
class Conv2dOperator : public Operator {
  public:
    Conv2dOperator() : Operator("Conv2d") {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

// Max pooling operator
class MaxPool2dOperator : public Operator {
  public:
    MaxPool2dOperator() : Operator("MaxPool2d") {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

// Fully connected operator
class LinearOperator : public Operator {
  public:
    LinearOperator() : Operator("Linear") {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

// ReLU activation
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

// Matrix multiplication
class MatMulOperator : public Operator {
  public:
    MatMulOperator() : Operator("MatMul") {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

// Addition operator
class AddOperator : public Operator {
  public:
    AddOperator() : Operator("Add") {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override;
};

}  // namespace inference
