#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "operator.h"
#include "tensor.h"

namespace inference {

// Computation graph node
struct GraphNode {
    std::shared_ptr<Operator> op;
    std::vector<int> input_indices;   // Input tensor indices
    std::vector<int> output_indices;  // Output tensor indices
};

// Model class
class Model {
  public:
    Model() = default;
    ~Model() = default;

    // Add operator
    void addOperator(std::shared_ptr<Operator> op, const std::vector<int>& inputs,
                     const std::vector<int>& outputs) {
        GraphNode node;
        node.op = op;
        node.input_indices = inputs;
        node.output_indices = outputs;
        graph_.push_back(node);
    }

    // Set input tensor
    void setInput(const std::string& name, const Tensor& tensor) { inputs_[name] = tensor; }

    // Get output tensor
    Tensor getOutput(const std::string& name) const {
        auto it = outputs_.find(name);
        if (it != outputs_.end()) {
            return it->second;
        }
        throw std::runtime_error("Output not found: " + name);
    }

    // Execute inference
    void forward();

    // Load model
    bool load(const std::string& path);

    // Save model
    bool save(const std::string& path) const;

    // Print model structure
    void print() const;

  private:
    std::vector<GraphNode> graph_;
    std::map<std::string, Tensor> inputs_;
    std::map<std::string, Tensor> outputs_;
    std::vector<Tensor> intermediate_tensors_;
};

}  // namespace inference
