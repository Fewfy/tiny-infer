#pragma once

#include "tensor.h"
#include "operator.h"
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace inference {

// 计算图节点
struct GraphNode {
    std::shared_ptr<Operator> op;
    std::vector<int> input_indices;  // 输入张量索引
    std::vector<int> output_indices; // 输出张量索引
};

// 模型类
class Model {
public:
    Model() = default;
    ~Model() = default;
    
    // 添加算子
    void addOperator(std::shared_ptr<Operator> op, 
                     const std::vector<int>& inputs,
                     const std::vector<int>& outputs) {
        GraphNode node;
        node.op = op;
        node.input_indices = inputs;
        node.output_indices = outputs;
        graph_.push_back(node);
    }
    
    // 设置输入张量
    void setInput(const std::string& name, const Tensor& tensor) {
        inputs_[name] = tensor;
    }
    
    // 获取输出张量
    Tensor getOutput(const std::string& name) const {
        auto it = outputs_.find(name);
        if (it != outputs_.end()) {
            return it->second;
        }
        throw std::runtime_error("Output not found: " + name);
    }
    
    // 执行推理
    void forward();
    
    // 加载模型
    bool load(const std::string& path);
    
    // 保存模型
    bool save(const std::string& path) const;
    
    // 打印模型结构
    void print() const;
    
private:
    std::vector<GraphNode> graph_;
    std::map<std::string, Tensor> inputs_;
    std::map<std::string, Tensor> outputs_;
    std::vector<Tensor> intermediate_tensors_;
};

} // namespace inference

