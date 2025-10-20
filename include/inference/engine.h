#pragma once

#include "model.h"
#include "tensor.h"
#include <memory>
#include <string>
#include <vector>

namespace inference {

// 推理引擎配置
struct EngineConfig {
    int num_threads = 1;          // 线程数
    bool use_fp16 = false;        // 是否使用FP16
    bool enable_profiling = false; // 是否启用性能分析
    size_t workspace_size = 1024 * 1024 * 256; // 工作空间大小（256MB）
};

// 推理引擎
class InferenceEngine {
public:
    InferenceEngine() = default;
    explicit InferenceEngine(const EngineConfig& config) : config_(config) {}
    ~InferenceEngine() = default;
    
    // 加载模型
    bool loadModel(const std::string& model_path);
    
    // 设置模型
    void setModel(std::shared_ptr<Model> model) {
        model_ = model;
    }
    
    // 执行推理
    std::vector<Tensor> infer(const std::vector<Tensor>& inputs);
    
    // 异步推理
    void inferAsync(const std::vector<Tensor>& inputs);
    
    // 获取推理结果
    std::vector<Tensor> getResults();
    
    // 预热
    void warmup(int iterations = 10);
    
    // 获取性能统计
    struct ProfilingInfo {
        double total_time_ms = 0.0;
        double avg_time_ms = 0.0;
        std::map<std::string, double> op_times_ms;
    };
    
    ProfilingInfo getProfilingInfo() const { return profiling_info_; }
    
    // 优化模型
    void optimizeModel();
    
private:
    EngineConfig config_;
    std::shared_ptr<Model> model_;
    ProfilingInfo profiling_info_;
};

} // namespace inference

