#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "tensor.h"

namespace inference {

// Inference engine configuration
struct EngineConfig {
    int num_threads = 1;                        // Number of threads
    bool use_fp16 = false;                      // Use FP16 precision
    bool enable_profiling = false;              // Enable profiling
    size_t workspace_size = 1024 * 1024 * 256;  // Workspace size (256MB)
};

// Inference engine
class InferenceEngine {
  public:
    InferenceEngine() = default;
    explicit InferenceEngine(const EngineConfig& config) : config_(config) {}
    ~InferenceEngine() = default;

    // Load model
    bool loadModel(const std::string& model_path);

    // Set model
    void setModel(std::shared_ptr<Model> model) { model_ = model; }

    // Execute inference
    std::vector<Tensor> infer(const std::vector<Tensor>& inputs);

    // Asynchronous inference
    void inferAsync(const std::vector<Tensor>& inputs);

    // Get inference results
    std::vector<Tensor> getResults();

    // Warmup
    void warmup(int iterations = 10);

    // Get profiling statistics
    struct ProfilingInfo {
        double total_time_ms = 0.0;
        double avg_time_ms = 0.0;
        std::map<std::string, double> op_times_ms;
    };

    ProfilingInfo getProfilingInfo() const { return profiling_info_; }

    // Optimize model
    void optimizeModel();

  private:
    EngineConfig config_;
    std::shared_ptr<Model> model_;
    ProfilingInfo profiling_info_;
};

}  // namespace inference
