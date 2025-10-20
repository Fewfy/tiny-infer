#include "inference/inference.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "Benchmark Example" << std::endl;
    std::cout << "=================" << std::endl;
    
    inference::initialize();
    
    // 配置
    const int warmup_iterations = 10;
    const int benchmark_iterations = 100;
    
    // 创建输入
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    inference::Tensor input(input_shape, inference::DataType::FLOAT32);
    input.fill(0.5f);
    
    // 创建引擎
    inference::EngineConfig config;
    config.num_threads = 4;
    config.enable_profiling = true;
    
    inference::InferenceEngine engine(config);
    
    // TODO: 加载模型
    // engine.loadModel("model.bin");
    
    // 预热
    std::cout << "Warming up..." << std::endl;
    engine.warmup(warmup_iterations);
    
    // 基准测试
    std::cout << "Running benchmark..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < benchmark_iterations; ++i) {
        // TODO: 执行推理
        // engine.infer({input});
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // 统计
    double avg_time = static_cast<double>(duration) / benchmark_iterations;
    double throughput = 1000.0 / avg_time; // 每秒推理次数
    
    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << "  Iterations: " << benchmark_iterations << std::endl;
    std::cout << "  Total time: " << duration << " ms" << std::endl;
    std::cout << "  Average time: " << avg_time << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput << " infer/s" << std::endl;
    
    inference::finalize();
    
    return 0;
}

