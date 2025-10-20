#include <chrono>
#include <iostream>

#include "inference/inference.h"

int main() {
    std::cout << "Benchmark Example" << std::endl;
    std::cout << "=================" << std::endl;

    inference::initialize();

    // Configuration
    const int warmup_iterations = 10;
    const int benchmark_iterations = 100;

    // Create input
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    inference::Tensor input(input_shape, inference::DataType::FLOAT32);
    input.fill(0.5f);

    // Create engine
    inference::EngineConfig config;
    config.num_threads = 4;
    config.enable_profiling = true;

    inference::InferenceEngine engine(config);

    // TODO: Load model
    // engine.loadModel("model.bin");

    // Warmup
    std::cout << "Warming up..." << std::endl;
    engine.warmup(warmup_iterations);

    // Benchmark
    std::cout << "Running benchmark..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < benchmark_iterations; ++i) {
        // TODO: Execute inference
        // engine.infer({input});
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Statistics
    double avg_time = static_cast<double>(duration) / benchmark_iterations;
    double throughput = 1000.0 / avg_time;  // Inferences per second

    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << "  Iterations: " << benchmark_iterations << std::endl;
    std::cout << "  Total time: " << duration << " ms" << std::endl;
    std::cout << "  Average time: " << avg_time << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput << " infer/s" << std::endl;

    inference::finalize();

    return 0;
}
