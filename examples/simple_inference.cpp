#include <iostream>
#include <vector>

#include "inference/inference.h"

int main() {
    std::cout << "Simple Inference Example" << std::endl;
    std::cout << "========================" << std::endl;

    // Initialize framework
    inference::initialize();

    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, 224, 224};  // NCHW format
    inference::Tensor input(input_shape, inference::DataType::FLOAT32);

    // TODO: Fill input data
    // input.fill(0.5f);

    // Create inference engine
    inference::EngineConfig config;
    config.num_threads = 4;
    config.enable_profiling = true;

    inference::InferenceEngine engine(config);

    // TODO: Load model
    // engine.loadModel("path/to/model.bin");

    // TODO: Execute inference
    // std::vector<inference::Tensor> outputs = engine.infer({input});

    // TODO: Process outputs
    // for (size_t i = 0; i < outputs.size(); ++i) {
    //     std::cout << "Output " << i << " shape: ";
    //     for (auto dim : outputs[i].shape()) {
    //         std::cout << dim << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Get profiling statistics
    auto profiling_info = engine.getProfilingInfo();
    std::cout << "Total inference time: " << profiling_info.total_time_ms << " ms" << std::endl;

    // Cleanup
    inference::finalize();

    return 0;
}
