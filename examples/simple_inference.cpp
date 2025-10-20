#include "inference/inference.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Simple Inference Example" << std::endl;
    std::cout << "========================" << std::endl;
    
    // 初始化框架
    inference::initialize();
    
    // 创建输入张量
    std::vector<int64_t> input_shape = {1, 3, 224, 224}; // NCHW格式
    inference::Tensor input(input_shape, inference::DataType::FLOAT32);
    
    // TODO: 填充输入数据
    // input.fill(0.5f);
    
    // 创建推理引擎
    inference::EngineConfig config;
    config.num_threads = 4;
    config.enable_profiling = true;
    
    inference::InferenceEngine engine(config);
    
    // TODO: 加载模型
    // engine.loadModel("path/to/model.bin");
    
    // TODO: 执行推理
    // std::vector<inference::Tensor> outputs = engine.infer({input});
    
    // TODO: 处理输出
    // for (size_t i = 0; i < outputs.size(); ++i) {
    //     std::cout << "Output " << i << " shape: ";
    //     for (auto dim : outputs[i].shape()) {
    //         std::cout << dim << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // 获取性能统计
    auto profiling_info = engine.getProfilingInfo();
    std::cout << "Total inference time: " << profiling_info.total_time_ms << " ms" << std::endl;
    
    // 清理
    inference::finalize();
    
    return 0;
}

