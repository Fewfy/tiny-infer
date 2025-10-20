#include "inference/inference.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    
    std::cout << "Model Loading Example" << std::endl;
    std::cout << "=====================" << std::endl;
    
    inference::initialize();
    
    // 创建模型
    auto model = std::make_shared<inference::Model>();
    
    // 加载模型文件
    std::string model_path = argv[1];
    if (!model->load(model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    // 打印模型结构
    model->print();
    
    // 创建推理引擎
    inference::InferenceEngine engine;
    engine.setModel(model);
    
    // 优化模型
    engine.optimizeModel();
    
    // TODO: 执行推理...
    
    inference::finalize();
    
    return 0;
}

