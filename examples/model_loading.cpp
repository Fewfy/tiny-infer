#include <iostream>

#include "inference/inference.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    std::cout << "Model Loading Example" << std::endl;
    std::cout << "=====================" << std::endl;

    inference::initialize();

    // Create model
    auto model = std::make_shared<inference::Model>();

    // Load model file
    std::string model_path = argv[1];
    if (!model->load(model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    // Print model structure
    model->print();

    // Create inference engine
    inference::InferenceEngine engine;
    engine.setModel(model);

    // Optimize model
    engine.optimizeModel();

    // TODO: Execute inference...

    inference::finalize();

    return 0;
}
