#include <iostream>

#include "inference/inference.h"

// Custom operator example
class CustomActivationOperator : public inference::Operator {
  public:
    CustomActivationOperator() : Operator("CustomActivation") {}

    std::vector<inference::Tensor> forward(const std::vector<inference::Tensor>& inputs) override {
        // TODO: Implement custom activation function
        // Example: output = x * sigmoid(x) (Swish activation)

        std::vector<inference::Tensor> outputs;

        if (inputs.empty()) {
            throw std::runtime_error("CustomActivation requires at least one input");
        }

        const auto& input = inputs[0];
        inference::Tensor output(input.shape(), input.dtype());

        // TODO: Implement specific activation function computation
        // float* in_ptr = input.data_ptr<float>();
        // float* out_ptr = output.data_ptr<float>();
        // for (int64_t i = 0; i < input.numel(); ++i) {
        //     float sigmoid = 1.0f / (1.0f + std::exp(-in_ptr[i]));
        //     out_ptr[i] = in_ptr[i] * sigmoid;
        // }

        outputs.push_back(output);
        return outputs;
    }
};

int main() {
    std::cout << "Custom Operator Example" << std::endl;
    std::cout << "=======================" << std::endl;

    inference::initialize();

    // Create input
    std::vector<int64_t> shape = {2, 4};
    inference::Tensor input(shape, inference::DataType::FLOAT32);

    // TODO: Fill test data
    // float test_data[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    // input.copyFrom(test_data);

    // Create custom operator
    auto custom_op = std::make_shared<CustomActivationOperator>();

    // Execute operator
    // auto outputs = custom_op->forward({input});

    // TODO: Print results

    inference::finalize();

    return 0;
}
