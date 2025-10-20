# Inference Framework

A lightweight, high-performance C++ deep learning inference framework supporting various neural network operators and model formats.

## Features

- Lightweight design, easy to integrate
- Support for multiple data types (FP32, FP16, INT8, etc.)
- Modular operator system, easy to extend
- Multi-threading support
- Built-in performance profiling tools
- Cross-platform support (Linux, Windows, macOS)

## Project Structure

```
inference-framework/
├── include/inference/      # Public header files
│   ├── tensor.h           # Tensor class
│   ├── operator.h         # Operator base class and built-in operators
│   ├── model.h            # Model class and computation graph
│   ├── engine.h           # Inference engine
│   └── inference.h        # Main header file
├── src/                   # Source code implementation
│   ├── tensor.cpp
│   ├── operator.cpp
│   ├── model.cpp
│   ├── engine.cpp
│   └── inference.cpp
├── examples/              # Example code
│   ├── simple_inference.cpp
│   ├── model_loading.cpp
│   ├── benchmark.cpp
│   └── custom_operator.cpp
├── tests/                 # Unit tests
│   ├── test_tensor.cpp
│   └── test_operators.cpp
├── models/                # Model files directory
└── CMakeLists.txt         # CMake build configuration
```

## Requirements

- C++17 or higher
- CMake 3.15+
- Supported compilers:
  - GCC 7+
  - Clang 5+
  - MSVC 2017+

## Build

### Linux/macOS

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Windows

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### CMake Options

- `BUILD_EXAMPLES`: Build example programs (default: ON)
- `BUILD_TESTS`: Build test programs (default: ON)
- `ENABLE_PROFILING`: Enable profiling (default: OFF)

Example:
```bash
cmake -DBUILD_EXAMPLES=ON -DENABLE_PROFILING=ON ..
```

## Quick Start

### Basic Inference

```cpp
#include "inference/inference.h"

int main() {
    // Initialize framework
    inference::initialize();
    
    // Create input tensor
    std::vector<int64_t> shape = {1, 3, 224, 224};
    inference::Tensor input(shape, inference::DataType::FLOAT32);
    
    // Create inference engine
    inference::EngineConfig config;
    config.num_threads = 4;
    inference::InferenceEngine engine(config);
    
    // Load model
    engine.loadModel("model.bin");
    
    // Execute inference
    auto outputs = engine.infer({input});
    
    // Cleanup
    inference::finalize();
    return 0;
}
```

### Custom Operator

```cpp
class MyCustomOperator : public inference::Operator {
public:
    MyCustomOperator() : Operator("MyCustomOp") {}
    
    std::vector<inference::Tensor> forward(
        const std::vector<inference::Tensor>& inputs) override {
        // Implement your operator logic
        // ...
        return outputs;
    }
};
```

## Supported Operators

Currently supported operators (TODO indicates pending implementation):

- ✓ Tensor (basic tensor operations)
- TODO Conv2d (2D convolution)
- TODO MaxPool2d (max pooling)
- TODO Linear (fully connected layer)
- TODO ReLU (activation function)
- TODO Softmax (normalization)
- TODO MatMul (matrix multiplication)
- TODO Add (addition)

More operators are under development...

## API Documentation

### Tensor

Tensor is the fundamental data structure of the framework.

```cpp
// Create tensor
Tensor tensor({2, 3, 4}, DataType::FLOAT32);

// Get properties
int64_t num_elements = tensor.numel();
size_t dimensions = tensor.ndim();
const auto& shape = tensor.shape();

// Data operations
tensor.fill(0.0f);
tensor.reshape({4, 6});
float* data = tensor.data_ptr<float>();
```

### Model

Model class manages computation graph and inference workflow.

```cpp
Model model;

// Add operator to computation graph
model.addOperator(op, input_indices, output_indices);

// Load/save model
model.load("model.bin");
model.save("model.bin");

// Execute inference
model.forward();
```

### InferenceEngine

Inference engine provides high-level inference interface.

```cpp
EngineConfig config;
config.num_threads = 4;
config.enable_profiling = true;

InferenceEngine engine(config);
engine.loadModel("model.bin");
auto outputs = engine.infer(inputs);
```

## Performance Optimization

### Multi-threading

```cpp
EngineConfig config;
config.num_threads = std::thread::hardware_concurrency();
```

### FP16 Inference

```cpp
EngineConfig config;
config.use_fp16 = true;  // Use half-precision floating point
```

### Warmup

```cpp
engine.warmup(10);  // Warmup for 10 iterations
```

## Example Programs

The project includes multiple example programs:

1. `simple_inference` - Basic inference example
2. `model_loading` - Model loading and structure printing
3. `benchmark` - Performance benchmark
4. `custom_operator` - Custom operator example

Run examples:
```bash
./build/examples/simple_inference
./build/examples/benchmark
```

## Testing

Run tests:
```bash
cd build
ctest --verbose
# or
./tests/test_tensor
./tests/test_operators
```

## Development Guide

### Development Environment Setup

#### Generate compile_commands.json (for IDE code completion and analysis)

The project provides a script to generate the `compile_commands.json` file, which is used by tools like clangd and VSCode for code completion and intelligent analysis:

```bash
./generate_compile_commands.sh
```

Or generate manually:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ln -sf build/compile_commands.json ../compile_commands.json
```

**Note**: `compile_commands.json` contains local path information and has been added to `.gitignore`, so it won't be committed to the repository. Each developer needs to generate this file in their own environment.

#### VSCode Configuration

The project includes recommended VSCode configuration (`.vscode/` directory), including:
- C++ code completion and intelligent hints (using clangd)
- Code formatting settings
- Recommended extensions

### Adding New Operators

1. Declare operator class in `operator.h`
2. Implement `forward()` method in `operator.cpp`
3. Add corresponding unit tests

### Code Style

- Use 4 spaces for indentation
- Class names use PascalCase
- Functions and variables use camelCase
- Private member variables end with underscore

## TODO List

- [ ] Implement all basic operators
- [ ] Add ONNX model format support
- [ ] GPU acceleration support (CUDA)
- [ ] Quantization support (INT8)
- [ ] Operator fusion optimization
- [ ] More complete documentation and tutorials
- [ ] More example programs

## License

MIT License

## Contributing

Issues and Pull Requests are welcome!

## Contact

For questions or suggestions, please contact via Issues.

