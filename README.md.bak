# Inference Framework

一个轻量级、高性能的C++深度学习推理框架，支持多种神经网络算子和模型格式。

## 特性

- 轻量级设计，易于集成
- 支持多种数据类型（FP32、FP16、INT8等）
- 模块化算子系统，易于扩展
- 多线程支持
- 内置性能分析工具
- 跨平台支持（Linux、Windows、macOS）

## 项目结构

```
inference-framework/
├── include/inference/      # 公共头文件
│   ├── tensor.h           # 张量类
│   ├── operator.h         # 算子基类和内置算子
│   ├── model.h            # 模型类和计算图
│   ├── engine.h           # 推理引擎
│   └── inference.h        # 主头文件
├── src/                   # 源代码实现
│   ├── tensor.cpp
│   ├── operator.cpp
│   ├── model.cpp
│   ├── engine.cpp
│   └── inference.cpp
├── examples/              # 示例代码
│   ├── simple_inference.cpp
│   ├── model_loading.cpp
│   ├── benchmark.cpp
│   └── custom_operator.cpp
├── tests/                 # 单元测试
│   ├── test_tensor.cpp
│   └── test_operators.cpp
├── models/                # 模型文件目录
└── CMakeLists.txt         # CMake构建配置
```

## 依赖要求

- C++17或更高版本
- CMake 3.15+
- 支持的编译器：
  - GCC 7+
  - Clang 5+
  - MSVC 2017+

## 构建

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

### CMake选项

- `BUILD_EXAMPLES`: 构建示例程序（默认：ON）
- `BUILD_TESTS`: 构建测试程序（默认：ON）
- `ENABLE_PROFILING`: 启用性能分析（默认：OFF）

示例：
```bash
cmake -DBUILD_EXAMPLES=ON -DENABLE_PROFILING=ON ..
```

## 快速开始

### 基本推理

```cpp
#include "inference/inference.h"

int main() {
    // 初始化框架
    inference::initialize();
    
    // 创建输入张量
    std::vector<int64_t> shape = {1, 3, 224, 224};
    inference::Tensor input(shape, inference::DataType::FLOAT32);
    
    // 创建推理引擎
    inference::EngineConfig config;
    config.num_threads = 4;
    inference::InferenceEngine engine(config);
    
    // 加载模型
    engine.loadModel("model.bin");
    
    // 执行推理
    auto outputs = engine.infer({input});
    
    // 清理
    inference::finalize();
    return 0;
}
```

### 自定义算子

```cpp
class MyCustomOperator : public inference::Operator {
public:
    MyCustomOperator() : Operator("MyCustomOp") {}
    
    std::vector<inference::Tensor> forward(
        const std::vector<inference::Tensor>& inputs) override {
        // 实现你的算子逻辑
        // ...
        return outputs;
    }
};
```

## 支持的算子

当前支持以下算子（TODO表示需要实现）：

- ✓ Tensor（张量基础操作）
- TODO Conv2d（二维卷积）
- TODO MaxPool2d（最大池化）
- TODO Linear（全连接层）
- TODO ReLU（激活函数）
- TODO Softmax（归一化）
- TODO MatMul（矩阵乘法）
- TODO Add（加法）

更多算子正在开发中...

## API文档

### Tensor（张量）

张量是框架的基本数据结构。

```cpp
// 创建张量
Tensor tensor({2, 3, 4}, DataType::FLOAT32);

// 获取属性
int64_t num_elements = tensor.numel();
size_t dimensions = tensor.ndim();
const auto& shape = tensor.shape();

// 数据操作
tensor.fill(0.0f);
tensor.reshape({4, 6});
float* data = tensor.data_ptr<float>();
```

### Model（模型）

模型类管理计算图和推理流程。

```cpp
Model model;

// 添加算子到计算图
model.addOperator(op, input_indices, output_indices);

// 加载/保存模型
model.load("model.bin");
model.save("model.bin");

// 执行推理
model.forward();
```

### InferenceEngine（推理引擎）

推理引擎提供高级推理接口。

```cpp
EngineConfig config;
config.num_threads = 4;
config.enable_profiling = true;

InferenceEngine engine(config);
engine.loadModel("model.bin");
auto outputs = engine.infer(inputs);
```

## 性能优化

### 多线程

```cpp
EngineConfig config;
config.num_threads = std::thread::hardware_concurrency();
```

### FP16推理

```cpp
EngineConfig config;
config.use_fp16 = true;  // 使用半精度浮点
```

### 预热

```cpp
engine.warmup(10);  // 预热10次迭代
```

## 示例程序

项目包含多个示例程序：

1. `simple_inference` - 基本推理示例
2. `model_loading` - 模型加载和结构打印
3. `benchmark` - 性能基准测试
4. `custom_operator` - 自定义算子示例

运行示例：
```bash
./build/examples/simple_inference
./build/examples/benchmark
```

## 测试

运行测试：
```bash
cd build
ctest --verbose
# 或者
./tests/test_tensor
./tests/test_operators
```

## 开发指南

### 开发环境设置

#### 生成 compile_commands.json（用于IDE代码补全和分析）

项目提供了一个脚本来生成 `compile_commands.json` 文件，该文件被 clangd、VSCode 等工具用于代码补全和智能分析：

```bash
./generate_compile_commands.sh
```

或者手动生成：

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ln -sf build/compile_commands.json ../compile_commands.json
```

**注意**：`compile_commands.json` 包含本地路径信息，已添加到 `.gitignore`，不会提交到仓库。每个开发者需要在自己的环境中生成此文件。

#### VSCode 配置

项目包含推荐的 VSCode 配置（`.vscode/` 目录），包括：
- C++ 代码补全和智能提示（使用 clangd）
- 代码格式化设置
- 推荐的扩展插件

### 添加新算子

1. 在 `operator.h` 中声明算子类
2. 在 `operator.cpp` 中实现 `forward()` 方法
3. 添加相应的单元测试

### 代码风格

- 使用4个空格缩进
- 类名使用大驼峰命名
- 函数和变量使用小驼峰命名
- 私有成员变量以下划线结尾

## 待办事项

- [ ] 实现所有基础算子
- [ ] 添加ONNX模型格式支持
- [ ] GPU加速支持（CUDA）
- [ ] 量化支持（INT8）
- [ ] 算子融合优化
- [ ] 更完整的文档和教程
- [ ] 更多示例程序

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题或建议，请通过Issue联系。

