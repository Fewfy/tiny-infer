# API参考文档

## 命名空间

所有API都在 `inference` 命名空间下。

---

## Tensor 类

### 构造函数

```cpp
Tensor()
Tensor(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32)
```

创建张量对象。

**参数：**
- `shape` - 张量形状
- `dtype` - 数据类型（默认FLOAT32）

**示例：**
```cpp
inference::Tensor tensor({2, 3, 4}, inference::DataType::FLOAT32);
```

### 成员函数

#### shape()
```cpp
const std::vector<int64_t>& shape() const
```
获取张量形状。

#### ndim()
```cpp
size_t ndim() const
```
获取维度数。

#### numel()
```cpp
int64_t numel() const
```
获取总元素数。

#### dtype()
```cpp
DataType dtype() const
```
获取数据类型。

#### data()
```cpp
void* data()
const void* data() const
```
获取原始数据指针。

#### data_ptr<T>()
```cpp
template<typename T>
T* data_ptr()
```
获取类型化数据指针。

#### reshape()
```cpp
void reshape(const std::vector<int64_t>& new_shape)
```
重塑张量（元素总数必须相同）。

#### fill()
```cpp
template<typename T>
void fill(T value)
```
用指定值填充张量。

#### copyFrom()
```cpp
template<typename T>
void copyFrom(const T* src)
```
从外部数据复制。

---

## Operator 类

### 基类

```cpp
class Operator {
public:
    virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs) = 0;
    const std::string& name() const;
    void setAttribute(const std::string& key, const Attribute& value);
    template<typename T>
    T getAttribute(const std::string& key, const T& default_value) const;
};
```

### 内置算子

#### Conv2dOperator
```cpp
class Conv2dOperator : public Operator
```

二维卷积算子。

**属性：**
- `stride` (int) - 步长
- `padding` (int) - 填充
- `dilation` (int) - 膨胀系数

**输入：**
- inputs[0]: 特征图 [N, C_in, H, W]
- inputs[1]: 卷积核 [C_out, C_in, K_H, K_W]
- inputs[2]: 偏置 [C_out]（可选）

**输出：**
- outputs[0]: 输出特征图 [N, C_out, H_out, W_out]

#### LinearOperator
```cpp
class LinearOperator : public Operator
```

全连接层。

**输入：**
- inputs[0]: 输入 [N, in_features]
- inputs[1]: 权重 [out_features, in_features]
- inputs[2]: 偏置 [out_features]（可选）

**输出：**
- outputs[0]: 输出 [N, out_features]

#### ReluOperator
```cpp
class ReluOperator : public Operator
```

ReLU激活函数。

**输入：**
- inputs[0]: 输入张量

**输出：**
- outputs[0]: 激活后的张量

#### MatMulOperator
```cpp
class MatMulOperator : public Operator
```

矩阵乘法。

**输入：**
- inputs[0]: 左矩阵 [M, K]
- inputs[1]: 右矩阵 [K, N]

**输出：**
- outputs[0]: 结果矩阵 [M, N]

---

## Model 类

### 构造函数

```cpp
Model()
```

创建空模型。

### 成员函数

#### addOperator()
```cpp
void addOperator(std::shared_ptr<Operator> op,
                const std::vector<int>& inputs,
                const std::vector<int>& outputs)
```

添加算子到计算图。

**参数：**
- `op` - 算子指针
- `inputs` - 输入张量索引列表
- `outputs` - 输出张量索引列表

#### setInput()
```cpp
void setInput(const std::string& name, const Tensor& tensor)
```

设置模型输入。

#### getOutput()
```cpp
Tensor getOutput(const std::string& name) const
```

获取模型输出。

#### forward()
```cpp
void forward()
```

执行前向推理。

#### load()
```cpp
bool load(const std::string& path)
```

从文件加载模型。

#### save()
```cpp
bool save(const std::string& path) const
```

保存模型到文件。

#### print()
```cpp
void print() const
```

打印模型结构。

---

## InferenceEngine 类

### 构造函数

```cpp
InferenceEngine()
InferenceEngine(const EngineConfig& config)
```

创建推理引擎。

### EngineConfig 结构

```cpp
struct EngineConfig {
    int num_threads = 1;
    bool use_fp16 = false;
    bool enable_profiling = false;
    size_t workspace_size = 1024 * 1024 * 256;
};
```

**字段：**
- `num_threads` - 线程数
- `use_fp16` - 是否使用FP16
- `enable_profiling` - 是否启用性能分析
- `workspace_size` - 工作空间大小（字节）

### 成员函数

#### loadModel()
```cpp
bool loadModel(const std::string& model_path)
```

加载模型文件。

#### setModel()
```cpp
void setModel(std::shared_ptr<Model> model)
```

设置模型对象。

#### infer()
```cpp
std::vector<Tensor> infer(const std::vector<Tensor>& inputs)
```

执行同步推理。

#### inferAsync()
```cpp
void inferAsync(const std::vector<Tensor>& inputs)
```

启动异步推理。

#### getResults()
```cpp
std::vector<Tensor> getResults()
```

获取异步推理结果。

#### warmup()
```cpp
void warmup(int iterations = 10)
```

预热引擎。

#### optimizeModel()
```cpp
void optimizeModel()
```

优化模型（算子融合、常量折叠等）。

#### getProfilingInfo()
```cpp
ProfilingInfo getProfilingInfo() const
```

获取性能统计信息。

---

## 全局函数

### initialize()
```cpp
void initialize()
```

初始化推理框架。

### finalize()
```cpp
void finalize()
```

清理框架资源。

### getVersion()
```cpp
const char* getVersion()
```

获取框架版本。

---

## 枚举类型

### DataType
```cpp
enum class DataType {
    FLOAT32,  // 32位浮点
    FLOAT16,  // 16位浮点
    INT32,    // 32位整数
    INT8,     // 8位整数
    UINT8     // 8位无符号整数
};
```

---

## 使用示例

### 完整推理流程

```cpp
#include "inference/inference.h"

int main() {
    // 1. 初始化
    inference::initialize();
    
    // 2. 配置引擎
    inference::EngineConfig config;
    config.num_threads = 4;
    config.enable_profiling = true;
    
    // 3. 创建引擎
    inference::InferenceEngine engine(config);
    
    // 4. 加载模型
    engine.loadModel("model.bin");
    
    // 5. 准备输入
    inference::Tensor input({1, 3, 224, 224}, inference::DataType::FLOAT32);
    input.fill(0.5f);
    
    // 6. 执行推理
    auto outputs = engine.infer({input});
    
    // 7. 处理输出
    for (const auto& output : outputs) {
        std::cout << "Output shape: ";
        for (auto dim : output.shape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
    
    // 8. 清理
    inference::finalize();
    
    return 0;
}
```

