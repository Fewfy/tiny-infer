#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace inference {

// 数据类型枚举
enum class DataType { FLOAT32, FLOAT16, INT32, INT8, UINT8 };

// 获取数据类型的字节大小
inline size_t getDataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return 4;
        case DataType::FLOAT16:
            return 2;
        case DataType::INT32:
            return 4;
        case DataType::INT8:
            return 1;
        case DataType::UINT8:
            return 1;
        default:
            return 0;
    }
}

// 张量类
class Tensor {
  public:
    Tensor() : data_(nullptr), dtype_(DataType::FLOAT32) {}

    Tensor(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32)
        : shape_(shape), dtype_(dtype) {
        allocate();
    }

    ~Tensor() { deallocate(); }

    // 拷贝构造

    Tensor(const Tensor& other) : shape_(other.shape_), dtype_(other.dtype_) {
        allocate();
        std::memcpy(data_, other.data_, totalSize());
    }

    /*
    Tensor(const Tensor& other)
        : shape_(other.shape_), dtype_(other.dtype_) {
        allocate();
        std::memcpy(data_, other.data_, totalSize());
    }
        */

    // 移动构造
    Tensor(Tensor&& other) noexcept
        : data_(other.data_), shape_(other.shape_), dtype_(other.dtype_) {
        other.data_ = nullptr;
    }

    // 赋值运算符
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            deallocate();
            shape_ = other.shape_;
            dtype_ = other.dtype_;
            allocate();
            std::memcpy(data_, other.data_, totalSize());
        }
        return *this;
    }

    // 获取形状
    const std::vector<int64_t>& shape() const { return shape_; }

    // 获取维度数
    size_t ndim() const { return shape_.size(); }

    // 获取总元素数
    int64_t numel() const {
        int64_t size = 1;
        for (auto dim : shape_) {
            size *= dim;
        }
        return size;
    }

    // 获取总字节数
    size_t totalSize() const { return numel() * getDataTypeSize(dtype_); }

    // 获取数据类型
    DataType dtype() const { return dtype_; }

    // 获取数据指针
    void* data() { return data_; }
    const void* data() const { return data_; }

    // 类型化访问
    template <typename T>
    T* data_ptr() {
        return static_cast<T*>(data_);
    }

    template <typename T>
    const T* data_ptr() const {
        return static_cast<const T*>(data_);
    }

    // 重塑张量
    void reshape(const std::vector<int64_t>& new_shape) {
        int64_t new_numel = 1;
        for (auto dim : new_shape) {
            new_numel *= dim;
        }
        if (new_numel != numel()) {
            throw std::runtime_error("Reshape: new shape must have same number of elements");
        }
        shape_ = new_shape;
    }

    // 填充数据
    template <typename T>
    void fill(T value) {
        T* ptr = data_ptr<T>();
        for (int64_t i = 0; i < numel(); ++i) {
            ptr[i] = value;
        }
    }

    // 从数据复制
    template <typename T>
    void copyFrom(const T* src) {
        std::memcpy(data_, src, totalSize());
    }

  private:
    void allocate() {
        if (numel() > 0) {
            data_ = ::operator new(totalSize());
        }
    }

    void deallocate() {
        if (data_) {
            ::operator delete(data_);
            data_ = nullptr;
        }
    }

    void* data_;
    std::vector<int64_t> shape_;
    DataType dtype_;
};

}  // namespace inference
