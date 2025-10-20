#pragma once

// 主头文件，包含所有公共API

#include "tensor.h"
#include "operator.h"
#include "model.h"
#include "engine.h"

namespace inference {

// 版本信息
constexpr const char* VERSION = "1.0.0";

// 初始化推理框架
void initialize();

// 清理资源
void finalize();

// 获取版本信息
const char* getVersion();

} // namespace inference

