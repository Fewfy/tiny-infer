#!/bin/bash

# 推理框架构建脚本

set -e

echo "========================================"
echo "  Inference Framework 构建脚本"
echo "========================================"
echo ""

# 创建构建目录
if [ ! -d "build" ]; then
    echo "创建构建目录..."
    mkdir build
fi

cd build

# 配置CMake
echo "配置CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_TESTS=ON

# 编译
echo "开始编译..."
make -j$(nproc)

echo ""
echo "========================================"
echo "  编译完成！"
echo "========================================"
echo ""
echo "可执行文件位置："
echo "  - 示例: ./build/examples/"
echo "  - 测试: ./build/tests/"
echo ""
echo "运行测试："
echo "  cd build && ctest"
echo ""

