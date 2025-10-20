#!/bin/bash
# 生成 compile_commands.json 文件的脚本

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo "正在生成 compile_commands.json..."

# 创建 build 目录（如果不存在）
if [ ! -d "$BUILD_DIR" ]; then
    echo "创建构建目录: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# 运行 CMake 配置（会自动生成 compile_commands.json）
cd "$BUILD_DIR"
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 创建软链接到项目根目录
if [ -f "${BUILD_DIR}/compile_commands.json" ]; then
    echo "创建软链接: ${PROJECT_ROOT}/compile_commands.json -> ${BUILD_DIR}/compile_commands.json"
    ln -sf "${BUILD_DIR}/compile_commands.json" "${PROJECT_ROOT}/compile_commands.json"
    echo "✓ compile_commands.json 生成成功！"
else
    echo "✗ 错误: compile_commands.json 未生成"
    exit 1
fi

echo ""
echo "提示: compile_commands.json 已添加到 .gitignore，不会被提交到仓库"
echo "      其他开发者可以运行此脚本来生成自己的版本"

