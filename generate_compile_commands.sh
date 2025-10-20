#!/bin/bash
# Script to generate compile_commands.json file

set -e

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo "Generating compile_commands.json..."

# Create build directory (if it doesn't exist)
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Run CMake configuration (will automatically generate compile_commands.json)
cd "$BUILD_DIR"
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Create symbolic link to project root
if [ -f "${BUILD_DIR}/compile_commands.json" ]; then
    echo "Creating symbolic link: ${PROJECT_ROOT}/compile_commands.json -> ${BUILD_DIR}/compile_commands.json"
    ln -sf "${BUILD_DIR}/compile_commands.json" "${PROJECT_ROOT}/compile_commands.json"
    echo "✓ compile_commands.json generated successfully!"
else
    echo "✗ Error: compile_commands.json not generated"
    exit 1
fi

echo ""
echo "Note: compile_commands.json is added to .gitignore and will not be committed to the repository"
echo "      Other developers can run this script to generate their own version"
