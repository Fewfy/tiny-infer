#!/bin/bash

# Inference Framework Build Script

set -e

echo "========================================"
echo "  Inference Framework Build Script"
echo "========================================"
echo ""

# Create build directory
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# Configure CMake
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_TESTS=ON

# Compile
echo "Starting compilation..."
make -j$(nproc)

echo ""
echo "========================================"
echo "  Build completed!"
echo "========================================"
echo ""
echo "Executable locations:"
echo "  - Examples: ./build/examples/"
echo "  - Tests: ./build/tests/"
echo ""
echo "Run tests:"
echo "  cd build && ctest"
echo ""
