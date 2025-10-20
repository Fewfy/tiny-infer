cmake_minimum_required(VERSION 3.15)
project(InferenceFramework VERSION 1.0.0 LANGUAGES CXX)

# 导出编译命令，用于IDE代码跳转
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    # 性能优化选项
    add_compile_options(-O3 -march=native)
elseif(MSVC)
    add_compile_options(/W4)
    add_compile_options(/O2)
endif()

# 选项
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_PROFILING "Enable profiling" OFF)

# 头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 收集源文件
file(GLOB_RECURSE SOURCES 
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# 创建库
add_library(inference SHARED ${SOURCES})
add_library(inference_static STATIC ${SOURCES})

# 设置库属性
set_target_properties(inference PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/inference/inference.h
)

set_target_properties(inference_static PROPERTIES
    OUTPUT_NAME inference
)

# 包含目录
target_include_directories(inference PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(inference_static PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 链接库
find_package(Threads REQUIRED)
target_link_libraries(inference PRIVATE Threads::Threads)
target_link_libraries(inference_static PRIVATE Threads::Threads)

# 编译定义
if(ENABLE_PROFILING)
    target_compile_definitions(inference PRIVATE ENABLE_PROFILING)
    target_compile_definitions(inference_static PRIVATE ENABLE_PROFILING)
endif()

# 构建示例
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 构建测试
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 安装规则
install(TARGETS inference inference_static
    EXPORT InferenceFrameworkTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/inference
)

install(DIRECTORY include/inference
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 导出配置
install(EXPORT InferenceFrameworkTargets
    FILE InferenceFrameworkTargets.cmake
    NAMESPACE InferenceFramework::
    DESTINATION lib/cmake/InferenceFramework
)

# 生成版本文件
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    InferenceFrameworkConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# 打印配置信息
message(STATUS "Inference Framework Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Enable Profiling: ${ENABLE_PROFILING}")

