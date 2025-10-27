cmake_minimum_required(VERSION 3.15)
project(InferenceFramework VERSION 1.0.0 LANGUAGES CXX)

# Export compile commands for IDE code navigation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set compilation options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    # Performance optimization options
    add_compile_options(-O3 -march=native)
elseif(MSVC)
    add_compile_options(/W4)
    add_compile_options(/O2)
endif()

# Options
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_PROFILING "Enable profiling" OFF)

# Header directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Collect source files
file(GLOB_RECURSE SOURCES 
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Create library
add_library(inference SHARED ${SOURCES})
add_library(inference_static STATIC ${SOURCES})

# Set library properties
set_target_properties(inference PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER include/inference/inference.h
)

set_target_properties(inference_static PROPERTIES
    OUTPUT_NAME inference
)

# Include directories
target_include_directories(inference PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(inference_static PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link libraries
find_package(Threads REQUIRED)
target_link_libraries(inference PRIVATE Threads::Threads)
target_link_libraries(inference_static PRIVATE Threads::Threads)

# Compile definitions
if(ENABLE_PROFILING)
    target_compile_definitions(inference PRIVATE ENABLE_PROFILING)
    target_compile_definitions(inference_static PRIVATE ENABLE_PROFILING)
endif()

# Build examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install rules
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

# Export configuration
install(EXPORT InferenceFrameworkTargets
    FILE InferenceFrameworkTargets.cmake
    NAMESPACE InferenceFramework::
    DESTINATION lib/cmake/InferenceFramework
)

# Generate version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    InferenceFrameworkConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Print configuration information
message(STATUS "Inference Framework Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Enable Profiling: ${ENABLE_PROFILING}")

