#pragma once

// Main header file containing all public APIs

#include "engine.h"
#include "model.h"
#include "operator.h"
#include "tensor.h"

namespace inference {

// Version information
constexpr const char* VERSION = "1.0.0";

// Initialize inference framework
void initialize();

// Cleanup resources
void finalize();

// Get version information
const char* getVersion();

}  // namespace inference
