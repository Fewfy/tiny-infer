# Inference Framework Makefile
# =============================

CXX = g++
CXXFLAGS = -std=c++2a -Wall -Wextra -Wpedantic -O3 -march=native -fPIC
LDFLAGS = -pthread

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
TESTS_DIR = tests
EXAMPLES_DIR = examples

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

TEST_SOURCES = $(wildcard $(TESTS_DIR)/*.cpp)
TEST_OBJECTS = $(TEST_SOURCES:$(TESTS_DIR)/%.cpp=$(BUILD_DIR)/test_%.o)
TEST_EXECUTABLES = $(TEST_SOURCES:$(TESTS_DIR)/%.cpp=$(BUILD_DIR)/test_%)

EXAMPLE_SOURCES = $(wildcard $(EXAMPLES_DIR)/*.cpp)
EXAMPLE_OBJECTS = $(EXAMPLE_SOURCES:$(EXAMPLES_DIR)/%.cpp=$(BUILD_DIR)/example_%.o)
EXAMPLE_EXECUTABLES = $(EXAMPLE_SOURCES:$(EXAMPLES_DIR)/%.cpp=$(BUILD_DIR)/%)

STATIC_LIB = $(BUILD_DIR)/libinference.a
SHARED_LIB = $(BUILD_DIR)/libinference.so

.PHONY: all clean test examples libs help install

all: libs test examples

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/test_%.o: $(TESTS_DIR)/%.cpp | $(BUILD_DIR)
	@echo "Compiling test $<"
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/example_%.o: $(EXAMPLES_DIR)/%.cpp | $(BUILD_DIR)
	@echo "Compiling example $<"
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(STATIC_LIB): $(OBJECTS) | $(BUILD_DIR)
	@echo "Creating static library $@"
	ar rcs $@ $^

$(SHARED_LIB): $(OBJECTS) | $(BUILD_DIR)
	@echo "Creating shared library $@"
	$(CXX) -shared -fPIC -o $@ $^ $(LDFLAGS)

libs: $(STATIC_LIB) $(SHARED_LIB)

test: $(TEST_EXECUTABLES)
	@echo "Running tests..."
	@export LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH; \
	for test in $(TEST_EXECUTABLES); do \
		echo "Running $$test"; \
		LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH ./$$test; \
	done

examples: $(EXAMPLE_EXECUTABLES)

$(BUILD_DIR)/test_%: $(BUILD_DIR)/test_%.o $(STATIC_LIB)
	@echo "Linking test $@"
	$(CXX) $< -L$(BUILD_DIR) -linference $(LDFLAGS) -o $@
	@echo "Test executable created: $@"

$(BUILD_DIR)/%: $(BUILD_DIR)/example_%.o $(STATIC_LIB)
	@echo "Linking example $@"
	$(CXX) $< -L$(BUILD_DIR) -linference $(LDFLAGS) -o $@

parallel:
	@echo "Building with parallel compilation..."
	$(MAKE) -j$(shell nproc) all

debug: CXXFLAGS = -std=c++2a -Wall -Wextra -Wpedantic -g -O0 -DDEBUG -fPIC
debug: clean all

release: CXXFLAGS = -std=c++2a -Wall -Wextra -Wpedantic -O3 -march=native -DNDEBUG -fPIC
release: clean all

test-all: test
	@echo "All tests completed"

test-tensor: $(BUILD_DIR)/test_tensor
	@echo "Running tensor tests..."
	./$(BUILD_DIR)/test_tensor

test-operators: $(BUILD_DIR)/test_operators
	@echo "Running operator tests..."
	./$(BUILD_DIR)/test_operators

run-simple: $(BUILD_DIR)/simple_inference
	@echo "Running simple inference example..."
	./$(BUILD_DIR)/simple_inference

run-benchmark: $(BUILD_DIR)/benchmark
	@echo "Running benchmark example..."
	./$(BUILD_DIR)/benchmark

install: libs
	@echo "Installing libraries..."
	@mkdir -p /usr/local/lib
	@mkdir -p /usr/local/include/inference
	cp $(STATIC_LIB) /usr/local/lib/
	cp $(SHARED_LIB) /usr/local/lib/
	cp -r $(INCLUDE_DIR)/inference/* /usr/local/include/inference/
	@echo "Installation completed"

clean:
	@echo "Cleaning build files..."
	rm -rf $(BUILD_DIR)
	@echo "Clean completed"

distclean: clean
	@echo "Distclean completed"

help:
	@echo "Inference Framework Makefile"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build everything (default)"
	@echo "  libs         - Build static and shared libraries"
	@echo "  test         - Build and run all tests"
	@echo "  examples     - Build all examples"
	@echo "  parallel     - Build with parallel compilation"
	@echo "  debug        - Build debug version"
	@echo "  release      - Build release version"
	@echo "  test-tensor  - Run tensor tests only"
	@echo "  test-operators - Run operator tests only"
	@echo "  run-simple   - Run simple inference example"
	@echo "  run-benchmark - Run benchmark example"
	@echo "  install      - Install libraries to system"
	@echo "  clean        - Remove build files"
	@echo "  distclean    - Remove all generated files"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make all              # Build everything"
	@echo "  make -j4 test         # Build and test with 4 threads"
	@echo "  make debug            # Build debug version"
	@echo "  make clean && make   # Clean rebuild"

info:
	@echo "Build Information:"
	@echo "  Compiler: $(CXX)"
	@echo "  Flags: $(CXXFLAGS)"
	@echo "  Sources: $(SOURCES)"
	@echo "  Objects: $(OBJECTS)"
	@echo "  Tests: $(TEST_EXECUTABLES)"
	@echo "  Examples: $(EXAMPLE_EXECUTABLES)"
