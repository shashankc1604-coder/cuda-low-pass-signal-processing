# =========================================
# CUDA + Python Makefile
# =========================================

# Compiler
NVCC := nvcc

# Executable name
TARGET := cuda_fir_filter

# Source files
SRC := src/main.cu \
       src/fir_filter.cu \
       src/csv_loader.cu

# Flags
NVCC_FLAGS := -std=c++17 -O2
ARCH := -arch=sm_70

# Directories
INPUT_DIR   := data/input
OUTPUT_DIR  := data/output
RESULTS_DIR := results
LOGS_DIR    := logs
PY_DIR      := python_script

# Python
PYTHON := python3
PLOT_SCRIPT := $(PY_DIR)/plot_signals.py

# Libraries
LIBS := -lcudart

# =========================================
# Targets
# =========================================

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	@echo "Building CUDA executable..."
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(SRC) $(LIBS) -o $(TARGET)

run: build dirs
	@echo "Running CUDA FIR filter..."
	./$(TARGET)

plot: run
	@echo "Running Python plotting script..."
	$(PYTHON) $(PLOT_SCRIPT)

dirs:
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p $(RESULTS_DIR)
	@mkdir -p $(LOGS_DIR)

clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET)
	rm -rf $(RESULTS_DIR)/*
	rm -rf $(LOGS_DIR)/*

help:
	@echo "Available targets:"
	@echo "  make build   - Compile CUDA code"
	@echo "  make run     - Run CUDA filtering"
	@echo "  make plot    - Run CUDA + plotting"
	@echo "  make clean   - Remove outputs"
