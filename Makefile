# ============================
# CUDA Build Configuration
# ============================
NVCC       = nvcc
CXXFLAGS   = -O2 -std=c++17

TARGET     = bin/butterworth_gpu

SRCS = \
	src/main.cu \
	src/butterworth.cu \
	src/csv_io.cu

PYTHON     = python3
PLOT_SCRIPT = src/plotter_tool.py

# ============================
# Default target
# ============================
all: clean build

# ============================
# Build CUDA binary
# ============================
build:
	mkdir -p bin
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# ============================
# Run full pipeline
# ============================
run: build
	@echo "=== Running GPU Butterworth Filter ==="
	./$(TARGET)
	@echo "=== Generating plots ==="
	$(PYTHON) $(PLOT_SCRIPT)

# ============================
# Clean
# ============================
clean:
	rm -rf bin/*
