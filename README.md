# GPU-Accelerated Butterworth Signal Filtering (CUDA Project)

## Overview

This project implements a **GPU-accelerated signal processing pipeline**
using **NVIDIA CUDA** to apply a **Butterworth low-pass filter** to
time-series data stored in CSV files. The goal of the project is to
demonstrate practical GPU programming concepts such as:

-   Host--device memory transfers
-   CUDA kernel launches
-   GPU-based digital signal processing
-   End-to-end automation with logging and visualization
------------------------------------------------------------------------

## Motivation

High-sampling-rate sensor data (vibration, industrial sensing, etc.) often requires efficient filtering. This
project explores how **GPU acceleration** can be applied to digital
signal processing workloads by offloading filtering computations to CUDA
while keeping I/O and visualization on the CPU.

------------------------------------------------------------------------

## Project Features

-   CUDA-based Butterworth low-pass filter
-   Host ↔ Device memory management
-   Batch processing of multiple CSV files
-   Unified logging (CUDA + Python)
-   Automated plotting pipeline
-   One-command execution using Makefile

------------------------------------------------------------------------

## Directory Structure

    cuda-low-pass-signal-processing/
    ├── bin/                    # Compiled CUDA binary
    ├── log/                    # Unified execution logs
    │   └── run.log
    ├── src/
    │   ├── main.cu             # CUDA application entry point
    │   ├── butterworth.cu      # GPU Butterworth filter implementation
    │   ├── butterworth.h
    │   ├── csv_io.cu           # CSV read/write utilities
    │   ├── csv_io.h
    │   ├── cuda_utils.h        # CUDA helpers
    │   └── plotter_tool.py     # Python plotting tool
    ├── data/
    │   ├── input/              # Input CSV files
    │   ├── output/             # Filtered outputs
    │   └── plots/              # Generated plots
    ├── Makefile
    └── README.md

------------------------------------------------------------------------

## Signal Processing Details

-   **Filter Type:** Butterworth Low-Pass
-   **Order:** 2nd-order (biquad)
-   **Sampling Frequency:** 2500 Hz
-   **Cutoff Frequency:** Configurable (default: 50 Hz)

> Note: A full 3rd-order Butterworth filter can be implemented by
> cascading first- and second-order sections. The current implementation
> focuses on correctness and clarity.

------------------------------------------------------------------------

## CUDA Design Rationale

IIR filters are recursive in nature and cannot be fully parallelized
across samples. In this project:

-   One CUDA thread processes one complete signal sequentially
-   Parallelism is achieved across independent input files/signals

This approach reflects standard industry practices for GPU-based IIR
filtering.

------------------------------------------------------------------------

## Input and Output Format

### Input CSV

    0.0123
    0.0187
    -0.0045
    ...

### Output CSV

    original_signal,filtered_signal
    0.0123,0.0091
    0.0187,0.0104
    -0.0045,0.0062
    ...

------------------------------------------------------------------------

## Build Instructions

### Prerequisites

-   NVIDIA GPU with CUDA support
-   CUDA Toolkit (nvcc)
-   Python 3
-   Python packages: numpy, pandas, matplotlib

Install Python dependencies:

    pip install numpy pandas matplotlib

------------------------------------------------------------------------

### Build

    make build

------------------------------------------------------------------------

### Run Full Pipeline (CUDA + Plotting)

    make run

This will: 1. Build the CUDA binary 2. Process all CSV files in
`data/input/` 3. Write filtered outputs to `data/output/` 4. Generate
plots in `data/plots/` 5. Log all activity to `log/run.log`

------------------------------------------------------------------------

## Visualization

For each input CSV, the following plots are generated: - Original
signal - CUDA-filtered signal

Plots are saved automatically as PNG files.

------------------------------------------------------------------------

## Logging

All CUDA and Python stages write to a unified log file:

    log/run.log

------------------------------------------------------------------------

## Validation

-   Visual comparison of original vs filtered signals
-   Optional comparison with Python reference filters
-   Frequency-domain inspection using FFT (optional)

------------------------------------------------------------------------

## Limitations and Future Work

-   Extend to full 3rd-order Butterworth cascade
-   Add high-pass filter support
-   Implement CUDA streams
-   Add quantitative validation metrics (RMSE)
-   Add CLI configuration

------------------------------------------------------------------------

## Author

**Shashank C**
CUDA Project
