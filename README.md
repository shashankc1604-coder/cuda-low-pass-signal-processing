# CUDA-Accelerated Low-Pass Filtering of High-Rate Sensor Signals

## Overview

This project implements a **GPU-accelerated digital signal processing pipeline** using **CUDA** to apply a **15 Hz low-pass FIR filter** to high-frequency sensor signals sampled at **2500 Hz**.  
The application processes **hundreds of CSV-based time-series signals**, performs filtering entirely on the GPU using custom CUDA kernels, and generates visual proof of filtering through plotted results.

The project is designed to meet the requirements of the **“CUDA at Scale for the Enterprise” course project**, demonstrating real GPU computation, scalability, and reproducible execution.

---

## Key Features

- Custom CUDA FIR low-pass filter kernel  
- Batch processing of large volumes of signal data (hundreds of CSV files)  
- 15 Hz cutoff frequency FIR filter (Fs = 2500 Hz)  
- Clear CPU–GPU separation  
- Detailed execution logging  
- Automatic raw vs filtered signal plots  
- One-command build and run using `Makefile`  

---

## Signal Processing Details

- **Sampling Rate:** 2500 Hz  
- **Filter Type:** FIR Low-Pass Filter  
- **Cutoff Frequency:** 15 Hz  
- **Filter Length:** 127 taps  
- **Design Method:** Hamming-windowed sinc  
- **Execution:** GPU (CUDA kernel)  

Each output sample is computed in parallel on the GPU using:

```
y[n] = Σ h[k] · x[n − k]
```

Where:
- `x[n]` = input sensor signal  
- `h[k]` = FIR coefficients  
- `y[n]` = filtered output  

---

## Project Structure

```
cuda-low-pass-signal-processing/
├── src/
│   ├── main.cu            # Application orchestration + logging
│   ├── fir_filter.cu      # CUDA FIR kernel
│   └── csv_loader.cu      # CSV input/output (CPU)
│
├── python_script/
│   └── plot_signals.py    # Raw vs filtered signal plots
│
├── data/
│   ├── input/             # Input CSV files (sensor data)
│   └── output/            # Filtered CSV outputs
│
├── results/               # Generated PNG plots
├── logs/                  # Execution logs
├── Makefile
└── README.md
```

---

## Input Data Format

Input CSV files must contain a column named **`sensor`**:

```csv
timestamp,sensor
0.0000,12.34
0.0004,12.36
0.0008,12.39
```

- One CSV file corresponds to one sensor signal  
- Signals are sampled at **2500 Hz**  
- Hundreds of CSV files can be processed in a single run  

---

## Build & Run Instructions

### Prerequisites

- NVIDIA GPU with CUDA support  
- CUDA Toolkit installed  
- Python 3 with the following libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`

---

### Build the CUDA application

```bash
make build
```

---

### Run GPU filtering

```bash
make run
```

This step:
- Loads CSV files from `data/input`
- Executes FIR low-pass filtering on the GPU
- Writes filtered signals to `data/output`
- Logs execution details to `logs/`

---

### Generate plots (recommended)

```bash
make plot
```

This will:
- Run CUDA filtering
- Generate **PNG plots** in the `results/` directory
- Each plot contains two subplots:
  - Raw sensor signal (top)
  - Filtered sensor signal (bottom)

---

## Proof of Execution

The repository includes the following proof artifacts:

- CUDA execution logs (`logs/*.log`)
- Filtered output CSV files (`data/output/*.csv`)
- PNG plots showing raw vs filtered signals (`results/*.png`)
- Explicit CUDA kernel implementation (`fir_filter.cu`)

These artifacts clearly demonstrate GPU-based computation and correct execution.

---

## Logging

Execution details are logged to:

```
logs/cuda_fir_filter.log
```

Example log entries:

```
[2026-01-02 10:41:02] Processing file: data/input/sensor_001.csv
[2026-01-02 10:41:02] CUDA FIR kernel executed
[2026-01-02 10:41:02] Saved filtered CSV
```

---

## Why CUDA?

- FIR filtering is highly parallelizable  
- Each output sample is computed independently  
- GPU acceleration enables efficient processing of large signal datasets  
- Demonstrates practical, real-world CUDA usage beyond toy examples  

---

## Assignment Compliance

| Requirement            | Status |
|------------------------|--------|
| GPU computation        | ✅ Custom CUDA kernel |
| Large dataset          | ✅ Hundreds of CSV signals |
| Signal processing      | ✅ FIR low-pass filtering |
| Proof of execution     | ✅ Logs and PNG plots |
| Public repository      | ✅ GitHub |

---

## Author

**Shashank C**  
CUDA Signal Processing Project  
2026

---
