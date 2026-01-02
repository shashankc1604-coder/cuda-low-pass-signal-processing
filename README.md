cuda-lowpass-signal-processing/
│
├── data/
│   ├── input/
│   │   ├── sensor_001.csv
│   │   ├── sensor_002.csv
│   │   └── ...
│   └── output/
│       ├── sensor_001_filtered.csv
│       └── ...
│
├── src/
│   ├── fir_filter.cu
│   ├── csv_loader.cpp
│   ├── main.cu
│
├── python/
│   └── plot_signals.py
│
├── results/
│   ├── sensor_001_plot.png
│   ├── sensor_002_plot.png
│   └── ...
│
├── logs/
│   └── execution_log.txt
│
├── CMakeLists.txt
├── README.md
└── LICENSE
