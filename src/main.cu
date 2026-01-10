#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <ctime>

#include "csv_io.h"
#include "butterworth.h"
#include "cuda_utils.h"

std::ofstream log_file;

std::string timestamp()
{
    std::time_t now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&now));
    return std::string(buf);
}

void log_msg(const std::string& msg)
{
    log_file << "[" << timestamp() << "] " << msg << std::endl;
}


int main(int argc, char* argv[])
{
    const std::string input_csv  = "data/input_signal.csv";
    const std::string output_csv = "data/output_filtered.csv";
    const std::string log_path   = "log/run.log";

    std::filesystem::create_directories("log");

    log_file.open(log_path, std::ios::out);
    if (!log_file.is_open()) {
        std::cerr << "ERROR: Unable to open log file\n";
        return -1;
    }

    log_msg("GPU Butterworth Filter Application Started");

    // Parameters (can be extended to CLI later)
    const float fs = 2500.0f;     // Sampling frequency (Hz)
    const float fc = 50.0f;       // Cutoff frequency (Hz)
    const ButterworthType filter_type = BUTTER_LOWPASS;
    const int filter_order = 3;

    log_msg("Sampling rate: " + std::to_string(fs));
    log_msg("Cutoff frequency: " + std::to_string(fc));
    log_msg("Filter type: Low-pass");
    log_msg("Filter order: " + std::to_string(filter_order));

    // Load input CSV
    log_msg("Loading input CSV: " + input_csv);
    std::vector<float> input_signal = load_csv(input_csv);

    if (input_signal.empty()) {
        log_msg("ERROR: Input signal is empty");
        return -1;
    }

    const int N = static_cast<int>(input_signal.size());
    log_msg("Signal length: " + std::to_string(N));

    // Allocate GPU memory
    float* d_input  = nullptr;
    float* d_output = nullptr;
    float* d_b      = nullptr;  // Numerator coeffs
    float* d_a      = nullptr;  // Denominator coeffs

    CUDA_CHECK(cudaMalloc(&d_input,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a, 2 * sizeof(float)));

    log_msg("Allocated GPU memory");

    
    // Copy input signal to GPU
    CUDA_CHECK(cudaMemcpy(d_input,
                          input_signal.data(),
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    log_msg("Copied input signal to GPU");

    // Generate Butterworth coefficients on GPU
    log_msg("Generating Butterworth coefficients on GPU");
    launch_butterworth_coeffs(
        d_b,
        d_a,
        fs,
        fc,
        filter_type
    );

    log_msg("Butterworth coefficients generated");

    
    // Apply Butterworth filter on GPU
    GpuTimer timer;
    timer.tic();

    launch_butterworth_filter(
        d_input,
        d_output,
        N,
        d_b,
        d_a
    );

    float elapsed_ms = timer.toc();
    log_msg("Filtering completed on GPU");
    log_msg("GPU execution time: " + std::to_string(elapsed_ms) + " ms");

    
    // Copy filtered signal back to host
    std::vector<float> output_signal(N);
    CUDA_CHECK(cudaMemcpy(output_signal.data(),
                          d_output,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    log_msg("Copied filtered signal back to CPU");

   
    // Save output CSV
    save_csv(output_csv, output_signal);
    log_msg("Saved output CSV: " + output_csv);

    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_b);
    cudaFree(d_a);

    log_msg("Freed GPU memory");
    log_msg("Application finished successfully");

    log_file.close();
    return 0;
}
