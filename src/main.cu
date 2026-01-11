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
    std::string line = "[" + timestamp() + "] " + msg;
    std::cout << line << std::endl;

    if (log_file.is_open()) {
        log_file << line << std::endl;
        log_file.flush();
    }
}

int main(int argc, char* argv[])
{
    const std::string input_dir  = "data/input";
    const std::string output_dir = "data/output";
    const std::string log_path   = "log/run.log";

    std::filesystem::create_directories(output_dir);
    std::filesystem::create_directories("log");

    log_file.open(log_path, std::ios::out);
    if (!log_file.is_open()) {
        std::cerr << "ERROR: Unable to open log file\n";
        return -1;
    }

    log_msg("GPU Butterworth Filter Application Started");

    const float fs = 2500.0f;     // Sampling frequency (Hz)
    const float fc = 15.0f;       // Cutoff frequency (Hz)
    const ButterworthType filter_type = BUTTER_LOWPASS;

    log_msg("Sampling rate: " + std::to_string(fs));
    log_msg("Cutoff frequency: " + std::to_string(fc));
    log_msg("Filter type: Low-pass");

    float* d_b = nullptr;   // b0, b1, b2
    float* d_a = nullptr;   // a1, a2

    CUDA_CHECK(cudaMalloc(&d_b, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a, 2 * sizeof(float)));

    launch_butterworth_coeffs(
        d_b,
        d_a,
        fs,
        fc,
        filter_type
    );

    log_msg("Butterworth coefficients generated on GPU");


    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {

        if (!entry.is_regular_file())
            continue;

        if (entry.path().extension() != ".csv")
            continue;

        std::string input_path  = entry.path().string();
        std::string filename    = entry.path().filename().string();
        std::string output_path = output_dir + "/output_" + filename;

        log_msg("--------------------------------------------------");
        log_msg("Processing file: " + input_path);

        std::vector<float> input_signal = load_csv(input_path);

        if (input_signal.empty()) {
            log_msg("WARNING: Empty or invalid CSV, skipping");
            continue;
        }

        int N = static_cast<int>(input_signal.size());
        log_msg("Signal length: " + std::to_string(N));

        float* d_input  = nullptr;
        float* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(&d_input,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input,
                              input_signal.data(),
                              N * sizeof(float),
                              cudaMemcpyHostToDevice));

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
        log_msg("GPU execution time: " +
                std::to_string(elapsed_ms) + " ms");

        std::vector<float> output_signal(N);
        CUDA_CHECK(cudaMemcpy(output_signal.data(),
                              d_output,
                              N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        
        save_csv_two_columns(output_path,
                     input_signal,
                     output_signal,
                     "original_signal",
                     "filtered_signal");
        log_msg("Saved output file: " + output_path);

        cudaFree(d_input);
        cudaFree(d_output);
    }

    cudaFree(d_b);
    cudaFree(d_a);

    log_msg("Freed Butterworth coefficient memory");
    log_msg("Application finished successfully");

    log_file.close();
    return 0;
}
