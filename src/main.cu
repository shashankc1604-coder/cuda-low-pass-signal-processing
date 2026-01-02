#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <ctime>

// Forward declarations
std::vector<float> load_sensor_csv(const std::string& filepath);
void save_sensor_csv(const std::string& filepath,
                     const std::vector<float>& data);

void launch_fir_lowpass(
    const float* d_input,
    float* d_output,
    const float* d_coeffs,
    int signal_length,
    int filter_length
);


// Simple logger
std::ofstream log_file;

std::string current_time() {
    std::time_t now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&now));
    return std::string(buf);
}

void log_msg(const std::string& msg) {
    log_file << "[" << current_time() << "] " << msg << std::endl;
}

// FIR coefficients (simple LPF)
std::vector<float> generate_fir_coeffs(int M) {
    const float fs = 2500.0f;   // Sampling rate
    const float fc = 5.0f;     // Cutoff frequency (Hz)

    std::vector<float> h(M);
    int mid = M / 2;
    float sum = 0.0f;

    for (int n = 0; n < M; n++) {
        if (n == mid) {
            h[n] = 2.0f * fc / fs;
        } else {
            float x = M_PI * (n - mid);
            h[n] = sinf(2.0f * M_PI * fc * (n - mid) / fs) / x;
        }

        // Hamming window
        float w = 0.54f - 0.46f * cosf(2.0f * M_PI * n / (M - 1));
        h[n] *= w;

        sum += h[n];
    }

    // Normalize gain to 1.0 (DC)
    for (int n = 0; n < M; n++) {
        h[n] /= sum;
    }

    return h;
}


// ==============================
// Main
// ==============================
int main() {
    const std::string input_dir  = "data/input";
    const std::string output_dir = "data/output";
    const std::string log_dir    = "logs";
    const std::string log_path   = log_dir + "/cuda_fir_filter.log";

    std::filesystem::create_directories(log_dir);

    log_file.open(log_path, std::ios::out);
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file\n";
        return -1;
    }

    log_msg("CUDA FIR Low-Pass Filter Application Started");

    const int filter_length = 64;

    log_msg("Generating FIR coefficients (length = " +
            std::to_string(filter_length) + ")");

    std::vector<float> h = generate_fir_coeffs(filter_length);

    // Allocate coefficients on GPU
    float* d_coeffs = nullptr;
    cudaMalloc(&d_coeffs, filter_length * sizeof(float));
    cudaMemcpy(d_coeffs, h.data(),
               filter_length * sizeof(float),
               cudaMemcpyHostToDevice);

    log_msg("Copied FIR coefficients to GPU");

    // Process CSV files
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        if (entry.path().extension() != ".csv")
            continue;

        std::string input_path  = entry.path().string();
        std::string output_path =
            output_dir + "/" + entry.path().filename().string();

        log_msg("Processing file: " + input_path);

        // Load CSV
        std::vector<float> signal = load_sensor_csv(input_path);
        int N = static_cast<int>(signal.size());

        log_msg("Signal length: " + std::to_string(N));

        if (N == 0) {
            log_msg("WARNING: Empty signal, skipping file");
            continue;
        }

        float* d_input  = nullptr;
        float* d_output = nullptr;

        cudaMalloc(&d_input,  N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));

        log_msg("Allocated GPU buffers");

        cudaMemcpy(d_input, signal.data(),
                   N * sizeof(float),
                   cudaMemcpyHostToDevice);

        log_msg("Copied input signal to GPU");

        launch_fir_lowpass(
            d_input,
            d_output,
            d_coeffs,
            N,
            filter_length
        );

        cudaDeviceSynchronize();
        log_msg("CUDA FIR kernel executed");

        std::vector<float> filtered(N);
        cudaMemcpy(filtered.data(), d_output,
                   N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        log_msg("Copied filtered signal back to CPU");

        save_sensor_csv(output_path, filtered);
        log_msg("Saved filtered CSV: " + output_path);

        cudaFree(d_input);
        cudaFree(d_output);

        log_msg("Freed GPU buffers");
    }

    cudaFree(d_coeffs);
    log_msg("Freed FIR coefficients from GPU");

    log_msg("Processing complete. Application exiting.");
    log_file.close();

    return 0;
}
