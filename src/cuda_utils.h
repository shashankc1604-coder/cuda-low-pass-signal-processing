#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA ERROR: "                                 \
                      << cudaGetErrorString(err)                        \
                      << " at " << __FILE__ << ":" << __LINE__          \
                      << std::endl;                                     \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic() {
        CUDA_CHECK(cudaEventRecord(start, 0));
    }

    float toc() {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};
