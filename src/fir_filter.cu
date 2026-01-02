#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

/*
 * FIR Low-Pass Filter CUDA Kernel
 *
 * Each thread computes one output sample:
 * y[n] = sum_{k=0}^{M-1} h[k] * x[n-k]
 */

__global__ void fir_lowpass_kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const float* __restrict__ d_coeffs,
    int signal_length,
    int filter_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= signal_length)
        return;

    float acc = 0.0f;

    #pragma unroll
    for (int k = 0; k < filter_length; k++) {
        int input_idx = idx - k;
        if (input_idx >= 0) {
            acc += d_coeffs[k] * d_input[input_idx];
        }
    }

    d_output[idx] = acc;
}

/*
 * This is called from main.cu
 */
void launch_fir_lowpass(
    const float* d_input,
    float* d_output,
    const float* d_coeffs,
    int signal_length,
    int filter_length
) {
    const int threads_per_block = 256;
    const int blocks =
        (signal_length + threads_per_block - 1) / threads_per_block;

    fir_lowpass_kernel<<<blocks, threads_per_block>>>(
        d_input,
        d_output,
        d_coeffs,
        signal_length,
        filter_length
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
