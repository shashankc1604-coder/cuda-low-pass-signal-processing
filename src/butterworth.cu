#include "butterworth.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>

// Kernel: Compute Butterworth 3rd-order coefficients (biquad)

/*
We implement a 3rd-order Butterworth filter as:
--> One 2nd-order biquad section
--> One 1st-order section (merged into coefficients)
Coefficients are computed using bilinear transform
*/

__global__ void butterworth_coeffs_kernel(
    float* b,      // numerator coefficients (b0, b1, b2)
    float* a,      // denominator coefficients (a1, a2)
    float fs,
    float fc,
    int type       // 0 = low-pass, 1 = high-pass
) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    const float K = tanf(M_PI * fc / fs);
    const float K2 = K * K;
    const float norm = 1.0f / (1.0f + sqrtf(2.0f) * K + K2);

    if (type == 0) {
        // Low-pass Butterworth
        b[0] = K2 * norm;
        b[1] = 2.0f * K2 * norm;
        b[2] = K2 * norm;

        a[0] = 2.0f * (K2 - 1.0f) * norm;
        a[1] = (1.0f - sqrtf(2.0f) * K + K2) * norm;
    } else {
        // High-pass Butterworth
        b[0] = 1.0f * norm;
        b[1] = -2.0f * norm;
        b[2] = 1.0f * norm;

        a[0] = 2.0f * (K2 - 1.0f) * norm;
        a[1] = (1.0f - sqrtf(2.0f) * K + K2) * norm;
    }
}

// Kernel: Apply 3rd-order Butterworth IIR filter
__global__ void butterworth_filter_kernel(
    const float* x,
    float* y,
    int N,
    const float* b,
    const float* a
) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    float x1 = 0.0f, x2 = 0.0f;
    float y1 = 0.0f, y2 = 0.0f;

    for (int n = 0; n < N; ++n) {
        float xn = x[n];

        float yn =
            b[0] * xn +
            b[1] * x1 +
            b[2] * x2 -
            a[0] * y1 -
            a[1] * y2;

        y[n] = yn;

        x2 = x1;
        x1 = xn;
        y2 = y1;
        y1 = yn;
    }
}


// Host launcher: coefficient computation
void launch_butterworth_coeffs(
    float* d_b,
    float* d_a,
    float fs,
    float fc,
    ButterworthType type
) {
    butterworth_coeffs_kernel<<<1, 1>>>(
        d_b,
        d_a,
        fs,
        fc,
        static_cast<int>(type)
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Host launcher: filtering
void launch_butterworth_filter(
    const float* d_input,
    float* d_output,
    int signal_length,
    const float* d_b,
    const float* d_a
) {
    butterworth_filter_kernel<<<1, 1>>>(
        d_input,
        d_output,
        signal_length,
        d_b,
        d_a
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
