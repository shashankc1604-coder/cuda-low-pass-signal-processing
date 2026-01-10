#pragma once

enum ButterworthType {
    BUTTER_LOWPASS = 0,
    BUTTER_HIGHPASS = 1
};

void launch_butterworth_coeffs(
    float* d_b,
    float* d_a,
    float fs,
    float fc,
    ButterworthType type
);

void launch_butterworth_filter(
    const float* d_input,
    float* d_output,
    int signal_length,
    const float* d_b,
    const float* d_a
);
