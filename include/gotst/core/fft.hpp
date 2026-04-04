#pragma once

#include <cstdint>
#include <vector>

namespace gotst {

struct FftOutput {
    std::vector<double> real;
    std::vector<double> imag;
};

void radix2_fft(const double *input_real, const double *input_imag, int64_t n, double *output_real, double *output_imag);

struct FftWorkspace {
    std::vector<double> real_in;
    std::vector<double> imag_in;
    std::vector<double> real_out;
    std::vector<double> imag_out;

    void resize(int64_t padded_size);
};

void compute_power_spectrum_fft(
    const float *samples,
    int64_t num_samples,
    int64_t offset,
    int64_t freq_bins,
    const float *window,
    int64_t fft_size,
    float *destination,
    FftWorkspace *workspace = nullptr
);

} // namespace gotst
