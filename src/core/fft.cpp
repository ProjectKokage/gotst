#include "gotst/core/fft.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace gotst {

namespace {

constexpr double PI = 3.14159265358979323846;

int64_t reflect_index(int64_t index, int64_t count) {
    if (count <= 1) {
        return 0;
    }
    int64_t reflected = index;
    while (reflected < 0 || reflected >= count) {
        if (reflected < 0) {
            reflected = -reflected;
        }
        if (reflected >= count) {
            reflected = (count * 2) - reflected - 2;
        }
    }
    return reflected;
}

int64_t next_power_of_two(int64_t n) {
    if (n <= 0) return 1;
    int64_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

void bit_reverse_copy(const double *src_real, const double *src_imag, int64_t n, int64_t log2n, double *dst_real, double *dst_imag) {
    for (int64_t i = 0; i < n; ++i) {
        int64_t rev = 0;
        int64_t val = i;
        for (int64_t bit = 0; bit < log2n; ++bit) {
            rev = (rev << 1) | (val & 1);
            val >>= 1;
        }
        dst_real[rev] = src_real[i];
        dst_imag[rev] = src_imag[i];
    }
}

} // namespace

void radix2_fft(const double *input_real, const double *input_imag, int64_t n, double *output_real, double *output_imag) {
    if (n <= 0) return;
    if (n == 1) {
        output_real[0] = input_real[0];
        output_imag[0] = input_imag[0];
        return;
    }

    int64_t log2n = 0;
    {
        int64_t temp = n;
        while (temp > 1) {
            temp >>= 1;
            ++log2n;
        }
    }

    bit_reverse_copy(input_real, input_imag, n, log2n, output_real, output_imag);

    for (int64_t stage = 1; stage <= log2n; ++stage) {
        const int64_t m = 1LL << stage;
        const int64_t half_m = m >> 1;
        const double angle = -2.0 * PI / static_cast<double>(m);
        const double wm_real = std::cos(angle);
        const double wm_imag = std::sin(angle);

        for (int64_t k = 0; k < n; k += m) {
            double w_real = 1.0;
            double w_imag = 0.0;
            for (int64_t j = 0; j < half_m; ++j) {
                const int64_t even = k + j;
                const int64_t odd = k + j + half_m;
                const double t_real = w_real * output_real[odd] - w_imag * output_imag[odd];
                const double t_imag = w_real * output_imag[odd] + w_imag * output_real[odd];
                output_real[odd] = output_real[even] - t_real;
                output_imag[odd] = output_imag[even] - t_imag;
                output_real[even] = output_real[even] + t_real;
                output_imag[even] = output_imag[even] + t_imag;
                const double next_w_real = w_real * wm_real - w_imag * wm_imag;
                w_imag = w_real * wm_imag + w_imag * wm_real;
                w_real = next_w_real;
            }
        }
    }
}

void FftWorkspace::resize(int64_t padded_size) {
    const auto n = static_cast<size_t>(padded_size);
    if (real_in.size() < n) {
        real_in.resize(n);
        imag_in.resize(n);
        real_out.resize(n);
        imag_out.resize(n);
    }
}

void compute_power_spectrum_fft(
    const float *samples,
    int64_t num_samples,
    int64_t offset,
    int64_t freq_bins,
    const float *window,
    int64_t fft_size,
    float *destination,
    FftWorkspace *workspace
) {
    const int64_t padded_size = next_power_of_two(fft_size);

    FftWorkspace local_ws;
    FftWorkspace &ws = workspace ? *workspace : local_ws;
    ws.resize(padded_size);

    std::fill_n(ws.real_in.data(), static_cast<size_t>(padded_size), 0.0);
    std::fill_n(ws.imag_in.data(), static_cast<size_t>(padded_size), 0.0);

    for (int64_t si = 0; si < fft_size; ++si) {
        const int64_t src_index = offset + si;
        double sample = 0.0;
        if (src_index < 0 || src_index >= num_samples) {
            const int64_t reflected = reflect_index(src_index, num_samples);
            if (reflected >= 0 && reflected < num_samples) {
                sample = samples[static_cast<size_t>(reflected)];
            }
        } else {
            sample = samples[static_cast<size_t>(src_index)];
        }
        ws.real_in[static_cast<size_t>(si)] = sample * window[static_cast<size_t>(si)];
    }

    radix2_fft(ws.real_in.data(), ws.imag_in.data(), padded_size, ws.real_out.data(), ws.imag_out.data());

    const int64_t usable_bins = (freq_bins <= padded_size / 2 + 1) ? freq_bins : padded_size / 2 + 1;
    for (int64_t freq = 0; freq < usable_bins; ++freq) {
        const double r = ws.real_out[freq];
        const double im = ws.imag_out[freq];
        destination[freq] = static_cast<float>(r * r + im * im);
    }
    for (int64_t freq = usable_bins; freq < freq_bins; ++freq) {
        destination[freq] = 0.0f;
    }
}

} // namespace gotst
