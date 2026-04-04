#include "gotst/core/speaker_mel.hpp"
#include "gotst/core/fft.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace gotst {

namespace {

constexpr double PI = 3.14159265358979323846;

double hz_to_slaney_mel(double hz) {
    double mel = (3.0 * hz) / 200.0;
    if (hz >= 1000.0) {
        mel = 15.0 + (std::log(hz / 1000.0) * (27.0 / std::log(6.4)));
    }
    return mel;
}

double slaney_mel_to_hz(double mel) {
    double hz = (200.0 * mel) / 3.0;
    if (mel >= 15.0) {
        hz = 1000.0 * std::exp((std::log(6.4) / 27.0) * (mel - 15.0));
    }
    return hz;
}

std::vector<float> resample_linear(
    const float *data,
    int64_t num_samples,
    int64_t from_rate,
    int64_t to_rate
) {
    if (!data || num_samples <= 0 || from_rate <= 0 || to_rate <= 0) {
        return {};
    }
    if (from_rate == to_rate) {
        return std::vector<float>(data, data + num_samples);
    }

    const int64_t target_size = std::max<int64_t>(
        1,
        static_cast<int64_t>(std::llround(
            static_cast<double>(num_samples) * static_cast<double>(to_rate) /
            static_cast<double>(from_rate)
        ))
    );
    std::vector<float> result(static_cast<size_t>(target_size));
    const double ratio = static_cast<double>(from_rate) / static_cast<double>(to_rate);
    for (int64_t i = 0; i < target_size; ++i) {
        const double pos = static_cast<double>(i) * ratio;
        const int64_t left = static_cast<int64_t>(std::floor(pos));
        const int64_t right = std::min<int64_t>(left + 1, num_samples - 1);
        const double frac = pos - static_cast<double>(left);
        result[static_cast<size_t>(i)] =
            static_cast<float>(data[left] + (data[right] - data[left]) * frac);
    }
    return result;
}

std::vector<float> build_hann_window(int64_t fft_size) {
    std::vector<float> window(static_cast<size_t>(fft_size));
    if (fft_size <= 1) {
        if (fft_size == 1) {
            window[0] = 1.0f;
        }
        return window;
    }
    for (int64_t i = 0; i < fft_size; ++i) {
        window[static_cast<size_t>(i)] =
            static_cast<float>(0.5 - 0.5 * std::cos(2.0 * PI * static_cast<double>(i) /
                                                      static_cast<double>(fft_size)));
    }
    return window;
}

struct SparseMelFilter {
    int64_t start_bin = 0;
    std::vector<float> weights;
};

std::vector<SparseMelFilter> build_mel_filter_bank(
    int64_t mel_bins,
    int64_t freq_bins,
    double fmin,
    double fmax
) {
    std::vector<SparseMelFilter> bank;
    bank.reserve(static_cast<size_t>(mel_bins));

    const double mel_min = hz_to_slaney_mel(fmin);
    const double mel_max = hz_to_slaney_mel(fmax);

    std::vector<double> filter_freqs;
    filter_freqs.reserve(static_cast<size_t>(mel_bins + 2));
    const double denom = static_cast<double>(std::max<int64_t>(mel_bins + 1, 1));
    for (int64_t i = 0; i < mel_bins + 2; ++i) {
        const double t = static_cast<double>(i) / denom;
        filter_freqs.push_back(slaney_mel_to_hz(mel_min + (mel_max - mel_min) * t));
    }

    std::vector<double> fft_freqs;
    fft_freqs.reserve(static_cast<size_t>(freq_bins));
    const double max_freq_index = static_cast<double>(std::max<int64_t>(freq_bins - 1, 1));
    for (int64_t fi = 0; fi < freq_bins; ++fi) {
        fft_freqs.push_back((fmax * static_cast<double>(fi)) / max_freq_index);
    }

    for (int64_t mi = 0; mi < mel_bins; ++mi) {
        const double left_hz = filter_freqs[static_cast<size_t>(mi)];
        const double center_hz = filter_freqs[static_cast<size_t>(mi) + 1];
        const double right_hz = filter_freqs[static_cast<size_t>(mi) + 2];
        const double down_width = std::max(center_hz - left_hz, 1e-12);
        const double up_width = std::max(right_hz - center_hz, 1e-12);
        const double slaney_norm = 2.0 / std::max(right_hz - left_hz, 1e-12);

        int64_t first_nonzero = freq_bins;
        int64_t last_nonzero = -1;
        for (int64_t fi = 0; fi < freq_bins; ++fi) {
            const double fft_hz = fft_freqs[static_cast<size_t>(fi)];
            const double down_slope = (fft_hz - left_hz) / down_width;
            const double up_slope = (right_hz - fft_hz) / up_width;
            const double w = std::max(0.0, std::min(down_slope, up_slope)) * slaney_norm;
            if (w > 0.0) {
                if (fi < first_nonzero) first_nonzero = fi;
                last_nonzero = fi;
            }
        }

        SparseMelFilter filter;
        if (first_nonzero <= last_nonzero) {
            filter.start_bin = first_nonzero;
            const int64_t count = last_nonzero - first_nonzero + 1;
            filter.weights.resize(static_cast<size_t>(count));
            for (int64_t fi = first_nonzero; fi <= last_nonzero; ++fi) {
                const double fft_hz = fft_freqs[static_cast<size_t>(fi)];
                const double down_slope = (fft_hz - left_hz) / down_width;
                const double up_slope = (right_hz - fft_hz) / up_width;
                filter.weights[static_cast<size_t>(fi - first_nonzero)] =
                    static_cast<float>(std::max(0.0, std::min(down_slope, up_slope)) * slaney_norm);
            }
        }
        bank.push_back(std::move(filter));
    }

    return bank;
}

} // namespace

Result<SpeakerMelResult> build_speaker_mel_features(
    const float *waveform,
    int64_t num_samples,
    int64_t input_sample_rate,
    int64_t target_sample_rate,
    int64_t mel_bins,
    int64_t fft_size,
    int64_t hop_length,
    double fmin,
    double fmax
) {
    if (!waveform || num_samples <= 0 || target_sample_rate <= 0 || mel_bins <= 0 ||
        fft_size <= 0 || hop_length <= 0 || fmax <= fmin) {
        return Error::invalid_argument("build_speaker_mel_features: invalid input parameters");
    }

    std::vector<float> working;
    if (input_sample_rate != target_sample_rate) {
        working = resample_linear(waveform, num_samples, input_sample_rate, target_sample_rate);
    } else {
        working.assign(waveform, waveform + num_samples);
    }

    if (working.empty()) {
        return Error::invalid_argument("build_speaker_mel_features: resampling produced empty output");
    }

    const int64_t source_samples = static_cast<int64_t>(working.size());
    const int64_t freq_bins = (fft_size / 2) + 1;

    const std::vector<float> window = build_hann_window(fft_size);

    const std::vector<SparseMelFilter> mel_bank =
        build_mel_filter_bank(mel_bins, freq_bins, fmin, fmax);

    const int64_t centered_frame_count = std::max<int64_t>(
        1,
        1 + static_cast<int64_t>(std::floor(
                static_cast<double>(source_samples) / static_cast<double>(hop_length)))
    );
    const int64_t frame_count = std::max<int64_t>(1, centered_frame_count - 1);

    SpeakerMelResult result;
    result.mel_dim = mel_bins;
    result.frames = frame_count;
    result.features.resize(static_cast<size_t>(frame_count * mel_bins));

    std::vector<float> spectrum(static_cast<size_t>(freq_bins));
    FftWorkspace fft_ws;

    for (int64_t frame = 0; frame < frame_count; ++frame) {
        const int64_t offset = (frame * hop_length) - (fft_size / 2);
        compute_power_spectrum_fft(
            working.data(), source_samples, offset, freq_bins,
            window.data(), fft_size, spectrum.data(), &fft_ws
        );

        for (int64_t mel = 0; mel < mel_bins; ++mel) {
            double energy = 0.0;
            const SparseMelFilter &filter = mel_bank[static_cast<size_t>(mel)];
            const int64_t wcount = static_cast<int64_t>(filter.weights.size());
            for (int64_t wi = 0; wi < wcount; ++wi) {
                energy += static_cast<double>(spectrum[static_cast<size_t>(filter.start_bin + wi)]) *
                          static_cast<double>(filter.weights[static_cast<size_t>(wi)]);
            }
            const double log_energy = std::log(std::max(energy, 1e-5));
            result.features[static_cast<size_t>(frame * mel_bins + mel)] =
                static_cast<float>(log_energy);
        }
    }

    return result;
}

} // namespace gotst
