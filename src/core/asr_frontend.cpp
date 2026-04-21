#include "gotst/core/asr_frontend.hpp"

#include "gotst/core/fft.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace gotst {

namespace {

constexpr double MIN_LOG_ENERGY = 1e-10;
constexpr double PI = 3.14159265358979323846;

double hz_to_slaney_mel(double value) {
    double mel = (3.0 * value) / 200.0;
    if(value >= 1000.0) {
        mel = 15.0 + (std::log(value / 1000.0) * (27.0 / std::log(6.4)));
    }
    return mel;
}

double slaney_mel_to_hz(double value) {
    double hz = (200.0 * value) / 3.0;
    if(value >= 15.0) {
        hz = 1000.0 * std::exp((std::log(6.4) / 27.0) * (value - 15.0));
    }
    return hz;
}

std::vector<float> resample_linear(const float *data,
                                   int64_t num_samples,
                                   int64_t from_rate,
                                   int64_t to_rate) {
    if(!data || num_samples <= 0 || from_rate <= 0 || to_rate <= 0) {
        return {};
    }
    if(from_rate == to_rate) {
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
    for(int64_t index = 0; index < target_size; ++index) {
        const double position = static_cast<double>(index) * ratio;
        const int64_t left = static_cast<int64_t>(std::floor(position));
        const int64_t right = std::min<int64_t>(left + 1, num_samples - 1);
        const double fraction = position - static_cast<double>(left);
        result[static_cast<size_t>(index)] = static_cast<float>(
            data[left] + ((data[right] - data[left]) * fraction)
        );
    }
    return result;
}

std::vector<float> build_window(int64_t fft_size) {
    std::vector<float> window(static_cast<size_t>(fft_size));
    if(fft_size <= 1) {
        if(fft_size == 1) {
            window[0] = 1.0f;
        }
        return window;
    }

    for(int64_t index = 0; index < fft_size; ++index) {
        window[static_cast<size_t>(index)] = static_cast<float>(
            0.5 - (0.5 * std::cos((2.0 * PI * static_cast<double>(index)) / static_cast<double>(fft_size)))
        );
    }
    return window;
}

struct SparseMelFilter {
    int64_t start_bin = 0;
    std::vector<float> weights;
};

std::vector<SparseMelFilter> build_sparse_mel_filter_bank(int64_t sample_rate,
                                                          int64_t mel_bins,
                                                          int64_t freq_bins) {
    std::vector<SparseMelFilter> bank;
    bank.reserve(static_cast<size_t>(mel_bins));

    const double mel_min = hz_to_slaney_mel(0.0);
    const double mel_max = hz_to_slaney_mel(static_cast<double>(sample_rate) * 0.5);

    std::vector<double> filter_freqs;
    filter_freqs.reserve(static_cast<size_t>(mel_bins + 2));
    const double denom = static_cast<double>(std::max<int64_t>(mel_bins + 1, 1));
    for(int64_t index = 0; index < mel_bins + 2; ++index) {
        const double t = static_cast<double>(index) / denom;
        const double mel_value = mel_min + ((mel_max - mel_min) * t);
        filter_freqs.push_back(slaney_mel_to_hz(mel_value));
    }

    std::vector<double> fft_freqs;
    fft_freqs.reserve(static_cast<size_t>(freq_bins));
    const double max_frequency = static_cast<double>(sample_rate) * 0.5;
    const double max_freq_index = static_cast<double>(std::max<int64_t>(freq_bins - 1, 1));
    for(int64_t freq_index = 0; freq_index < freq_bins; ++freq_index) {
        fft_freqs.push_back((max_frequency * static_cast<double>(freq_index)) / max_freq_index);
    }

    for(int64_t mel_index = 0; mel_index < mel_bins; ++mel_index) {
        const double left_hz = filter_freqs[static_cast<size_t>(mel_index)];
        const double center_hz = filter_freqs[static_cast<size_t>(mel_index) + 1];
        const double right_hz = filter_freqs[static_cast<size_t>(mel_index) + 2];
        const double down_width = std::max(center_hz - left_hz, 1e-12);
        const double up_width = std::max(right_hz - center_hz, 1e-12);
        const double slaney_norm = 2.0 / std::max(right_hz - left_hz, 1e-12);

        int64_t first_nonzero = freq_bins;
        int64_t last_nonzero = -1;
        for(int64_t freq_index = 0; freq_index < freq_bins; ++freq_index) {
            const double fft_hz = fft_freqs[static_cast<size_t>(freq_index)];
            const double down_slope = (fft_hz - left_hz) / down_width;
            const double up_slope = (right_hz - fft_hz) / up_width;
            const double weight = std::max(0.0, std::min(down_slope, up_slope)) * slaney_norm;
            if(weight > 0.0) {
                first_nonzero = std::min(first_nonzero, freq_index);
                last_nonzero = std::max(last_nonzero, freq_index);
            }
        }

        SparseMelFilter filter;
        if(first_nonzero <= last_nonzero) {
            filter.start_bin = first_nonzero;
            filter.weights.resize(static_cast<size_t>(last_nonzero - first_nonzero + 1));
            for(int64_t freq_index = first_nonzero; freq_index <= last_nonzero; ++freq_index) {
                const double fft_hz = fft_freqs[static_cast<size_t>(freq_index)];
                const double down_slope = (fft_hz - left_hz) / down_width;
                const double up_slope = (right_hz - fft_hz) / up_width;
                filter.weights[static_cast<size_t>(freq_index - first_nonzero)] = static_cast<float>(
                    std::max(0.0, std::min(down_slope, up_slope)) * slaney_norm
                );
            }
        }
        bank.push_back(std::move(filter));
    }

    return bank;
}

struct AsrMelPlanKey {
    int64_t sample_rate = 0;
    int64_t mel_bins = 0;
    int64_t fft_size = 0;

    bool operator<(const AsrMelPlanKey &other) const {
        if(sample_rate != other.sample_rate) {
            return sample_rate < other.sample_rate;
        }
        if(mel_bins != other.mel_bins) {
            return mel_bins < other.mel_bins;
        }
        return fft_size < other.fft_size;
    }
};

struct AsrMelPlan {
    int64_t freq_bins = 0;
    std::vector<float> window;
    std::vector<SparseMelFilter> filter_bank;
};

struct AsrMelPlanCacheState {
    std::mutex mutex;
    std::map<AsrMelPlanKey, std::shared_ptr<const AsrMelPlan>> cache;
};

AsrMelPlanCacheState &asr_mel_plan_cache_state() {
    // Keep the frontend cache alive until process exit so shutdown cannot race
    // static destruction of the mutex or cached mel plans.
    static auto *state = new AsrMelPlanCacheState();
    return *state;
}

std::shared_ptr<const AsrMelPlan> get_asr_mel_plan(int64_t sample_rate,
                                                   int64_t mel_bins,
                                                   int64_t fft_size) {
    AsrMelPlanCacheState &cache_state = asr_mel_plan_cache_state();

    const AsrMelPlanKey key{sample_rate, mel_bins, fft_size};
    {
        std::lock_guard<std::mutex> lock(cache_state.mutex);
        auto it = cache_state.cache.find(key);
        if(it != cache_state.cache.end()) {
            return it->second;
        }
    }

    auto plan = std::make_shared<AsrMelPlan>();
    plan->freq_bins = (fft_size / 2) + 1;
    plan->window = build_window(fft_size);
    plan->filter_bank = build_sparse_mel_filter_bank(sample_rate, mel_bins, plan->freq_bins);

    std::lock_guard<std::mutex> lock(cache_state.mutex);
    auto [it, inserted] = cache_state.cache.emplace(key, plan);
    if(!inserted) {
        return it->second;
    }
    return plan;
}

} // namespace

Result<AsrLogMelResult> build_asr_log_mel_features(const float *waveform,
                                                   int64_t num_samples,
                                                   int64_t input_sample_rate,
                                                   int64_t sample_rate,
                                                   int64_t mel_bins,
                                                   int64_t fft_size,
                                                   int64_t hop_length,
                                                   double chunk_length_seconds) {
    if(!waveform || num_samples <= 0 || input_sample_rate <= 0 || sample_rate <= 0 ||
       mel_bins <= 0 || fft_size <= 0 || hop_length <= 0) {
        return Error::invalid_argument("build_asr_log_mel_features: invalid input parameters");
    }

    std::vector<float> working;
    if(input_sample_rate != sample_rate) {
        working = resample_linear(waveform, num_samples, input_sample_rate, sample_rate);
    } else {
        working.assign(waveform, waveform + num_samples);
    }
    if(working.empty()) {
        return Error::invalid_argument("build_asr_log_mel_features: resampling produced empty output");
    }

    const int64_t max_chunk_samples = std::max<int64_t>(
        1600,
        static_cast<int64_t>(std::ceil(std::max(chunk_length_seconds, 1.0) * static_cast<double>(sample_rate)))
    );
    if(static_cast<int64_t>(working.size()) > max_chunk_samples) {
        const auto start = static_cast<size_t>(working.size() - static_cast<size_t>(max_chunk_samples));
        working.assign(working.begin() + static_cast<std::ptrdiff_t>(start), working.end());
    }

    const auto plan = get_asr_mel_plan(sample_rate, mel_bins, fft_size);
    const int64_t source_samples = static_cast<int64_t>(working.size());
    const int64_t frame_count = std::max<int64_t>(
        1,
        std::max<int64_t>(
            1,
            1 + static_cast<int64_t>(
                    std::floor(static_cast<double>(source_samples) / static_cast<double>(hop_length))
                )
        ) -
            1
    );

    AsrLogMelResult result;
    result.frame_count = frame_count;
    result.valid_frame_count = frame_count;
    result.features.resize(static_cast<size_t>(mel_bins * frame_count));

    std::vector<float> spectrum(static_cast<size_t>(plan->freq_bins));
    FftWorkspace fft_workspace;
    double max_log = -std::numeric_limits<double>::infinity();

    for(int64_t frame_index = 0; frame_index < frame_count; ++frame_index) {
        const int64_t offset = (frame_index * hop_length) - (fft_size / 2);
        compute_power_spectrum_fft(
            working.data(),
            source_samples,
            offset,
            plan->freq_bins,
            plan->window.data(),
            fft_size,
            spectrum.data(),
            &fft_workspace
        );

        for(int64_t mel_index = 0; mel_index < mel_bins; ++mel_index) {
            double energy = 0.0;
            const SparseMelFilter &filter = plan->filter_bank[static_cast<size_t>(mel_index)];
            const int64_t weight_count = static_cast<int64_t>(filter.weights.size());
            for(int64_t weight_index = 0; weight_index < weight_count; ++weight_index) {
                energy += static_cast<double>(
                    spectrum[static_cast<size_t>(filter.start_bin + weight_index)]
                ) *
                    static_cast<double>(filter.weights[static_cast<size_t>(weight_index)]);
            }

            const double log_energy = std::log10(std::max(MIN_LOG_ENERGY, energy));
            result.features[static_cast<size_t>((mel_index * frame_count) + frame_index)] =
                static_cast<float>(log_energy);
            max_log = std::max(max_log, log_energy);
        }
    }

    if(!std::isfinite(max_log)) {
        return Error::inference_failed("build_asr_log_mel_features: produced non-finite log energy");
    }

    const double floor_value = max_log - 8.0;
    for(float &feature : result.features) {
        const double normalized = std::max(static_cast<double>(feature), floor_value);
        feature = static_cast<float>((normalized + 4.0) * 0.25);
    }

    return result;
}

} // namespace gotst
