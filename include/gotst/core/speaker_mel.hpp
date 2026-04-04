#pragma once

#include "gotst/core/result.hpp"

#include <cstdint>
#include <vector>

namespace gotst {

struct SpeakerMelResult {
    std::vector<float> features;
    int64_t frames = 0;
    int64_t mel_dim = 0;
};

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
);

} // namespace gotst
