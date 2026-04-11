#pragma once

#include "gotst/core/result.hpp"

#include <cstdint>
#include <vector>

namespace gotst {

struct AsrLogMelResult {
    std::vector<float> features;
    int64_t frame_count = 0;
    int64_t valid_frame_count = 0;
};

Result<AsrLogMelResult> build_asr_log_mel_features(const float *waveform,
                                                   int64_t num_samples,
                                                   int64_t input_sample_rate,
                                                   int64_t sample_rate,
                                                   int64_t mel_bins,
                                                   int64_t fft_size,
                                                   int64_t hop_length,
                                                   double chunk_length_seconds);

} // namespace gotst
