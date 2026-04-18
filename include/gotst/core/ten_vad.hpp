#pragma once

#include "gotst/core/result.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gotst {

struct TenVadConfig {
    std::string model_path;
    int32_t model_sample_rate = 16000;
    int32_t hop_size = 256;
    float threshold = 0.5f;
    int32_t reset_frame_count = 1875;
    float pitch_est_voiced_threshold = 0.4f;
};

struct TenVadFrameResult {
    float probability = 0.0f;
    bool is_voice = false;
    float frame_rms = 0.0f;
    float frame_energy = 0.0f;
    float pitch_frequency_hz = 0.0f;
};

struct TenVadChunkResult {
    std::vector<TenVadFrameResult> frames;
    float last_probability = 0.0f;
    bool last_is_voice = false;
    bool any_voice = false;
    int32_t processed_frame_count = 0;
    int32_t hop_size = 256;
    int32_t model_sample_rate = 16000;
};

class TenVad {
public:
    TenVad();
    ~TenVad();

    TenVad(const TenVad &) = delete;
    TenVad &operator=(const TenVad &) = delete;
    TenVad(TenVad &&) noexcept;
    TenVad &operator=(TenVad &&) noexcept;

    Result<void> load(const TenVadConfig &config);
    bool is_loaded() const;
    Result<void> reset();
    Result<TenVadChunkResult> process(std::span<const float> samples, int32_t input_sample_rate);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gotst
