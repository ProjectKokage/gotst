#pragma once

#include "gotst/core/result.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gotst {

class SpeakerEncoderSession {
public:
    SpeakerEncoderSession();
    ~SpeakerEncoderSession();

    SpeakerEncoderSession(const SpeakerEncoderSession &) = delete;
    SpeakerEncoderSession &operator=(const SpeakerEncoderSession &) = delete;
    SpeakerEncoderSession(SpeakerEncoderSession &&) noexcept = default;
    SpeakerEncoderSession &operator=(SpeakerEncoderSession &&) noexcept = default;

    Result<void> load(const std::string &model_path);
    bool is_loaded() const;
    Result<std::vector<float>> extract_embedding(const float *mel_features, int64_t frames, int64_t mel_dim) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct SpeechTokenizerEncodeResult {
    std::vector<int32_t> codes;
    int64_t frames = 0;
    int64_t codebooks = 16;
};

class SpeechTokenizerEncoderSession {
public:
    SpeechTokenizerEncoderSession();
    ~SpeechTokenizerEncoderSession();

    SpeechTokenizerEncoderSession(const SpeechTokenizerEncoderSession &) = delete;
    SpeechTokenizerEncoderSession &operator=(const SpeechTokenizerEncoderSession &) = delete;
    SpeechTokenizerEncoderSession(SpeechTokenizerEncoderSession &&) noexcept = default;
    SpeechTokenizerEncoderSession &operator=(SpeechTokenizerEncoderSession &&) noexcept = default;

    Result<void> load(const std::string &model_path);
    bool is_loaded() const;
    Result<SpeechTokenizerEncodeResult> encode(const float *audio, int64_t num_samples) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gotst
