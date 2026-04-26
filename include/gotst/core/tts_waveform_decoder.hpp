#pragma once

#include "gotst/core/result.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gotst {

struct TtsWaveformDecoderConfig {
    std::string decoder_onnx_path;
    std::string provider_requested;
    std::string provider = "CPU";
    int32_t intra_op_threads = 0;
    int32_t inter_op_threads = 0;
    int32_t optimization_level = 99;
    std::string optimized_model_path;
    int32_t sample_rate = 24000;
    bool normalize_waveform = false;
    float waveform_gain = 1.0f;
    int32_t stateful_chunk_frames = 12;
};

struct TtsWaveformDecodeResult {
    std::vector<float> waveform;
    int32_t frame_count = 0;
    int32_t codes_per_frame = 0;
    int32_t sample_count = 0;
    double elapsed_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    std::string backend = "gotst_native";
    std::string provider_requested = "CPU";
    std::string provider_effective = "CPU";
    int32_t cpu_fallback_node_count = -1;
    bool fixed_shape = false;
};

class TtsWaveformDecoderStream;

class TtsWaveformDecoder {
public:
    TtsWaveformDecoder();
    ~TtsWaveformDecoder();

    TtsWaveformDecoder(const TtsWaveformDecoder &) = delete;
    TtsWaveformDecoder &operator=(const TtsWaveformDecoder &) = delete;
    TtsWaveformDecoder(TtsWaveformDecoder &&) noexcept;
    TtsWaveformDecoder &operator=(TtsWaveformDecoder &&) noexcept;

    Result<void> load(const TtsWaveformDecoderConfig &config);
    bool is_loaded() const;

    Result<TtsWaveformDecodeResult> decode(
        std::span<const int64_t> audio_codes,
        int32_t frame_count
    ) const;

    Result<std::unique_ptr<TtsWaveformDecoderStream>> create_stream() const;

private:
    friend class TtsWaveformDecoderStream;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class TtsWaveformDecoderStream {
public:
    ~TtsWaveformDecoderStream();

    TtsWaveformDecoderStream(const TtsWaveformDecoderStream &) = delete;
    TtsWaveformDecoderStream &operator=(const TtsWaveformDecoderStream &) = delete;
    TtsWaveformDecoderStream(TtsWaveformDecoderStream &&) noexcept;
    TtsWaveformDecoderStream &operator=(TtsWaveformDecoderStream &&) noexcept;

    void reset();

    Result<TtsWaveformDecodeResult> decode(
        std::span<const int64_t> audio_codes,
        int32_t frame_count,
        bool is_final
    );

private:
    friend class TtsWaveformDecoder;

    explicit TtsWaveformDecoderStream(const TtsWaveformDecoder &decoder);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gotst
