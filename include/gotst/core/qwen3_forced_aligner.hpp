#pragma once

#include "gotst/core/cancellation_token.hpp"
#include "gotst/core/result.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gotst {

struct Qwen3ForcedAlignerModelPaths {
    std::string audio_conv_onnx_path;
    std::string audio_encoder_onnx_path;
    std::string embedding_onnx_path;
    std::string backbone_gguf_path;
    std::string classifier_head_path;
};

struct Qwen3ForcedAlignerSessionConfig {
    int32_t sample_rate = 16000;
    int32_t mel_bins = 128;
    int32_t fft_size = 400;
    int32_t hop_length = 160;
    double chunk_length_seconds = 300.0;
    int32_t audio_conv_chunk_frames = 100;

    int32_t timestamp_token_id = 151705;
    int32_t timestamp_segment_ms = 80;
    int32_t classify_num = 5000;

    int32_t n_ctx = 8192;
    int32_t n_batch = 1024;
    int32_t n_threads = -1;
    int32_t n_gpu_layers = 0;
    bool use_mmap = true;
    bool use_mlock = false;
    int32_t flash_attn_type = -1;
    int32_t type_k = -1;
    int32_t type_v = -1;
    int32_t position_components = 3;

    std::string onnx_provider = "CPU";
    int32_t onnx_device_id = 0;
    int32_t onnx_intra_op_threads = 0;
    int32_t onnx_inter_op_threads = 0;
    int32_t onnx_optimization_level = 99;
};

struct Qwen3ForcedAlignmentRequest {
    std::vector<float> waveform;
    int32_t input_sample_rate = 0;
    std::string language;
    std::vector<std::string> text_units;
    std::vector<int64_t> token_ids;
    std::vector<int32_t> timestamp_token_indices;
    int32_t audio_placeholder_start = -1;
    int32_t audio_placeholder_count = 0;
    bool allow_audio_truncation = false;
    double max_duration_seconds = 300.0;
};

struct Qwen3ForcedAlignmentSpan {
    std::string text;
    double start_sec = 0.0;
    double end_sec = 0.0;
    int32_t start_bin = 0;
    int32_t end_bin = 0;
    double confidence = 0.0;
};

struct Qwen3ForcedAlignmentResult {
    std::vector<Qwen3ForcedAlignmentSpan> spans;
    int32_t token_count = 0;
    int32_t audio_token_count = 0;
    double elapsed_ms = 0.0;
    double frontend_ms = 0.0;
    double audio_conv_ms = 0.0;
    double audio_encoder_ms = 0.0;
    double embedding_ms = 0.0;
    double backbone_ms = 0.0;
    double classifier_ms = 0.0;
};

struct Qwen3TimestampPrediction {
    int32_t bin = 0;
    double confidence = 0.0;
};

struct Qwen3ForcedAlignerClassifierHead {
    int32_t classify_num = 0;
    int32_t hidden_size = 0;
    std::vector<float> weights;

    bool is_valid() const;
    Result<void> load(const std::string &path);
    Result<Qwen3TimestampPrediction> classify(std::span<const float> hidden) const;
};

std::vector<int32_t> repair_monotonic_timestamp_bins(std::span<const int32_t> bins);
std::vector<std::string> split_forced_alignment_units(const std::string &text, const std::string &mode);
Result<void> validate_qwen3_forced_alignment_request(
    const Qwen3ForcedAlignmentRequest &request,
    const Qwen3ForcedAlignerSessionConfig &config
);

class Qwen3ForcedAligner {
public:
    Qwen3ForcedAligner();
    ~Qwen3ForcedAligner();

    Qwen3ForcedAligner(const Qwen3ForcedAligner &) = delete;
    Qwen3ForcedAligner &operator=(const Qwen3ForcedAligner &) = delete;
    Qwen3ForcedAligner(Qwen3ForcedAligner &&) noexcept;
    Qwen3ForcedAligner &operator=(Qwen3ForcedAligner &&) noexcept;

    Result<void> load(
        const Qwen3ForcedAlignerModelPaths &paths,
        const Qwen3ForcedAlignerSessionConfig &config
    );
    bool is_loaded() const;

    Result<Qwen3ForcedAlignmentResult> align(
        const Qwen3ForcedAlignmentRequest &request,
        CancellationToken *cancel = nullptr
    );

    const Qwen3ForcedAlignerSessionConfig &config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gotst
