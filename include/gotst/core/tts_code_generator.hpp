#pragma once

#include "gotst/core/cancellation_token.hpp"
#include "gotst/core/result.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gotst {

struct TtsModelPaths {
    std::string talker_gguf_path;
    std::string predictor_gguf_path;
    std::string codec_embedding_onnx_path;
    std::string predictor_embedding_onnx_path;
};

struct TtsSessionConfig {
    int32_t talker_n_ctx = 1024;
    int32_t talker_n_batch = 1024;
    int32_t predictor_n_ctx = 128;
    int32_t predictor_n_batch = 128;
    int32_t n_threads = -1;
    int32_t n_gpu_layers = 0;
    bool use_mmap = true;
    bool use_mlock = false;
    int32_t flash_attn_type = -1;
    int32_t type_k = -1;
    int32_t type_v = -1;
    int32_t talker_position_components = 4;
    int32_t predictor_position_components = 1;
};

struct TtsSamplingConfig {
    int32_t codebook_size = 2048;
    int32_t residual_groups = 15;
    int32_t target_frames = 96;
    int32_t min_frames_before_eos = 8;
    int32_t hidden_size = 1024;
    int32_t eos_token_id = 2150;
    float eos_logit_margin = 0.0f;
    bool do_sample = true;
    int32_t top_k = 50;
    float top_p = 1.0f;
    float temperature = 0.9f;
    bool sub_do_sample = true;
    int32_t sub_top_k = 50;
    float sub_top_p = 1.0f;
    float sub_temperature = 0.9f;
    float repetition_penalty = 1.05f;
    int64_t rng_seed = 1;
};

struct TtsGenerateResult {
    std::vector<int64_t> codes;
    int32_t frame_count = 0;
    int32_t codes_per_frame = 0;
};

struct TtsFrameChunk {
    std::vector<int64_t> codes;
    int32_t frame_count = 0;
    int32_t codes_per_frame = 0;
    bool is_final = false;
};

using FrameChunkCallback = std::function<void(TtsFrameChunk)>;

class TtsCodeGenerator {
public:
    TtsCodeGenerator();
    ~TtsCodeGenerator();

    TtsCodeGenerator(const TtsCodeGenerator &) = delete;
    TtsCodeGenerator &operator=(const TtsCodeGenerator &) = delete;
    TtsCodeGenerator(TtsCodeGenerator &&) noexcept;
    TtsCodeGenerator &operator=(TtsCodeGenerator &&) noexcept;

    Result<void> load(const TtsModelPaths &paths, const TtsSessionConfig &config);
    bool is_loaded() const;

    Result<TtsGenerateResult> generate(
        std::span<const float> initial_language_input,
        int32_t initial_sequence_length,
        std::span<const float> trailing_text_hidden,
        int32_t trailing_text_length,
        std::span<const float> tts_pad_embedding,
        const TtsSamplingConfig &params,
        CancellationToken *cancel = nullptr
    );

    Result<TtsGenerateResult> generate_streaming(
        std::span<const float> initial_language_input,
        int32_t initial_sequence_length,
        std::span<const float> trailing_text_hidden,
        int32_t trailing_text_length,
        std::span<const float> tts_pad_embedding,
        const TtsSamplingConfig &params,
        int32_t chunk_frames,
        FrameChunkCallback on_chunk,
        CancellationToken *cancel = nullptr
    );

private:
    Result<TtsGenerateResult> run_generation_impl(
        std::span<const float> initial_language_input,
        int32_t initial_sequence_length,
        std::span<const float> trailing_text_hidden,
        int32_t trailing_text_length,
        std::span<const float> tts_pad_embedding,
        const TtsSamplingConfig &params,
        int32_t chunk_frames,
        FrameChunkCallback on_chunk,
        CancellationToken *cancel,
        const char *log_tag
    );

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gotst
