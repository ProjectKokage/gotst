#pragma once

#include "gotst/core/cancellation_token.hpp"
#include "gotst/core/result.hpp"
#include "gotst/core/tts_code_generator.hpp"
#include "gotst/core/tts_prompt_assembly.hpp"
#include "gotst/core/tts_waveform_decoder.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gotst {

struct QwenTtsPipelineConfig {
    std::string tokenizer_json_path;
    std::string model_config_path;
    std::string generation_config_path;
    std::string text_embedding_path;
    std::string text_projection_path;
    std::string codec_embedding_path;
    std::string predictor_embedding_path;
    std::string talker_gguf_path;
    std::string predictor_gguf_path;
    std::string decoder_onnx_path;
    std::string speaker_embedding_path;
    std::string custom_voice_config_path;
    std::string custom_voice_speaker_name;
    std::string mode = "base";
    std::string provider = "CPU";
    std::string decoder_provider_requested = "CPU";
    std::string decoder_provider = "CPU";
    int32_t intra_op_threads = 0;
    int32_t inter_op_threads = 0;
    int32_t optimization_level = 99;
    std::string decoder_optimized_model_path;
    int32_t decoder_intra_op_threads = 0;
    int32_t decoder_inter_op_threads = 0;
    int32_t decoder_optimization_level = 99;
    int32_t sample_rate = 24000;
    int32_t max_text_tokens = 512;
    float target_frames_per_text_token = 5.0f;
    int32_t target_frame_padding = 2;
    int32_t min_frames_before_eos = 8;
    int32_t max_frames = 96;
    bool force_japanese_language = true;
    bool use_style_instruction = true;
    bool use_icl_voice_clone = false;
    std::string icl_ref_text;
    bool normalize_waveform = true;
    float waveform_gain = 0.9f;
    int32_t stateful_chunk_frames = 12;
    int32_t talker_n_ctx = 1024;
    int32_t talker_n_batch = 1024;
    int32_t predictor_n_ctx = 128;
    int32_t predictor_n_batch = 128;
    int32_t n_threads = -1;
    int32_t n_gpu_layers = 0;
    int32_t predictor_n_gpu_layers = -2;
    bool use_mmap = true;
    bool use_mlock = false;
    int32_t flash_attn_type = -1;
    int32_t type_k = -1;
    int32_t type_v = -1;
};

struct QwenTtsPipelineRequest {
    std::string text;
    std::string mode;
    std::string style_instruction;
    std::string voice_design;
    int64_t seed = 1;
};

struct QwenTtsPreparedPrompt {
    TtsPromptAssemblyResult prompt;
    TtsSamplingConfig sampling;
    int32_t target_frames = 0;
    int32_t visible_frame_count = 0;
    int32_t visible_code_count = 0;
    int32_t decode_prefix_frames = 0;
    int32_t codec_groups = 16;
    std::vector<int64_t> decoder_prefix_codes;
};

struct QwenTtsStreamChunk {
    std::vector<float> waveform;
    int32_t sample_rate = 24000;
    int32_t frame_count = 0;
    int32_t code_count = 0;
    int32_t visible_frame_count = 0;
    int32_t visible_code_count = 0;
    int32_t chunk_index = 0;
    bool is_final = false;
    double elapsed_ms = 0.0;
    double codegen_ms = 0.0;
    double talker_prefill_ms = 0.0;
    double talker_decode_ms = 0.0;
    double predictor_ms = 0.0;
    double onnx_embedding_ms = 0.0;
    double codegen_other_ms = 0.0;
    double decoder_ms = 0.0;
    double decoder_inference_ms = 0.0;
    double decoder_postprocess_ms = 0.0;
    std::string decoder_provider_requested;
    std::string decoder_provider_effective;
    int32_t decoder_cpu_fallback_node_count = -1;
    bool decoder_fixed_shape = false;
};

struct QwenTtsSynthesisResult {
    std::vector<float> waveform;
    int32_t sample_rate = 24000;
    int32_t frame_count = 0;
    int32_t code_count = 0;
    int32_t visible_frame_count = 0;
    int32_t visible_code_count = 0;
    double elapsed_ms = 0.0;
    double codegen_ms = 0.0;
    double talker_prefill_ms = 0.0;
    double talker_decode_ms = 0.0;
    double predictor_ms = 0.0;
    double onnx_embedding_ms = 0.0;
    double codegen_other_ms = 0.0;
    double decoder_ms = 0.0;
    double decoder_inference_ms = 0.0;
    double decoder_postprocess_ms = 0.0;
};

using QwenTtsStreamCallback = std::function<void(const QwenTtsStreamChunk &)>;

class QwenTtsPipeline {
public:
    QwenTtsPipeline();
    ~QwenTtsPipeline();

    QwenTtsPipeline(const QwenTtsPipeline &) = delete;
    QwenTtsPipeline &operator=(const QwenTtsPipeline &) = delete;
    QwenTtsPipeline(QwenTtsPipeline &&) noexcept;
    QwenTtsPipeline &operator=(QwenTtsPipeline &&) noexcept;

    Result<void> load(const QwenTtsPipelineConfig &config);
    bool is_loaded() const;

    Result<QwenTtsPreparedPrompt> prepare_prompt(const QwenTtsPipelineRequest &request);
    Result<QwenTtsSynthesisResult> synthesize(
        const QwenTtsPipelineRequest &request,
        CancellationToken *cancel = nullptr
    );
    Result<QwenTtsSynthesisResult> synthesize_streaming(
        const QwenTtsPipelineRequest &request,
        int32_t chunk_frames,
        QwenTtsStreamCallback on_chunk,
        CancellationToken *cancel = nullptr
    );

    std::vector<std::string> custom_voice_speaker_names() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gotst
