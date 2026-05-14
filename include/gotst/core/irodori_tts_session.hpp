#pragma once

#include "gotst/core/cancellation_token.hpp"
#include "gotst/core/result.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace gotst {

enum class IrodoriTtsMode {
    Unknown,
    BaseV3,
    BaseV2,
    VoiceDesignV2,
};

struct IrodoriTtsBucket {
    int32_t latent_steps = 0;
    int32_t text_tokens = 0;
    int32_t caption_tokens = 0;
    int32_t ref_steps = 0;
};

struct IrodoriTtsArtifactPaths {
    std::string tokenizer_json_path;
    std::string tokenizer_config_path;
    std::string model_config_json_path;
    std::string text_encoder_onnx_path;
    std::string caption_encoder_onnx_path;
    std::string speaker_encoder_onnx_path;
    std::string duration_predictor_onnx_path;
    std::string dit_step_onnx_path;
    std::string dacvae_encoder_onnx_path;
    std::string dacvae_decoder_onnx_path;
};

struct IrodoriTtsProviderRoute {
    std::string provider_requested;
    std::string provider;
};

struct IrodoriTtsProviderRoutes {
    IrodoriTtsProviderRoute text_encoder;
    IrodoriTtsProviderRoute caption_encoder;
    IrodoriTtsProviderRoute speaker_encoder;
    IrodoriTtsProviderRoute duration_predictor;
    IrodoriTtsProviderRoute dit_step;
    IrodoriTtsProviderRoute dacvae_encoder;
    IrodoriTtsProviderRoute dacvae_decoder;
};

struct IrodoriTtsStaticArtifact {
    IrodoriTtsBucket bucket;
    std::string dit_step_onnx_path;
    std::string dacvae_decoder_onnx_path;
    std::string rf_sampler_6_step_onnx_path;
    std::string rf_sampler_8_step_onnx_path;
};

struct IrodoriTtsSessionConfig {
    std::string bundle_root;
    std::string manifest_path;
    IrodoriTtsMode mode = IrodoriTtsMode::Unknown;
    std::string provider_profile = "cpu";
    std::string provider_requested = "CPU";
    std::string provider = "CPU";
    bool strict_provider = false;
    int32_t intra_op_threads = 0;
    int32_t inter_op_threads = 0;
    int32_t device_id = 0;
    int32_t optimization_level = 99;
    std::string optimized_model_path;
    std::map<std::string, std::string> provider_options;
    std::map<std::string, std::string> session_options;
    bool ort_enable_profiling = false;
    std::string ort_profiling_prefix;
    int32_t ort_log_severity_level = -1;
    std::string runtime_dispatch = "force_cpu";
    std::string rf_execution_mode = "auto";
    std::string dispatch_recommendation_path;
    bool print_provider_diagnostics = false;
    int32_t sample_rate = 48000;
    int32_t latent_dim = 32;
    int32_t latent_patch_size = 1;
    int32_t speaker_patch_size = 1;
    int32_t codec_hop_length = 640;
    int32_t duration_aux_dim = 14;
    int32_t max_text_tokens = 256;
    int32_t max_caption_tokens = 512;
    int32_t max_ref_seconds = 30;
    int32_t default_num_steps = 8;
    std::string default_t_schedule_mode = "sway";
    float default_sway_coeff = -1.0f;
    std::string default_cfg_guidance_mode;
    bool require_all_artifacts = true;
    bool enable_context_kv_cache = true;
    bool enable_ref_latent_cache = true;
    bool enable_coreml_unrolled_rf_sampler = false;
    std::vector<IrodoriTtsBucket> buckets;
    IrodoriTtsArtifactPaths artifacts;
    IrodoriTtsProviderRoutes provider_routes;
    std::vector<IrodoriTtsStaticArtifact> coreml_static_artifacts;
};

struct IrodoriTtsRequest {
    std::string text;
    std::string caption;
    std::vector<int64_t> text_token_ids;
    std::vector<uint8_t> text_token_mask;
    std::vector<int64_t> caption_token_ids;
    std::vector<uint8_t> caption_token_mask;
    std::string ref_wav_path;
    std::string ref_latent_path;
    std::vector<float> ref_latent;
    int32_t ref_latent_steps = 0;
    bool no_ref = false;
    int64_t seed = -1;
    int32_t num_steps = 8;
    float duration_scale = 1.0f;
    std::optional<float> seconds;
    float cfg_scale_text = 3.0f;
    float cfg_scale_caption = 3.0f;
    float cfg_scale_speaker = 5.0f;
    std::optional<float> cfg_scale;
    float cfg_min_t = 0.5f;
    float cfg_max_t = 1.0f;
    std::string cfg_guidance_mode;
    std::string t_schedule_mode = "sway";
    float sway_coeff = -1.0f;
    bool context_kv_cache = true;
    bool ref_latent_cache = true;
    std::string decode_mode = "sequential";
};

struct IrodoriTtsSynthesisResult {
    std::vector<float> waveform;
    int32_t sample_rate = 48000;
    int32_t frame_count = 0;
    int32_t selected_bucket_latent_steps = 0;
    std::string selected_bucket;
    std::string mode;
    std::string provider_profile;
    std::string provider_requested = "CPU";
    std::string provider_effective = "CPU";
    std::map<std::string, std::string> provider_requested_by_stage;
    std::map<std::string, std::string> provider_effective_by_stage;
    std::map<std::string, double> timings_ms;
    std::map<std::string, double> instrumentation_ms;
    std::map<std::string, std::string> diagnostics;
    double elapsed_ms = 0.0;
    bool cache_hit = false;
};

IrodoriTtsMode parse_irodori_tts_mode(const std::string &mode);
std::string irodori_tts_mode_name(IrodoriTtsMode mode);
bool irodori_tts_mode_uses_caption(IrodoriTtsMode mode);
bool irodori_tts_mode_uses_speaker(IrodoriTtsMode mode);

Result<std::vector<float>> build_irodori_sway_schedule(
    int32_t num_steps,
    const std::string &mode,
    float sway_coeff
);

Result<IrodoriTtsBucket> select_irodori_bucket(
    const std::vector<IrodoriTtsBucket> &buckets,
    int32_t latent_steps,
    int32_t text_tokens,
    int32_t caption_tokens,
    int32_t ref_steps
);

Result<std::vector<std::string>> build_irodori_cfg_branches(
    IrodoriTtsMode mode,
    const std::string &guidance_mode,
    float cfg_scale_text,
    float cfg_scale_caption,
    float cfg_scale_speaker
);

std::string build_irodori_condition_cache_key(
    IrodoriTtsMode mode,
    const std::string &text,
    const std::string &caption,
    const std::string &ref_latent_path,
    bool no_ref
);

class IrodoriTtsSession {
public:
    IrodoriTtsSession();
    ~IrodoriTtsSession();

    IrodoriTtsSession(const IrodoriTtsSession &) = delete;
    IrodoriTtsSession &operator=(const IrodoriTtsSession &) = delete;
    IrodoriTtsSession(IrodoriTtsSession &&) noexcept;
    IrodoriTtsSession &operator=(IrodoriTtsSession &&) noexcept;

    Result<void> load(const IrodoriTtsSessionConfig &config);
    bool is_loaded() const;
    bool is_execution_ready() const;

    Result<IrodoriTtsSynthesisResult> synthesize(
        const IrodoriTtsRequest &request,
        CancellationToken *cancel
    );

    const IrodoriTtsSessionConfig &config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gotst
