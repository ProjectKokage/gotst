#include "cli_common.hpp"

#include "gotst/core/cancellation_token.hpp"
#include "gotst/core/irodori_tts_session.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace gotst_cli {

namespace {

gotst::IrodoriTtsProviderRoute stage_route(const std::string &provider) {
    const std::string effective = provider.empty() ? "CPU" : provider;
    return {effective, effective};
}

std::filesystem::path resolve_artifact(
    const ParsedArgs &args,
    const std::filesystem::path &bundle_root,
    const JsonValue *artifacts,
    const std::string &option_name,
    const std::string &manifest_key
) {
    const std::string override_path = args.value(option_name);
    if(!override_path.empty()) {
        return resolve_path(bundle_root, override_path);
    }
    return resolve_path(bundle_root, json_string(artifacts, manifest_key));
}

std::vector<gotst::IrodoriTtsBucket> parse_buckets(const JsonValue *manifest) {
    const JsonValue *bucket_array = manifest == nullptr ? nullptr : manifest->get("buckets");
    if(bucket_array == nullptr || !bucket_array->is_array()) {
        bucket_array = manifest == nullptr ? nullptr : manifest->get("bucket_definitions");
    }
    std::vector<gotst::IrodoriTtsBucket> buckets;
    if(bucket_array == nullptr || !bucket_array->is_array()) {
        return buckets;
    }
    for(const JsonValue &entry : bucket_array->array_values) {
        if(!entry.is_object()) {
            continue;
        }
        gotst::IrodoriTtsBucket bucket;
        bucket.latent_steps = json_int(&entry, "latent_steps", json_int(&entry, "latent", 0));
        bucket.text_tokens = json_int(&entry, "text_tokens", json_int(&entry, "text", 0));
        bucket.caption_tokens = json_int(&entry, "caption_tokens", json_int(&entry, "caption", 0));
        bucket.ref_steps = json_int(&entry, "ref_steps", json_int(&entry, "ref", 0));
        if(bucket.latent_steps > 0 && bucket.text_tokens > 0) {
            buckets.push_back(bucket);
        }
    }
    return buckets;
}

void apply_common_config_defaults(
    gotst::IrodoriTtsSessionConfig &config,
    const ParsedArgs &args,
    const JsonValue *manifest
) {
    const JsonValue *codec = manifest == nullptr ? nullptr : manifest->get("codec_metadata");
    const JsonValue *dimensions = manifest == nullptr ? nullptr : manifest->get("model_dimensions");

    config.sample_rate = json_int(codec, "sample_rate", config.sample_rate);
    config.latent_dim = json_int(codec, "latent_dim", config.latent_dim);
    config.codec_hop_length = json_int(codec, "hop_length", config.codec_hop_length);
    config.latent_patch_size = json_int(dimensions, "latent_patch_size", config.latent_patch_size);
    config.speaker_patch_size = json_int(dimensions, "speaker_patch_size", config.speaker_patch_size);
    config.duration_aux_dim = json_int(dimensions, "duration_aux_dim", config.duration_aux_dim);
    config.max_text_tokens = args.int_value(
        "max-text-tokens",
        json_int(dimensions, "max_text_tokens", config.max_text_tokens)
    );
    config.max_caption_tokens = args.int_value(
        "max-caption-tokens",
        json_int(dimensions, "max_caption_tokens", config.max_caption_tokens)
    );
    config.default_num_steps = args.int_value("num-steps", 6);
    config.default_t_schedule_mode = args.value("t-schedule-mode", config.default_t_schedule_mode);
    config.default_sway_coeff = args.float_value("sway-coeff", config.default_sway_coeff);
    config.default_cfg_guidance_mode = args.value("cfg-guidance-mode", "alternating");
}

gotst::Result<gotst::IrodoriTtsSessionConfig> build_config(
    const ParsedArgs &args,
    const std::filesystem::path &bundle_root,
    const std::filesystem::path &manifest_path,
    const JsonValue &manifest
) {
    const JsonValue *artifacts = manifest.get("artifacts");
    gotst::IrodoriTtsSessionConfig config;
    config.bundle_root = bundle_root.string();
    config.manifest_path = manifest_path.string();
    config.mode = gotst::parse_irodori_tts_mode(args.value("mode", json_string(&manifest, "mode", "base_v3")));
    if(config.mode == gotst::IrodoriTtsMode::Unknown) {
        return gotst::Error::invalid_argument("unsupported Irodori mode; expected base_v3, base_v2, voice_design_v2, or voice_design_v3");
    }
    config.provider_profile = args.value("provider-profile", "cpu");
    config.provider_requested = args.value("provider", "CPU");
    config.provider = args.value("provider", "CPU");
    config.strict_provider = args.bool_value("strict-provider", false);
    config.intra_op_threads = args.int_value("intra-op-threads", 0);
    config.inter_op_threads = args.int_value("inter-op-threads", 0);
    config.device_id = args.int_value("device-id", 0);
    config.optimization_level = args.int_value("optimization-level", 99);
    config.runtime_dispatch = args.value("runtime-dispatch", "force_cpu");
    config.rf_execution_mode = args.value("rf-execution-mode", "auto");
    config.require_all_artifacts = args.bool_value("require-all-artifacts", true);
    config.enable_context_kv_cache = !args.bool_value("no-context-cache", false);
    config.enable_ref_latent_cache = !args.bool_value("no-ref-cache", false);

    apply_common_config_defaults(config, args, &manifest);

    config.artifacts.tokenizer_json_path = resolve_artifact(args, bundle_root, artifacts, "tokenizer-json", "tokenizer_json_path").string();
    config.artifacts.tokenizer_config_path = resolve_artifact(args, bundle_root, artifacts, "tokenizer-config", "tokenizer_config_path").string();
    config.artifacts.model_config_json_path = resolve_artifact(args, bundle_root, artifacts, "model-config", "model_config_json_path").string();
    config.artifacts.text_encoder_onnx_path = resolve_artifact(args, bundle_root, artifacts, "text-encoder", "text_encoder_onnx_path").string();
    config.artifacts.caption_encoder_onnx_path = resolve_artifact(args, bundle_root, artifacts, "caption-encoder", "caption_encoder_onnx_path").string();
    config.artifacts.speaker_encoder_onnx_path = resolve_artifact(args, bundle_root, artifacts, "speaker-encoder", "speaker_encoder_onnx_path").string();
    config.artifacts.duration_predictor_onnx_path = resolve_artifact(args, bundle_root, artifacts, "duration-predictor", "duration_predictor_onnx_path").string();
    config.artifacts.dit_step_onnx_path = resolve_artifact(args, bundle_root, artifacts, "dit-step", "dit_step_onnx_path").string();
    config.artifacts.dacvae_encoder_onnx_path = resolve_artifact(args, bundle_root, artifacts, "dacvae-encoder", "dacvae_encoder_onnx_path").string();
    config.artifacts.dacvae_decoder_onnx_path = resolve_artifact(args, bundle_root, artifacts, "dacvae-decoder", "dacvae_decoder_onnx_path").string();

    const std::string provider = args.value("provider", "CPU");
    config.provider_routes.text_encoder = stage_route(provider);
    config.provider_routes.caption_encoder = stage_route(provider);
    config.provider_routes.speaker_encoder = stage_route(provider);
    config.provider_routes.duration_predictor = stage_route(provider);
    config.provider_routes.dit_step = stage_route(provider);
    config.provider_routes.dacvae_encoder = stage_route(provider);
    config.provider_routes.dacvae_decoder = stage_route(provider);
    config.buckets = parse_buckets(&manifest);
    return config;
}

gotst::IrodoriTtsRequest build_request(
    const ParsedArgs &args,
    const std::filesystem::path &bundle_root,
    const gotst::IrodoriTtsSessionConfig &config
) {
    gotst::IrodoriTtsRequest request;
    request.text = args.value("text");
    request.caption = args.value("caption");
    request.ref_wav_path = resolve_path(bundle_root, args.value("ref-wav")).string();
    request.ref_latent_path = resolve_path(bundle_root, args.value("ref-latent")).string();
    request.no_ref = args.bool_value("no-ref", false);
    request.seed = args.int64_value("seed", -1);
    request.num_steps = args.int_value("num-steps", config.default_num_steps);
    request.duration_scale = args.float_value("duration-scale", 1.0f);
    if(args.has("seconds")) {
        request.seconds = args.float_value("seconds", 0.0f);
    }
    request.cfg_scale_text = args.float_value("cfg-scale-text", request.cfg_scale_text);
    request.cfg_scale_caption = args.float_value("cfg-scale-caption", request.cfg_scale_caption);
    request.cfg_scale_speaker = args.float_value("cfg-scale-speaker", request.cfg_scale_speaker);
    if(args.has("cfg-scale")) {
        request.cfg_scale = args.float_value("cfg-scale", request.cfg_scale_text);
    }
    request.cfg_min_t = args.float_value("cfg-min-t", request.cfg_min_t);
    request.cfg_max_t = args.float_value("cfg-max-t", request.cfg_max_t);
    request.cfg_guidance_mode = args.value("cfg-guidance-mode", config.default_cfg_guidance_mode);
    request.t_schedule_mode = args.value("t-schedule-mode", config.default_t_schedule_mode);
    request.sway_coeff = args.float_value("sway-coeff", config.default_sway_coeff);
    request.context_kv_cache = config.enable_context_kv_cache;
    request.ref_latent_cache = config.enable_ref_latent_cache;
    request.decode_mode = args.value("decode-mode", request.decode_mode);
    return request;
}

void print_help() {
    std::cout
        << "Usage: gotst irodori-tts --bundle DIR --text TEXT --output speech.wav [options]\n\n"
        << "Required:\n"
        << "  --bundle DIR              Irodori artifact bundle root\n"
        << "  --text TEXT               Text to synthesize\n"
        << "  --output PATH             Output WAV path\n\n"
        << "Common options:\n"
        << "  --manifest PATH           Manifest path relative to bundle (default: irodori_bundle.json)\n"
        << "  --mode MODE               base_v3, base_v2, voice_design_v2, or voice_design_v3\n"
        << "  --caption TEXT            VoiceDesign caption text\n"
        << "  --no-ref                  Run without reference audio/latent\n"
        << "  --ref-wav PATH            Reference WAV path, relative to bundle unless absolute\n"
        << "  --ref-latent PATH         Reference latent path, relative to bundle unless absolute\n"
        << "  --provider PROVIDER       ONNX provider name (default: CPU)\n"
        << "  --seed N                  RNG seed\n"
        << "  --num-steps N             RF sampling steps (default: 6)\n"
        << "  --seconds S               Fixed output duration in seconds\n"
        << "  --cfg-scale-text F        Text CFG scale (default: 3.0)\n"
        << "  --cfg-scale-speaker F     Speaker CFG scale (default: 5.0)\n";
}

} // namespace

int command_irodori_tts(const ParsedArgs &args) {
    if(args.has("help") || args.has("h")) {
        print_help();
        return 0;
    }

    const std::filesystem::path bundle_root = args.value("bundle");
    const std::string text = args.value("text");
    const std::filesystem::path output_path = args.value("output");
    if(bundle_root.empty()) {
        return print_error("irodori-tts requires --bundle DIR", 2);
    }
    if(text.empty()) {
        return print_error("irodori-tts requires --text TEXT", 2);
    }
    if(output_path.empty()) {
        return print_error("irodori-tts requires --output PATH", 2);
    }

    const std::filesystem::path manifest_path = resolve_path(bundle_root, args.value("manifest", "irodori_bundle.json"));
    auto manifest = read_json_file(manifest_path);
    if(!manifest.is_ok()) {
        return print_error(manifest.get_error());
    }

    auto config = build_config(args, bundle_root, manifest_path, manifest.value());
    if(!config.is_ok()) {
        return print_error(config.get_error());
    }

    gotst::IrodoriTtsRequest request = build_request(
        args,
        bundle_root,
        config.value()
    );
    if(gotst::irodori_tts_mode_uses_caption(config.value().mode) && request.caption.empty()) {
        return print_error("Irodori VoiceDesign requires --caption TEXT", 2);
    }

    gotst::IrodoriTtsSession session;
    auto load_result = session.load(config.value());
    if(!load_result.is_ok()) {
        return print_error(load_result.get_error());
    }

    gotst::CancellationToken cancel;
    auto synthesis = session.synthesize(request, &cancel);
    if(!synthesis.is_ok()) {
        return print_error(synthesis.get_error());
    }

    auto write_result = write_wav_mono_f32(output_path, synthesis.value().waveform, synthesis.value().sample_rate);
    if(!write_result.is_ok()) {
        return print_error(write_result.get_error());
    }

    std::cout << "wrote " << output_path.string()
              << " (samples=" << synthesis.value().waveform.size()
              << ", sample_rate=" << synthesis.value().sample_rate
              << ", elapsed_ms=" << synthesis.value().elapsed_ms
              << ")\n";
    return 0;
}

} // namespace gotst_cli
