#include "cli_common.hpp"

#include "gotst/core/cancellation_token.hpp"
#include "gotst/core/qwen_tts_pipeline.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace gotst_cli {

namespace {

struct QwenCliConfig {
    gotst::QwenTtsPipelineConfig pipeline;
    gotst::QwenTtsPipelineRequest request;
    std::filesystem::path bundle_root;
    std::filesystem::path output_path;
};

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::filesystem::path resolve_bundle_path(
    const QwenCliConfig &config,
    const ParsedArgs &args,
    const std::string &option_name,
    const std::string &fallback
) {
    return resolve_path(config.bundle_root, args.value(option_name, fallback));
}

std::filesystem::path resolve_cli_path(const ParsedArgs &args, const std::string &option_name) {
    return resolve_path({}, args.value(option_name));
}

gotst::Result<QwenCliConfig> build_cli_config(const ParsedArgs &args) {
    QwenCliConfig config;
    config.bundle_root = args.value("bundle");
    config.request.text = args.value("text");
    config.output_path = args.value("output");
    if(config.bundle_root.empty()) {
        return gotst::Error::invalid_argument("qwen-tts requires --bundle DIR");
    }
    if(config.request.text.empty()) {
        return gotst::Error::invalid_argument("qwen-tts requires --text TEXT");
    }
    if(config.output_path.empty()) {
        return gotst::Error::invalid_argument("qwen-tts requires --output PATH");
    }

    gotst::QwenTtsPipelineConfig &pipeline = config.pipeline;
    pipeline.mode = lower_ascii(args.value("mode", pipeline.mode));
    if(pipeline.mode != "base" && pipeline.mode != "voice_design" && pipeline.mode != "custom_voice") {
        return gotst::Error::invalid_argument("unsupported mode; expected base, voice_design, or custom_voice");
    }

    pipeline.provider = args.value("provider", pipeline.provider);
    pipeline.decoder_provider_requested = args.value("decoder-provider", pipeline.decoder_provider_requested);
    pipeline.decoder_provider = args.value("decoder-provider", pipeline.decoder_provider);
    pipeline.intra_op_threads = args.int_value("intra-op-threads", pipeline.intra_op_threads);
    pipeline.inter_op_threads = args.int_value("inter-op-threads", pipeline.inter_op_threads);
    pipeline.optimization_level = args.int_value("optimization-level", pipeline.optimization_level);
    pipeline.decoder_intra_op_threads = pipeline.intra_op_threads;
    pipeline.decoder_inter_op_threads = pipeline.inter_op_threads;
    pipeline.decoder_optimization_level = pipeline.optimization_level;
    pipeline.sample_rate = args.int_value("sample-rate", pipeline.sample_rate);
    pipeline.max_text_tokens = args.int_value("max-text-tokens", pipeline.max_text_tokens);
    pipeline.target_frames_per_text_token =
        args.float_value("target-frames-per-text-token", pipeline.target_frames_per_text_token);
    pipeline.target_frame_padding = args.int_value("target-frame-padding", pipeline.target_frame_padding);
    pipeline.min_frames_before_eos = args.int_value("min-frames-before-eos", pipeline.min_frames_before_eos);
    pipeline.max_frames = args.int_value("max-frames", pipeline.max_frames);
    pipeline.force_japanese_language = !args.bool_value("no-force-japanese-language", false);
    pipeline.use_style_instruction = !args.bool_value("no-style", false);
    pipeline.use_icl_voice_clone = args.bool_value("icl", pipeline.use_icl_voice_clone);
    pipeline.icl_ref_text = args.value("icl-ref-text", pipeline.icl_ref_text);
    pipeline.normalize_waveform = !args.bool_value("no-normalize", false);
    pipeline.waveform_gain = args.float_value("waveform-gain", pipeline.waveform_gain);
    pipeline.stateful_chunk_frames = args.int_value("stateful-chunk-frames", pipeline.stateful_chunk_frames);
    pipeline.talker_n_ctx = args.int_value("talker-n-ctx", pipeline.talker_n_ctx);
    pipeline.talker_n_batch = args.int_value("talker-n-batch", pipeline.talker_n_batch);
    pipeline.predictor_n_ctx = args.int_value("predictor-n-ctx", pipeline.predictor_n_ctx);
    pipeline.predictor_n_batch = args.int_value("predictor-n-batch", pipeline.predictor_n_batch);
    pipeline.n_threads = args.int_value("n-threads", pipeline.n_threads);
    pipeline.n_gpu_layers = args.int_value("n-gpu-layers", 99);
    pipeline.predictor_n_gpu_layers = args.int_value("predictor-n-gpu-layers", pipeline.predictor_n_gpu_layers);
    pipeline.use_mmap = !args.bool_value("no-mmap", false);
    pipeline.use_mlock = args.bool_value("mlock", pipeline.use_mlock);
    pipeline.flash_attn_type = args.int_value("flash-attn-type", 1);
    pipeline.type_k = args.int_value("type-k", pipeline.type_k);
    pipeline.type_v = args.int_value("type-v", pipeline.type_v);

    pipeline.tokenizer_json_path =
        resolve_bundle_path(config, args, "tokenizer-json", "tokenizer.json").string();
    pipeline.model_config_path =
        resolve_bundle_path(config, args, "config-json", "config.json").string();
    pipeline.generation_config_path =
        resolve_bundle_path(config, args, "generation-config", "generation_config.json").string();
    pipeline.text_embedding_path =
        resolve_bundle_path(config, args, "text-embedding", "talker_text_embedding.onnx").string();
    pipeline.text_projection_path =
        resolve_bundle_path(config, args, "text-projection", "talker_text_projection.opt.onnx").string();
    pipeline.codec_embedding_path =
        resolve_bundle_path(config, args, "codec-embedding", "talker_embedding.onnx").string();
    pipeline.predictor_embedding_path =
        resolve_bundle_path(config, args, "predictor-embedding", "talker_code_predictor_embedding.onnx").string();
    pipeline.talker_gguf_path =
        resolve_bundle_path(config, args, "talker-gguf", "talker.q4_k_m.gguf").string();
    pipeline.predictor_gguf_path =
        resolve_bundle_path(config, args, "predictor-gguf", "predictor.gguf").string();
    pipeline.decoder_onnx_path =
        resolve_bundle_path(config, args, "decoder", "speech_tokenizer_decoder_stateful.onnx").string();
    pipeline.speaker_embedding_path = resolve_cli_path(args, "speaker-embedding").string();
    pipeline.custom_voice_config_path = resolve_cli_path(args, "custom-voice-config").string();
    pipeline.custom_voice_speaker_name = args.value("speaker");

    config.request.mode = pipeline.mode;
    config.request.style_instruction = args.value(
        "style-instruction",
        "Please read the following assistant response naturally in Japanese."
    );
    config.request.voice_design = args.value("voice-design", "");
    config.request.seed = args.int64_value("seed", 1);
    return config;
}

[[noreturn]] void fast_exit_after_qwen_cli(int exit_code) {
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    std::_Exit(exit_code);
}

[[noreturn]] void print_error_and_fast_exit(const gotst::Error &error) {
    const int exit_code = print_error(error);
    fast_exit_after_qwen_cli(exit_code);
}

void print_help() {
    std::cout
        << "Usage: gotst qwen-tts --bundle DIR --text TEXT --output speech.wav [options]\n\n"
        << "Required:\n"
        << "  --bundle DIR              Qwen3-TTS runtime artifact bundle root\n"
        << "  --text TEXT               Text to synthesize\n"
        << "  --output PATH             Output WAV path\n\n"
        << "Common options:\n"
        << "  --mode MODE               base, voice_design, or custom_voice (default: base)\n"
        << "  --speaker-embedding PATH  Base-mode speaker embedding JSON\n"
        << "  --style-instruction TEXT  Leading style instruction for base/custom voice\n"
        << "  --voice-design TEXT       Leading voice design text for voice_design mode\n"
        << "  --custom-voice-config PATH  JSON with spk_id mapping for custom_voice\n"
        << "  --speaker NAME            CustomVoice speaker name\n"
        << "  --icl                     Enable base-mode ICL voice clone when ref_codes exist\n"
        << "  --provider PROVIDER       ONNX prep provider (default: CPU)\n"
        << "  --decoder-provider PROVIDER  Waveform decoder provider (default: CPU)\n"
        << "  --seed N                  Sampling seed (default: 1)\n"
        << "  --max-frames N            Maximum generated codec frames (default: 96)\n"
        << "  --no-style                Disable base/custom style instruction\n";
}

} // namespace

int command_qwen_tts(const ParsedArgs &args) {
    if(args.has("help") || args.has("h")) {
        print_help();
        return 0;
    }

    auto cli_config = build_cli_config(args);
    if(!cli_config.is_ok()) {
        return print_error(cli_config.get_error());
    }
    QwenCliConfig config = std::move(cli_config.value());

    gotst::QwenTtsPipeline pipeline;
    auto loaded = pipeline.load(config.pipeline);
    if(!loaded.is_ok()) {
        print_error_and_fast_exit(loaded.get_error());
    }

    gotst::CancellationToken cancel;
    const auto started = std::chrono::steady_clock::now();
    auto synthesized = pipeline.synthesize(config.request, &cancel);
    if(!synthesized.is_ok()) {
        print_error_and_fast_exit(synthesized.get_error());
    }

    auto written = write_wav_mono_f32(
        config.output_path,
        synthesized.value().waveform,
        synthesized.value().sample_rate
    );
    if(!written.is_ok()) {
        print_error_and_fast_exit(written.get_error());
    }

    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started
    ).count();
    std::cout << "wrote " << config.output_path.string()
              << " (samples=" << synthesized.value().waveform.size()
              << ", sample_rate=" << synthesized.value().sample_rate
              << ", frames=" << synthesized.value().frame_count
              << ", codes=" << synthesized.value().code_count
              << ", codegen_ms=" << synthesized.value().codegen_ms
              << ", decoder_ms=" << synthesized.value().decoder_ms
              << ", elapsed_ms=" << wall_ms
              << ")\n";
    fast_exit_after_qwen_cli(0);
}

} // namespace gotst_cli
