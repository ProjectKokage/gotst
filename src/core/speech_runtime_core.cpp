#include "gotst/core/speech_runtime_core.hpp"

#include <gonx/core/session.hpp>

#include <godot_llama/llama_params.hpp>
#include <godot_llama/worker.hpp>

#include <filesystem>

namespace gotst {

namespace {

bool file_exists(const std::string &path_value) {
    if(path_value.empty()) {
        return false;
    }
    return std::filesystem::exists(std::filesystem::path(path_value));
}

} // namespace

SpeechRuntimeCore::SpeechRuntimeCore() = default;

BackendSummary SpeechRuntimeCore::inspect(const RuntimeConfig &config) const {
    BackendSummary summary;
    summary.uses_llama_core = true;
    summary.uses_onnx_core = true;
    summary.asr_hybrid_requested = config.asr.prefer_hybrid;
    summary.tts_hybrid_requested = config.tts.prefer_hybrid;

    gonx::InferenceSession onnx_probe_session;
    summary.onnx_session_loaded = onnx_probe_session.is_loaded();

    godot_llama::InferenceWorker llama_worker;
    summary.llama_worker_running = llama_worker.is_running();

    godot_llama::ModelConfig llama_defaults;
    llama_defaults.disable_thinking = true;
    (void)llama_defaults;

    if(summary.asr_hybrid_requested) {
        append_missing_if_needed("asr_frontend_model_path", config.asr.frontend_model_path, summary.missing_paths);
        append_missing_if_needed(
            "asr_text_embedding_model_path",
            config.asr.text_embedding_model_path,
            summary.missing_paths
        );
        append_missing_if_needed("asr_thinker_model_path", config.asr.thinker_model_path, summary.missing_paths);
        summary.asr_hybrid_ready =
            file_exists(config.asr.frontend_model_path) &&
            file_exists(config.asr.text_embedding_model_path) &&
            file_exists(config.asr.thinker_model_path);
    }

    if(summary.tts_hybrid_requested) {
        append_missing_if_needed(
            "tts_text_embedding_model_path",
            config.tts.text_embedding_model_path,
            summary.missing_paths
        );
        append_missing_if_needed(
            "tts_speaker_embedding_model_path",
            config.tts.speaker_embedding_model_path,
            summary.missing_paths
        );
        append_missing_if_needed(
            "tts_audio_decoder_model_path",
            config.tts.audio_decoder_model_path,
            summary.missing_paths
        );
        append_missing_if_needed("tts_talker_model_path", config.tts.talker_model_path, summary.missing_paths);
        append_missing_if_needed("tts_predictor_model_path", config.tts.predictor_model_path, summary.missing_paths);
        summary.tts_hybrid_ready =
            file_exists(config.tts.text_embedding_model_path) &&
            file_exists(config.tts.speaker_embedding_model_path) &&
            file_exists(config.tts.audio_decoder_model_path) &&
            file_exists(config.tts.talker_model_path) &&
            file_exists(config.tts.predictor_model_path);

        if(!config.tts.speaker_encoder_model_path.empty()) {
            append_missing_if_needed(
                "tts_speaker_encoder_model_path",
                config.tts.speaker_encoder_model_path,
                summary.missing_paths
            );
            summary.speaker_encoder_ready = file_exists(config.tts.speaker_encoder_model_path);
        }

        if(!config.tts.speech_tokenizer_encoder_model_path.empty()) {
            append_missing_if_needed(
                "tts_speech_tokenizer_encoder_model_path",
                config.tts.speech_tokenizer_encoder_model_path,
                summary.missing_paths
            );
            summary.speech_tokenizer_encoder_ready = file_exists(config.tts.speech_tokenizer_encoder_model_path);
        }
    }

    summary.notes.emplace_back("Scaffold only: gotst does not load models or run inference yet.");
    summary.notes.emplace_back("The current API verifies dependency wiring and speech bundle configuration only.");

    return summary;
}

void SpeechRuntimeCore::append_missing_if_needed(const std::string &path_label,
                                                 const std::string &path_value,
                                                 std::vector<std::string> &missing_paths) const {
    if(!file_exists(path_value)) {
        missing_paths.push_back(path_label);
    }
}

} // namespace gotst
