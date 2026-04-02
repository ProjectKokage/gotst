#include "gotst/core/speech_runtime_core.hpp"

#include <algorithm>
#include <cassert>
#include <string>

int main() {
    gotst::SpeechRuntimeCore runtime;

    {
        gotst::RuntimeConfig config;
        config.asr.prefer_hybrid = false;
        config.tts.prefer_hybrid = false;

        const gotst::BackendSummary summary = runtime.inspect(config);
        assert(summary.uses_llama_core);
        assert(summary.uses_onnx_core);
        assert(!summary.asr_hybrid_requested);
        assert(!summary.tts_hybrid_requested);
        assert(summary.missing_paths.empty());
        assert(!summary.notes.empty());
    }

    {
        gotst::RuntimeConfig config;
        config.asr.frontend_model_path = "/definitely/missing/asr_frontend.onnx";
        config.asr.text_embedding_model_path = "/definitely/missing/asr_text_embedding.onnx";
        config.asr.thinker_model_path = "/definitely/missing/asr_thinker.gguf";
        config.tts.text_embedding_model_path = "/definitely/missing/tts_text_embedding.onnx";
        config.tts.speaker_embedding_model_path = "/definitely/missing/tts_speaker_embedding.onnx";
        config.tts.audio_decoder_model_path = "/definitely/missing/tts_audio_decoder.onnx";
        config.tts.talker_model_path = "/definitely/missing/tts_talker.gguf";
        config.tts.predictor_model_path = "/definitely/missing/tts_predictor.gguf";

        const gotst::BackendSummary summary = runtime.inspect(config);
        assert(summary.asr_hybrid_requested);
        assert(summary.tts_hybrid_requested);
        assert(!summary.asr_hybrid_ready);
        assert(!summary.tts_hybrid_ready);
        assert(std::find(summary.missing_paths.begin(), summary.missing_paths.end(), "asr_thinker_model_path") != summary.missing_paths.end());
        assert(std::find(summary.missing_paths.begin(), summary.missing_paths.end(), "tts_predictor_model_path") != summary.missing_paths.end());
    }

    return 0;
}
