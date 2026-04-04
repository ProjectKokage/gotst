#include "gotst/core/speech_runtime_core.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

using namespace gotst;

TEST_CASE("SpeakerMelParams defaults", "[config]") {
    SpeakerMelParams p;
    CHECK(p.sample_rate == 24000);
    CHECK(p.n_fft == 1024);
    CHECK(p.hop_length == 256);
    CHECK(p.n_mels == 128);
    CHECK(p.fmin == 0.0);
    CHECK(p.fmax == 12000.0);
}

TEST_CASE("HybridAsrConfig defaults", "[config]") {
    HybridAsrConfig c;
    CHECK(c.prefer_hybrid == true);
    CHECK(c.frontend_model_path.empty());
    CHECK(c.text_embedding_model_path.empty());
    CHECK(c.thinker_model_path.empty());
    CHECK(c.default_language.empty());
    CHECK(c.supported_languages.empty());
}

TEST_CASE("HybridTtsConfig defaults", "[config]") {
    HybridTtsConfig c;
    CHECK(c.prefer_hybrid == true);
    CHECK(c.text_embedding_model_path.empty());
    CHECK(c.speaker_embedding_model_path.empty());
    CHECK(c.audio_decoder_model_path.empty());
    CHECK(c.talker_model_path.empty());
    CHECK(c.predictor_model_path.empty());
    CHECK(c.voice_mode.empty());
    CHECK(c.speaker_encoder_model_path.empty());
    CHECK(c.speech_tokenizer_encoder_model_path.empty());
    CHECK(c.speech_tokenizer_encoder_sample_rate == 24000);
    CHECK(c.speaker_mel_params.sample_rate == 24000);
}

TEST_CASE("BackendSummary defaults", "[config]") {
    BackendSummary s;
    CHECK_FALSE(s.uses_llama_core);
    CHECK_FALSE(s.uses_onnx_core);
    CHECK_FALSE(s.asr_hybrid_requested);
    CHECK_FALSE(s.tts_hybrid_requested);
    CHECK_FALSE(s.asr_hybrid_ready);
    CHECK_FALSE(s.tts_hybrid_ready);
    CHECK_FALSE(s.speaker_encoder_ready);
    CHECK_FALSE(s.speech_tokenizer_encoder_ready);
    CHECK_FALSE(s.onnx_session_loaded);
    CHECK_FALSE(s.llama_worker_running);
    CHECK(s.missing_paths.empty());
    CHECK(s.notes.empty());
}

TEST_CASE("inspect with non-hybrid config returns basic flags", "[inspect]") {
    SpeechRuntimeCore runtime;
    RuntimeConfig config;
    config.asr.prefer_hybrid = false;
    config.tts.prefer_hybrid = false;

    const auto summary = runtime.inspect(config);

    CHECK(summary.uses_llama_core);
    CHECK(summary.uses_onnx_core);
    CHECK_FALSE(summary.asr_hybrid_requested);
    CHECK_FALSE(summary.tts_hybrid_requested);
    CHECK_FALSE(summary.asr_hybrid_ready);
    CHECK_FALSE(summary.tts_hybrid_ready);
    CHECK(summary.missing_paths.empty());
    REQUIRE_FALSE(summary.notes.empty());
}

TEST_CASE("inspect with hybrid config flags missing paths", "[inspect]") {
    SpeechRuntimeCore runtime;
    RuntimeConfig config;
    config.asr.frontend_model_path = "/definitely/missing/asr_frontend.onnx";
    config.asr.text_embedding_model_path = "/definitely/missing/asr_text_embedding.onnx";
    config.asr.thinker_model_path = "/definitely/missing/asr_thinker.gguf";
    config.tts.text_embedding_model_path = "/definitely/missing/tts_text_embedding.onnx";
    config.tts.speaker_embedding_model_path = "/definitely/missing/tts_speaker_embedding.onnx";
    config.tts.audio_decoder_model_path = "/definitely/missing/tts_audio_decoder.onnx";
    config.tts.talker_model_path = "/definitely/missing/tts_talker.gguf";
    config.tts.predictor_model_path = "/definitely/missing/tts_predictor.gguf";

    const auto summary = runtime.inspect(config);

    CHECK(summary.asr_hybrid_requested);
    CHECK(summary.tts_hybrid_requested);
    CHECK_FALSE(summary.asr_hybrid_ready);
    CHECK_FALSE(summary.tts_hybrid_ready);

    auto has = [&](const std::string &label) {
        return std::find(summary.missing_paths.begin(), summary.missing_paths.end(), label) !=
               summary.missing_paths.end();
    };
    CHECK(has("asr_thinker_model_path"));
    CHECK(has("tts_predictor_model_path"));
}

TEST_CASE("inspect with speaker encoder path flags missing", "[inspect]") {
    SpeechRuntimeCore runtime;
    RuntimeConfig config;
    config.asr.prefer_hybrid = false;
    config.tts.speaker_encoder_model_path = "/definitely/missing/speaker_encoder.onnx";

    const auto summary = runtime.inspect(config);

    CHECK(summary.tts_hybrid_requested);
    CHECK_FALSE(summary.speaker_encoder_ready);

    auto has = [&](const std::string &label) {
        return std::find(summary.missing_paths.begin(), summary.missing_paths.end(), label) !=
               summary.missing_paths.end();
    };
    CHECK(has("tts_speaker_encoder_model_path"));
}

TEST_CASE("inspect with speech tokenizer encoder path flags missing", "[inspect]") {
    SpeechRuntimeCore runtime;
    RuntimeConfig config;
    config.asr.prefer_hybrid = false;
    config.tts.speech_tokenizer_encoder_model_path = "/definitely/missing/tokenizer_enc.onnx";

    const auto summary = runtime.inspect(config);

    CHECK(summary.tts_hybrid_requested);
    CHECK_FALSE(summary.speech_tokenizer_encoder_ready);

    auto has = [&](const std::string &label) {
        return std::find(summary.missing_paths.begin(), summary.missing_paths.end(), label) !=
               summary.missing_paths.end();
    };
    CHECK(has("tts_speech_tokenizer_encoder_model_path"));
}

TEST_CASE("inspect skips speaker encoder when path is empty", "[inspect]") {
    SpeechRuntimeCore runtime;
    RuntimeConfig config;
    config.asr.prefer_hybrid = false;

    const auto summary = runtime.inspect(config);

    CHECK_FALSE(summary.speaker_encoder_ready);
    CHECK_FALSE(summary.speech_tokenizer_encoder_ready);

    auto has = [&](const std::string &label) {
        return std::find(summary.missing_paths.begin(), summary.missing_paths.end(), label) !=
               summary.missing_paths.end();
    };
    CHECK_FALSE(has("tts_speaker_encoder_model_path"));
    CHECK_FALSE(has("tts_speech_tokenizer_encoder_model_path"));
}
