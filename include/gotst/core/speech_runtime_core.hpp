#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gotst {

struct SpeakerMelParams {
    int64_t sample_rate = 24000;
    int64_t n_fft = 1024;
    int64_t hop_length = 256;
    int64_t n_mels = 128;
    double fmin = 0.0;
    double fmax = 12000.0;
};

struct CustomVoiceSpeakerEntry {
    std::string name;
    std::vector<int64_t> token_ids;
};

struct HybridAsrConfig {
    std::string frontend_model_path;
    std::string text_embedding_model_path;
    std::string thinker_model_path;
    bool prefer_hybrid = true;
    std::string default_language;
    std::vector<std::string> supported_languages;
};

struct HybridTtsConfig {
    std::string text_embedding_model_path;
    std::string speaker_embedding_model_path;
    std::string audio_decoder_model_path;
    std::string talker_model_path;
    std::string predictor_model_path;
    bool prefer_hybrid = true;
    std::string voice_mode;
    std::string speaker_encoder_model_path;
    std::string speech_tokenizer_encoder_model_path;
    int64_t speech_tokenizer_encoder_sample_rate = 24000;
    SpeakerMelParams speaker_mel_params;
    std::string custom_voice_config_path;
    std::vector<CustomVoiceSpeakerEntry> custom_voice_speakers;
};

struct RuntimeConfig {
    HybridAsrConfig asr;
    HybridTtsConfig tts;
};

struct BackendSummary {
    bool uses_llama_core = false;
    bool uses_onnx_core = false;
    bool asr_hybrid_requested = false;
    bool tts_hybrid_requested = false;
    bool asr_hybrid_ready = false;
    bool tts_hybrid_ready = false;
    bool speaker_encoder_ready = false;
    bool speech_tokenizer_encoder_ready = false;
    bool onnx_session_loaded = false;
    bool llama_worker_running = false;
    std::vector<std::string> missing_paths;
    std::vector<std::string> notes;
};

class SpeechRuntimeCore {
public:
    SpeechRuntimeCore();

    [[nodiscard]] BackendSummary inspect(const RuntimeConfig &config) const;

private:
    void append_missing_if_needed(const std::string &path_label,
                                  const std::string &path_value,
                                  std::vector<std::string> &missing_paths) const;
};

} // namespace gotst
