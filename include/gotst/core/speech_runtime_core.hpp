#pragma once

#include <string>
#include <vector>

namespace gotst {

struct HybridAsrConfig {
    std::string frontend_model_path;
    std::string text_embedding_model_path;
    std::string thinker_model_path;
    bool prefer_hybrid = true;
};

struct HybridTtsConfig {
    std::string text_embedding_model_path;
    std::string speaker_embedding_model_path;
    std::string audio_decoder_model_path;
    std::string talker_model_path;
    std::string predictor_model_path;
    bool prefer_hybrid = true;
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
