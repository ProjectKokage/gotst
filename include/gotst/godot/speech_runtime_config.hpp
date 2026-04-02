#pragma once

#include "gotst/core/speech_runtime_core.hpp"

#include <godot_cpp/classes/resource.hpp>

namespace godot {

class GotstSpeechRuntimeConfig : public Resource {
    GDCLASS(GotstSpeechRuntimeConfig, Resource)

public:
    void set_asr_frontend_model_path(const String &value);
    String get_asr_frontend_model_path() const;

    void set_asr_text_embedding_model_path(const String &value);
    String get_asr_text_embedding_model_path() const;

    void set_asr_thinker_model_path(const String &value);
    String get_asr_thinker_model_path() const;

    void set_tts_text_embedding_model_path(const String &value);
    String get_tts_text_embedding_model_path() const;

    void set_tts_speaker_embedding_model_path(const String &value);
    String get_tts_speaker_embedding_model_path() const;

    void set_tts_audio_decoder_model_path(const String &value);
    String get_tts_audio_decoder_model_path() const;

    void set_tts_talker_model_path(const String &value);
    String get_tts_talker_model_path() const;

    void set_tts_predictor_model_path(const String &value);
    String get_tts_predictor_model_path() const;

    void set_prefer_hybrid_asr(bool value);
    bool is_prefer_hybrid_asr() const;

    void set_prefer_hybrid_tts(bool value);
    bool is_prefer_hybrid_tts() const;

    [[nodiscard]] gotst::RuntimeConfig build_native_config() const;

protected:
    static void _bind_methods();

private:
    String asr_frontend_model_path_;
    String asr_text_embedding_model_path_;
    String asr_thinker_model_path_;
    String tts_text_embedding_model_path_;
    String tts_speaker_embedding_model_path_;
    String tts_audio_decoder_model_path_;
    String tts_talker_model_path_;
    String tts_predictor_model_path_;
    bool prefer_hybrid_asr_ = true;
    bool prefer_hybrid_tts_ = true;
};

} // namespace godot
