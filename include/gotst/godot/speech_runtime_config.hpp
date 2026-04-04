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

    void set_asr_default_language(const String &value);
    String get_asr_default_language() const;

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

    void set_tts_voice_mode(const String &value);
    String get_tts_voice_mode() const;

    void set_tts_speaker_encoder_model_path(const String &value);
    String get_tts_speaker_encoder_model_path() const;

    void set_tts_speech_tokenizer_encoder_model_path(const String &value);
    String get_tts_speech_tokenizer_encoder_model_path() const;

    void set_tts_speech_tokenizer_encoder_sample_rate(int64_t value);
    int64_t get_tts_speech_tokenizer_encoder_sample_rate() const;

    void set_tts_speaker_mel_sample_rate(int64_t value);
    int64_t get_tts_speaker_mel_sample_rate() const;

    void set_tts_speaker_mel_n_fft(int64_t value);
    int64_t get_tts_speaker_mel_n_fft() const;

    void set_tts_speaker_mel_hop_length(int64_t value);
    int64_t get_tts_speaker_mel_hop_length() const;

    void set_tts_speaker_mel_n_mels(int64_t value);
    int64_t get_tts_speaker_mel_n_mels() const;

    void set_tts_speaker_mel_fmin(double value);
    double get_tts_speaker_mel_fmin() const;

    void set_tts_speaker_mel_fmax(double value);
    double get_tts_speaker_mel_fmax() const;

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
    String asr_default_language_;
    String tts_text_embedding_model_path_;
    String tts_speaker_embedding_model_path_;
    String tts_audio_decoder_model_path_;
    String tts_talker_model_path_;
    String tts_predictor_model_path_;
    String tts_voice_mode_;
    String tts_speaker_encoder_model_path_;
    String tts_speech_tokenizer_encoder_model_path_;
    int64_t tts_speech_tokenizer_encoder_sample_rate_ = 24000;
    int64_t tts_speaker_mel_sample_rate_ = 24000;
    int64_t tts_speaker_mel_n_fft_ = 1024;
    int64_t tts_speaker_mel_hop_length_ = 256;
    int64_t tts_speaker_mel_n_mels_ = 128;
    double tts_speaker_mel_fmin_ = 0.0;
    double tts_speaker_mel_fmax_ = 12000.0;
    bool prefer_hybrid_asr_ = true;
    bool prefer_hybrid_tts_ = true;
};

} // namespace godot
