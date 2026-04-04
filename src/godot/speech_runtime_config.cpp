#include "gotst/godot/speech_runtime_config.hpp"

#include <godot_cpp/core/class_db.hpp>

namespace godot {

namespace {

std::string to_utf8_string(const String &value) {
    CharString utf8 = value.utf8();
    return std::string(utf8.get_data());
}

} // namespace

void GotstSpeechRuntimeConfig::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_asr_frontend_model_path", "value"), &GotstSpeechRuntimeConfig::set_asr_frontend_model_path);
    ClassDB::bind_method(D_METHOD("get_asr_frontend_model_path"), &GotstSpeechRuntimeConfig::get_asr_frontend_model_path);
    ClassDB::bind_method(
        D_METHOD("set_asr_text_embedding_model_path", "value"),
        &GotstSpeechRuntimeConfig::set_asr_text_embedding_model_path
    );
    ClassDB::bind_method(
        D_METHOD("get_asr_text_embedding_model_path"),
        &GotstSpeechRuntimeConfig::get_asr_text_embedding_model_path
    );
    ClassDB::bind_method(D_METHOD("set_asr_thinker_model_path", "value"), &GotstSpeechRuntimeConfig::set_asr_thinker_model_path);
    ClassDB::bind_method(D_METHOD("get_asr_thinker_model_path"), &GotstSpeechRuntimeConfig::get_asr_thinker_model_path);
    ClassDB::bind_method(D_METHOD("set_asr_default_language", "value"), &GotstSpeechRuntimeConfig::set_asr_default_language);
    ClassDB::bind_method(D_METHOD("get_asr_default_language"), &GotstSpeechRuntimeConfig::get_asr_default_language);
    ClassDB::bind_method(
        D_METHOD("set_tts_text_embedding_model_path", "value"),
        &GotstSpeechRuntimeConfig::set_tts_text_embedding_model_path
    );
    ClassDB::bind_method(
        D_METHOD("get_tts_text_embedding_model_path"),
        &GotstSpeechRuntimeConfig::get_tts_text_embedding_model_path
    );
    ClassDB::bind_method(
        D_METHOD("set_tts_speaker_embedding_model_path", "value"),
        &GotstSpeechRuntimeConfig::set_tts_speaker_embedding_model_path
    );
    ClassDB::bind_method(
        D_METHOD("get_tts_speaker_embedding_model_path"),
        &GotstSpeechRuntimeConfig::get_tts_speaker_embedding_model_path
    );
    ClassDB::bind_method(
        D_METHOD("set_tts_audio_decoder_model_path", "value"),
        &GotstSpeechRuntimeConfig::set_tts_audio_decoder_model_path
    );
    ClassDB::bind_method(
        D_METHOD("get_tts_audio_decoder_model_path"),
        &GotstSpeechRuntimeConfig::get_tts_audio_decoder_model_path
    );
    ClassDB::bind_method(D_METHOD("set_tts_talker_model_path", "value"), &GotstSpeechRuntimeConfig::set_tts_talker_model_path);
    ClassDB::bind_method(D_METHOD("get_tts_talker_model_path"), &GotstSpeechRuntimeConfig::get_tts_talker_model_path);
    ClassDB::bind_method(
        D_METHOD("set_tts_predictor_model_path", "value"),
        &GotstSpeechRuntimeConfig::set_tts_predictor_model_path
    );
    ClassDB::bind_method(D_METHOD("get_tts_predictor_model_path"), &GotstSpeechRuntimeConfig::get_tts_predictor_model_path);
    ClassDB::bind_method(D_METHOD("set_tts_voice_mode", "value"), &GotstSpeechRuntimeConfig::set_tts_voice_mode);
    ClassDB::bind_method(D_METHOD("get_tts_voice_mode"), &GotstSpeechRuntimeConfig::get_tts_voice_mode);
    ClassDB::bind_method(
        D_METHOD("set_tts_speaker_encoder_model_path", "value"),
        &GotstSpeechRuntimeConfig::set_tts_speaker_encoder_model_path
    );
    ClassDB::bind_method(
        D_METHOD("get_tts_speaker_encoder_model_path"),
        &GotstSpeechRuntimeConfig::get_tts_speaker_encoder_model_path
    );
    ClassDB::bind_method(
        D_METHOD("set_tts_speech_tokenizer_encoder_model_path", "value"),
        &GotstSpeechRuntimeConfig::set_tts_speech_tokenizer_encoder_model_path
    );
    ClassDB::bind_method(
        D_METHOD("get_tts_speech_tokenizer_encoder_model_path"),
        &GotstSpeechRuntimeConfig::get_tts_speech_tokenizer_encoder_model_path
    );
    ClassDB::bind_method(
        D_METHOD("set_tts_speech_tokenizer_encoder_sample_rate", "value"),
        &GotstSpeechRuntimeConfig::set_tts_speech_tokenizer_encoder_sample_rate
    );
    ClassDB::bind_method(
        D_METHOD("get_tts_speech_tokenizer_encoder_sample_rate"),
        &GotstSpeechRuntimeConfig::get_tts_speech_tokenizer_encoder_sample_rate
    );
    ClassDB::bind_method(
        D_METHOD("set_tts_speaker_mel_sample_rate", "value"),
        &GotstSpeechRuntimeConfig::set_tts_speaker_mel_sample_rate
    );
    ClassDB::bind_method(
        D_METHOD("get_tts_speaker_mel_sample_rate"),
        &GotstSpeechRuntimeConfig::get_tts_speaker_mel_sample_rate
    );
    ClassDB::bind_method(
        D_METHOD("set_tts_speaker_mel_n_fft", "value"),
        &GotstSpeechRuntimeConfig::set_tts_speaker_mel_n_fft
    );
    ClassDB::bind_method(D_METHOD("get_tts_speaker_mel_n_fft"), &GotstSpeechRuntimeConfig::get_tts_speaker_mel_n_fft);
    ClassDB::bind_method(
        D_METHOD("set_tts_speaker_mel_hop_length", "value"),
        &GotstSpeechRuntimeConfig::set_tts_speaker_mel_hop_length
    );
    ClassDB::bind_method(
        D_METHOD("get_tts_speaker_mel_hop_length"),
        &GotstSpeechRuntimeConfig::get_tts_speaker_mel_hop_length
    );
    ClassDB::bind_method(
        D_METHOD("set_tts_speaker_mel_n_mels", "value"),
        &GotstSpeechRuntimeConfig::set_tts_speaker_mel_n_mels
    );
    ClassDB::bind_method(D_METHOD("get_tts_speaker_mel_n_mels"), &GotstSpeechRuntimeConfig::get_tts_speaker_mel_n_mels);
    ClassDB::bind_method(D_METHOD("set_tts_speaker_mel_fmin", "value"), &GotstSpeechRuntimeConfig::set_tts_speaker_mel_fmin);
    ClassDB::bind_method(D_METHOD("get_tts_speaker_mel_fmin"), &GotstSpeechRuntimeConfig::get_tts_speaker_mel_fmin);
    ClassDB::bind_method(D_METHOD("set_tts_speaker_mel_fmax", "value"), &GotstSpeechRuntimeConfig::set_tts_speaker_mel_fmax);
    ClassDB::bind_method(D_METHOD("get_tts_speaker_mel_fmax"), &GotstSpeechRuntimeConfig::get_tts_speaker_mel_fmax);
    ClassDB::bind_method(D_METHOD("set_prefer_hybrid_asr", "value"), &GotstSpeechRuntimeConfig::set_prefer_hybrid_asr);
    ClassDB::bind_method(D_METHOD("is_prefer_hybrid_asr"), &GotstSpeechRuntimeConfig::is_prefer_hybrid_asr);
    ClassDB::bind_method(D_METHOD("set_prefer_hybrid_tts", "value"), &GotstSpeechRuntimeConfig::set_prefer_hybrid_tts);
    ClassDB::bind_method(D_METHOD("is_prefer_hybrid_tts"), &GotstSpeechRuntimeConfig::is_prefer_hybrid_tts);

    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "asr_frontend_model_path"),
        "set_asr_frontend_model_path",
        "get_asr_frontend_model_path"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "asr_text_embedding_model_path"),
        "set_asr_text_embedding_model_path",
        "get_asr_text_embedding_model_path"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "asr_thinker_model_path"),
        "set_asr_thinker_model_path",
        "get_asr_thinker_model_path"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "asr_default_language"),
        "set_asr_default_language",
        "get_asr_default_language"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "tts_text_embedding_model_path"),
        "set_tts_text_embedding_model_path",
        "get_tts_text_embedding_model_path"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "tts_speaker_embedding_model_path"),
        "set_tts_speaker_embedding_model_path",
        "get_tts_speaker_embedding_model_path"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "tts_audio_decoder_model_path"),
        "set_tts_audio_decoder_model_path",
        "get_tts_audio_decoder_model_path"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "tts_talker_model_path"),
        "set_tts_talker_model_path",
        "get_tts_talker_model_path"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "tts_predictor_model_path"),
        "set_tts_predictor_model_path",
        "get_tts_predictor_model_path"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "tts_voice_mode", PROPERTY_HINT_ENUM, "base,custom_voice,voice_design"),
        "set_tts_voice_mode",
        "get_tts_voice_mode"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "tts_speaker_encoder_model_path"),
        "set_tts_speaker_encoder_model_path",
        "get_tts_speaker_encoder_model_path"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::STRING, "tts_speech_tokenizer_encoder_model_path"),
        "set_tts_speech_tokenizer_encoder_model_path",
        "get_tts_speech_tokenizer_encoder_model_path"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "tts_speech_tokenizer_encoder_sample_rate"),
        "set_tts_speech_tokenizer_encoder_sample_rate",
        "get_tts_speech_tokenizer_encoder_sample_rate"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "tts_speaker_mel_sample_rate"),
        "set_tts_speaker_mel_sample_rate",
        "get_tts_speaker_mel_sample_rate"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "tts_speaker_mel_n_fft"),
        "set_tts_speaker_mel_n_fft",
        "get_tts_speaker_mel_n_fft"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "tts_speaker_mel_hop_length"),
        "set_tts_speaker_mel_hop_length",
        "get_tts_speaker_mel_hop_length"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "tts_speaker_mel_n_mels"),
        "set_tts_speaker_mel_n_mels",
        "get_tts_speaker_mel_n_mels"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::FLOAT, "tts_speaker_mel_fmin"),
        "set_tts_speaker_mel_fmin",
        "get_tts_speaker_mel_fmin"
    );
    ADD_PROPERTY(
        PropertyInfo(Variant::FLOAT, "tts_speaker_mel_fmax"),
        "set_tts_speaker_mel_fmax",
        "get_tts_speaker_mel_fmax"
    );
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "prefer_hybrid_asr"), "set_prefer_hybrid_asr", "is_prefer_hybrid_asr");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "prefer_hybrid_tts"), "set_prefer_hybrid_tts", "is_prefer_hybrid_tts");
}

void GotstSpeechRuntimeConfig::set_asr_frontend_model_path(const String &value) {
    asr_frontend_model_path_ = value;
}

String GotstSpeechRuntimeConfig::get_asr_frontend_model_path() const {
    return asr_frontend_model_path_;
}

void GotstSpeechRuntimeConfig::set_asr_text_embedding_model_path(const String &value) {
    asr_text_embedding_model_path_ = value;
}

String GotstSpeechRuntimeConfig::get_asr_text_embedding_model_path() const {
    return asr_text_embedding_model_path_;
}

void GotstSpeechRuntimeConfig::set_asr_thinker_model_path(const String &value) {
    asr_thinker_model_path_ = value;
}

String GotstSpeechRuntimeConfig::get_asr_thinker_model_path() const {
    return asr_thinker_model_path_;
}

void GotstSpeechRuntimeConfig::set_asr_default_language(const String &value) {
    asr_default_language_ = value;
}

String GotstSpeechRuntimeConfig::get_asr_default_language() const {
    return asr_default_language_;
}

void GotstSpeechRuntimeConfig::set_tts_text_embedding_model_path(const String &value) {
    tts_text_embedding_model_path_ = value;
}

String GotstSpeechRuntimeConfig::get_tts_text_embedding_model_path() const {
    return tts_text_embedding_model_path_;
}

void GotstSpeechRuntimeConfig::set_tts_speaker_embedding_model_path(const String &value) {
    tts_speaker_embedding_model_path_ = value;
}

String GotstSpeechRuntimeConfig::get_tts_speaker_embedding_model_path() const {
    return tts_speaker_embedding_model_path_;
}

void GotstSpeechRuntimeConfig::set_tts_audio_decoder_model_path(const String &value) {
    tts_audio_decoder_model_path_ = value;
}

String GotstSpeechRuntimeConfig::get_tts_audio_decoder_model_path() const {
    return tts_audio_decoder_model_path_;
}

void GotstSpeechRuntimeConfig::set_tts_talker_model_path(const String &value) {
    tts_talker_model_path_ = value;
}

String GotstSpeechRuntimeConfig::get_tts_talker_model_path() const {
    return tts_talker_model_path_;
}

void GotstSpeechRuntimeConfig::set_tts_predictor_model_path(const String &value) {
    tts_predictor_model_path_ = value;
}

String GotstSpeechRuntimeConfig::get_tts_predictor_model_path() const {
    return tts_predictor_model_path_;
}

void GotstSpeechRuntimeConfig::set_tts_voice_mode(const String &value) {
    tts_voice_mode_ = value;
}

String GotstSpeechRuntimeConfig::get_tts_voice_mode() const {
    return tts_voice_mode_;
}

void GotstSpeechRuntimeConfig::set_tts_speaker_encoder_model_path(const String &value) {
    tts_speaker_encoder_model_path_ = value;
}

String GotstSpeechRuntimeConfig::get_tts_speaker_encoder_model_path() const {
    return tts_speaker_encoder_model_path_;
}

void GotstSpeechRuntimeConfig::set_tts_speech_tokenizer_encoder_model_path(const String &value) {
    tts_speech_tokenizer_encoder_model_path_ = value;
}

String GotstSpeechRuntimeConfig::get_tts_speech_tokenizer_encoder_model_path() const {
    return tts_speech_tokenizer_encoder_model_path_;
}

void GotstSpeechRuntimeConfig::set_tts_speech_tokenizer_encoder_sample_rate(int64_t value) {
    tts_speech_tokenizer_encoder_sample_rate_ = value;
}

int64_t GotstSpeechRuntimeConfig::get_tts_speech_tokenizer_encoder_sample_rate() const {
    return tts_speech_tokenizer_encoder_sample_rate_;
}

void GotstSpeechRuntimeConfig::set_tts_speaker_mel_sample_rate(int64_t value) {
    tts_speaker_mel_sample_rate_ = value;
}

int64_t GotstSpeechRuntimeConfig::get_tts_speaker_mel_sample_rate() const {
    return tts_speaker_mel_sample_rate_;
}

void GotstSpeechRuntimeConfig::set_tts_speaker_mel_n_fft(int64_t value) {
    tts_speaker_mel_n_fft_ = value;
}

int64_t GotstSpeechRuntimeConfig::get_tts_speaker_mel_n_fft() const {
    return tts_speaker_mel_n_fft_;
}

void GotstSpeechRuntimeConfig::set_tts_speaker_mel_hop_length(int64_t value) {
    tts_speaker_mel_hop_length_ = value;
}

int64_t GotstSpeechRuntimeConfig::get_tts_speaker_mel_hop_length() const {
    return tts_speaker_mel_hop_length_;
}

void GotstSpeechRuntimeConfig::set_tts_speaker_mel_n_mels(int64_t value) {
    tts_speaker_mel_n_mels_ = value;
}

int64_t GotstSpeechRuntimeConfig::get_tts_speaker_mel_n_mels() const {
    return tts_speaker_mel_n_mels_;
}

void GotstSpeechRuntimeConfig::set_tts_speaker_mel_fmin(double value) {
    tts_speaker_mel_fmin_ = value;
}

double GotstSpeechRuntimeConfig::get_tts_speaker_mel_fmin() const {
    return tts_speaker_mel_fmin_;
}

void GotstSpeechRuntimeConfig::set_tts_speaker_mel_fmax(double value) {
    tts_speaker_mel_fmax_ = value;
}

double GotstSpeechRuntimeConfig::get_tts_speaker_mel_fmax() const {
    return tts_speaker_mel_fmax_;
}

void GotstSpeechRuntimeConfig::set_prefer_hybrid_asr(bool value) {
    prefer_hybrid_asr_ = value;
}

bool GotstSpeechRuntimeConfig::is_prefer_hybrid_asr() const {
    return prefer_hybrid_asr_;
}

void GotstSpeechRuntimeConfig::set_prefer_hybrid_tts(bool value) {
    prefer_hybrid_tts_ = value;
}

bool GotstSpeechRuntimeConfig::is_prefer_hybrid_tts() const {
    return prefer_hybrid_tts_;
}

gotst::RuntimeConfig GotstSpeechRuntimeConfig::build_native_config() const {
    gotst::RuntimeConfig config;
    config.asr.frontend_model_path = to_utf8_string(asr_frontend_model_path_);
    config.asr.text_embedding_model_path = to_utf8_string(asr_text_embedding_model_path_);
    config.asr.thinker_model_path = to_utf8_string(asr_thinker_model_path_);
    config.asr.prefer_hybrid = prefer_hybrid_asr_;
    config.asr.default_language = to_utf8_string(asr_default_language_);

    config.tts.text_embedding_model_path = to_utf8_string(tts_text_embedding_model_path_);
    config.tts.speaker_embedding_model_path = to_utf8_string(tts_speaker_embedding_model_path_);
    config.tts.audio_decoder_model_path = to_utf8_string(tts_audio_decoder_model_path_);
    config.tts.talker_model_path = to_utf8_string(tts_talker_model_path_);
    config.tts.predictor_model_path = to_utf8_string(tts_predictor_model_path_);
    config.tts.prefer_hybrid = prefer_hybrid_tts_;
    config.tts.voice_mode = to_utf8_string(tts_voice_mode_);
    config.tts.speaker_encoder_model_path = to_utf8_string(tts_speaker_encoder_model_path_);
    config.tts.speech_tokenizer_encoder_model_path = to_utf8_string(tts_speech_tokenizer_encoder_model_path_);
    config.tts.speech_tokenizer_encoder_sample_rate = tts_speech_tokenizer_encoder_sample_rate_;
    config.tts.speaker_mel_params.sample_rate = tts_speaker_mel_sample_rate_;
    config.tts.speaker_mel_params.n_fft = tts_speaker_mel_n_fft_;
    config.tts.speaker_mel_params.hop_length = tts_speaker_mel_hop_length_;
    config.tts.speaker_mel_params.n_mels = tts_speaker_mel_n_mels_;
    config.tts.speaker_mel_params.fmin = tts_speaker_mel_fmin_;
    config.tts.speaker_mel_params.fmax = tts_speaker_mel_fmax_;
    return config;
}

} // namespace godot
