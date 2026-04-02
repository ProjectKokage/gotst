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

    config.tts.text_embedding_model_path = to_utf8_string(tts_text_embedding_model_path_);
    config.tts.speaker_embedding_model_path = to_utf8_string(tts_speaker_embedding_model_path_);
    config.tts.audio_decoder_model_path = to_utf8_string(tts_audio_decoder_model_path_);
    config.tts.talker_model_path = to_utf8_string(tts_talker_model_path_);
    config.tts.predictor_model_path = to_utf8_string(tts_predictor_model_path_);
    config.tts.prefer_hybrid = prefer_hybrid_tts_;
    return config;
}

} // namespace godot
