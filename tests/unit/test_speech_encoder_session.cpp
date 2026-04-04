#include "gotst/core/speech_encoder_session.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace gotst;

TEST_CASE("SpeakerEncoderSession starts unloaded", "[speaker_encoder]") {
    SpeakerEncoderSession s;
    CHECK_FALSE(s.is_loaded());
}

TEST_CASE("SpeakerEncoderSession load with empty path returns error", "[speaker_encoder]") {
    SpeakerEncoderSession s;
    auto result = s.load("");
    CHECK(!result.is_ok());
    CHECK_FALSE(s.is_loaded());
}

TEST_CASE("SpeakerEncoderSession extract_embedding returns error when unloaded", "[speaker_encoder]") {
    SpeakerEncoderSession s;
    float dummy_mel = 1.0f;
    auto result = s.extract_embedding(&dummy_mel, 1, 1);
    CHECK(!result.is_ok());
}

TEST_CASE("SpeakerEncoderSession extract_embedding returns error on null input", "[speaker_encoder]") {
    SpeakerEncoderSession s;
    auto result = s.extract_embedding(nullptr, 1, 1);
    CHECK(!result.is_ok());
}

TEST_CASE("SpeakerEncoderSession extract_embedding returns error on zero dims", "[speaker_encoder]") {
    SpeakerEncoderSession s;
    float dummy = 0.0f;
    CHECK(!s.extract_embedding(&dummy, 0, 1).is_ok());
    CHECK(!s.extract_embedding(&dummy, 1, 0).is_ok());
    CHECK(!s.extract_embedding(&dummy, -1, 1).is_ok());
    CHECK(!s.extract_embedding(&dummy, 1, -1).is_ok());
}

TEST_CASE("SpeechTokenizerEncoderSession starts unloaded", "[speech_tokenizer_encoder]") {
    SpeechTokenizerEncoderSession s;
    CHECK_FALSE(s.is_loaded());
}

TEST_CASE("SpeechTokenizerEncoderSession load with empty path returns error", "[speech_tokenizer_encoder]") {
    SpeechTokenizerEncoderSession s;
    auto result = s.load("");
    CHECK(!result.is_ok());
    CHECK_FALSE(s.is_loaded());
}

TEST_CASE("SpeechTokenizerEncoderSession encode returns error when unloaded", "[speech_tokenizer_encoder]") {
    SpeechTokenizerEncoderSession s;
    float dummy_audio = 0.5f;
    auto result = s.encode(&dummy_audio, 1);
    CHECK(!result.is_ok());
}

TEST_CASE("SpeechTokenizerEncoderSession encode returns error on null input", "[speech_tokenizer_encoder]") {
    SpeechTokenizerEncoderSession s;
    auto result = s.encode(nullptr, 1);
    CHECK(!result.is_ok());
}

TEST_CASE("SpeechTokenizerEncoderSession encode returns error on zero samples", "[speech_tokenizer_encoder]") {
    SpeechTokenizerEncoderSession s;
    float dummy = 0.0f;
    CHECK(!s.encode(&dummy, 0).is_ok());
    CHECK(!s.encode(&dummy, -1).is_ok());
}

TEST_CASE("SpeechTokenizerEncodeResult defaults", "[speech_tokenizer_encoder]") {
    SpeechTokenizerEncodeResult r;
    CHECK(r.codes.empty());
    CHECK(r.frames == 0);
    CHECK(r.codebooks == 16);
}
