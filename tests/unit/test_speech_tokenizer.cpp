#include "gotst/core/speech_encoder_session.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::string get_fixtures_dir() {
    const char *dir = std::getenv("GOTST_TEST_FIXTURES_DIR");
    if (dir && dir[0] != '\0') {
        return std::string(dir);
    }
#ifdef GOTST_TEST_FIXTURES_DIR
    return GOTST_TEST_FIXTURES_DIR;
#else
    return "./fixtures";
#endif
}

std::string read_file_contents(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

}

using namespace gotst;

TEST_CASE("SpeechTokenizerEncoderSession encode with unloaded session returns error", "[speech_tokenizer]") {
    SpeechTokenizerEncoderSession s;
    float audio[256] = {};
    auto result = s.encode(audio, 256);
    CHECK(!result.is_ok());
}

TEST_CASE("SpeechTokenizerEncoderSession encode returns error on negative sample count", "[speech_tokenizer]") {
    SpeechTokenizerEncoderSession s;
    float dummy = 0.0f;
    auto result = s.encode(&dummy, -100);
    CHECK(!result.is_ok());
}

TEST_CASE("SpeechTokenizerEncoderSession encode returns error on zero sample count", "[speech_tokenizer]") {
    SpeechTokenizerEncoderSession s;
    float dummy = 0.0f;
    auto result = s.encode(&dummy, 0);
    CHECK(!result.is_ok());
}

TEST_CASE("SpeechTokenizerEncodeResult codebooks defaults to 16", "[speech_tokenizer]") {
    SpeechTokenizerEncodeResult r;
    CHECK(r.codebooks == 16);
    CHECK(r.frames == 0);
    CHECK(r.codes.empty());
}

TEST_CASE("Voice clone fixture file exists and is readable", "[voice_clone_fixture]") {
    const std::string fixtures_dir = get_fixtures_dir();
    const std::string path = fixtures_dir + "/test_voice_clone.json";
    const std::string contents = read_file_contents(path);
    REQUIRE_FALSE(contents.empty());
    CHECK(contents.find("ref_audio") != std::string::npos);
    CHECK(contents.find("ref_text") != std::string::npos);
    CHECK(contents.find("ref_codes") != std::string::npos);
    CHECK(contents.find("ref_codes_shape") != std::string::npos);
    CHECK(contents.find("ref_speaker_embedding") != std::string::npos);
}

TEST_CASE("Voice clone fixture has expected ref_text", "[voice_clone_fixture]") {
    const std::string fixtures_dir = get_fixtures_dir();
    const std::string path = fixtures_dir + "/test_voice_clone.json";
    const std::string contents = read_file_contents(path);
    REQUIRE_FALSE(contents.empty());
    CHECK(contents.find("Hello, this is a test.") != std::string::npos);
}

TEST_CASE("Voice clone fixture ref_codes_shape is [2, 16]", "[voice_clone_fixture]") {
    const std::string fixtures_dir = get_fixtures_dir();
    const std::string path = fixtures_dir + "/test_voice_clone.json";
    const std::string contents = read_file_contents(path);
    REQUIRE_FALSE(contents.empty());
    CHECK(contents.find("[2, 16]") != std::string::npos);
}

TEST_CASE("Voice clone fixture missing file returns empty contents", "[voice_clone_fixture]") {
    const std::string contents = read_file_contents("/nonexistent/path/missing.json");
    CHECK(contents.empty());
}

TEST_CASE("SpeechTokenizerEncodeResult codes vector size matches frames * codebooks", "[speech_tokenizer]") {
    SpeechTokenizerEncodeResult r;
    r.frames = 5;
    r.codebooks = 16;
    r.codes.resize(static_cast<size_t>(r.frames * r.codebooks), 42);
    CHECK(static_cast<int64_t>(r.codes.size()) == r.frames * r.codebooks);
}

TEST_CASE("SpeechTokenizerEncoderSession double load with empty path stays unloaded", "[speech_tokenizer]") {
    SpeechTokenizerEncoderSession s;
    CHECK(!s.load("").is_ok());
    CHECK(!s.load("").is_ok());
    CHECK_FALSE(s.is_loaded());
}

TEST_CASE("SpeechTokenizerEncoderSession load with nonexistent path returns error", "[speech_tokenizer]") {
    SpeechTokenizerEncoderSession s;
    CHECK(!s.load("/nonexistent/path/model.onnx").is_ok());
    CHECK_FALSE(s.is_loaded());
}
