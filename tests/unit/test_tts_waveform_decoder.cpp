#include "gotst/core/tts_waveform_decoder.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <vector>

using namespace gotst;

namespace {

std::filesystem::path resolve_decoder_model_path() {
    if(const char *env_path = std::getenv("KOKAGE_TTS_DECODER_ONNX_PATH")) {
        if(*env_path != '\0') {
            return std::filesystem::path(env_path);
        }
    }
    return {};
}

} // namespace

TEST_CASE("TtsWaveformDecoder starts unloaded", "[tts_waveform_decoder]") {
    TtsWaveformDecoder decoder;
    CHECK_FALSE(decoder.is_loaded());
}

TEST_CASE("TtsWaveformDecoder load with empty path returns error", "[tts_waveform_decoder]") {
    TtsWaveformDecoder decoder;
    TtsWaveformDecoderConfig config;
    config.decoder_onnx_path.clear();

    auto result = decoder.load(config);
    CHECK_FALSE(result.is_ok());
    CHECK_FALSE(decoder.is_loaded());
}

TEST_CASE("TtsWaveformDecoder decode returns error when unloaded", "[tts_waveform_decoder]") {
    TtsWaveformDecoder decoder;
    std::vector<int64_t> codes(16, 0);

    auto result = decoder.decode(codes, 1);
    CHECK_FALSE(result.is_ok());
}

TEST_CASE("TtsWaveformDecoder can load and decode the real tokenizer wrapper", "[tts_waveform_decoder][integration]") {
    const std::filesystem::path decoder_model_path = resolve_decoder_model_path();
    if(decoder_model_path.empty() || !std::filesystem::exists(decoder_model_path)) {
        SKIP("Set KOKAGE_TTS_DECODER_ONNX_PATH to run the real tokenizer decoder integration test.");
    }

    TtsWaveformDecoder decoder;
    TtsWaveformDecoderConfig config;
    config.decoder_onnx_path = decoder_model_path.string();
    config.sample_rate = 24000;
    config.normalize_waveform = false;
    config.waveform_gain = 1.0f;

    auto load_result = decoder.load(config);
    REQUIRE(load_result.is_ok());
    REQUIRE(decoder.is_loaded());

    std::vector<int64_t> codes(12 * 16, 0);
    for(size_t index = 0; index < codes.size(); ++index) {
        codes[index] = static_cast<int64_t>(index % 2048);
    }

    auto decode_result = decoder.decode(codes, 12);
    REQUIRE(decode_result.is_ok());

    const auto &decoded = decode_result.value();
    CHECK(decoded.backend == "gotst_native");
    CHECK(decoded.frame_count == 12);
    CHECK(decoded.codes_per_frame == 16);
    CHECK(decoded.sample_count > 0);
    CHECK_FALSE(decoded.waveform.empty());
    CHECK(decoded.elapsed_ms >= 0.0);
    CHECK(decoded.inference_ms >= 0.0);
    CHECK(decoded.postprocess_ms >= 0.0);
}
