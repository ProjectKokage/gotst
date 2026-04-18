#include "gotst/core/ten_vad.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <vector>

using namespace gotst;

namespace {

std::filesystem::path resolve_ten_vad_model_path() {
    if(const char *env_path = std::getenv("KOKAGE_TEN_VAD_ONNX_PATH")) {
        if(*env_path != '\0') {
            return std::filesystem::path(env_path);
        }
    }
    return {};
}

} // namespace

TEST_CASE("TenVad starts unloaded", "[ten_vad]") {
    TenVad detector;
    CHECK_FALSE(detector.is_loaded());
}

TEST_CASE("TenVad rejects empty load path", "[ten_vad]") {
    TenVad detector;
    TenVadConfig config;
    config.model_path.clear();

    auto result = detector.load(config);
    CHECK_FALSE(result.is_ok());
    CHECK_FALSE(detector.is_loaded());
}

TEST_CASE("TenVad process rejects unloaded runtime", "[ten_vad]") {
    TenVad detector;
    const std::vector<float> silence(256, 0.0f);

    auto result = detector.process(silence, 16000);
    CHECK_FALSE(result.is_ok());
}

TEST_CASE("TenVad can load and process silence with the real ONNX model", "[ten_vad][integration]") {
    const std::filesystem::path model_path = resolve_ten_vad_model_path();
    if(model_path.empty() || !std::filesystem::exists(model_path)) {
        SKIP("Set KOKAGE_TEN_VAD_ONNX_PATH to run the TEN-Vad integration test.");
    }

    TenVad detector;
    TenVadConfig config;
    config.model_path = model_path.string();

    auto load_result = detector.load(config);
    REQUIRE(load_result.is_ok());
    REQUIRE(detector.is_loaded());

    const std::vector<float> partial(128, 0.0f);
    auto first_result = detector.process(partial, 16000);
    REQUIRE(first_result.is_ok());
    CHECK(first_result.value().processed_frame_count == 0);

    auto second_result = detector.process(partial, 16000);
    REQUIRE(second_result.is_ok());
    REQUIRE(second_result.value().processed_frame_count == 1);
    CHECK_FALSE(second_result.value().any_voice);
    CHECK(second_result.value().frames.size() == 1);
    CHECK(second_result.value().frames.front().probability >= 0.0f);
}

TEST_CASE("TenVad resamples 48kHz input and buffers partial frames", "[ten_vad][integration]") {
    const std::filesystem::path model_path = resolve_ten_vad_model_path();
    if(model_path.empty() || !std::filesystem::exists(model_path)) {
        SKIP("Set KOKAGE_TEN_VAD_ONNX_PATH to run the TEN-Vad integration test.");
    }

    TenVad detector;
    TenVadConfig config;
    config.model_path = model_path.string();

    auto load_result = detector.load(config);
    REQUIRE(load_result.is_ok());

    const std::vector<float> partial_48k(384, 0.0f);
    auto first_result = detector.process(partial_48k, 48000);
    REQUIRE(first_result.is_ok());
    CHECK(first_result.value().processed_frame_count == 0);
    CHECK(first_result.value().frames.empty());

    auto second_result = detector.process(partial_48k, 48000);
    REQUIRE(second_result.is_ok());
    REQUIRE(second_result.value().processed_frame_count == 1);
    CHECK(second_result.value().hop_size == 256);
    CHECK(second_result.value().model_sample_rate == 16000);
    CHECK_FALSE(second_result.value().any_voice);
    CHECK(second_result.value().frames.size() == 1);
    CHECK(second_result.value().frames.front().probability >= 0.0f);
}

TEST_CASE("TenVad reset clears buffered partial frames", "[ten_vad][integration]") {
    const std::filesystem::path model_path = resolve_ten_vad_model_path();
    if(model_path.empty() || !std::filesystem::exists(model_path)) {
        SKIP("Set KOKAGE_TEN_VAD_ONNX_PATH to run the TEN-Vad integration test.");
    }

    TenVad detector;
    TenVadConfig config;
    config.model_path = model_path.string();

    auto load_result = detector.load(config);
    REQUIRE(load_result.is_ok());

    const std::vector<float> partial_48k(384, 0.0f);
    auto first_result = detector.process(partial_48k, 48000);
    REQUIRE(first_result.is_ok());
    CHECK(first_result.value().processed_frame_count == 0);

    auto reset_result = detector.reset();
    REQUIRE(reset_result.is_ok());

    auto second_result = detector.process(partial_48k, 48000);
    REQUIRE(second_result.is_ok());
    CHECK(second_result.value().processed_frame_count == 0);

    auto third_result = detector.process(partial_48k, 48000);
    REQUIRE(third_result.is_ok());
    CHECK(third_result.value().processed_frame_count == 1);
}

TEST_CASE("TenVad rejects invalid sample rates after load", "[ten_vad][integration]") {
    const std::filesystem::path model_path = resolve_ten_vad_model_path();
    if(model_path.empty() || !std::filesystem::exists(model_path)) {
        SKIP("Set KOKAGE_TEN_VAD_ONNX_PATH to run the TEN-Vad integration test.");
    }

    TenVad detector;
    TenVadConfig config;
    config.model_path = model_path.string();

    auto load_result = detector.load(config);
    REQUIRE(load_result.is_ok());

    const std::vector<float> silence(256, 0.0f);
    auto result = detector.process(silence, 0);
    CHECK_FALSE(result.is_ok());
    CHECK(result.error_code() == ErrorCode::InvalidArgument);
}
