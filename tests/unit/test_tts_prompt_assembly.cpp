#include "gotst/core/tts_prompt_assembly.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

using namespace gotst;

TEST_CASE("build_voice_clone_icl_overlay with equal lengths adds element-wise", "[icl_overlay]") {
    constexpr int64_t hidden_size = 4;
    constexpr int64_t seq_len = 3;

    std::vector<float> ref_text = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> ref_codec = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};

    auto result = build_voice_clone_icl_overlay(ref_text.data(), seq_len, ref_codec.data(), seq_len, hidden_size);

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    REQUIRE(val.icl_length == seq_len);
    REQUIRE(static_cast<int64_t>(val.icl_overlay.size()) == seq_len * hidden_size);

    for (int64_t i = 0; i < seq_len * hidden_size; ++i) {
        float expected = ref_text[static_cast<size_t>(i)] + ref_codec[static_cast<size_t>(i)];
        CHECK(std::abs(val.icl_overlay[static_cast<size_t>(i)] - expected) < 1e-5f);
    }
}

TEST_CASE("build_voice_clone_icl_overlay with text longer than codec", "[icl_overlay]") {
    constexpr int64_t hidden_size = 2;
    std::vector<float> ref_text = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> ref_codec = {0.5f, 0.6f, 0.7f, 0.8f};

    auto result = build_voice_clone_icl_overlay(ref_text.data(), 3, ref_codec.data(), 2, hidden_size);

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    REQUIRE(val.icl_length == 3);
    CHECK(std::abs(val.icl_overlay[0] - 1.5f) < 1e-5f);
    CHECK(std::abs(val.icl_overlay[1] - 2.6f) < 1e-5f);
    CHECK(std::abs(val.icl_overlay[2] - 3.7f) < 1e-5f);
    CHECK(std::abs(val.icl_overlay[3] - 4.8f) < 1e-5f);
    CHECK(std::abs(val.icl_overlay[4] - 5.0f) < 1e-5f);
    CHECK(std::abs(val.icl_overlay[5] - 6.0f) < 1e-5f);
}

TEST_CASE("build_voice_clone_icl_overlay with codec longer than text", "[icl_overlay]") {
    constexpr int64_t hidden_size = 2;
    std::vector<float> ref_text = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> ref_codec = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    auto result = build_voice_clone_icl_overlay(ref_text.data(), 2, ref_codec.data(), 3, hidden_size);

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    REQUIRE(val.icl_length == 3);
    CHECK(std::abs(val.icl_overlay[0] - 1.5f) < 1e-5f);
    CHECK(std::abs(val.icl_overlay[1] - 2.6f) < 1e-5f);
    CHECK(std::abs(val.icl_overlay[2] - 3.7f) < 1e-5f);
    CHECK(std::abs(val.icl_overlay[3] - 4.8f) < 1e-5f);
    CHECK(std::abs(val.icl_overlay[4] - 0.9f) < 1e-5f);
    CHECK(std::abs(val.icl_overlay[5] - 1.0f) < 1e-5f);
}

TEST_CASE("build_voice_clone_icl_overlay with null text returns codec only", "[icl_overlay]") {
    constexpr int64_t hidden_size = 3;
    std::vector<float> ref_codec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto result = build_voice_clone_icl_overlay(nullptr, 0, ref_codec.data(), 2, hidden_size);

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    REQUIRE(val.icl_length == 2);
    for (int64_t i = 0; i < 6; ++i) {
        CHECK(val.icl_overlay[static_cast<size_t>(i)] == ref_codec[static_cast<size_t>(i)]);
    }
}

TEST_CASE("build_voice_clone_icl_overlay with null codec returns text only", "[icl_overlay]") {
    constexpr int64_t hidden_size = 3;
    std::vector<float> ref_text = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto result = build_voice_clone_icl_overlay(ref_text.data(), 2, nullptr, 0, hidden_size);

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    REQUIRE(val.icl_length == 2);
    for (int64_t i = 0; i < 6; ++i) {
        CHECK(val.icl_overlay[static_cast<size_t>(i)] == ref_text[static_cast<size_t>(i)]);
    }
}

TEST_CASE("build_voice_clone_icl_overlay with both null returns error", "[icl_overlay]") {
    auto result = build_voice_clone_icl_overlay(nullptr, 0, nullptr, 0, 4);
    CHECK(!result.is_ok());
}

TEST_CASE("build_voice_clone_icl_overlay with zero hidden_size returns error", "[icl_overlay]") {
    std::vector<float> dummy = {1.0f};
    auto result = build_voice_clone_icl_overlay(dummy.data(), 1, dummy.data(), 1, 0);
    CHECK(!result.is_ok());
}

TEST_CASE("build_voice_clone_icl_overlay with single frame single hidden", "[icl_overlay]") {
    std::vector<float> ref_text = {3.0f};
    std::vector<float> ref_codec = {7.0f};

    auto result = build_voice_clone_icl_overlay(ref_text.data(), 1, ref_codec.data(), 1, 1);

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    REQUIRE(val.icl_length == 1);
    REQUIRE(val.icl_overlay.size() == 1);
    CHECK(std::abs(val.icl_overlay[0] - 10.0f) < 1e-5f);
}

TEST_CASE("VoiceCloneIclResult default values", "[icl_overlay]") {
    VoiceCloneIclResult r;
    CHECK(r.icl_length == 0);
    CHECK(r.icl_overlay.empty());
}
