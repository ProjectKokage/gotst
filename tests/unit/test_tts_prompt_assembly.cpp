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

TEST_CASE("build_tts_prompt_assembly builds the base prompt in one pass", "[prompt_assembly]") {
    constexpr int64_t hidden_size = 2;

    const std::vector<float> text = {
        20.0f, 21.0f,
        30.0f, 40.0f,
        50.0f, 60.0f,
        70.0f, 80.0f,
    };
    const std::vector<float> special = {
        100.0f, 200.0f,
        300.0f, 400.0f,
        10.0f, 20.0f,
    };
    const std::vector<float> codec_prefill = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    const std::vector<float> instruction = {
        7.0f, 8.0f,
    };

    auto result = build_tts_prompt_assembly({
        .text_projected_states = text,
        .text_sequence_length = 4,
        .special_projected_states = special,
        .codec_prefill_embeddings = codec_prefill,
        .codec_prefill_length = 3,
        .codec_prompt_insert = {},
        .leading_prompt_states = instruction,
        .leading_prompt_length = 1,
        .icl_overlay = {},
        .icl_length = 0,
        .hidden_size = hidden_size,
        .wrapped_prefix_token_count = 1,
        .wrapped_suffix_token_count = 1,
    });

    REQUIRE(result.is_ok());
    const auto &value = result.value();
    CHECK(value.language_sequence_length == 5);
    CHECK(value.trailing_text_length == 2);
    CHECK(value.hidden_size == hidden_size);
    CHECK(value.produced_frames == 0);
    CHECK(value.icl_length == 0);

    const std::vector<float> expected_language = {
        7.0f, 8.0f,
        20.0f, 21.0f,
        11.0f, 22.0f,
        103.0f, 204.0f,
        35.0f, 46.0f,
    };
    CHECK(value.language_sequence == expected_language);

    const std::vector<float> expected_trailing = {
        50.0f, 60.0f,
        300.0f, 400.0f,
    };
    CHECK(value.trailing_text_hidden == expected_trailing);
    CHECK(value.tts_pad_embedding == std::vector<float>({10.0f, 20.0f}));
}

TEST_CASE("build_tts_prompt_assembly inserts speaker rows before codec pad and bos", "[prompt_assembly]") {
    constexpr int64_t hidden_size = 2;

    const std::vector<float> text = {
        20.0f, 21.0f,
        30.0f, 40.0f,
        50.0f, 60.0f,
        70.0f, 80.0f,
    };
    const std::vector<float> special = {
        100.0f, 200.0f,
        300.0f, 400.0f,
        10.0f, 20.0f,
    };
    const std::vector<float> codec_prefill = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    const std::vector<float> speaker = {
        9.0f, 10.0f,
    };

    auto result = build_tts_prompt_assembly({
        .text_projected_states = text,
        .text_sequence_length = 4,
        .special_projected_states = special,
        .codec_prefill_embeddings = codec_prefill,
        .codec_prefill_length = 3,
        .codec_prompt_insert = speaker,
        .leading_prompt_states = {},
        .leading_prompt_length = 0,
        .icl_overlay = {},
        .icl_length = 0,
        .hidden_size = hidden_size,
        .wrapped_prefix_token_count = 1,
        .wrapped_suffix_token_count = 1,
    });

    REQUIRE(result.is_ok());
    const auto &value = result.value();
    CHECK(value.language_sequence_length == 5);

    const std::vector<float> expected_language = {
        20.0f, 21.0f,
        11.0f, 22.0f,
        19.0f, 30.0f,
        103.0f, 204.0f,
        35.0f, 46.0f,
    };
    CHECK(value.language_sequence == expected_language);
}

TEST_CASE("build_tts_prompt_assembly inserts ICL rows between the role and codec prefill", "[prompt_assembly]") {
    constexpr int64_t hidden_size = 2;

    const std::vector<float> text = {
        20.0f, 21.0f,
        30.0f, 40.0f,
        50.0f, 60.0f,
        70.0f, 80.0f,
    };
    const std::vector<float> special = {
        100.0f, 200.0f,
        300.0f, 400.0f,
        10.0f, 20.0f,
    };
    const std::vector<float> codec_prefill = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    const std::vector<float> ref_text = {
        1.0f, 1.0f,
        2.0f, 2.0f,
    };
    const std::vector<float> ref_codec = {
        3.0f, 4.0f,
    };

    auto icl = build_voice_clone_icl_overlay(ref_text.data(), 2, ref_codec.data(), 1, hidden_size);
    REQUIRE(icl.is_ok());

    auto result = build_tts_prompt_assembly({
        .text_projected_states = text,
        .text_sequence_length = 4,
        .special_projected_states = special,
        .codec_prefill_embeddings = codec_prefill,
        .codec_prefill_length = 3,
        .codec_prompt_insert = {},
        .leading_prompt_states = {},
        .leading_prompt_length = 0,
        .icl_overlay = icl.value().icl_overlay,
        .icl_length = icl.value().icl_length,
        .hidden_size = hidden_size,
        .wrapped_prefix_token_count = 1,
        .wrapped_suffix_token_count = 1,
    });

    REQUIRE(result.is_ok());
    const auto &value = result.value();
    CHECK(value.language_sequence_length == 6);
    CHECK(value.icl_length == 2);

    const std::vector<float> expected_language = {
        20.0f, 21.0f,
        4.0f, 5.0f,
        2.0f, 2.0f,
        11.0f, 22.0f,
        103.0f, 204.0f,
        35.0f, 46.0f,
    };
    CHECK(value.language_sequence == expected_language);
}

TEST_CASE("build_tts_prompt_assembly rejects misaligned insert rows", "[prompt_assembly]") {
    constexpr int64_t hidden_size = 2;

    const std::vector<float> text = {
        20.0f, 21.0f,
        30.0f, 40.0f,
    };
    const std::vector<float> special = {
        100.0f, 200.0f,
        300.0f, 400.0f,
        10.0f, 20.0f,
    };
    const std::vector<float> codec_prefill = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    const std::vector<float> misaligned_insert = {9.0f};

    auto result = build_tts_prompt_assembly({
        .text_projected_states = text,
        .text_sequence_length = 2,
        .special_projected_states = special,
        .codec_prefill_embeddings = codec_prefill,
        .codec_prefill_length = 2,
        .codec_prompt_insert = misaligned_insert,
        .leading_prompt_states = {},
        .leading_prompt_length = 0,
        .icl_overlay = {},
        .icl_length = 0,
        .hidden_size = hidden_size,
        .wrapped_prefix_token_count = 0,
        .wrapped_suffix_token_count = 0,
    });

    CHECK_FALSE(result.is_ok());
}
