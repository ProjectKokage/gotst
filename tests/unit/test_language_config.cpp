#include "gotst/core/language_config.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>
#include <vector>

using namespace gotst;

TEST_CASE("LanguageConfig has all 10 TTS languages", "[language]") {
    LanguageConfig config;
    auto names = config.get_supported_tts_language_names();
    CHECK(names.size() == 10);
}

TEST_CASE("LanguageConfig contains all expected language display names", "[language]") {
    LanguageConfig config;
    const std::vector<std::string> expected = {
        "Chinese", "English", "French", "German", "Italian",
        "Japanese", "Korean", "Portuguese", "Russian", "Spanish"
    };
    auto names = config.get_supported_tts_language_names();
    for (const auto &exp : expected) {
        CHECK(std::find(names.begin(), names.end(), exp) != names.end());
    }
}

TEST_CASE("LanguageConfig get_tts_language_token_id returns correct IDs", "[language]") {
    LanguageConfig config;
    CHECK(config.get_tts_language_token_id("chinese") == 2055);
    CHECK(config.get_tts_language_token_id("english") == 2050);
    CHECK(config.get_tts_language_token_id("japanese") == 2058);
    CHECK(config.get_tts_language_token_id("korean") == 2064);
    CHECK(config.get_tts_language_token_id("german") == 2053);
    CHECK(config.get_tts_language_token_id("french") == 2061);
    CHECK(config.get_tts_language_token_id("spanish") == 2054);
    CHECK(config.get_tts_language_token_id("russian") == 2069);
    CHECK(config.get_tts_language_token_id("portuguese") == 2071);
    CHECK(config.get_tts_language_token_id("italian") == 2070);
}

TEST_CASE("LanguageConfig get_tts_language_token_id returns -1 for unknown language", "[language]") {
    LanguageConfig config;
    CHECK(config.get_tts_language_token_id("unknown") == -1);
    CHECK(config.get_tts_language_token_id("") == -1);
}

TEST_CASE("LanguageConfig has_tts_language works correctly", "[language]") {
    LanguageConfig config;
    CHECK(config.has_tts_language("japanese"));
    CHECK(config.has_tts_language("english"));
    CHECK_FALSE(config.has_tts_language("unknown"));
    CHECK_FALSE(config.has_tts_language(""));
}

TEST_CASE("build_codec_prefix_tokens with known language produces 6 tokens", "[language]") {
    auto prefix = LanguageConfig::build_codec_prefix_tokens(
        2058, 2154, 2155, 2156, 2157, 2148, 2149
    );
    REQUIRE(prefix.size() == 6);
    CHECK(prefix[0] == 2154);
    CHECK(prefix[1] == 2156);
    CHECK(prefix[2] == 2058);
    CHECK(prefix[3] == 2157);
    CHECK(prefix[4] == 2148);
    CHECK(prefix[5] == 2149);
}

TEST_CASE("build_codec_prefix_tokens with -1 language produces 5 tokens (auto-detect)", "[language]") {
    auto prefix = LanguageConfig::build_codec_prefix_tokens(
        -1, 2154, 2155, 2156, 2157, 2148, 2149
    );
    REQUIRE(prefix.size() == 5);
    CHECK(prefix[0] == 2155);
    CHECK(prefix[1] == 2156);
    CHECK(prefix[2] == 2157);
    CHECK(prefix[3] == 2148);
    CHECK(prefix[4] == 2149);
}

TEST_CASE("build_codec_prefix_tokens Japanese matches expected pattern", "[language]") {
    LanguageConfig config;
    int64_t lang_id = config.get_tts_language_token_id("japanese");
    auto prefix = LanguageConfig::build_codec_prefix_tokens(
        lang_id, 2154, 2155, 2156, 2157, 2148, 2149
    );
    REQUIRE(prefix.size() == 6);
    std::vector<int64_t> expected = {2154, 2156, 2058, 2157, 2148, 2149};
    CHECK(prefix == expected);
}

TEST_CASE("LanguageEntry default values", "[language]") {
    LanguageEntry entry;
    CHECK(entry.name.empty());
    CHECK(entry.codec_language_token_id == -1);
    CHECK(entry.codec_prefix_mode.empty());
}
