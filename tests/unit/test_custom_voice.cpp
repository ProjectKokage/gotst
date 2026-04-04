#include "gotst/core/speech_runtime_core.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace gotst;

TEST_CASE("CustomVoiceSpeakerEntry default values", "[custom_voice]") {
    CustomVoiceSpeakerEntry entry;
    CHECK(entry.name.empty());
    CHECK(entry.token_ids.empty());
}

TEST_CASE("CustomVoiceSpeakerEntry holds name and token IDs", "[custom_voice]") {
    CustomVoiceSpeakerEntry entry;
    entry.name = "Vivian";
    entry.token_ids = {151673, 151674, 151675};
    CHECK(entry.name == "Vivian");
    CHECK(entry.token_ids.size() == 3);
    CHECK(entry.token_ids[0] == 151673);
    CHECK(entry.token_ids[2] == 151675);
}

TEST_CASE("HybridTtsConfig custom_voice_config_path defaults empty", "[custom_voice]") {
    HybridTtsConfig config;
    CHECK(config.custom_voice_config_path.empty());
}

TEST_CASE("HybridTtsConfig custom_voice_speakers defaults empty", "[custom_voice]") {
    HybridTtsConfig config;
    CHECK(config.custom_voice_speakers.empty());
}

TEST_CASE("HybridTtsConfig can hold custom voice speakers", "[custom_voice]") {
    HybridTtsConfig config;
    CustomVoiceSpeakerEntry speaker;
    speaker.name = "Serena";
    speaker.token_ids = {151676, 151677, 151678};
    config.custom_voice_speakers.push_back(speaker);
    CHECK(config.custom_voice_speakers.size() == 1);
    CHECK(config.custom_voice_speakers[0].name == "Serena");
    CHECK(config.custom_voice_speakers[0].token_ids.size() == 3);
}

TEST_CASE("Custom voice config fixture file exists and is readable", "[custom_voice][fixture]") {
    const std::string path = GOTST_TEST_FIXTURES_DIR "/test_custom_voice_config.json";
    std::ifstream file(path);
    REQUIRE(file.is_open());
    file.close();
}

TEST_CASE("Custom voice config fixture has talker_config.spk_id structure", "[custom_voice][fixture]") {
    const std::string path = GOTST_TEST_FIXTURES_DIR "/test_custom_voice_config.json";
    FILE *file = fopen(path.c_str(), "rb");
    REQUIRE(file != nullptr);

    fseek(file, 0, SEEK_END);
    const long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    REQUIRE(file_size > 0);

    std::vector<char> buffer(static_cast<size_t>(file_size + 1));
    fread(buffer.data(), 1, static_cast<size_t>(file_size), file);
    fclose(file);
    buffer[static_cast<size_t>(file_size)] = '\0';

    const std::string content(buffer.data());

    CHECK(content.find("\"talker_config\"") != std::string::npos);
    CHECK(content.find("\"spk_id\"") != std::string::npos);
}

TEST_CASE("Custom voice config fixture has all 9 preset speakers", "[custom_voice][fixture]") {
    const std::string path = GOTST_TEST_FIXTURES_DIR "/test_custom_voice_config.json";
    FILE *file = fopen(path.c_str(), "rb");
    REQUIRE(file != nullptr);

    fseek(file, 0, SEEK_END);
    const long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    std::vector<char> buffer(static_cast<size_t>(file_size + 1));
    fread(buffer.data(), 1, static_cast<size_t>(file_size), file);
    fclose(file);
    buffer[static_cast<size_t>(file_size)] = '\0';

    const std::string content(buffer.data());

    const std::vector<std::string> speakers = {
        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
        "Ryan", "Aiden", "Ono_Anna", "Sohee"
    };

    for (const auto &speaker : speakers) {
        CHECK(content.find("\"" + speaker + "\"") != std::string::npos);
    }
}

TEST_CASE("Custom voice config fixture speaker entries have token ID arrays", "[custom_voice][fixture]") {
    const std::string path = GOTST_TEST_FIXTURES_DIR "/test_custom_voice_config.json";
    FILE *file = fopen(path.c_str(), "rb");
    REQUIRE(file != nullptr);

    fseek(file, 0, SEEK_END);
    const long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    std::vector<char> buffer(static_cast<size_t>(file_size + 1));
    fread(buffer.data(), 1, static_cast<size_t>(file_size), file);
    fclose(file);
    buffer[static_cast<size_t>(file_size)] = '\0';

    const std::string content(buffer.data());

    const std::vector<std::string> speakers = {
        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
        "Ryan", "Aiden", "Ono_Anna", "Sohee"
    };

    for (const auto &speaker : speakers) {
        const std::string search = "\"" + speaker + "\": [";
        CHECK(content.find(search) != std::string::npos);
    }
}

TEST_CASE("HybridTtsConfig voice_mode defaults empty", "[voice_design]") {
    HybridTtsConfig config;
    CHECK(config.voice_mode.empty());
}

TEST_CASE("HybridTtsConfig can store voice_design mode", "[voice_design]") {
    HybridTtsConfig config;
    config.voice_mode = "voice_design";
    CHECK(config.voice_mode == "voice_design");
}

TEST_CASE("BackendSummary tracks speaker encoder and speech tokenizer readiness", "[backend]") {
    BackendSummary summary;
    CHECK_FALSE(summary.speaker_encoder_ready);
    CHECK_FALSE(summary.speech_tokenizer_encoder_ready);
}
