#include "gotst/core/qwen3_forced_aligner.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace gotst;

namespace {

struct TestClassifierHeader {
    char magic[8] = {'G', 'F', 'A', 'H', '0', '0', '0', '1'};
    uint32_t version = 1;
    int32_t classify_num = 0;
    int32_t hidden_size = 0;
    int32_t reserved = 0;
};

std::filesystem::path write_classifier_fixture(
    int32_t classify_num,
    int32_t hidden_size,
    const std::vector<float> &weights
) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("gotst_qwen3_fa_classifier_" + std::to_string(stamp) + ".bin");
    std::ofstream stream(path, std::ios::binary);
    TestClassifierHeader header;
    header.classify_num = classify_num;
    header.hidden_size = hidden_size;
    stream.write(reinterpret_cast<const char *>(&header), sizeof(header));
    stream.write(reinterpret_cast<const char *>(weights.data()), static_cast<std::streamsize>(weights.size() * sizeof(float)));
    return path;
}

Qwen3ForcedAlignmentRequest valid_request() {
    Qwen3ForcedAlignmentRequest request;
    request.waveform = std::vector<float>(1600, 0.0f);
    request.input_sample_rate = 16000;
    request.text_units = {"こ", "こ"};
    request.token_ids = {1, 151705, 2, 151705, 3, 151705, 4, 151705, 5};
    request.timestamp_token_indices = {1, 3, 5, 7};
    request.audio_placeholder_start = 0;
    request.audio_placeholder_count = 1;
    return request;
}

} // namespace

TEST_CASE("Qwen3 forced-aligner request validation rejects invalid timestamp slots", "[qwen3_forced_aligner]") {
    Qwen3ForcedAlignerSessionConfig config;
    Qwen3ForcedAlignmentRequest request = valid_request();
    CHECK(validate_qwen3_forced_alignment_request(request, config).is_ok());

    request.timestamp_token_indices = {1, 3, 5};
    CHECK_FALSE(validate_qwen3_forced_alignment_request(request, config).is_ok());

    request = valid_request();
    request.token_ids[static_cast<size_t>(1)] = 42;
    CHECK_FALSE(validate_qwen3_forced_alignment_request(request, config).is_ok());

    request = valid_request();
    request.audio_placeholder_start = 8;
    request.audio_placeholder_count = 2;
    CHECK_FALSE(validate_qwen3_forced_alignment_request(request, config).is_ok());
}

TEST_CASE("Qwen3 forced-aligner monotonic repair follows official LIS behavior", "[qwen3_forced_aligner]") {
    const std::vector<int32_t> descending = {10, 9};
    CHECK(repair_monotonic_timestamp_bins(descending) == std::vector<int32_t>{10, 10});

    const std::vector<int32_t> short_anomaly = {0, 10, 5, 20};
    CHECK(repair_monotonic_timestamp_bins(short_anomaly) == std::vector<int32_t>{0, 10, 10, 20});

    const std::vector<int32_t> long_anomaly = {0, 10, 4, 3, 2, 20};
    CHECK(repair_monotonic_timestamp_bins(long_anomaly) == std::vector<int32_t>{0, 10, 12, 15, 17, 20});
}

TEST_CASE("Qwen3 forced-aligner classifier head loads and predicts argmax", "[qwen3_forced_aligner]") {
    const std::vector<float> weights = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 4.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    const std::filesystem::path path = write_classifier_fixture(4, 3, weights);

    Qwen3ForcedAlignerClassifierHead head;
    auto load = head.load(path.string());
    REQUIRE(load.is_ok());
    CHECK(head.classify_num == 4);
    CHECK(head.hidden_size == 3);

    const std::vector<float> hidden = {0.0f, 1.0f, 0.0f};
    auto prediction = head.classify(hidden);
    REQUIRE(prediction.is_ok());
    CHECK(prediction.value().bin == 2);
    CHECK(prediction.value().confidence > 0.8);

    std::filesystem::remove(path);
}

TEST_CASE("Qwen3 forced-aligner helper segmentation stays lightweight", "[qwen3_forced_aligner]") {
    CHECK(split_forced_alignment_units("hello world", "english") == std::vector<std::string>{"hello", "world"});
    CHECK(split_forced_alignment_units("ここ", "japanese") == std::vector<std::string>{"こ", "こ"});
}
