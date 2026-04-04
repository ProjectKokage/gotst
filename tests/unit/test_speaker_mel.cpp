#include "gotst/core/speaker_mel.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

using namespace gotst;
using Catch::Matchers::WithinAbs;

TEST_CASE("build_speaker_mel_features returns error on null input", "[speaker_mel]") {
    auto result = build_speaker_mel_features(nullptr, 100, 24000, 24000, 128, 1024, 256, 0.0, 12000.0);
    CHECK(!result.is_ok());
}

TEST_CASE("build_speaker_mel_features returns error on zero samples", "[speaker_mel]") {
    float dummy = 0.0f;
    auto result = build_speaker_mel_features(&dummy, 0, 24000, 24000, 128, 1024, 256, 0.0, 12000.0);
    CHECK(!result.is_ok());
}

TEST_CASE("build_speaker_mel_features returns error on negative samples", "[speaker_mel]") {
    float dummy = 0.0f;
    auto result = build_speaker_mel_features(&dummy, -1, 24000, 24000, 128, 1024, 256, 0.0, 12000.0);
    CHECK(!result.is_ok());
}

TEST_CASE("build_speaker_mel_features returns error on invalid params", "[speaker_mel]") {
    std::vector<float> audio(48000, 0.1f);
    CHECK(!build_speaker_mel_features(audio.data(), 48000, 0, 24000, 128, 1024, 256, 0.0, 12000.0).is_ok());
    CHECK(!build_speaker_mel_features(audio.data(), 48000, 24000, 0, 128, 1024, 256, 0.0, 12000.0).is_ok());
    CHECK(!build_speaker_mel_features(audio.data(), 48000, 24000, 24000, 0, 1024, 256, 0.0, 12000.0).is_ok());
    CHECK(!build_speaker_mel_features(audio.data(), 48000, 24000, 24000, 128, 0, 256, 0.0, 12000.0).is_ok());
    CHECK(!build_speaker_mel_features(audio.data(), 48000, 24000, 24000, 128, 1024, 0, 0.0, 12000.0).is_ok());
    CHECK(!build_speaker_mel_features(audio.data(), 48000, 24000, 24000, 128, 1024, 256, 12000.0, 0.0).is_ok());
}

TEST_CASE("build_speaker_mel_features produces correct output dimensions", "[speaker_mel]") {
    const int64_t sample_rate = 24000;
    const int64_t fft_size = 1024;
    const int64_t hop_length = 256;
    const int64_t mel_bins = 128;
    const int64_t duration_samples = sample_rate;

    std::vector<float> audio(static_cast<size_t>(duration_samples), 0.1f);
    auto result = build_speaker_mel_features(
        audio.data(), duration_samples, sample_rate, sample_rate,
        mel_bins, fft_size, hop_length, 0.0, 12000.0
    );

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    CHECK(val.mel_dim == mel_bins);
    CHECK(val.frames > 0);
    CHECK(static_cast<int64_t>(val.features.size()) == val.frames * val.mel_dim);
}

TEST_CASE("build_speaker_mel_features output layout is frames x mel_dim", "[speaker_mel]") {
    const int64_t sample_rate = 24000;
    const int64_t mel_bins = 80;
    const int64_t fft_size = 512;
    const int64_t hop_length = 128;
    const int64_t duration_samples = sample_rate / 2;

    std::vector<float> audio(static_cast<size_t>(duration_samples), 0.0f);
    for (int64_t i = 0; i < duration_samples; ++i) {
        audio[static_cast<size_t>(i)] =
            static_cast<float>(0.5 * std::sin(2.0 * 3.14159265358979323846 * 440.0 *
                                              static_cast<double>(i) / static_cast<double>(sample_rate)));
    }

    auto result = build_speaker_mel_features(
        audio.data(), duration_samples, sample_rate, sample_rate,
        mel_bins, fft_size, hop_length, 0.0, 12000.0
    );

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    CHECK(val.mel_dim == mel_bins);
    CHECK(val.frames > 0);
    CHECK(static_cast<int64_t>(val.features.size()) == val.frames * val.mel_dim);
}

TEST_CASE("build_speaker_mel_features uses log normalization not whisper", "[speaker_mel]") {
    const int64_t sample_rate = 16000;
    const int64_t mel_bins = 128;
    const int64_t fft_size = 1024;
    const int64_t hop_length = 256;

    std::vector<float> audio(static_cast<size_t>(sample_rate), 0.1f);
    auto result = build_speaker_mel_features(
        audio.data(), sample_rate, sample_rate, sample_rate,
        mel_bins, fft_size, hop_length, 0.0, 8000.0
    );

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    REQUIRE_FALSE(val.features.empty());

    for (const float &feat : val.features) {
        CHECK(std::isfinite(static_cast<double>(feat)));
        CHECK(feat >= static_cast<float>(std::log(1e-5)));
        CHECK(feat < 50.0f);
    }
}

TEST_CASE("build_speaker_mel_features with resampling", "[speaker_mel]") {
    const int64_t input_rate = 16000;
    const int64_t target_rate = 24000;
    const int64_t mel_bins = 128;
    const int64_t fft_size = 1024;
    const int64_t hop_length = 256;

    std::vector<float> audio(static_cast<size_t>(input_rate), 0.1f);
    auto result = build_speaker_mel_features(
        audio.data(), input_rate, input_rate, target_rate,
        mel_bins, fft_size, hop_length, 0.0, 12000.0
    );

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    CHECK(val.mel_dim == mel_bins);
    CHECK(val.frames > 0);
    CHECK_FALSE(val.features.empty());

    for (const float &feat : val.features) {
        CHECK(std::isfinite(static_cast<double>(feat)));
    }
}

TEST_CASE("build_speaker_mel_features sine wave has non-zero energy", "[speaker_mel]") {
    const int64_t sample_rate = 24000;
    const int64_t mel_bins = 128;
    const int64_t fft_size = 1024;
    const int64_t hop_length = 256;
    const int64_t duration_samples = sample_rate;

    std::vector<float> audio(static_cast<size_t>(duration_samples));
    for (int64_t i = 0; i < duration_samples; ++i) {
        audio[static_cast<size_t>(i)] =
            static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * 1000.0 *
                                        static_cast<double>(i) / static_cast<double>(sample_rate)));
    }

    auto result = build_speaker_mel_features(
        audio.data(), duration_samples, sample_rate, sample_rate,
        mel_bins, fft_size, hop_length, 0.0, 12000.0
    );

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    REQUIRE_FALSE(val.features.empty());

    double max_val = -std::numeric_limits<double>::infinity();
    for (const float &feat : val.features) {
        max_val = std::max(max_val, static_cast<double>(feat));
    }
    CHECK(max_val > -5.0);
}

TEST_CASE("build_speaker_mel_features silence produces floor values", "[speaker_mel]") {
    const int64_t sample_rate = 24000;
    const int64_t mel_bins = 128;
    const int64_t fft_size = 1024;
    const int64_t hop_length = 256;
    const int64_t duration_samples = sample_rate;

    std::vector<float> audio(static_cast<size_t>(duration_samples), 0.0f);
    auto result = build_speaker_mel_features(
        audio.data(), duration_samples, sample_rate, sample_rate,
        mel_bins, fft_size, hop_length, 0.0, 12000.0
    );

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    REQUIRE_FALSE(val.features.empty());

    for (const float &feat : val.features) {
        CHECK(std::isfinite(static_cast<double>(feat)));
    }
}

TEST_CASE("build_speaker_mel_features fmin fmax constrain mel range", "[speaker_mel]") {
    const int64_t sample_rate = 24000;
    const int64_t mel_bins = 64;
    const int64_t fft_size = 1024;
    const int64_t hop_length = 256;
    const int64_t duration_samples = sample_rate;

    std::vector<float> audio(static_cast<size_t>(duration_samples));
    for (int64_t i = 0; i < duration_samples; ++i) {
        audio[static_cast<size_t>(i)] =
            static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * 500.0 *
                                        static_cast<double>(i) / static_cast<double>(sample_rate)));
    }

    auto result = build_speaker_mel_features(
        audio.data(), duration_samples, sample_rate, sample_rate,
        mel_bins, fft_size, hop_length, 200.0, 4000.0
    );

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    CHECK(val.mel_dim == mel_bins);
    CHECK(val.frames > 0);
    CHECK_FALSE(val.features.empty());

    for (const float &feat : val.features) {
        CHECK(std::isfinite(static_cast<double>(feat)));
    }
}

TEST_CASE("build_speaker_mel_features short audio produces at least one frame", "[speaker_mel]") {
    const int64_t sample_rate = 24000;
    const int64_t mel_bins = 128;
    const int64_t fft_size = 1024;
    const int64_t hop_length = 256;

    std::vector<float> audio(static_cast<size_t>(hop_length), 0.1f);
    auto result = build_speaker_mel_features(
        audio.data(), hop_length, sample_rate, sample_rate,
        mel_bins, fft_size, hop_length, 0.0, 12000.0
    );

    REQUIRE(result.is_ok());
    const auto &val = result.value();
    CHECK(val.frames >= 1);
    CHECK(val.mel_dim == mel_bins);
}

TEST_CASE("SpeakerMelResult default values", "[speaker_mel]") {
    SpeakerMelResult r;
    CHECK(r.features.empty());
    CHECK(r.frames == 0);
    CHECK(r.mel_dim == 0);
}
