#include "gotst/core/asr_frontend.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using Catch::Approx;
using namespace gotst;

namespace {

std::vector<float> make_sine_wave(int64_t sample_count, int64_t sample_rate, double frequency_hz) {
    std::vector<float> waveform(static_cast<size_t>(sample_count));
    for(int64_t index = 0; index < sample_count; ++index) {
        waveform[static_cast<size_t>(index)] = static_cast<float>(
            std::sin((2.0 * 3.14159265358979323846 * frequency_hz * static_cast<double>(index)) /
                     static_cast<double>(sample_rate))
        );
    }
    return waveform;
}

} // namespace

TEST_CASE("build_asr_log_mel_features rejects invalid input", "[asr_frontend]") {
    float dummy = 0.0f;
    CHECK_FALSE(build_asr_log_mel_features(nullptr, 1, 16000, 16000, 8, 32, 16, 1.0).is_ok());
    CHECK_FALSE(build_asr_log_mel_features(&dummy, 0, 16000, 16000, 8, 32, 16, 1.0).is_ok());
    CHECK_FALSE(build_asr_log_mel_features(&dummy, 1, 0, 16000, 8, 32, 16, 1.0).is_ok());
    CHECK_FALSE(build_asr_log_mel_features(&dummy, 1, 16000, 0, 8, 32, 16, 1.0).is_ok());
    CHECK_FALSE(build_asr_log_mel_features(&dummy, 1, 16000, 16000, 0, 32, 16, 1.0).is_ok());
    CHECK_FALSE(build_asr_log_mel_features(&dummy, 1, 16000, 16000, 8, 0, 16, 1.0).is_ok());
    CHECK_FALSE(build_asr_log_mel_features(&dummy, 1, 16000, 16000, 8, 32, 0, 1.0).is_ok());
}

TEST_CASE("build_asr_log_mel_features produces expected dimensions and finite values", "[asr_frontend]") {
    const int64_t sample_rate = 16000;
    const int64_t hop_length = 160;
    const int64_t mel_bins = 8;
    const int64_t fft_size = 400;
    const std::vector<float> waveform = make_sine_wave(1600, sample_rate, 440.0);

    auto result = build_asr_log_mel_features(
        waveform.data(),
        static_cast<int64_t>(waveform.size()),
        sample_rate,
        sample_rate,
        mel_bins,
        fft_size,
        hop_length,
        1.0
    );

    REQUIRE(result.is_ok());
    const auto &value = result.value();
    CHECK(value.frame_count == 10);
    CHECK(value.valid_frame_count == value.frame_count);
    REQUIRE(value.features.size() == static_cast<size_t>(mel_bins * value.frame_count));
    for(float feature : value.features) {
        CHECK(std::isfinite(static_cast<double>(feature)));
    }
    CHECK(*std::max_element(value.features.begin(), value.features.end()) > 0.0f);
}

TEST_CASE("build_asr_log_mel_features trims to the latest chunk", "[asr_frontend]") {
    const int64_t sample_rate = 16000;
    const int64_t mel_bins = 8;
    const int64_t fft_size = 400;
    const int64_t hop_length = 160;
    const std::vector<float> waveform = make_sine_wave(20000, sample_rate, 330.0);
    const std::vector<float> tail(
        waveform.end() - static_cast<std::ptrdiff_t>(sample_rate),
        waveform.end()
    );

    auto full = build_asr_log_mel_features(
        waveform.data(),
        static_cast<int64_t>(waveform.size()),
        sample_rate,
        sample_rate,
        mel_bins,
        fft_size,
        hop_length,
        1.0
    );
    auto trimmed = build_asr_log_mel_features(
        tail.data(),
        static_cast<int64_t>(tail.size()),
        sample_rate,
        sample_rate,
        mel_bins,
        fft_size,
        hop_length,
        1.0
    );

    REQUIRE(full.is_ok());
    REQUIRE(trimmed.is_ok());
    REQUIRE(full.value().features.size() == trimmed.value().features.size());
    CHECK(full.value().frame_count == trimmed.value().frame_count);
    for(size_t index = 0; index < full.value().features.size(); ++index) {
        CHECK(full.value().features[index] == Approx(trimmed.value().features[index]).margin(1e-6));
    }
}

TEST_CASE("build_asr_log_mel_features maps silence to a stable floor", "[asr_frontend]") {
    const std::vector<float> silence(1600, 0.0f);

    auto result = build_asr_log_mel_features(
        silence.data(),
        static_cast<int64_t>(silence.size()),
        16000,
        16000,
        8,
        400,
        160,
        1.0
    );

    REQUIRE(result.is_ok());
    REQUIRE_FALSE(result.value().features.empty());
    const float first = result.value().features.front();
    for(float feature : result.value().features) {
        CHECK(feature == Approx(first).margin(1e-6));
    }
}
