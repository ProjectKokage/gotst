#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "gotst/core/fft.hpp"

#include <cmath>
#include <vector>

using Catch::Approx;

TEST_CASE("radix2_fft size 1 returns input", "[fft]") {
    double real_in[] = {3.0};
    double imag_in[] = {0.0};
    double real_out[1];
    double imag_out[1];
    gotst::radix2_fft(real_in, imag_in, 1, real_out, imag_out);
    REQUIRE(real_out[0] == Catch::Approx(3.0));
    REQUIRE(imag_out[0] == Catch::Approx(0.0));
}

TEST_CASE("radix2_fft DC signal", "[fft]") {
    constexpr int64_t N = 8;
    double real_in[N], imag_in[N];
    for (int64_t i = 0; i < N; ++i) {
        real_in[i] = 1.0;
        imag_in[i] = 0.0;
    }
    double real_out[N], imag_out[N];
    gotst::radix2_fft(real_in, imag_in, N, real_out, imag_out);
    REQUIRE(real_out[0] == Catch::Approx(static_cast<double>(N)));
    for (int64_t k = 1; k < N; ++k) {
        REQUIRE(std::abs(real_out[k]) < 1e-9);
        REQUIRE(std::abs(imag_out[k]) < 1e-9);
    }
}

TEST_CASE("radix2_fft single frequency sine", "[fft]") {
    constexpr int64_t N = 8;
    double real_in[N], imag_in[N];
    for (int64_t i = 0; i < N; ++i) {
        real_in[i] = std::sin(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(N));
        imag_in[i] = 0.0;
    }
    double real_out[N], imag_out[N];
    gotst::radix2_fft(real_in, imag_in, N, real_out, imag_out);

    REQUIRE(std::abs(real_out[0]) < 1e-9);
    REQUIRE(std::abs(real_out[N / 2]) < 1e-9);

    double power_bin1 = real_out[1] * real_out[1] + imag_out[1] * imag_out[1];
    double power_bin7 = real_out[7] * real_out[7] + imag_out[7] * imag_out[7];
    REQUIRE(power_bin1 > 1.0);
    REQUIRE(power_bin7 > 1.0);
}

TEST_CASE("radix2_fft Parseval's theorem holds", "[fft]") {
    constexpr int64_t N = 16;
    double real_in[N], imag_in[N];
    for (int64_t i = 0; i < N; ++i) {
        real_in[i] = std::cos(2.0 * M_PI * 3.0 * static_cast<double>(i) / static_cast<double>(N)) +
                      0.5 * std::sin(2.0 * M_PI * 7.0 * static_cast<double>(i) / static_cast<double>(N));
        imag_in[i] = 0.0;
    }
    double real_out[N], imag_out[N];
    gotst::radix2_fft(real_in, imag_in, N, real_out, imag_out);

    double time_energy = 0.0;
    for (int64_t i = 0; i < N; ++i) {
        time_energy += real_in[i] * real_in[i];
    }

    double freq_energy = 0.0;
    for (int64_t k = 0; k < N; ++k) {
        freq_energy += real_out[k] * real_out[k] + imag_out[k] * imag_out[k];
    }

    REQUIRE(freq_energy == Catch::Approx(time_energy * static_cast<double>(N)).margin(1e-6));
}

TEST_CASE("radix2_fft size 4 impulse", "[fft]") {
    double real_in[] = {1.0, 0.0, 0.0, 0.0};
    double imag_in[] = {0.0, 0.0, 0.0, 0.0};
    double real_out[4], imag_out[4];
    gotst::radix2_fft(real_in, imag_in, 4, real_out, imag_out);
    for (int64_t k = 0; k < 4; ++k) {
        REQUIRE(real_out[k] == Catch::Approx(1.0));
        REQUIRE(imag_out[k] == Catch::Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("compute_power_spectrum_fft produces non-negative values", "[fft]") {
    constexpr int64_t N = 64;
    std::vector<float> samples(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        samples[static_cast<size_t>(i)] = static_cast<float>(
            std::sin(2.0 * M_PI * 440.0 * static_cast<double>(i) / 16000.0));
    }
    std::vector<float> window(static_cast<size_t>(N), 1.0f);
    const int64_t freq_bins = N / 2 + 1;
    std::vector<float> spectrum(static_cast<size_t>(freq_bins));

    gotst::compute_power_spectrum_fft(
        samples.data(), N, 0, freq_bins, window.data(), N, spectrum.data()
    );

    for (int64_t k = 0; k < freq_bins; ++k) {
        REQUIRE(spectrum[static_cast<size_t>(k)] >= 0.0f);
    }
}

TEST_CASE("compute_power_spectrum_fft silence is near zero", "[fft]") {
    constexpr int64_t N = 64;
    std::vector<float> samples(static_cast<size_t>(N), 0.0f);
    std::vector<float> window(static_cast<size_t>(N), 1.0f);
    const int64_t freq_bins = N / 2 + 1;
    std::vector<float> spectrum(static_cast<size_t>(freq_bins));

    gotst::compute_power_spectrum_fft(
        samples.data(), N, 0, freq_bins, window.data(), N, spectrum.data()
    );

    for (int64_t k = 0; k < freq_bins; ++k) {
        REQUIRE(spectrum[static_cast<size_t>(k)] == Catch::Approx(0.0f).margin(1e-6f));
    }
}

TEST_CASE("compute_power_spectrum_fft sine peaks at correct bin", "[fft]") {
    constexpr int64_t N = 512;
    constexpr int64_t target_bin = 16;
    std::vector<float> samples(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        samples[static_cast<size_t>(i)] = static_cast<float>(
            std::cos(2.0 * M_PI * static_cast<double>(target_bin) * static_cast<double>(i) / static_cast<double>(N)));
    }
    std::vector<float> window(static_cast<size_t>(N), 1.0f);
    const int64_t freq_bins = N / 2 + 1;
    std::vector<float> spectrum(static_cast<size_t>(freq_bins));

    gotst::compute_power_spectrum_fft(
        samples.data(), N, 0, freq_bins, window.data(), N, spectrum.data()
    );

    int64_t peak_bin = 0;
    float peak_val = -1.0f;
    for (int64_t k = 0; k < freq_bins; ++k) {
        if (spectrum[static_cast<size_t>(k)] > peak_val) {
            peak_val = spectrum[static_cast<size_t>(k)];
            peak_bin = k;
        }
    }
    REQUIRE(peak_bin == target_bin);
    REQUIRE(peak_val > 10.0f);
}

TEST_CASE("compute_power_spectrum_fft handles negative offset with reflection", "[fft]") {
    constexpr int64_t N = 64;
    std::vector<float> samples(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        samples[static_cast<size_t>(i)] = static_cast<float>(i);
    }
    std::vector<float> window(static_cast<size_t>(N), 1.0f);
    const int64_t freq_bins = N / 2 + 1;
    std::vector<float> spectrum(static_cast<size_t>(freq_bins));

    gotst::compute_power_spectrum_fft(
        samples.data(), N, -16, freq_bins, window.data(), N, spectrum.data()
    );

    bool has_nonzero = false;
    for (int64_t k = 0; k < freq_bins; ++k) {
        if (spectrum[static_cast<size_t>(k)] > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    REQUIRE(has_nonzero);
}

TEST_CASE("compute_power_spectrum_fft reuses workspace without changing results", "[fft]") {
    constexpr int64_t N = 128;
    const int64_t freq_bins = N / 2 + 1;
    std::vector<float> samples(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        samples[static_cast<size_t>(i)] = static_cast<float>(
            std::sin(2.0 * M_PI * 220.0 * static_cast<double>(i) / 16000.0)
        );
    }
    std::vector<float> window(static_cast<size_t>(N), 1.0f);
    std::vector<float> reference(static_cast<size_t>(freq_bins));
    std::vector<float> reused_first(static_cast<size_t>(freq_bins));
    std::vector<float> reused_second(static_cast<size_t>(freq_bins));
    gotst::FftWorkspace workspace;

    gotst::compute_power_spectrum_fft(
        samples.data(), N, 0, freq_bins, window.data(), N, reference.data()
    );
    gotst::compute_power_spectrum_fft(
        samples.data(), N, 0, freq_bins, window.data(), N, reused_first.data(), &workspace
    );
    gotst::compute_power_spectrum_fft(
        samples.data(), N, 16, freq_bins, window.data(), N, reused_second.data(), &workspace
    );

    for (int64_t bin = 0; bin < freq_bins; ++bin) {
        CHECK(reused_first[static_cast<size_t>(bin)] == Approx(reference[static_cast<size_t>(bin)]).margin(1e-6f));
    }

    bool changed_with_offset = false;
    for (int64_t bin = 0; bin < freq_bins; ++bin) {
        if (std::abs(reused_second[static_cast<size_t>(bin)] - reused_first[static_cast<size_t>(bin)]) > 1e-6f) {
            changed_with_offset = true;
            break;
        }
    }
    REQUIRE(changed_with_offset);
}
