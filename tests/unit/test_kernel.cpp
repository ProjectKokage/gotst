#include "gotst/core/tts_prompt_assembly.hpp"
#include "gotst/core/cancellation_token.hpp"
#include "gotst/core/language_config.hpp"
#include "gotst/core/fft.hpp"
#include "gotst/core/speech_runtime_core.hpp"
#include "gotst/core/result.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

using Catch::Approx;

TEST_CASE("VoiceCloneIclResult defaults are empty", "[kernel]") {
    gotst::VoiceCloneIclResult r;
    CHECK(r.icl_overlay.empty());
    CHECK(r.icl_length == 0);
}

TEST_CASE("build_voice_clone_icl_overlay with null inputs returns error", "[kernel]") {
    auto r = gotst::build_voice_clone_icl_overlay(nullptr, 0, nullptr, 0, 256);
    CHECK(!r.is_ok());
}

TEST_CASE("build_voice_clone_icl_overlay with codec zero frames returns text only", "[kernel]") {
    std::vector<float> text(256, 0.5f);
    auto r = gotst::build_voice_clone_icl_overlay(text.data(), 1, nullptr, 0, 256);
    REQUIRE(r.is_ok());
    CHECK(r.value().icl_length == 1);
    CHECK(static_cast<int64_t>(r.value().icl_overlay.size()) == 256);
}

TEST_CASE("build_voice_clone_icl_overlay produces max_len output", "[kernel]") {
    const int64_t hidden = 64;
    const int64_t text_len = 3;
    const int64_t codec_frames = 5;
    std::vector<float> text(static_cast<size_t>(text_len * hidden), 0.1f);
    std::vector<float> codec(static_cast<size_t>(codec_frames * hidden), 0.2f);

    auto r = gotst::build_voice_clone_icl_overlay(text.data(), text_len, codec.data(), codec_frames, hidden);
    REQUIRE(r.is_ok());
    CHECK(r.value().icl_length == std::max(text_len, codec_frames));
    CHECK(static_cast<int64_t>(r.value().icl_overlay.size()) == std::max(text_len, codec_frames) * hidden);
}

TEST_CASE("build_voice_clone_icl_overlay adds overlapping rows", "[kernel]") {
    const int64_t hidden = 4;
    const int64_t text_len = 2;
    const int64_t codec_frames = 2;
    std::vector<float> text(static_cast<size_t>(text_len * hidden));
    std::vector<float> codec(static_cast<size_t>(codec_frames * hidden));
    for (int64_t i = 0; i < text_len * hidden; ++i) text[static_cast<size_t>(i)] = 1.0f;
    for (int64_t i = 0; i < codec_frames * hidden; ++i) codec[static_cast<size_t>(i)] = 2.0f;

    auto r = gotst::build_voice_clone_icl_overlay(text.data(), text_len, codec.data(), codec_frames, hidden);

    CHECK(r.value().icl_overlay[0] == Approx(3.0f));
    CHECK(r.value().icl_length == 2);
}

TEST_CASE("CancellationToken concurrent cancel is safe", "[kernel]") {
    gotst::CancellationToken token;
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&token]() {
            token.cancel();
        });
    }
    for (auto &t : threads) t.join();
    CHECK(token.is_cancelled());
}

TEST_CASE("CancellationToken reset after cancel allows reuse", "[kernel]") {
    gotst::CancellationToken token;
    token.cancel();
    CHECK(token.is_cancelled());
    token.reset();
    CHECK_FALSE(token.is_cancelled());
}

TEST_CASE("LanguageConfig build_codec_prefix_tokens default token IDs", "[kernel]") {
    auto tokens = gotst::LanguageConfig::build_codec_prefix_tokens(2050, 2154, 2155, 2156, 2157, 2148, 2149);
    REQUIRE(tokens.size() == 6);
    CHECK(tokens[0] == 2154);
    CHECK(tokens[1] == 2156);
    CHECK(tokens[2] == 2050);
    CHECK(tokens[3] == 2157);
    CHECK(tokens[4] == 2148);
    CHECK(tokens[5] == 2149);
}

TEST_CASE("LanguageConfig build_codec_prefix_tokens auto-detect mode", "[kernel]") {
    auto tokens = gotst::LanguageConfig::build_codec_prefix_tokens(-1, 2154, 2155, 2156, 2157, 2148, 2149);
    REQUIRE(tokens.size() == 5);
    CHECK(tokens[0] == 2155);
    CHECK(tokens[1] == 2156);
    CHECK(tokens[2] == 2157);
    CHECK(tokens[3] == 2148);
    CHECK(tokens[4] == 2149);
}

TEST_CASE("radix2_fft linearity property", "[kernel]") {
    constexpr int64_t N = 8;
    double a_real[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    double a_imag[N] = {0};
    double b_real[N] = {8, 7, 6, 5, 4, 3, 2, 1};
    double b_imag[N] = {0};
    double sum_real[N], sum_imag[N];
    for (int64_t i = 0; i < N; ++i) {
        sum_real[i] = a_real[i] + b_real[i];
        sum_imag[i] = a_imag[i] + b_imag[i];
    }

    double fft_sum_real[N], fft_sum_imag[N];
    gotst::radix2_fft(sum_real, sum_imag, N, fft_sum_real, fft_sum_imag);

    double fft_a_real[N], fft_a_imag[N];
    gotst::radix2_fft(a_real, a_imag, N, fft_a_real, fft_a_imag);

    double fft_b_real[N], fft_b_imag[N];
    gotst::radix2_fft(b_real, b_imag, N, fft_b_real, fft_b_imag);

    for (int64_t k = 0; k < N; ++k) {
        double expected_real = fft_a_real[k] + fft_b_real[k];
        double expected_imag = fft_a_imag[k] + fft_b_imag[k];
        CHECK(fft_sum_real[k] == Approx(expected_real));
        CHECK(fft_sum_imag[k] == Approx(expected_imag));
    }
}

TEST_CASE("CustomVoiceSpeakerEntry default initialization", "[kernel]") {
    gotst::CustomVoiceSpeakerEntry entry;
    CHECK(entry.token_ids.empty());
}

TEST_CASE("build_voice_clone_icl_overlay single frame codec", "[kernel]") {
    const int64_t hidden = 8;
    std::vector<float> text(static_cast<size_t>(2 * hidden), 1.0f);
    std::vector<float> codec(static_cast<size_t>(1 * hidden), 2.0f);

    auto r = gotst::build_voice_clone_icl_overlay(text.data(), 2, codec.data(), 1, hidden);
    REQUIRE(r.is_ok());
    CHECK(r.value().icl_length == 2);
    CHECK(static_cast<int64_t>(r.value().icl_overlay.size()) == 2 * hidden);
    CHECK(r.value().icl_overlay[0] == Approx(3.0f));
    CHECK(r.value().icl_overlay[static_cast<size_t>(hidden)] == Approx(1.0f));
}

TEST_CASE("BackendSummary default flags are false", "[kernel]") {
    gotst::BackendSummary s;
    CHECK_FALSE(s.uses_llama_core);
    CHECK_FALSE(s.uses_onnx_core);
    CHECK_FALSE(s.asr_hybrid_requested);
    CHECK_FALSE(s.tts_hybrid_requested);
    CHECK(s.missing_paths.empty());
    CHECK(s.notes.empty());
}

TEST_CASE("SpeakerMelParams default values", "[kernel]") {
    gotst::SpeakerMelParams p;
    CHECK(p.sample_rate == 24000);
    CHECK(p.n_fft == 1024);
    CHECK(p.hop_length == 256);
    CHECK(p.n_mels == 128);
    CHECK(p.fmin == 0.0);
    CHECK(p.fmax == 12000.0);
}

TEST_CASE("Result<T> with value is ok", "[result]") {
    gotst::Result<int> r(42);
    CHECK(r.is_ok());
    CHECK(r.value() == 42);
    CHECK(r.error_code() == gotst::ErrorCode::Ok);
}

TEST_CASE("Result<T> with error is not ok", "[result]") {
    gotst::Result<int> r(gotst::Error::not_found("missing file"));
    CHECK_FALSE(r.is_ok());
    CHECK(r.error_code() == gotst::ErrorCode::NotFound);
    CHECK(r.error_message() == "missing file");
}

TEST_CASE("Result<void> default is ok", "[result]") {
    gotst::Result<void> r;
    CHECK(r.is_ok());
    CHECK(r.error_code() == gotst::ErrorCode::Ok);
}

TEST_CASE("Result<void> with error is not ok", "[result]") {
    gotst::Result<void> r(gotst::Error::cancelled("request aborted"));
    CHECK_FALSE(r.is_ok());
    CHECK(r.error_code() == gotst::ErrorCode::Cancelled);
}

TEST_CASE("Error factory methods produce correct codes", "[result]") {
    CHECK(gotst::Error::ok().is_ok());
    CHECK(gotst::Error::invalid_argument("x").code == gotst::ErrorCode::InvalidArgument);
    CHECK(gotst::Error::invalid_state("x").code == gotst::ErrorCode::InvalidState);
    CHECK(gotst::Error::not_found("x").code == gotst::ErrorCode::NotFound);
    CHECK(gotst::Error::io_error("x").code == gotst::ErrorCode::IoError);
    CHECK(gotst::Error::model_not_loaded("x").code == gotst::ErrorCode::ModelNotLoaded);
    CHECK(gotst::Error::inference_failed("x").code == gotst::ErrorCode::InferenceFailed);
    CHECK(gotst::Error::shape_mismatch("x").code == gotst::ErrorCode::ShapeMismatch);
    CHECK(gotst::Error::empty_input("x").code == gotst::ErrorCode::EmptyInput);
    CHECK(gotst::Error::cancelled("x").code == gotst::ErrorCode::Cancelled);
}

TEST_CASE("Error message is preserved", "[result]") {
    auto e = gotst::Error::io_error("cannot read /tmp/model.onnx");
    CHECK(e.message == "cannot read /tmp/model.onnx");
    CHECK_FALSE(e.is_ok());
}
