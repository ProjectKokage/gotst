#include "gotst/core/irodori_tts_session.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;

TEST_CASE("Irodori mode parser accepts base and voice design aliases") {
    CHECK(gotst::parse_irodori_tts_mode("base_v3") == gotst::IrodoriTtsMode::BaseV3);
    CHECK(gotst::parse_irodori_tts_mode("base") == gotst::IrodoriTtsMode::BaseV3);
    CHECK(gotst::parse_irodori_tts_mode("base_v2") == gotst::IrodoriTtsMode::BaseV2);
    CHECK(gotst::parse_irodori_tts_mode("voice_design_v2") == gotst::IrodoriTtsMode::VoiceDesignV2);
    CHECK(gotst::parse_irodori_tts_mode("voice_design") == gotst::IrodoriTtsMode::VoiceDesignV2);
    CHECK(gotst::parse_irodori_tts_mode("voice_design_v3") == gotst::IrodoriTtsMode::VoiceDesignV3);
    CHECK(gotst::parse_irodori_tts_mode("other") == gotst::IrodoriTtsMode::Unknown);
}

TEST_CASE("Irodori Sway schedule matches upstream shape constraints") {
    auto linear = gotst::build_irodori_sway_schedule(4, "linear", -1.0f);
    REQUIRE(linear.is_ok());
    REQUIRE(linear.value().size() == 5);
    CHECK(linear.value().front() == 0.999f);
    CHECK(linear.value().back() == 0.0f);
    for(size_t index = 0; index + 1 < linear.value().size(); ++index) {
        CHECK(linear.value()[index] > linear.value()[index + 1]);
    }

    auto sway = gotst::build_irodori_sway_schedule(8, "sway", -1.0f);
    REQUIRE(sway.is_ok());
    REQUIRE(sway.value().size() == 9);
    CHECK(sway.value().front() == 0.999f);
    CHECK(sway.value().back() == 0.0f);
    for(size_t index = 0; index + 1 < sway.value().size(); ++index) {
        CHECK(sway.value()[index] > sway.value()[index + 1]);
    }

    auto invalid = gotst::build_irodori_sway_schedule(0, "sway", -1.0f);
    CHECK_FALSE(invalid.is_ok());
}

TEST_CASE("Irodori bucket selector chooses the smallest fitting shape") {
    std::vector<gotst::IrodoriTtsBucket> buckets = {
        {128, 128, 0, 128},
        {256, 192, 0, 256},
        {512, 256, 0, 512},
    };

    auto selected = gotst::select_irodori_bucket(buckets, 160, 150, 0, 128);
    REQUIRE(selected.is_ok());
    CHECK(selected.value().latent_steps == 256);
    CHECK(selected.value().text_tokens == 192);

    auto too_large = gotst::select_irodori_bucket(buckets, 768, 150, 0, 128);
    CHECK_FALSE(too_large.is_ok());
    CHECK(too_large.error_code() == gotst::ErrorCode::ShapeMismatch);
}

TEST_CASE("Irodori CFG branch construction follows mode-specific conditions") {
    auto base = gotst::build_irodori_cfg_branches(
        gotst::IrodoriTtsMode::BaseV3,
        "alternating",
        3.0f,
        3.0f,
        5.0f
    );
    REQUIRE(base.is_ok());
    CHECK(base.value() == std::vector<std::string>{"conditional", "drop_text", "drop_speaker"});

    auto voice_design = gotst::build_irodori_cfg_branches(
        gotst::IrodoriTtsMode::VoiceDesignV2,
        "joint",
        3.0f,
        3.0f,
        5.0f
    );
    REQUIRE(voice_design.is_ok());
    CHECK(voice_design.value() == std::vector<std::string>{"conditional", "drop_all"});

    auto voice_design_v3 = gotst::build_irodori_cfg_branches(
        gotst::IrodoriTtsMode::VoiceDesignV3,
        "alternating",
        3.0f,
        3.0f,
        5.0f
    );
    REQUIRE(voice_design_v3.is_ok());
    CHECK(voice_design_v3.value() == std::vector<std::string>{"conditional", "drop_text", "drop_speaker", "drop_caption"});

    auto invalid_joint = gotst::build_irodori_cfg_branches(
        gotst::IrodoriTtsMode::VoiceDesignV2,
        "joint",
        3.0f,
        4.0f,
        5.0f
    );
    CHECK_FALSE(invalid_joint.is_ok());
    CHECK_THAT(invalid_joint.error_message(), ContainsSubstring("joint CFG"));
}

TEST_CASE("Irodori CFG step branch planner matches guidance modes") {
    std::vector<std::string> branches = {
        "conditional",
        "drop_text",
        "drop_speaker",
        "drop_caption",
    };

    auto independent = gotst::build_irodori_cfg_step_drop_branches(branches, "independent", 0);
    REQUIRE(independent.is_ok());
    CHECK(independent.value() == std::vector<std::string>{"text", "speaker", "caption"});

    auto alternating0 = gotst::build_irodori_cfg_step_drop_branches(branches, "alternating", 0);
    REQUIRE(alternating0.is_ok());
    CHECK(alternating0.value() == std::vector<std::string>{"text"});

    auto alternating2 = gotst::build_irodori_cfg_step_drop_branches(branches, "alternating", 2);
    REQUIRE(alternating2.is_ok());
    CHECK(alternating2.value() == std::vector<std::string>{"caption"});

    auto joint = gotst::build_irodori_cfg_step_drop_branches(
        std::vector<std::string>{"conditional", "drop_all"},
        "joint",
        0
    );
    REQUIRE(joint.is_ok());
    CHECK(joint.value() == std::vector<std::string>{"all"});

    auto cfg_off = gotst::build_irodori_cfg_step_drop_branches(
        std::vector<std::string>{"conditional"},
        "alternating",
        0
    );
    REQUIRE(cfg_off.is_ok());
    CHECK(cfg_off.value().empty());
}

TEST_CASE("Irodori cache keys separate mode, caption, and reference state") {
    std::string base_key = gotst::build_irodori_condition_cache_key(
        gotst::IrodoriTtsMode::BaseV3,
        "konnichiwa",
        "",
        "/voices/ref.latent",
        false
    );
    std::string voice_design_key = gotst::build_irodori_condition_cache_key(
        gotst::IrodoriTtsMode::VoiceDesignV2,
        "konnichiwa",
        "bright voice",
        "",
        true
    );

    CHECK(base_key != voice_design_key);
    CHECK(base_key.find("/voices/ref.latent") != std::string::npos);
    CHECK(voice_design_key.find("caption:") != std::string::npos);
}

TEST_CASE("Irodori session supports manifest-only load for tooling tests") {
    gotst::IrodoriTtsSessionConfig config;
    config.mode = gotst::IrodoriTtsMode::VoiceDesignV2;
    config.require_all_artifacts = false;
    config.default_cfg_guidance_mode = "joint";
    config.buckets = {
        {128, 128, 256, 0},
    };

    gotst::IrodoriTtsSession session;
    auto load_result = session.load(config);
    REQUIRE(load_result.is_ok());
    CHECK(session.is_loaded());
    CHECK_FALSE(session.is_execution_ready());

    gotst::IrodoriTtsRequest request;
    request.text = "konnichiwa";
    request.caption = "bright voice";
    request.no_ref = true;
    request.cfg_guidance_mode = "joint";
    request.cfg_scale_text = 3.0f;
    request.cfg_scale_caption = 3.0f;

    gotst::CancellationToken cancelled;
    cancelled.cancel();
    auto cancelled_result = session.synthesize(request, &cancelled);
    CHECK_FALSE(cancelled_result.is_ok());
    CHECK(cancelled_result.error_code() == gotst::ErrorCode::Cancelled);

    auto synth_result = session.synthesize(request, nullptr);
    CHECK_FALSE(synth_result.is_ok());
    CHECK(synth_result.error_code() == gotst::ErrorCode::ModelNotLoaded);
}

TEST_CASE("Irodori static CoreML strict load requires bucket artifacts") {
    gotst::IrodoriTtsSessionConfig config;
    config.mode = gotst::IrodoriTtsMode::BaseV3;
    config.provider_profile = "coreml_static";
    config.strict_provider = true;
    config.require_all_artifacts = false;
    config.provider_routes.dit_step.provider_requested = "CPU";
    config.provider_routes.dit_step.provider = "CPU";

    gotst::IrodoriTtsSession session;
    auto load_result = session.load(config);
    CHECK_FALSE(load_result.is_ok());
    CHECK(load_result.error_code() == gotst::ErrorCode::ModelNotLoaded);
    CHECK_THAT(load_result.error_message(), ContainsSubstring("no static DiT artifacts"));
}
