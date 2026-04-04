#pragma once

#include "gotst/core/result.hpp"

#include <cstdint>
#include <vector>

namespace gotst {

struct VoiceCloneIclResult {
    std::vector<float> icl_overlay;
    int64_t icl_length = 0;
};

Result<VoiceCloneIclResult> build_voice_clone_icl_overlay(
    const float *ref_text_projected,
    int64_t ref_text_length,
    const float *ref_codec_projected,
    int64_t ref_codec_frames,
    int64_t hidden_size
);

}
