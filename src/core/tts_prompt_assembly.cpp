#include "gotst/core/tts_prompt_assembly.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace gotst {

Result<VoiceCloneIclResult> build_voice_clone_icl_overlay(
    const float *ref_text_projected,
    int64_t ref_text_length,
    const float *ref_codec_projected,
    int64_t ref_codec_frames,
    int64_t hidden_size
) {
    if (hidden_size <= 0) {
        return Error::invalid_argument("build_voice_clone_icl_overlay: hidden_size must be positive");
    }

    if ((!ref_text_projected || ref_text_length <= 0) &&
        (!ref_codec_projected || ref_codec_frames <= 0)) {
        return Error::empty_input("build_voice_clone_icl_overlay: both inputs are empty");
    }

    const int64_t text_len = (ref_text_projected && ref_text_length > 0) ? ref_text_length : 0;
    const int64_t codec_len = (ref_codec_projected && ref_codec_frames > 0) ? ref_codec_frames : 0;

    const int64_t overlap_len = std::min(text_len, codec_len);
    const int64_t max_len = std::max(text_len, codec_len);

    if (max_len <= 0) {
        return Error::empty_input("build_voice_clone_icl_overlay: computed max_len is zero");
    }

    VoiceCloneIclResult result;
    result.icl_length = max_len;
    result.icl_overlay.resize(static_cast<size_t>(max_len * hidden_size), 0.0f);

    for (int64_t i = 0; i < overlap_len; ++i) {
        for (int64_t h = 0; h < hidden_size; ++h) {
            const int64_t out_idx = i * hidden_size + h;
            float text_val = (i < text_len) ? ref_text_projected[i * hidden_size + h] : 0.0f;
            float codec_val = (i < codec_len) ? ref_codec_projected[i * hidden_size + h] : 0.0f;
            result.icl_overlay[static_cast<size_t>(out_idx)] = text_val + codec_val;
        }
    }

    for (int64_t i = overlap_len; i < text_len; ++i) {
        for (int64_t h = 0; h < hidden_size; ++h) {
            const int64_t out_idx = i * hidden_size + h;
            result.icl_overlay[static_cast<size_t>(out_idx)] = ref_text_projected[i * hidden_size + h];
        }
    }

    for (int64_t i = overlap_len; i < codec_len; ++i) {
        for (int64_t h = 0; h < hidden_size; ++h) {
            const int64_t out_idx = i * hidden_size + h;
            result.icl_overlay[static_cast<size_t>(out_idx)] = ref_codec_projected[i * hidden_size + h];
        }
    }

    return result;
}

}
