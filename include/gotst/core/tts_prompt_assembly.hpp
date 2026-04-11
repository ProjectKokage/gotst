#pragma once

#include "gotst/core/result.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace gotst {

struct VoiceCloneIclResult {
    std::vector<float> icl_overlay;
    int64_t icl_length = 0;
};

struct TtsPromptAssemblyInputs {
    std::span<const float> text_projected_states;
    int64_t text_sequence_length = 0;
    std::span<const float> special_projected_states;
    std::span<const float> codec_prefill_embeddings;
    int64_t codec_prefill_length = 0;
    std::span<const float> codec_prompt_insert;
    std::span<const float> leading_prompt_states;
    int64_t leading_prompt_length = 0;
    std::span<const float> icl_overlay;
    int64_t icl_length = 0;
    int64_t hidden_size = 0;
    int64_t wrapped_prefix_token_count = 0;
    int64_t wrapped_suffix_token_count = 0;
};

struct TtsPromptAssemblyResult {
    std::vector<float> language_sequence;
    int64_t language_sequence_length = 0;
    std::vector<float> trailing_text_hidden;
    int64_t trailing_text_length = 0;
    std::vector<float> tts_pad_embedding;
    int64_t hidden_size = 0;
    int64_t produced_frames = 0;
    int64_t icl_length = 0;
};

Result<VoiceCloneIclResult> build_voice_clone_icl_overlay(
    const float *ref_text_projected,
    int64_t ref_text_length,
    const float *ref_codec_projected,
    int64_t ref_codec_frames,
    int64_t hidden_size
);

Result<TtsPromptAssemblyResult> build_tts_prompt_assembly(const TtsPromptAssemblyInputs &inputs);

}
