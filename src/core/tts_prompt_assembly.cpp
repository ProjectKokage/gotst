#include "gotst/core/tts_prompt_assembly.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace gotst {

namespace {

bool has_rows(std::span<const float> values, int64_t row_count, int64_t hidden_size) {
    if(row_count < 0 || hidden_size <= 0) {
        return false;
    }
    const size_t required =
        static_cast<size_t>(row_count) * static_cast<size_t>(hidden_size);
    return values.size() >= required;
}

const float *row_ptr(std::span<const float> values, int64_t row_index, int64_t hidden_size) {
    return values.data() + (static_cast<size_t>(row_index) * static_cast<size_t>(hidden_size));
}

void copy_rows(std::span<const float> source,
               int64_t start_row,
               int64_t row_count,
               int64_t hidden_size,
               std::vector<float> &destination,
               size_t &cursor) {
    if(row_count <= 0) {
        return;
    }

    const size_t float_count =
        static_cast<size_t>(row_count) * static_cast<size_t>(hidden_size);
    std::memcpy(
        destination.data() + cursor,
        source.data() + (static_cast<size_t>(start_row) * static_cast<size_t>(hidden_size)),
        float_count * sizeof(float)
    );
    cursor += float_count;
}

void copy_span(std::span<const float> source, std::vector<float> &destination, size_t &cursor) {
    if(source.empty()) {
        return;
    }

    std::memcpy(destination.data() + cursor, source.data(), source.size() * sizeof(float));
    cursor += source.size();
}

void add_rows(const float *left,
              const float *right,
              int64_t hidden_size,
              std::vector<float> &destination,
              size_t &cursor) {
    for(int64_t index = 0; index < hidden_size; ++index) {
        destination[cursor++] = left[index] + right[index];
    }
}

} // namespace

Result<VoiceCloneIclResult> build_voice_clone_icl_overlay(
    const float *ref_text_projected,
    int64_t ref_text_length,
    const float *ref_codec_projected,
    int64_t ref_codec_frames,
    int64_t hidden_size
) {
    if(hidden_size <= 0) {
        return Error::invalid_argument("build_voice_clone_icl_overlay: hidden_size must be positive");
    }

    if((!ref_text_projected || ref_text_length <= 0) &&
       (!ref_codec_projected || ref_codec_frames <= 0)) {
        return Error::empty_input("build_voice_clone_icl_overlay: both inputs are empty");
    }

    const int64_t text_len = (ref_text_projected && ref_text_length > 0) ? ref_text_length : 0;
    const int64_t codec_len = (ref_codec_projected && ref_codec_frames > 0) ? ref_codec_frames : 0;
    const int64_t overlap_len = std::min(text_len, codec_len);
    const int64_t max_len = std::max(text_len, codec_len);

    if(max_len <= 0) {
        return Error::empty_input("build_voice_clone_icl_overlay: computed max_len is zero");
    }

    VoiceCloneIclResult result;
    result.icl_length = max_len;
    result.icl_overlay.resize(static_cast<size_t>(max_len * hidden_size), 0.0f);

    for(int64_t row = 0; row < overlap_len; ++row) {
        for(int64_t hidden = 0; hidden < hidden_size; ++hidden) {
            const int64_t out_index = (row * hidden_size) + hidden;
            const float text_value = ref_text_projected[(row * hidden_size) + hidden];
            const float codec_value = ref_codec_projected[(row * hidden_size) + hidden];
            result.icl_overlay[static_cast<size_t>(out_index)] = text_value + codec_value;
        }
    }

    for(int64_t row = overlap_len; row < text_len; ++row) {
        for(int64_t hidden = 0; hidden < hidden_size; ++hidden) {
            const int64_t out_index = (row * hidden_size) + hidden;
            result.icl_overlay[static_cast<size_t>(out_index)] =
                ref_text_projected[(row * hidden_size) + hidden];
        }
    }

    for(int64_t row = overlap_len; row < codec_len; ++row) {
        for(int64_t hidden = 0; hidden < hidden_size; ++hidden) {
            const int64_t out_index = (row * hidden_size) + hidden;
            result.icl_overlay[static_cast<size_t>(out_index)] =
                ref_codec_projected[(row * hidden_size) + hidden];
        }
    }

    return result;
}

Result<TtsPromptAssemblyResult> build_tts_prompt_assembly(const TtsPromptAssemblyInputs &inputs) {
    if(inputs.hidden_size <= 0 || inputs.text_sequence_length <= 0 || inputs.codec_prefill_length < 2) {
        return Error::invalid_argument("build_tts_prompt_assembly: invalid shape parameters");
    }
    if(inputs.wrapped_prefix_token_count < 0 || inputs.wrapped_suffix_token_count < 0) {
        return Error::invalid_argument("build_tts_prompt_assembly: negative wrapped token count");
    }
    if(!has_rows(inputs.text_projected_states, inputs.text_sequence_length, inputs.hidden_size)) {
        return Error::shape_mismatch("build_tts_prompt_assembly: text states are smaller than text_sequence_length");
    }
    if(!has_rows(inputs.special_projected_states, 3, inputs.hidden_size)) {
        return Error::shape_mismatch("build_tts_prompt_assembly: special states must contain at least three rows");
    }
    if(!has_rows(inputs.codec_prefill_embeddings, inputs.codec_prefill_length, inputs.hidden_size)) {
        return Error::shape_mismatch("build_tts_prompt_assembly: codec prefill is smaller than codec_prefill_length");
    }
    if(inputs.leading_prompt_length > 0 &&
       !has_rows(inputs.leading_prompt_states, inputs.leading_prompt_length, inputs.hidden_size)) {
        return Error::shape_mismatch("build_tts_prompt_assembly: leading prompt is smaller than leading_prompt_length");
    }
    if(inputs.icl_length > 0 && !has_rows(inputs.icl_overlay, inputs.icl_length, inputs.hidden_size)) {
        return Error::shape_mismatch("build_tts_prompt_assembly: ICL overlay is smaller than icl_length");
    }
    if((inputs.codec_prompt_insert.size() % static_cast<size_t>(inputs.hidden_size)) != 0) {
        return Error::shape_mismatch("build_tts_prompt_assembly: codec prompt insert is not aligned to hidden_size");
    }

    const int64_t first_text_index = inputs.wrapped_prefix_token_count;
    const int64_t trailing_text_start = first_text_index + 1;
    const int64_t trailing_text_end_exclusive =
        inputs.text_sequence_length - inputs.wrapped_suffix_token_count;
    if(first_text_index >= inputs.text_sequence_length ||
       trailing_text_end_exclusive < trailing_text_start) {
        return Error::invalid_argument("build_tts_prompt_assembly: wrapped token counts exceed the text sequence");
    }

    const int64_t codec_tag_length = inputs.codec_prefill_length - 2;
    const int64_t insert_rows = static_cast<int64_t>(
        inputs.codec_prompt_insert.size() / static_cast<size_t>(inputs.hidden_size)
    );
    const int64_t prefill_core_length = codec_tag_length + insert_rows + 1;
    const int64_t trailing_text_count = trailing_text_end_exclusive - trailing_text_start;

    TtsPromptAssemblyResult result;
    result.hidden_size = inputs.hidden_size;
    result.produced_frames = 0;
    result.icl_length = inputs.icl_length;
    result.language_sequence_length =
        inputs.leading_prompt_length +
        inputs.wrapped_prefix_token_count +
        inputs.icl_length +
        prefill_core_length +
        1;
    result.trailing_text_length = trailing_text_count + 1;
    result.language_sequence.resize(
        static_cast<size_t>(result.language_sequence_length) * static_cast<size_t>(inputs.hidden_size)
    );
    result.trailing_text_hidden.resize(
        static_cast<size_t>(result.trailing_text_length) * static_cast<size_t>(inputs.hidden_size)
    );
    result.tts_pad_embedding.resize(static_cast<size_t>(inputs.hidden_size));

    const float *tts_bos = row_ptr(inputs.special_projected_states, 0, inputs.hidden_size);
    const float *tts_eos = row_ptr(inputs.special_projected_states, 1, inputs.hidden_size);
    const float *tts_pad = row_ptr(inputs.special_projected_states, 2, inputs.hidden_size);
    std::memcpy(
        result.tts_pad_embedding.data(),
        tts_pad,
        static_cast<size_t>(inputs.hidden_size) * sizeof(float)
    );

    size_t language_cursor = 0;
    copy_rows(
        inputs.leading_prompt_states,
        0,
        inputs.leading_prompt_length,
        inputs.hidden_size,
        result.language_sequence,
        language_cursor
    );
    copy_rows(
        inputs.text_projected_states,
        0,
        inputs.wrapped_prefix_token_count,
        inputs.hidden_size,
        result.language_sequence,
        language_cursor
    );
    copy_span(
        inputs.icl_overlay.first(
            static_cast<size_t>(inputs.icl_length) * static_cast<size_t>(inputs.hidden_size)
        ),
        result.language_sequence,
        language_cursor
    );

    const float *codec_pad = row_ptr(inputs.codec_prefill_embeddings, codec_tag_length, inputs.hidden_size);
    const float *codec_seed = row_ptr(
        inputs.codec_prefill_embeddings,
        inputs.codec_prefill_length - 1,
        inputs.hidden_size
    );
    for(int64_t row = 0; row < prefill_core_length; ++row) {
        const float *codec_row = codec_pad;
        if(row < codec_tag_length) {
            codec_row = row_ptr(inputs.codec_prefill_embeddings, row, inputs.hidden_size);
        } else if(row < codec_tag_length + insert_rows) {
            codec_row = row_ptr(inputs.codec_prompt_insert, row - codec_tag_length, inputs.hidden_size);
        }

        const float *talker_row = (row == (prefill_core_length - 1)) ? tts_bos : tts_pad;
        add_rows(codec_row, talker_row, inputs.hidden_size, result.language_sequence, language_cursor);
    }

    const float *first_text = row_ptr(inputs.text_projected_states, first_text_index, inputs.hidden_size);
    add_rows(first_text, codec_seed, inputs.hidden_size, result.language_sequence, language_cursor);

    size_t trailing_cursor = 0;
    copy_rows(
        inputs.text_projected_states,
        trailing_text_start,
        trailing_text_count,
        inputs.hidden_size,
        result.trailing_text_hidden,
        trailing_cursor
    );
    std::memcpy(
        result.trailing_text_hidden.data() + trailing_cursor,
        tts_eos,
        static_cast<size_t>(inputs.hidden_size) * sizeof(float)
    );

    return result;
}

} // namespace gotst
