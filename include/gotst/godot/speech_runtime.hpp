#pragma once

#include "gotst/core/speech_runtime_core.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>

namespace godot {

class GotstSpeechRuntimeConfig;

class GotstSpeechRuntime : public RefCounted {
    GDCLASS(GotstSpeechRuntime, RefCounted)

public:
    void set_config(const Ref<GotstSpeechRuntimeConfig> &config);
    Ref<GotstSpeechRuntimeConfig> get_config() const;

    void clear_config();
    Dictionary inspect_backends() const;
    PackedInt64Array packed_int64_range(int64_t length) const;
    PackedInt32Array packed_int32_range(int64_t length) const;
    PackedInt64Array packed_int64_ones(int64_t length) const;
    PackedInt64Array build_triplicate_position_ids(int64_t sequence_length) const;
    PackedFloat32Array slice_rows(
        const PackedFloat32Array &source,
        int64_t start_row,
        int64_t row_count,
        int64_t hidden_size
    ) const;
    PackedFloat32Array concat_rows(const Array &parts) const;
    PackedFloat32Array repeat_row(const PackedFloat32Array &row, int64_t repeat_count) const;
    PackedFloat32Array add_vectors(const PackedFloat32Array &a, const PackedFloat32Array &b) const;
    PackedFloat32Array sum_vectors(const Array &parts) const;
    Dictionary extract_sequence(const PackedFloat32Array &values, const PackedInt64Array &shape) const;
    PackedFloat32Array extract_last_hidden_row(
        const PackedFloat32Array &hidden_states,
        const PackedInt64Array &hidden_shape,
        int64_t hidden_size
    ) const;
    bool contains_only_finite_values(const PackedFloat32Array &values) const;
    int64_t select_last_row_argmax(
        const PackedFloat32Array &logits,
        const PackedInt64Array &logits_shape,
        int64_t start_index,
        int64_t count
    ) const;
    Dictionary select_last_row_token(
        const PackedFloat32Array &logits,
        const PackedInt64Array &logits_shape,
        int64_t start_index,
        int64_t count,
        bool do_sample,
        int64_t top_k,
        double top_p,
        double temperature,
        const Dictionary &prior_tokens,
        double repetition_penalty,
        int64_t rng_state
    ) const;
    bool should_stop_on_eos(
        const PackedFloat32Array &logits,
        const PackedInt64Array &logits_shape,
        int64_t eos_token_id,
        int64_t codec_size,
        double eos_logit_margin
    ) const;
    PackedFloat32Array convert_decoder_output_to_waveform(
        const PackedFloat32Array &output,
        const PackedInt64Array &output_shape,
        int64_t sample_rate,
        bool normalize_waveform,
        double waveform_gain
    ) const;
    Dictionary build_log_mel_features(
        const PackedFloat32Array &waveform,
        int64_t input_sample_rate,
        int64_t sample_rate,
        int64_t mel_bins,
        int64_t fft_size,
        int64_t hop_length,
        double chunk_length_seconds
    ) const;
    Dictionary build_tts_initial_language_input(
        const PackedFloat32Array &text_projected_states,
        int64_t text_sequence_length,
        const PackedFloat32Array &special_projected_states,
        const PackedFloat32Array &codec_pad_embedding,
        const PackedFloat32Array &codec_prefill_embeddings,
        int64_t codec_prefill_length,
        const PackedFloat32Array &speaker_prompt_embedding,
        const PackedFloat32Array &instruction_projected_states,
        int64_t instruction_sequence_length,
        int64_t hidden_size,
        int64_t wrapped_prefix_token_count,
        int64_t wrapped_suffix_token_count
    ) const;

protected:
    static void _bind_methods();

private:
    Ref<GotstSpeechRuntimeConfig> config_;
    gotst::SpeechRuntimeCore core_;
};

} // namespace godot
