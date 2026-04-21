#pragma once

#include "gotst/core/speech_runtime_core.hpp"
#include "gotst/core/speech_encoder_session.hpp"
#include "gotst/core/tts_code_generator.hpp"
#include "gotst/core/tts_waveform_decoder.hpp"
#include "gotst/core/asr_token_decoder.hpp"
#include "gotst/core/ten_vad.hpp"
#include "gotst/core/language_config.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <queue>

namespace godot {

class GotstSpeechRuntimeConfig;

class GotstSpeechRuntime : public RefCounted {
    GDCLASS(GotstSpeechRuntime, RefCounted)

public:
    ~GotstSpeechRuntime() override;

    void set_config(const Ref<GotstSpeechRuntimeConfig> &config);
    Ref<GotstSpeechRuntimeConfig> get_config() const;

    void clear_config();
    Dictionary inspect_backends() const;
    PackedInt64Array packed_int64_range(int64_t length) const;
    PackedInt32Array packed_int32_range(int64_t length) const;
    PackedInt64Array packed_int64_ones(int64_t length) const;
    PackedInt64Array build_triplicate_position_ids(int64_t sequence_length) const;

    bool load_speaker_encoder(const String &model_path);
    bool is_speaker_encoder_loaded() const;
    PackedFloat32Array extract_speaker_embedding(
        const PackedFloat32Array &mel_features,
        int64_t frames,
        int64_t mel_dim
    ) const;
    Dictionary build_speaker_mel_features(
        const PackedFloat32Array &waveform,
        int64_t input_sample_rate,
        int64_t target_sample_rate,
        int64_t mel_bins,
        int64_t fft_size,
        int64_t hop_length,
        double fmin,
        double fmax
    ) const;
    PackedFloat32Array compute_speaker_embedding_from_audio(
        const PackedFloat32Array &waveform,
        int64_t input_sample_rate
    ) const;
    bool load_speech_tokenizer_encoder(const String &model_path);
    bool is_speech_tokenizer_encoder_loaded() const;
    Dictionary encode_speech_codes(
        const PackedFloat32Array &waveform,
        int64_t input_sample_rate
    ) const;
    Dictionary load_voice_clone_fixture(const String &json_path) const;
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
    Dictionary build_voice_clone_language_input(
        const PackedFloat32Array &text_projected_states,
        int64_t text_sequence_length,
        const PackedFloat32Array &ref_text_projected_states,
        int64_t ref_text_sequence_length,
        const PackedFloat32Array &ref_codec_projected_states,
        int64_t ref_codec_frames,
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
    Dictionary build_custom_voice_language_input(
        const PackedFloat32Array &text_projected_states,
        int64_t text_sequence_length,
        const PackedFloat32Array &special_projected_states,
        const PackedFloat32Array &codec_pad_embedding,
        const PackedFloat32Array &codec_prefill_embeddings,
        int64_t codec_prefill_length,
        const PackedFloat32Array &speaker_token_embedding,
        const PackedFloat32Array &instruction_projected_states,
        int64_t instruction_sequence_length,
        int64_t hidden_size,
        int64_t wrapped_prefix_token_count,
        int64_t wrapped_suffix_token_count
    ) const;
    Dictionary build_voice_design_language_input(
        const PackedFloat32Array &text_projected_states,
        int64_t text_sequence_length,
        const PackedFloat32Array &special_projected_states,
        const PackedFloat32Array &codec_pad_embedding,
        const PackedFloat32Array &codec_prefill_embeddings,
        int64_t codec_prefill_length,
        const PackedFloat32Array &voice_description_projected_states,
        int64_t voice_description_sequence_length,
        int64_t hidden_size,
        int64_t wrapped_prefix_token_count,
        int64_t wrapped_suffix_token_count
    ) const;
    Dictionary get_custom_voice_speaker_ids(const String &speaker_name) const;
    Array get_custom_voice_speaker_names() const;
    bool load_custom_voice_config(const String &json_path);

    Dictionary get_supported_tts_languages() const;
    int64_t get_tts_language_token_id(const String &language_key) const;
    PackedInt64Array build_codec_prefix_tokens(
        int64_t language_token_id,
        int64_t think_token_id,
        int64_t nothink_token_id,
        int64_t think_bos_token_id,
        int64_t think_eos_token_id,
        int64_t pad_token_id,
        int64_t bos_token_id
    ) const;

    void emit_partial_synthesis(int64_t request_id, const PackedFloat32Array &pcm_chunk, int64_t sample_rate);
    void emit_partial_transcription(int64_t request_id, const String &partial_text);

    bool load_tts_code_generator(const Dictionary &config);
    bool is_tts_code_generator_loaded() const;
    Dictionary generate_tts_codes(const Dictionary &params);
    Dictionary generate_tts_codes_streaming(const Dictionary &params, int64_t request_id, int64_t chunk_frames);
    Array poll_tts_stream();
    void cancel_tts_stream();
    bool load_tts_waveform_decoder(const Dictionary &config);
    bool is_tts_waveform_decoder_loaded() const;
    Dictionary decode_tts_codes_to_waveform(const PackedInt64Array &audio_codes, int64_t frame_count) const;

    bool load_asr_token_decoder(const Dictionary &config);
    bool is_asr_token_decoder_loaded() const;
    Dictionary decode_asr_tokens(const Dictionary &params);
    bool load_ten_vad(const Dictionary &config);
    bool is_ten_vad_loaded() const;
    void reset_ten_vad();
    Dictionary process_ten_vad_samples(const PackedFloat32Array &samples, int64_t input_sample_rate);

protected:
    static void _bind_methods();

private:
    Ref<GotstSpeechRuntimeConfig> config_;
    gotst::SpeechRuntimeCore core_;
    gotst::LanguageConfig language_config_;
    std::unique_ptr<gotst::SpeakerEncoderSession> speaker_encoder_;
    std::unique_ptr<gotst::SpeechTokenizerEncoderSession> speech_tokenizer_encoder_;
    std::unique_ptr<gotst::TtsCodeGenerator> tts_code_generator_;
    std::unique_ptr<gotst::TtsWaveformDecoder> tts_waveform_decoder_;
    std::unique_ptr<gotst::AsrTokenDecoder> asr_token_decoder_;
    std::unique_ptr<gotst::TenVad> ten_vad_;
    std::map<std::string, std::vector<int64_t>> custom_voice_speakers_;

    struct StreamEvent {
        int64_t request_id = 0;
        PackedInt64Array codes;
        int32_t frame_count = 0;
        int32_t codes_per_frame = 0;
        bool is_final = false;
        bool is_error = false;
        String error_message;
    };

    std::mutex stream_mutex_;
    std::queue<StreamEvent> stream_queue_;
    std::atomic<bool> stream_active_{false};
    gotst::CancellationToken stream_cancel_;
};

} // namespace godot
