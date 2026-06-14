#pragma once

#include "gotst/core/speech_runtime_core.hpp"
#include "gotst/core/speech_encoder_session.hpp"
#include "gotst/core/tts_code_generator.hpp"
#include "gotst/core/qwen_tts_pipeline.hpp"
#include "gotst/core/tts_waveform_decoder.hpp"
#include "gotst/core/irodori_tts_session.hpp"
#include "gotst/core/text_tokenization.hpp"
#include "gotst/core/asr_token_decoder.hpp"
#include "gotst/core/qwen3_forced_aligner.hpp"
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
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

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
    PackedFloat32Array resolve_custom_voice_speaker_embedding(
        const String &config_json_path,
        const String &speaker_name,
        const String &codec_embedding_onnx_path,
        int64_t hidden_size
    );
    Dictionary prepare_voice_clone_decoder_codes(
        const PackedInt64Array &generated_codes,
        int64_t generated_frame_count,
        const PackedInt64Array &ref_codes,
        int64_t ref_frame_count,
        int64_t codec_groups
    ) const;
    PackedFloat32Array trim_voice_clone_waveform_prefix(
        const PackedFloat32Array &waveform,
        int64_t trim_prefix_frames,
        int64_t decode_frame_count
    ) const;

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
    Dictionary start_tts_waveform_stream(const Dictionary &params, int64_t request_id, int64_t chunk_frames);
    Array poll_tts_waveform_stream();
    void cancel_tts_waveform_stream(int64_t request_id);
    bool load_qwen_tts(const Dictionary &config);
    bool is_qwen_tts_loaded() const;
    Dictionary prepare_qwen_tts_prompt(const Dictionary &params);
    Dictionary start_qwen_tts_stream(const Dictionary &params, int64_t request_id, int64_t chunk_frames);
    Array poll_qwen_tts_stream();
    void cancel_qwen_tts_stream(int64_t request_id);
    Array get_qwen_custom_voice_speaker_names() const;
    bool load_tts_waveform_decoder(const Dictionary &config);
    bool is_tts_waveform_decoder_loaded() const;
    Dictionary decode_tts_codes_to_waveform(const PackedInt64Array &audio_codes, int64_t frame_count) const;

    bool load_irodori_tts(const Dictionary &config);
    bool is_irodori_tts_loaded() const;
    bool load_irodori_tokenizer(const String &tokenizer_json_path, const String &tokenizer_config_path);
    String normalize_irodori_text(const String &text) const;
    Dictionary tokenize_irodori_text(const String &text, int64_t max_tokens, bool force_empty_mask) const;
    Dictionary start_irodori_tts_stream(const Dictionary &params, int64_t request_id);
    Array poll_irodori_tts_stream();
    void cancel_irodori_tts_stream(int64_t request_id);

    bool load_asr_token_decoder(const Dictionary &config);
    bool is_asr_token_decoder_loaded() const;
    Dictionary decode_asr_tokens(const Dictionary &params);
    bool load_qwen3_forced_aligner(const Dictionary &config);
    bool is_qwen3_forced_aligner_loaded() const;
    Dictionary start_qwen3_forced_alignment(const Dictionary &params, int64_t request_id);
    Array poll_qwen3_forced_alignment();
    void cancel_qwen3_forced_alignment(int64_t request_id);
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
    std::unique_ptr<gotst::QwenTtsPipeline> qwen_tts_pipeline_;
    std::unique_ptr<gotst::TtsWaveformDecoder> tts_waveform_decoder_;
    std::unique_ptr<gotst::IrodoriTtsSession> irodori_tts_session_;
    std::unique_ptr<gotst::IrodoriTextTokenizer> irodori_text_tokenizer_;
    int32_t tts_waveform_decoder_sample_rate_ = 24000;
    std::unique_ptr<gotst::AsrTokenDecoder> asr_token_decoder_;
    std::unique_ptr<gotst::Qwen3ForcedAligner> qwen3_forced_aligner_;
    std::unique_ptr<gotst::TenVad> ten_vad_;
    std::map<std::string, std::vector<int64_t>> custom_voice_speakers_;

    struct WaveformStreamEvent {
        int64_t request_id = 0;
        PackedFloat32Array waveform;
        int32_t sample_rate = 0;
        int32_t frame_count = 0;
        int32_t code_count = 0;
        int32_t chunk_index = 0;
        int32_t chunk_samples = 0;
        bool is_final = false;
        bool is_stats = false;
        bool is_error = false;
        String error_message;
        int64_t elapsed_ms = 0;
        int64_t codegen_ms = 0;
        int64_t decoder_ms = 0;
        int64_t decoder_inference_ms = 0;
        int64_t decoder_postprocess_ms = 0;
        int64_t queue_wait_ms = 0;
        int64_t pipeline_queue_wait_ms = 0;
        int64_t native_pack_ms = 0;
        String decoder_provider_requested;
        String decoder_provider_effective;
        int64_t decoder_cpu_fallback_node_count = -1;
        bool decoder_fixed_shape = false;
        int64_t talker_prefill_ms = 0;
        int64_t talker_decode_ms = 0;
        int64_t predictor_ms = 0;
        int64_t onnx_embedding_ms = 0;
        int64_t codegen_other_ms = 0;
    };

    struct WaveformStreamRequest {
        int64_t request_id = 0;
        std::vector<float> initial_input;
        int32_t initial_length = 0;
        std::vector<float> trailing_hidden;
        int32_t trailing_length = 0;
        std::vector<float> pad_embedding;
        gotst::TtsSamplingConfig sampling;
        int32_t chunk_frames = 0;
        std::chrono::steady_clock::time_point queued_at;
        std::shared_ptr<gotst::CancellationToken> cancel;
        bool use_qwen_pipeline = false;
        gotst::QwenTtsPipelineRequest qwen_request;
    };

    struct WaveformDecodeJob {
        int64_t request_id = 0;
        std::shared_ptr<gotst::TtsWaveformDecoderStream> decoder_stream;
        std::shared_ptr<gotst::CancellationToken> cancel;
        std::vector<int64_t> codes;
        int32_t frame_count = 0;
        int32_t codes_per_frame = 0;
        int32_t chunk_index = 0;
        bool is_final = false;
        double codegen_ms = 0.0;
        double pipeline_queue_wait_ms = 0.0;
        std::chrono::steady_clock::time_point stream_start;
        std::chrono::steady_clock::time_point queued_at;
    };

    std::mutex waveform_stream_mutex_;
    std::queue<WaveformStreamEvent> waveform_stream_queue_;
    std::atomic<bool> waveform_stream_active_{false};
    std::mutex waveform_request_mutex_;
    std::condition_variable waveform_request_cv_;
    std::deque<WaveformStreamRequest> waveform_request_queue_;
    std::mutex waveform_decode_mutex_;
    std::condition_variable waveform_decode_cv_;
    std::deque<WaveformDecodeJob> waveform_decode_queue_;
    std::thread waveform_generation_worker_;
    std::thread waveform_decoder_worker_;
    std::atomic<bool> waveform_workers_stop_{false};
    int64_t waveform_active_request_id_ = 0;
    std::shared_ptr<gotst::CancellationToken> waveform_active_cancel_;

    struct IrodoriStreamEvent {
        int64_t request_id = 0;
        PackedFloat32Array waveform;
        int32_t sample_rate = 48000;
        int32_t frame_count = 0;
        bool is_final = false;
        bool is_error = false;
        bool is_cancelled = false;
        String error_message;
        int64_t elapsed_ms = 0;
        String mode;
        String selected_bucket;
        int32_t selected_bucket_latent_steps = 0;
        String provider_profile;
        String provider_requested;
        String provider_effective;
        Dictionary provider_requested_by_stage;
        Dictionary provider_effective_by_stage;
        bool cache_hit = false;
        Dictionary timings_ms;
        Dictionary instrumentation_ms;
        Dictionary diagnostics;
    };

    std::mutex irodori_stream_mutex_;
    std::queue<IrodoriStreamEvent> irodori_stream_queue_;
    std::thread irodori_worker_;
    std::atomic<bool> irodori_stream_active_{false};
    int64_t irodori_active_request_id_ = 0;
    std::shared_ptr<gotst::CancellationToken> irodori_active_cancel_;

    struct ForcedAlignmentEvent {
        int64_t request_id = 0;
        bool is_completed = false;
        bool is_error = false;
        bool is_cancelled = false;
        String error_message;
        Array spans;
        Dictionary timings_ms;
        int32_t token_count = 0;
        int32_t audio_token_count = 0;
    };

    std::mutex forced_alignment_mutex_;
    std::queue<ForcedAlignmentEvent> forced_alignment_queue_;
    std::thread forced_alignment_worker_;
    std::atomic<bool> forced_alignment_active_{false};
    int64_t forced_alignment_active_request_id_ = 0;
    std::shared_ptr<gotst::CancellationToken> forced_alignment_active_cancel_;

    void ensure_waveform_stream_workers_started();
    void stop_waveform_stream_workers();
    void waveform_generation_worker_main();
    void waveform_decoder_worker_main();
    void push_waveform_event(WaveformStreamEvent event);
    void clear_waveform_events();
    void stop_irodori_stream_worker();
    void push_irodori_event(IrodoriStreamEvent event);
    void clear_irodori_events();
    void stop_forced_alignment_worker();
    void push_forced_alignment_event(ForcedAlignmentEvent event);
    void clear_forced_alignment_events();
};

} // namespace godot
