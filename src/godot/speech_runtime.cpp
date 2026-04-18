#include "gotst/godot/speech_runtime.hpp"

#include "gotst/godot/speech_runtime_config.hpp"
#include "gotst/core/asr_frontend.hpp"
#include "core/sampling_utils.hpp"
#include "gotst/core/speaker_mel.hpp"
#include "gotst/core/tts_prompt_assembly.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace godot {

namespace {

struct DecoderLayout {
    bool has_layout = false;
    int64_t time_steps = 0;
    int64_t time_stride = 0;
    int64_t channel_stride = 0;
    int64_t channel_count = 1;
};

PackedFloat32Array resample_linear(
    const PackedFloat32Array &source,
    int64_t from_rate,
    int64_t to_rate
) {
    if(source.is_empty() || from_rate <= 0 || to_rate <= 0 || from_rate == to_rate) {
        return source;
    }

    const int64_t target_size = std::max<int64_t>(
        1,
        static_cast<int64_t>(std::llround(
            static_cast<double>(source.size()) * static_cast<double>(to_rate) / static_cast<double>(from_rate)
        ))
    );
    PackedFloat32Array result;
    result.resize(target_size);
    const double ratio = static_cast<double>(from_rate) / static_cast<double>(to_rate);
    for(int64_t index = 0; index < target_size; ++index) {
        const double position = static_cast<double>(index) * ratio;
        const int64_t left = static_cast<int64_t>(std::floor(position));
        const int64_t right = std::min<int64_t>(left + 1, source.size() - 1);
        const double fraction = position - static_cast<double>(left);
        const double left_value = source[left];
        const double right_value = source[right];
        result.set(index, static_cast<float>(left_value + ((right_value - left_value) * fraction)));
    }
    return result;
}

DecoderLayout resolve_decoder_output_layout(const PackedInt64Array &shape, int64_t output_length) {
    if(shape.size() < 2 || output_length <= 0) {
        return {
            .has_layout = false,
            .time_steps = output_length,
            .time_stride = 0,
            .channel_stride = 0,
            .channel_count = 1,
        };
    }

    std::vector<int64_t> dims;
    dims.reserve(shape.size());
    for(int64_t index = 0; index < shape.size(); ++index) {
        dims.push_back(shape[index]);
    }

    std::vector<int64_t> strides(dims.size(), 1);
    int64_t running_stride = 1;
    for(int64_t index = static_cast<int64_t>(dims.size()) - 1; index >= 0; --index) {
        strides[index] = running_stride;
        running_stride *= std::max<int64_t>(dims[index], 1);
    }

    const int64_t time_axis = std::max<int64_t>(0, static_cast<int64_t>(dims.size()) - 1);
    const int64_t time_steps = std::max<int64_t>(1, dims[time_axis]);
    int64_t channel_axis = -1;
    int64_t channel_count = 1;
    for(int64_t axis = 0; axis < time_axis; ++axis) {
        if(dims[axis] > 1) {
            channel_axis = axis;
            channel_count = dims[axis];
            break;
        }
    }

    return {
        .has_layout = true,
        .time_steps = time_steps,
        .time_stride = strides[time_axis],
        .channel_stride = channel_axis >= 0 ? strides[channel_axis] : 0,
        .channel_count = channel_count,
    };
}

PackedFloat32Array pack_float_array(const std::vector<float> &values) {
    PackedFloat32Array packed;
    if(values.empty()) {
        return packed;
    }

    packed.resize(static_cast<int64_t>(values.size()));
    std::memcpy(packed.ptrw(), values.data(), values.size() * sizeof(float));
    return packed;
}

Dictionary pack_tts_prompt_result(const gotst::TtsPromptAssemblyResult &result) {
    Dictionary payload;
    payload["language_sequence"] = pack_float_array(result.language_sequence);
    payload["language_sequence_length"] = result.language_sequence_length;
    payload["trailing_text_hidden"] = pack_float_array(result.trailing_text_hidden);
    payload["trailing_text_length"] = result.trailing_text_length;
    payload["tts_pad_embedding"] = pack_float_array(result.tts_pad_embedding);
    payload["hidden_size"] = result.hidden_size;
    payload["produced_frames"] = result.produced_frames;
    if(result.icl_length > 0) {
        payload["icl_length"] = result.icl_length;
    }
    return payload;
}

} // namespace

void GotstSpeechRuntime::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_config", "config"), &GotstSpeechRuntime::set_config);
    ClassDB::bind_method(D_METHOD("get_config"), &GotstSpeechRuntime::get_config);
    ClassDB::bind_method(D_METHOD("clear_config"), &GotstSpeechRuntime::clear_config);
    ClassDB::bind_method(D_METHOD("inspect_backends"), &GotstSpeechRuntime::inspect_backends);
    ClassDB::bind_method(D_METHOD("packed_int64_range", "length"), &GotstSpeechRuntime::packed_int64_range);
    ClassDB::bind_method(D_METHOD("packed_int32_range", "length"), &GotstSpeechRuntime::packed_int32_range);
    ClassDB::bind_method(D_METHOD("packed_int64_ones", "length"), &GotstSpeechRuntime::packed_int64_ones);
    ClassDB::bind_method(
        D_METHOD("build_triplicate_position_ids", "sequence_length"),
        &GotstSpeechRuntime::build_triplicate_position_ids
    );
    ClassDB::bind_method(
        D_METHOD("slice_rows", "source", "start_row", "row_count", "hidden_size"),
        &GotstSpeechRuntime::slice_rows
    );
    ClassDB::bind_method(D_METHOD("concat_rows", "parts"), &GotstSpeechRuntime::concat_rows);
    ClassDB::bind_method(D_METHOD("repeat_row", "row", "repeat_count"), &GotstSpeechRuntime::repeat_row);
    ClassDB::bind_method(D_METHOD("add_vectors", "a", "b"), &GotstSpeechRuntime::add_vectors);
    ClassDB::bind_method(D_METHOD("sum_vectors", "parts"), &GotstSpeechRuntime::sum_vectors);
    ClassDB::bind_method(D_METHOD("extract_sequence", "values", "shape"), &GotstSpeechRuntime::extract_sequence);
    ClassDB::bind_method(
        D_METHOD("extract_last_hidden_row", "hidden_states", "hidden_shape", "hidden_size"),
        &GotstSpeechRuntime::extract_last_hidden_row
    );
    ClassDB::bind_method(
        D_METHOD("contains_only_finite_values", "values"),
        &GotstSpeechRuntime::contains_only_finite_values
    );
    ClassDB::bind_method(
        D_METHOD("select_last_row_argmax", "logits", "logits_shape", "start_index", "count"),
        &GotstSpeechRuntime::select_last_row_argmax
    );
    ClassDB::bind_method(
        D_METHOD(
            "select_last_row_token",
            "logits",
            "logits_shape",
            "start_index",
            "count",
            "do_sample",
            "top_k",
            "top_p",
            "temperature",
            "prior_tokens",
            "repetition_penalty",
            "rng_state"
        ),
        &GotstSpeechRuntime::select_last_row_token
    );
    ClassDB::bind_method(
        D_METHOD("should_stop_on_eos", "logits", "logits_shape", "eos_token_id", "codec_size", "eos_logit_margin"),
        &GotstSpeechRuntime::should_stop_on_eos
    );
    ClassDB::bind_method(
        D_METHOD("convert_decoder_output_to_waveform", "output", "output_shape", "sample_rate", "normalize_waveform", "waveform_gain"),
        &GotstSpeechRuntime::convert_decoder_output_to_waveform
    );
    ClassDB::bind_method(
        D_METHOD(
            "build_log_mel_features",
            "waveform",
            "input_sample_rate",
            "sample_rate",
            "mel_bins",
            "fft_size",
            "hop_length",
            "chunk_length_seconds"
        ),
        &GotstSpeechRuntime::build_log_mel_features
    );
    ClassDB::bind_method(
        D_METHOD(
            "build_tts_initial_language_input",
            "text_projected_states",
            "text_sequence_length",
            "special_projected_states",
            "codec_pad_embedding",
            "codec_prefill_embeddings",
            "codec_prefill_length",
            "speaker_prompt_embedding",
            "instruction_projected_states",
            "instruction_sequence_length",
            "hidden_size",
            "wrapped_prefix_token_count",
            "wrapped_suffix_token_count"
        ),
        &GotstSpeechRuntime::build_tts_initial_language_input
    );
    ClassDB::bind_method(
        D_METHOD(
            "build_voice_clone_language_input",
            "text_projected_states",
            "text_sequence_length",
            "ref_text_projected_states",
            "ref_text_sequence_length",
            "ref_codec_projected_states",
            "ref_codec_frames",
            "special_projected_states",
            "codec_pad_embedding",
            "codec_prefill_embeddings",
            "codec_prefill_length",
            "speaker_prompt_embedding",
            "instruction_projected_states",
            "instruction_sequence_length",
            "hidden_size",
            "wrapped_prefix_token_count",
            "wrapped_suffix_token_count"
        ),
        &GotstSpeechRuntime::build_voice_clone_language_input
    );
    ClassDB::bind_method(
        D_METHOD(
            "build_custom_voice_language_input",
            "text_projected_states",
            "text_sequence_length",
            "special_projected_states",
            "codec_pad_embedding",
            "codec_prefill_embeddings",
            "codec_prefill_length",
            "speaker_token_embedding",
            "instruction_projected_states",
            "instruction_sequence_length",
            "hidden_size",
            "wrapped_prefix_token_count",
            "wrapped_suffix_token_count"
        ),
        &GotstSpeechRuntime::build_custom_voice_language_input
    );
    ClassDB::bind_method(
        D_METHOD(
            "build_voice_design_language_input",
            "text_projected_states",
            "text_sequence_length",
            "special_projected_states",
            "codec_pad_embedding",
            "codec_prefill_embeddings",
            "codec_prefill_length",
            "voice_description_projected_states",
            "voice_description_sequence_length",
            "hidden_size",
            "wrapped_prefix_token_count",
            "wrapped_suffix_token_count"
        ),
        &GotstSpeechRuntime::build_voice_design_language_input
    );
    ClassDB::bind_method(
        D_METHOD("get_custom_voice_speaker_ids", "speaker_name"),
        &GotstSpeechRuntime::get_custom_voice_speaker_ids
    );
    ClassDB::bind_method(
        D_METHOD("get_custom_voice_speaker_names"),
        &GotstSpeechRuntime::get_custom_voice_speaker_names
    );
    ClassDB::bind_method(
        D_METHOD("load_custom_voice_config", "json_path"),
        &GotstSpeechRuntime::load_custom_voice_config
    );
    ClassDB::bind_method(
        D_METHOD("emit_partial_synthesis", "request_id", "pcm_chunk", "sample_rate"),
        &GotstSpeechRuntime::emit_partial_synthesis
    );
    ClassDB::bind_method(
        D_METHOD("emit_partial_transcription", "request_id", "partial_text"),
        &GotstSpeechRuntime::emit_partial_transcription
    );
    ClassDB::bind_method(
        D_METHOD("load_tts_code_generator", "config"),
        &GotstSpeechRuntime::load_tts_code_generator
    );
    ClassDB::bind_method(
        D_METHOD("is_tts_code_generator_loaded"),
        &GotstSpeechRuntime::is_tts_code_generator_loaded
    );
    ClassDB::bind_method(
        D_METHOD("generate_tts_codes", "params"),
        &GotstSpeechRuntime::generate_tts_codes
    );
    ClassDB::bind_method(
        D_METHOD("generate_tts_codes_streaming", "params", "request_id", "chunk_frames"),
        &GotstSpeechRuntime::generate_tts_codes_streaming
    );
    ClassDB::bind_method(
        D_METHOD("poll_tts_stream"),
        &GotstSpeechRuntime::poll_tts_stream
    );
    ClassDB::bind_method(
        D_METHOD("cancel_tts_stream"),
        &GotstSpeechRuntime::cancel_tts_stream
    );
    ClassDB::bind_method(
        D_METHOD("load_tts_waveform_decoder", "config"),
        &GotstSpeechRuntime::load_tts_waveform_decoder
    );
    ClassDB::bind_method(
        D_METHOD("is_tts_waveform_decoder_loaded"),
        &GotstSpeechRuntime::is_tts_waveform_decoder_loaded
    );
    ClassDB::bind_method(
        D_METHOD("decode_tts_codes_to_waveform", "audio_codes", "frame_count"),
        &GotstSpeechRuntime::decode_tts_codes_to_waveform
    );
    ClassDB::bind_method(
        D_METHOD("load_asr_token_decoder", "config"),
        &GotstSpeechRuntime::load_asr_token_decoder
    );
    ClassDB::bind_method(
        D_METHOD("is_asr_token_decoder_loaded"),
        &GotstSpeechRuntime::is_asr_token_decoder_loaded
    );
    ClassDB::bind_method(
        D_METHOD("decode_asr_tokens", "params"),
        &GotstSpeechRuntime::decode_asr_tokens
    );
    ClassDB::bind_method(
        D_METHOD("load_ten_vad", "config"),
        &GotstSpeechRuntime::load_ten_vad
    );
    ClassDB::bind_method(
        D_METHOD("is_ten_vad_loaded"),
        &GotstSpeechRuntime::is_ten_vad_loaded
    );
    ClassDB::bind_method(
        D_METHOD("reset_ten_vad"),
        &GotstSpeechRuntime::reset_ten_vad
    );
    ClassDB::bind_method(
        D_METHOD("process_ten_vad_samples", "samples", "input_sample_rate"),
        &GotstSpeechRuntime::process_ten_vad_samples
    );
    ClassDB::bind_method(
        D_METHOD("load_speaker_encoder", "model_path"),
        &GotstSpeechRuntime::load_speaker_encoder
    );
    ClassDB::bind_method(
        D_METHOD("is_speaker_encoder_loaded"),
        &GotstSpeechRuntime::is_speaker_encoder_loaded
    );
    ClassDB::bind_method(
        D_METHOD("extract_speaker_embedding", "mel_features", "frames", "mel_dim"),
        &GotstSpeechRuntime::extract_speaker_embedding
    );
    ClassDB::bind_method(
        D_METHOD(
            "build_speaker_mel_features",
            "waveform",
            "input_sample_rate",
            "target_sample_rate",
            "mel_bins",
            "fft_size",
            "hop_length",
            "fmin",
            "fmax"
        ),
        &GotstSpeechRuntime::build_speaker_mel_features
    );
    ClassDB::bind_method(
        D_METHOD("compute_speaker_embedding_from_audio", "waveform", "input_sample_rate"),
        &GotstSpeechRuntime::compute_speaker_embedding_from_audio
    );
    ClassDB::bind_method(
        D_METHOD("load_speech_tokenizer_encoder", "model_path"),
        &GotstSpeechRuntime::load_speech_tokenizer_encoder
    );
    ClassDB::bind_method(
        D_METHOD("is_speech_tokenizer_encoder_loaded"),
        &GotstSpeechRuntime::is_speech_tokenizer_encoder_loaded
    );
    ClassDB::bind_method(
        D_METHOD("encode_speech_codes", "waveform", "input_sample_rate"),
        &GotstSpeechRuntime::encode_speech_codes
    );
    ClassDB::bind_method(
        D_METHOD("load_voice_clone_fixture", "json_path"),
        &GotstSpeechRuntime::load_voice_clone_fixture
    );
    ClassDB::bind_method(
        D_METHOD("get_supported_tts_languages"),
        &GotstSpeechRuntime::get_supported_tts_languages
    );
    ClassDB::bind_method(
        D_METHOD("get_tts_language_token_id", "language_key"),
        &GotstSpeechRuntime::get_tts_language_token_id
    );
    ClassDB::bind_method(
        D_METHOD(
            "build_codec_prefix_tokens",
            "language_token_id",
            "think_token_id",
            "nothink_token_id",
            "think_bos_token_id",
            "think_eos_token_id",
            "pad_token_id",
            "bos_token_id"
        ),
        &GotstSpeechRuntime::build_codec_prefix_tokens
    );

    ADD_PROPERTY(
        PropertyInfo(Variant::OBJECT, "config", PROPERTY_HINT_RESOURCE_TYPE, "GotstSpeechRuntimeConfig"),
        "set_config",
        "get_config"
    );

    ADD_SIGNAL(MethodInfo("partial_synthesis_available",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "pcm_chunk"),
        PropertyInfo(Variant::INT, "sample_rate")
    ));
    ADD_SIGNAL(MethodInfo("partial_transcription_available",
        PropertyInfo(Variant::INT, "request_id"),
        PropertyInfo(Variant::STRING, "partial_text")
    ));
}

void GotstSpeechRuntime::set_config(const Ref<GotstSpeechRuntimeConfig> &config) {
    config_ = config;
}

Ref<GotstSpeechRuntimeConfig> GotstSpeechRuntime::get_config() const {
    return config_;
}

void GotstSpeechRuntime::clear_config() {
    config_.unref();
}

Dictionary GotstSpeechRuntime::inspect_backends() const {
    gotst::RuntimeConfig native_config;
    if(config_.is_valid()) {
        native_config = config_->build_native_config();
    }

    const gotst::BackendSummary summary = core_.inspect(native_config);

    PackedStringArray missing_paths;
    for(const std::string &item : summary.missing_paths) {
        missing_paths.append(String::utf8(item.c_str()));
    }

    PackedStringArray notes;
    for(const std::string &item : summary.notes) {
        notes.append(String::utf8(item.c_str()));
    }

    Dictionary payload;
    payload["uses_llama_core"] = summary.uses_llama_core;
    payload["uses_onnx_core"] = summary.uses_onnx_core;
    payload["asr_hybrid_requested"] = summary.asr_hybrid_requested;
    payload["tts_hybrid_requested"] = summary.tts_hybrid_requested;
    payload["asr_hybrid_ready"] = summary.asr_hybrid_ready;
    payload["tts_hybrid_ready"] = summary.tts_hybrid_ready;
    payload["speaker_encoder_ready"] = summary.speaker_encoder_ready;
    payload["speech_tokenizer_encoder_ready"] = summary.speech_tokenizer_encoder_ready;
    payload["onnx_session_loaded"] = summary.onnx_session_loaded;
    payload["llama_worker_running"] = summary.llama_worker_running;
    payload["missing_paths"] = missing_paths;
    payload["notes"] = notes;
    return payload;
}

PackedInt64Array GotstSpeechRuntime::packed_int64_range(int64_t length) const {
    PackedInt64Array packed;
    const int64_t resolved_length = std::max<int64_t>(length, 0);
    packed.resize(resolved_length);
    for(int64_t index = 0; index < resolved_length; ++index) {
        packed.set(index, index);
    }
    return packed;
}

PackedInt32Array GotstSpeechRuntime::packed_int32_range(int64_t length) const {
    PackedInt32Array packed;
    const int64_t resolved_length = std::max<int64_t>(length, 0);
    packed.resize(resolved_length);
    for(int64_t index = 0; index < resolved_length; ++index) {
        packed.set(index, static_cast<int32_t>(index));
    }
    return packed;
}

PackedInt64Array GotstSpeechRuntime::packed_int64_ones(int64_t length) const {
    PackedInt64Array packed;
    const int64_t resolved_length = std::max<int64_t>(length, 0);
    packed.resize(resolved_length);
    for(int64_t index = 0; index < resolved_length; ++index) {
        packed.set(index, 1);
    }
    return packed;
}

PackedInt64Array GotstSpeechRuntime::build_triplicate_position_ids(int64_t sequence_length) const {
    const int64_t resolved_length = std::max<int64_t>(sequence_length, 1);
    PackedInt64Array packed;
    packed.resize(resolved_length * 3);
    for(int64_t index = 0; index < resolved_length; ++index) {
        packed.set(index, index);
        packed.set(resolved_length + index, index);
        packed.set((resolved_length * 2) + index, index);
    }
    return packed;
}

PackedFloat32Array GotstSpeechRuntime::slice_rows(
    const PackedFloat32Array &source,
    int64_t start_row,
    int64_t row_count,
    int64_t hidden_size
) const {
    if(source.is_empty() || row_count <= 0 || hidden_size <= 0) {
        return PackedFloat32Array();
    }

    const int64_t safe_start_row = std::max<int64_t>(start_row, 0);
    const int64_t available_rows = source.size() / hidden_size;
    if(safe_start_row >= available_rows) {
        return PackedFloat32Array();
    }

    const int64_t safe_row_count = std::min<int64_t>(row_count, available_rows - safe_start_row);
    if(safe_row_count <= 0) {
        return PackedFloat32Array();
    }

    PackedFloat32Array slice;
    slice.resize(safe_row_count * hidden_size);
    const int64_t read_offset = safe_start_row * hidden_size;
    for(int64_t index = 0; index < slice.size(); ++index) {
        slice.set(index, source[read_offset + index]);
    }
    return slice;
}

PackedFloat32Array GotstSpeechRuntime::concat_rows(const Array &parts) const {
    std::vector<PackedFloat32Array> cached;
    cached.reserve(static_cast<size_t>(parts.size()));
    int64_t total_size = 0;
    for(int64_t index = 0; index < parts.size(); ++index) {
        cached.push_back(parts[index]);
        total_size += cached.back().size();
    }

    PackedFloat32Array combined;
    combined.resize(total_size);
    int64_t cursor = 0;
    for(const PackedFloat32Array &part : cached) {
        if(!part.is_empty()) {
            memcpy(combined.ptrw() + cursor, part.ptr(), static_cast<size_t>(part.size()) * sizeof(float));
            cursor += part.size();
        }
    }
    return combined;
}

PackedFloat32Array GotstSpeechRuntime::repeat_row(const PackedFloat32Array &row, int64_t repeat_count) const {
    if(row.is_empty() || repeat_count <= 0) {
        return PackedFloat32Array();
    }

    PackedFloat32Array repeated;
    repeated.resize(row.size() * repeat_count);
    int64_t cursor = 0;
    for(int64_t repeat_index = 0; repeat_index < repeat_count; ++repeat_index) {
        for(int64_t row_index = 0; row_index < row.size(); ++row_index) {
            repeated.set(cursor++, row[row_index]);
        }
    }
    return repeated;
}

PackedFloat32Array GotstSpeechRuntime::add_vectors(const PackedFloat32Array &a, const PackedFloat32Array &b) const {
    if(a.is_empty() || a.size() != b.size()) {
        return PackedFloat32Array();
    }

    PackedFloat32Array result;
    result.resize(a.size());
    for(int64_t index = 0; index < a.size(); ++index) {
        result.set(index, a[index] + b[index]);
    }
    return result;
}

PackedFloat32Array GotstSpeechRuntime::sum_vectors(const Array &parts) const {
    PackedFloat32Array seed;
    for(int64_t index = 0; index < parts.size(); ++index) {
        const PackedFloat32Array part = parts[index];
        if(!part.is_empty()) {
            seed = part;
            break;
        }
    }
    if(seed.is_empty()) {
        return PackedFloat32Array();
    }

    PackedFloat32Array result;
    result.resize(seed.size());
    for(int64_t index = 0; index < seed.size(); ++index) {
        result.set(index, 0.0f);
    }

    for(int64_t part_index = 0; part_index < parts.size(); ++part_index) {
        const PackedFloat32Array part = parts[part_index];
        if(part.is_empty()) {
            continue;
        }
        if(part.size() != result.size()) {
            return PackedFloat32Array();
        }
        for(int64_t value_index = 0; value_index < result.size(); ++value_index) {
            result.set(value_index, result[value_index] + part[value_index]);
        }
    }
    return result;
}

Dictionary GotstSpeechRuntime::extract_sequence(const PackedFloat32Array &values, const PackedInt64Array &shape) const {
    if(values.is_empty()) {
        return {};
    }

    const int64_t shape_size = shape.size();
    const int64_t hidden_size = shape_size >= 1 ? shape[shape_size - 1] : values.size();
    const int64_t sequence_length = shape_size >= 2 ? shape[shape_size - 2] : 1;
    const int64_t required_length = hidden_size * sequence_length;
    if(required_length <= 0 || required_length > values.size()) {
        return {};
    }

    PackedFloat32Array sequence_values;
    sequence_values.resize(required_length);
    for(int64_t index = 0; index < required_length; ++index) {
        sequence_values.set(index, values[index]);
    }
    Dictionary payload;
    payload["values"] = sequence_values;
    payload["sequence_length"] = sequence_length;
    payload["hidden_size"] = hidden_size;
    return payload;
}

PackedFloat32Array GotstSpeechRuntime::extract_last_hidden_row(
    const PackedFloat32Array &hidden_states,
    const PackedInt64Array &hidden_shape,
    int64_t hidden_size
) const {
    if(hidden_states.is_empty() || hidden_size <= 0) {
        return PackedFloat32Array();
    }
    int64_t resolved_hidden_size = hidden_size;
    if(hidden_shape.size() >= 1) {
        resolved_hidden_size = hidden_shape[hidden_shape.size() - 1];
    }
    if(resolved_hidden_size != hidden_size) {
        return PackedFloat32Array();
    }
    const int64_t row_count = std::max<int64_t>(1, hidden_states.size() / hidden_size);
    return slice_rows(hidden_states, row_count - 1, 1, hidden_size);
}

bool GotstSpeechRuntime::contains_only_finite_values(const PackedFloat32Array &values) const {
    for(int64_t index = 0; index < values.size(); ++index) {
        if(!std::isfinite(static_cast<double>(values[index]))) {
            return false;
        }
    }
    return true;
}

int64_t GotstSpeechRuntime::select_last_row_argmax(
    const PackedFloat32Array &logits,
    const PackedInt64Array &logits_shape,
    int64_t start_index,
    int64_t count
) const {
    if(logits.is_empty() || logits_shape.is_empty()) {
        return -1;
    }

    const int64_t vocab_size = logits_shape[logits_shape.size() - 1];
    if(vocab_size <= 0) {
        return -1;
    }

    const int64_t rows = std::max<int64_t>(1, logits.size() / vocab_size);
    const int64_t row_start = (rows - 1) * vocab_size;
    const int64_t safe_start = std::clamp<int64_t>(start_index, 0, std::max<int64_t>(0, vocab_size - 1));
    const int64_t safe_count = std::clamp<int64_t>(count, 1, std::max<int64_t>(1, vocab_size - safe_start));
    const float *row_logits = logits.ptr() + row_start;
    int64_t best_index = safe_start;
    float best_value = -std::numeric_limits<float>::infinity();
    for(int64_t offset = 0; offset < safe_count; ++offset) {
        const int64_t token_index = safe_start + offset;
        const float value = row_logits[token_index];
        if(value > best_value) {
            best_value = value;
            best_index = token_index;
        }
    }
    return best_index;
}

Dictionary GotstSpeechRuntime::select_last_row_token(
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
) const {
    if(!do_sample) {
        Dictionary payload;
        payload["token"] = select_last_row_argmax(logits, logits_shape, start_index, count);
        payload["rng_state"] = rng_state;
        return payload;
    }

    if(logits.is_empty() || logits_shape.is_empty()) {
        Dictionary payload;
        payload["token"] = -1;
        payload["rng_state"] = rng_state;
        return payload;
    }

    const int64_t vocab_size = logits_shape[logits_shape.size() - 1];
    if(vocab_size <= 0) {
        Dictionary payload;
        payload["token"] = -1;
        payload["rng_state"] = rng_state;
        return payload;
    }

    const int64_t rows = std::max<int64_t>(1, logits.size() / std::max<int64_t>(vocab_size, 1));
    const int64_t row_start = (rows - 1) * vocab_size;
    const int64_t safe_start = std::clamp<int64_t>(start_index, 0, std::max<int64_t>(0, vocab_size - 1));
    const int64_t safe_count = std::clamp<int64_t>(count, 1, std::max<int64_t>(1, vocab_size - safe_start));
    const float *row_logits = logits.ptr() + row_start;
    gotst::detail::SamplingScratch sampling_scratch;
    const gotst::detail::SampledToken sampled = gotst::detail::sample_token(
        row_logits,
        static_cast<int32_t>(safe_start),
        static_cast<int32_t>(safe_count),
        true,
        static_cast<int32_t>(top_k),
        static_cast<float>(top_p),
        static_cast<float>(temperature),
        [&prior_tokens](int64_t token_index) {
            return prior_tokens.has(token_index);
        },
        static_cast<float>(repetition_penalty),
        rng_state,
        sampling_scratch
    );

    Dictionary payload;
    payload["token"] = sampled.token;
    payload["rng_state"] = sampled.rng_state;
    return payload;
}

bool GotstSpeechRuntime::should_stop_on_eos(
    const PackedFloat32Array &logits,
    const PackedInt64Array &logits_shape,
    int64_t eos_token_id,
    int64_t codec_size,
    double eos_logit_margin
) const {
    if(logits.is_empty() || logits_shape.is_empty()) {
        return false;
    }
    const int64_t vocab_size = logits_shape[logits_shape.size() - 1];
    if(eos_token_id < 0 || eos_token_id >= vocab_size) {
        return false;
    }
    const int64_t rows = std::max<int64_t>(1, logits.size() / std::max<int64_t>(vocab_size, 1));
    const int64_t row_start = (rows - 1) * vocab_size;
    const float eos_logit = logits[row_start + eos_token_id];
    float best_codec_logit = -std::numeric_limits<float>::infinity();
    const int64_t safe_codec_size = std::clamp<int64_t>(codec_size, 1, vocab_size);
    for(int64_t index = 0; index < safe_codec_size; ++index) {
        best_codec_logit = std::max(best_codec_logit, logits[row_start + index]);
    }
    return static_cast<double>(eos_logit) > (static_cast<double>(best_codec_logit) + eos_logit_margin);
}

PackedFloat32Array GotstSpeechRuntime::convert_decoder_output_to_waveform(
    const PackedFloat32Array &output,
    const PackedInt64Array &output_shape,
    int64_t sample_rate,
    bool normalize_waveform,
    double waveform_gain
) const {
    if(output.is_empty()) {
        return PackedFloat32Array();
    }

    const DecoderLayout layout = resolve_decoder_output_layout(output_shape, output.size());
    const int64_t sample_count = layout.time_steps;
    if(sample_count <= 0) {
        return PackedFloat32Array();
    }

    PackedFloat32Array waveform;
    waveform.resize(sample_count);
    double sum = 0.0;
    double peak = 0.0;

    if(!layout.has_layout) {
        for(int64_t index = 0; index < sample_count; ++index) {
            const double sample = std::clamp<double>(output[index], -1.5, 1.5) * waveform_gain;
            waveform.set(index, static_cast<float>(sample));
            sum += sample;
            peak = std::max(peak, std::abs(sample));
        }
    } else {
        const bool downmix = layout.channel_count > 1 && layout.channel_stride > 0;
        for(int64_t time_index = 0; time_index < sample_count; ++time_index) {
            double raw = 0.0;
            const int64_t base_index = time_index * layout.time_stride;
            if(!downmix) {
                if(base_index >= 0 && base_index < output.size()) {
                    raw = output[base_index];
                }
            } else {
                for(int64_t channel_index = 0; channel_index < layout.channel_count; ++channel_index) {
                    const int64_t sample_index = base_index + (channel_index * layout.channel_stride);
                    if(sample_index >= 0 && sample_index < output.size()) {
                        raw += output[sample_index];
                    }
                }
                raw /= static_cast<double>(std::max<int64_t>(layout.channel_count, 1));
            }
            const double sample = std::clamp<double>(raw, -1.5, 1.5) * waveform_gain;
            waveform.set(time_index, static_cast<float>(sample));
            sum += sample;
            peak = std::max(peak, std::abs(sample));
        }
    }

    const double dc_offset = sum / static_cast<double>(std::max<int64_t>(sample_count, 1));
    for(int64_t index = 0; index < waveform.size(); ++index) {
        waveform.set(index, waveform[index] - dc_offset);
    }

    peak = 0.0;
    for(int64_t index = 0; index < waveform.size(); ++index) {
        peak = std::max(peak, std::abs(static_cast<double>(waveform[index])));
    }

    if(normalize_waveform && peak > 0.00001) {
        const double normalize_scale = std::min(1.0, 0.95 / peak);
        for(int64_t index = 0; index < waveform.size(); ++index) {
            waveform.set(index, static_cast<float>(waveform[index] * normalize_scale));
        }
    }

    for(int64_t index = 0; index < waveform.size(); ++index) {
        waveform.set(index, std::clamp<float>(waveform[index], -1.0f, 1.0f));
    }

    const int64_t edge_fade = std::clamp<int64_t>(
        static_cast<int64_t>(std::llround(0.004 * static_cast<double>(sample_rate))),
        8,
        std::max<int64_t>(8, sample_count / 10)
    );
    const int64_t safe_edge_fade = std::min<int64_t>(edge_fade, waveform.size() / 2);
    for(int64_t index = 0; index < safe_edge_fade; ++index) {
        const double scale = static_cast<double>(index) / static_cast<double>(std::max<int64_t>(safe_edge_fade, 1));
        waveform.set(index, static_cast<float>(waveform[index] * scale));
        const int64_t right_index = waveform.size() - 1 - index;
        waveform.set(right_index, static_cast<float>(waveform[right_index] * scale));
    }

    return waveform;
}

Dictionary GotstSpeechRuntime::build_log_mel_features(
    const PackedFloat32Array &waveform,
    int64_t input_sample_rate,
    int64_t sample_rate,
    int64_t mel_bins,
    int64_t fft_size,
    int64_t hop_length,
    double chunk_length_seconds
) const {
    if(waveform.is_empty() || input_sample_rate <= 0 || sample_rate <= 0 ||
       mel_bins <= 0 || fft_size <= 0 || hop_length <= 0) {
        return {};
    }
    auto mel_result = gotst::build_asr_log_mel_features(
        waveform.ptr(),
        waveform.size(),
        input_sample_rate,
        sample_rate,
        mel_bins,
        fft_size,
        hop_length,
        chunk_length_seconds
    );
    if(!mel_result.is_ok()) {
        return {};
    }

    Dictionary payload;
    payload["features"] = pack_float_array(mel_result.value().features);
    payload["frame_count"] = mel_result.value().frame_count;
    payload["valid_frame_count"] = mel_result.value().valid_frame_count;
    return payload;
}

Dictionary GotstSpeechRuntime::build_tts_initial_language_input(
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
) const {
    if(text_projected_states.is_empty() || special_projected_states.is_empty() ||
       codec_pad_embedding.is_empty() || codec_prefill_embeddings.is_empty()) {
        return {};
    }
    auto prompt_result = gotst::build_tts_prompt_assembly({
        .text_projected_states =
            std::span<const float>(text_projected_states.ptr(), static_cast<size_t>(text_projected_states.size())),
        .text_sequence_length = text_sequence_length,
        .special_projected_states =
            std::span<const float>(special_projected_states.ptr(), static_cast<size_t>(special_projected_states.size())),
        .codec_prefill_embeddings =
            std::span<const float>(codec_prefill_embeddings.ptr(), static_cast<size_t>(codec_prefill_embeddings.size())),
        .codec_prefill_length = codec_prefill_length,
        .codec_prompt_insert = speaker_prompt_embedding.is_empty()
            ? std::span<const float>()
            : std::span<const float>(speaker_prompt_embedding.ptr(), static_cast<size_t>(speaker_prompt_embedding.size())),
        .leading_prompt_states = instruction_projected_states.is_empty()
            ? std::span<const float>()
            : std::span<const float>(
                  instruction_projected_states.ptr(),
                  static_cast<size_t>(instruction_projected_states.size())
              ),
        .leading_prompt_length = instruction_sequence_length,
        .icl_overlay = std::span<const float>(),
        .icl_length = 0,
        .hidden_size = hidden_size,
        .wrapped_prefix_token_count = wrapped_prefix_token_count,
        .wrapped_suffix_token_count = wrapped_suffix_token_count,
    });
    if(!prompt_result.is_ok()) {
        return {};
    }
    return pack_tts_prompt_result(prompt_result.value());
}

Dictionary GotstSpeechRuntime::build_voice_clone_language_input(
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
) const {
    if(text_projected_states.is_empty() || special_projected_states.is_empty() ||
       codec_pad_embedding.is_empty() || codec_prefill_embeddings.is_empty()) {
        return {};
    }

    auto icl_raw = gotst::build_voice_clone_icl_overlay(
        ref_text_projected_states.is_empty() ? nullptr : ref_text_projected_states.ptr(),
        ref_text_sequence_length,
        ref_codec_projected_states.is_empty() ? nullptr : ref_codec_projected_states.ptr(),
        ref_codec_frames,
        hidden_size
    );

    if (!icl_raw.is_ok()) {
        return {};
    }
    const gotst::VoiceCloneIclResult &icl_result = icl_raw.value();

    auto prompt_result = gotst::build_tts_prompt_assembly({
        .text_projected_states =
            std::span<const float>(text_projected_states.ptr(), static_cast<size_t>(text_projected_states.size())),
        .text_sequence_length = text_sequence_length,
        .special_projected_states =
            std::span<const float>(special_projected_states.ptr(), static_cast<size_t>(special_projected_states.size())),
        .codec_prefill_embeddings =
            std::span<const float>(codec_prefill_embeddings.ptr(), static_cast<size_t>(codec_prefill_embeddings.size())),
        .codec_prefill_length = codec_prefill_length,
        .codec_prompt_insert = speaker_prompt_embedding.is_empty()
            ? std::span<const float>()
            : std::span<const float>(speaker_prompt_embedding.ptr(), static_cast<size_t>(speaker_prompt_embedding.size())),
        .leading_prompt_states = instruction_projected_states.is_empty()
            ? std::span<const float>()
            : std::span<const float>(
                  instruction_projected_states.ptr(),
                  static_cast<size_t>(instruction_projected_states.size())
              ),
        .leading_prompt_length = instruction_sequence_length,
        .icl_overlay = icl_result.icl_overlay.empty()
            ? std::span<const float>()
            : std::span<const float>(icl_result.icl_overlay.data(), icl_result.icl_overlay.size()),
        .icl_length = icl_result.icl_length,
        .hidden_size = hidden_size,
        .wrapped_prefix_token_count = wrapped_prefix_token_count,
        .wrapped_suffix_token_count = wrapped_suffix_token_count,
    });
    if(!prompt_result.is_ok()) {
        return {};
    }

    return pack_tts_prompt_result(prompt_result.value());
}

bool GotstSpeechRuntime::load_speaker_encoder(const String &model_path) {
    if (!speaker_encoder_) {
        speaker_encoder_ = std::make_unique<gotst::SpeakerEncoderSession>();
    }
    return speaker_encoder_->load(std::string(model_path.utf8().get_data())).is_ok();
}

bool GotstSpeechRuntime::is_speaker_encoder_loaded() const {
    return speaker_encoder_ && speaker_encoder_->is_loaded();
}

PackedFloat32Array GotstSpeechRuntime::extract_speaker_embedding(
    const PackedFloat32Array &mel_features,
    int64_t frames,
    int64_t mel_dim
) const {
    if (!speaker_encoder_ || mel_features.is_empty() || frames <= 0 || mel_dim <= 0) {
        return PackedFloat32Array();
    }
    if (mel_features.size() != frames * mel_dim) {
        return PackedFloat32Array();
    }
    auto emb_result = speaker_encoder_->extract_embedding(
        mel_features.ptr(), frames, mel_dim
    );
    if (!emb_result.is_ok()) {
        return PackedFloat32Array();
    }
    const std::vector<float> &embedding = emb_result.value();
    PackedFloat32Array result;
    result.resize(static_cast<int64_t>(embedding.size()));
    memcpy(result.ptrw(), embedding.data(), embedding.size() * sizeof(float));
    return result;
}

Dictionary GotstSpeechRuntime::build_speaker_mel_features(
    const PackedFloat32Array &waveform,
    int64_t input_sample_rate,
    int64_t target_sample_rate,
    int64_t mel_bins,
    int64_t fft_size,
    int64_t hop_length,
    double fmin,
    double fmax
) const {
    if (waveform.is_empty()) {
        return {};
    }

    int64_t effective_target_rate = target_sample_rate;
    int64_t effective_mel_bins = mel_bins;
    int64_t effective_fft = fft_size;
    int64_t effective_hop = hop_length;
    double effective_fmin = fmin;
    double effective_fmax = fmax;

    if (config_.is_valid()) {
        gotst::RuntimeConfig native_config = config_->build_native_config();
        const auto &mp = native_config.tts.speaker_mel_params;
        if (effective_target_rate <= 0) effective_target_rate = mp.sample_rate;
        if (effective_mel_bins <= 0) effective_mel_bins = mp.n_mels;
        if (effective_fft <= 0) effective_fft = mp.n_fft;
        if (effective_hop <= 0) effective_hop = mp.hop_length;
        if (effective_fmin < 0.0) effective_fmin = mp.fmin;
        if (effective_fmax <= 0.0) effective_fmax = mp.fmax;
    }

    if (effective_target_rate <= 0) effective_target_rate = 24000;
    if (effective_mel_bins <= 0) effective_mel_bins = 128;
    if (effective_fft <= 0) effective_fft = 1024;
    if (effective_hop <= 0) effective_hop = 256;
    if (effective_fmin < 0.0) effective_fmin = 0.0;
    if (effective_fmax <= 0.0) effective_fmax = 12000.0;

    auto mel_raw = gotst::build_speaker_mel_features(
        waveform.ptr(),
        waveform.size(),
        input_sample_rate,
        effective_target_rate,
        effective_mel_bins,
        effective_fft,
        effective_hop,
        effective_fmin,
        effective_fmax
    );

    if (!mel_raw.is_ok()) {
        return {};
    }
    const gotst::SpeakerMelResult &mel_result = mel_raw.value();

    if (mel_result.features.empty() || mel_result.frames <= 0 || mel_result.mel_dim <= 0) {
        return {};
    }

    PackedFloat32Array mel_packed;
    mel_packed.resize(static_cast<int64_t>(mel_result.features.size()));
    memcpy(mel_packed.ptrw(), mel_result.features.data(), mel_result.features.size() * sizeof(float));

    Dictionary payload;
    payload["mel_features"] = mel_packed;
    payload["frames"] = mel_result.frames;
    payload["mel_dim"] = mel_result.mel_dim;
    return payload;
}

PackedFloat32Array GotstSpeechRuntime::compute_speaker_embedding_from_audio(
    const PackedFloat32Array &waveform,
    int64_t input_sample_rate
) const {
    if (waveform.is_empty() || input_sample_rate <= 0) {
        return PackedFloat32Array();
    }
    if (!speaker_encoder_ || !speaker_encoder_->is_loaded()) {
        return PackedFloat32Array();
    }

    int64_t target_rate = 24000;
    int64_t mel_bins = 128;
    int64_t fft_size = 1024;
    int64_t hop_length = 256;
    double fmin = 0.0;
    double fmax = 12000.0;

    if (config_.is_valid()) {
        gotst::RuntimeConfig native_config = config_->build_native_config();
        const auto &mp = native_config.tts.speaker_mel_params;
        target_rate = mp.sample_rate;
        mel_bins = mp.n_mels;
        fft_size = mp.n_fft;
        hop_length = mp.hop_length;
        fmin = mp.fmin;
        fmax = mp.fmax;
    }

    auto mel_raw = gotst::build_speaker_mel_features(
        waveform.ptr(),
        waveform.size(),
        input_sample_rate,
        target_rate,
        mel_bins,
        fft_size,
        hop_length,
        fmin,
        fmax
    );

    if (!mel_raw.is_ok()) {
        return PackedFloat32Array();
    }
    const gotst::SpeakerMelResult &mel_result = mel_raw.value();

    if (mel_result.features.empty() || mel_result.frames <= 0 || mel_result.mel_dim <= 0) {
        return PackedFloat32Array();
    }

    auto emb_result = speaker_encoder_->extract_embedding(
        mel_result.features.data(), mel_result.frames, mel_result.mel_dim
    );
    if (!emb_result.is_ok()) {
        return PackedFloat32Array();
    }
    const std::vector<float> &embedding = emb_result.value();

    if (embedding.empty()) {
        return PackedFloat32Array();
    }

    PackedFloat32Array result;
    result.resize(static_cast<int64_t>(embedding.size()));
    memcpy(result.ptrw(), embedding.data(), embedding.size() * sizeof(float));
    return result;
}

bool GotstSpeechRuntime::load_speech_tokenizer_encoder(const String &model_path) {
    if (!speech_tokenizer_encoder_) {
        speech_tokenizer_encoder_ = std::make_unique<gotst::SpeechTokenizerEncoderSession>();
    }
    return speech_tokenizer_encoder_->load(std::string(model_path.utf8().get_data())).is_ok();
}

bool GotstSpeechRuntime::is_speech_tokenizer_encoder_loaded() const {
    return speech_tokenizer_encoder_ && speech_tokenizer_encoder_->is_loaded();
}

Dictionary GotstSpeechRuntime::encode_speech_codes(
    const PackedFloat32Array &waveform,
    int64_t input_sample_rate
) const {
    if (!speech_tokenizer_encoder_ || !speech_tokenizer_encoder_->is_loaded()) {
        return {};
    }
    if (waveform.is_empty() || input_sample_rate <= 0) {
        return {};
    }

    int64_t target_sample_rate = 24000;
    if (config_.is_valid()) {
        gotst::RuntimeConfig native_config = config_->build_native_config();
        if (native_config.tts.speech_tokenizer_encoder_sample_rate > 0) {
            target_sample_rate = native_config.tts.speech_tokenizer_encoder_sample_rate;
        }
    }

    PackedFloat32Array resampled = waveform;
    if (input_sample_rate != target_sample_rate) {
        resampled = resample_linear(waveform, input_sample_rate, target_sample_rate);
    }

    auto encode_raw = speech_tokenizer_encoder_->encode(
        resampled.ptr(), resampled.size()
    );

    if (!encode_raw.is_ok()) {
        return {};
    }
    const gotst::SpeechTokenizerEncodeResult &encode_result = encode_raw.value();

    if (encode_result.codes.empty() || encode_result.frames <= 0) {
        return {};
    }

    PackedInt32Array codes_packed;
    codes_packed.resize(static_cast<int64_t>(encode_result.codes.size()));
    memcpy(codes_packed.ptrw(), encode_result.codes.data(), encode_result.codes.size() * sizeof(int32_t));

    Array shape_array;
    shape_array.append(encode_result.frames);
    shape_array.append(encode_result.codebooks);

    Dictionary payload;
    payload["codes"] = codes_packed;
    payload["shape"] = shape_array;
    payload["frames"] = encode_result.frames;
    payload["codebooks"] = encode_result.codebooks;
    return payload;
}

Dictionary GotstSpeechRuntime::load_voice_clone_fixture(const String &json_path) const {
    const std::string path_str = std::string(json_path.utf8().get_data());
    if (path_str.empty()) {
        return {};
    }

    FILE *file = fopen(path_str.c_str(), "rb");
    if (!file) {
        return {};
    }

    fseek(file, 0, SEEK_END);
    const long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size <= 0 || file_size > 64 * 1024 * 1024) {
        fclose(file);
        return {};
    }

    std::vector<char> buffer(static_cast<size_t>(file_size + 1));
    const size_t bytes_read = fread(buffer.data(), 1, static_cast<size_t>(file_size), file);
    fclose(file);
    buffer[bytes_read] = '\0';

    const String json_text = String::utf8(buffer.data());

    if (json_text.is_empty()) {
        return {};
    }

    JSON json_parser;
    const Error parse_error = json_parser.parse(json_text);
    if (parse_error != OK) {
        return {};
    }

    const Variant parsed = json_parser.get_data();
    if (parsed.get_type() != Variant::DICTIONARY) {
        return {};
    }

    Dictionary fixture_data = parsed;

    Dictionary result;
    result["ref_audio"] = fixture_data.get("ref_audio", String());
    result["ref_text"] = fixture_data.get("ref_text", String());

    if (fixture_data.has("ref_codes")) {
        const Variant codes_variant = fixture_data["ref_codes"];
        if (codes_variant.get_type() == Variant::ARRAY) {
            const Array codes_array = codes_variant;
            PackedInt32Array codes;
            codes.resize(codes_array.size());
            for (int64_t i = 0; i < codes_array.size(); ++i) {
                codes.set(i, static_cast<int32_t>(codes_array[i]));
            }
            result["ref_codes"] = codes;
        }
    }

    if (fixture_data.has("ref_codes_shape")) {
        const Variant shape_variant = fixture_data["ref_codes_shape"];
        if (shape_variant.get_type() == Variant::ARRAY) {
            const Array shape_array = shape_variant;
            Array shape_result;
            for (int64_t i = 0; i < shape_array.size(); ++i) {
                shape_result.append(shape_array[i]);
            }
            result["ref_codes_shape"] = shape_result;
        }
    }

    if (fixture_data.has("ref_speaker_embedding")) {
        const Variant emb_variant = fixture_data["ref_speaker_embedding"];
        if (emb_variant.get_type() == Variant::ARRAY) {
            const Array emb_array = emb_variant;
            PackedFloat32Array embedding;
            embedding.resize(emb_array.size());
            for (int64_t i = 0; i < emb_array.size(); ++i) {
                embedding.set(i, static_cast<float>(double(emb_array[i])));
            }
            result["ref_speaker_embedding"] = embedding;
        }
    }

    return result;
}

Dictionary GotstSpeechRuntime::build_custom_voice_language_input(
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
) const {
    if(text_projected_states.is_empty() || special_projected_states.is_empty() ||
       codec_pad_embedding.is_empty() || codec_prefill_embeddings.is_empty()) {
        return {};
    }
    auto prompt_result = gotst::build_tts_prompt_assembly({
        .text_projected_states =
            std::span<const float>(text_projected_states.ptr(), static_cast<size_t>(text_projected_states.size())),
        .text_sequence_length = text_sequence_length,
        .special_projected_states =
            std::span<const float>(special_projected_states.ptr(), static_cast<size_t>(special_projected_states.size())),
        .codec_prefill_embeddings =
            std::span<const float>(codec_prefill_embeddings.ptr(), static_cast<size_t>(codec_prefill_embeddings.size())),
        .codec_prefill_length = codec_prefill_length,
        .codec_prompt_insert = speaker_token_embedding.is_empty()
            ? std::span<const float>()
            : std::span<const float>(speaker_token_embedding.ptr(), static_cast<size_t>(speaker_token_embedding.size())),
        .leading_prompt_states = instruction_projected_states.is_empty()
            ? std::span<const float>()
            : std::span<const float>(
                  instruction_projected_states.ptr(),
                  static_cast<size_t>(instruction_projected_states.size())
              ),
        .leading_prompt_length = instruction_sequence_length,
        .icl_overlay = std::span<const float>(),
        .icl_length = 0,
        .hidden_size = hidden_size,
        .wrapped_prefix_token_count = wrapped_prefix_token_count,
        .wrapped_suffix_token_count = wrapped_suffix_token_count,
    });
    if(!prompt_result.is_ok()) {
        return {};
    }

    return pack_tts_prompt_result(prompt_result.value());
}

Dictionary GotstSpeechRuntime::build_voice_design_language_input(
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
) const {
    if(text_projected_states.is_empty() || special_projected_states.is_empty() ||
       codec_pad_embedding.is_empty() || codec_prefill_embeddings.is_empty()) {
        return {};
    }
    auto prompt_result = gotst::build_tts_prompt_assembly({
        .text_projected_states =
            std::span<const float>(text_projected_states.ptr(), static_cast<size_t>(text_projected_states.size())),
        .text_sequence_length = text_sequence_length,
        .special_projected_states =
            std::span<const float>(special_projected_states.ptr(), static_cast<size_t>(special_projected_states.size())),
        .codec_prefill_embeddings =
            std::span<const float>(codec_prefill_embeddings.ptr(), static_cast<size_t>(codec_prefill_embeddings.size())),
        .codec_prefill_length = codec_prefill_length,
        .codec_prompt_insert = std::span<const float>(),
        .leading_prompt_states = voice_description_projected_states.is_empty()
            ? std::span<const float>()
            : std::span<const float>(
                  voice_description_projected_states.ptr(),
                  static_cast<size_t>(voice_description_projected_states.size())
              ),
        .leading_prompt_length = voice_description_sequence_length,
        .icl_overlay = std::span<const float>(),
        .icl_length = 0,
        .hidden_size = hidden_size,
        .wrapped_prefix_token_count = wrapped_prefix_token_count,
        .wrapped_suffix_token_count = wrapped_suffix_token_count,
    });
    if(!prompt_result.is_ok()) {
        return {};
    }

    return pack_tts_prompt_result(prompt_result.value());
}

Dictionary GotstSpeechRuntime::get_custom_voice_speaker_ids(const String &speaker_name) const {
    Dictionary result;

    const std::string name = std::string(speaker_name.utf8().get_data());
    auto it = custom_voice_speakers_.find(name);
    if (it == custom_voice_speakers_.end()) {
        return result;
    }

    PackedInt64Array token_ids;
    token_ids.resize(static_cast<int64_t>(it->second.size()));
    memcpy(token_ids.ptrw(), it->second.data(), it->second.size() * sizeof(int64_t));

    result["token_ids"] = token_ids;
    return result;
}

Array GotstSpeechRuntime::get_custom_voice_speaker_names() const {
    Array names;
    for (const auto &pair : custom_voice_speakers_) {
        names.append(String::utf8(pair.first.c_str()));
    }
    return names;
}

bool GotstSpeechRuntime::load_custom_voice_config(const String &json_path) {
    const std::string path_str = std::string(json_path.utf8().get_data());
    if (path_str.empty()) {
        return false;
    }

    FILE *file = fopen(path_str.c_str(), "rb");
    if (!file) {
        return false;
    }

    fseek(file, 0, SEEK_END);
    const long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size <= 0 || file_size > 64 * 1024 * 1024) {
        fclose(file);
        return false;
    }

    std::vector<char> buffer(static_cast<size_t>(file_size + 1));
    const size_t bytes_read = fread(buffer.data(), 1, static_cast<size_t>(file_size), file);
    fclose(file);
    buffer[bytes_read] = '\0';

    const String json_text = String::utf8(buffer.data());
    if (json_text.is_empty()) {
        return false;
    }

    JSON json_parser;
    const Error parse_error = json_parser.parse(json_text);
    if (parse_error != OK) {
        return false;
    }

    const Variant parsed = json_parser.get_data();
    if (parsed.get_type() != Variant::DICTIONARY) {
        return false;
    }

    Dictionary root = parsed;

    if (!root.has("talker_config")) {
        return false;
    }

    const Variant talker_variant = root["talker_config"];
    if (talker_variant.get_type() != Variant::DICTIONARY) {
        return false;
    }

    const Dictionary talker_config = talker_variant;
    if (!talker_config.has("spk_id")) {
        return false;
    }

    const Variant spk_variant = talker_config["spk_id"];
    if (spk_variant.get_type() != Variant::DICTIONARY) {
        return false;
    }

    const Dictionary spk_id = spk_variant;

    custom_voice_speakers_.clear();

    const Array keys = spk_id.keys();
    for (int64_t i = 0; i < keys.size(); ++i) {
        const String key = keys[i];
        const Variant value = spk_id[key];

        if (value.get_type() == Variant::ARRAY) {
            const Array ids_array = value;
            std::vector<int64_t> token_ids;
            token_ids.reserve(static_cast<size_t>(ids_array.size()));
            for (int64_t j = 0; j < ids_array.size(); ++j) {
                token_ids.push_back(static_cast<int64_t>(ids_array[j]));
            }
            custom_voice_speakers_[std::string(key.utf8().get_data())] = std::move(token_ids);
        }
    }

    return !custom_voice_speakers_.empty();
}

void GotstSpeechRuntime::emit_partial_synthesis(int64_t request_id, const PackedFloat32Array &pcm_chunk, int64_t sample_rate) {
    emit_signal("partial_synthesis_available", request_id, pcm_chunk, sample_rate);
}

void GotstSpeechRuntime::emit_partial_transcription(int64_t request_id, const String &partial_text) {
    emit_signal("partial_transcription_available", request_id, partial_text);
}

Dictionary GotstSpeechRuntime::get_supported_tts_languages() const {
    Dictionary result;
    for (const auto &pair : language_config_.get_tts_language_map()) {
        result[String::utf8(pair.first.c_str())] = String::utf8(pair.second.name.c_str());
    }
    return result;
}

int64_t GotstSpeechRuntime::get_tts_language_token_id(const String &language_key) const {
    const std::string key = std::string(language_key.utf8().get_data());
    return language_config_.get_tts_language_token_id(key);
}

PackedInt64Array GotstSpeechRuntime::build_codec_prefix_tokens(
    int64_t language_token_id,
    int64_t think_token_id,
    int64_t nothink_token_id,
    int64_t think_bos_token_id,
    int64_t think_eos_token_id,
    int64_t pad_token_id,
    int64_t bos_token_id
) const {
    std::vector<int64_t> tokens = gotst::LanguageConfig::build_codec_prefix_tokens(
        language_token_id,
        think_token_id,
        nothink_token_id,
        think_bos_token_id,
        think_eos_token_id,
        pad_token_id,
        bos_token_id
    );

    PackedInt64Array result;
    result.resize(static_cast<int64_t>(tokens.size()));
    memcpy(result.ptrw(), tokens.data(), tokens.size() * sizeof(int64_t));
    return result;
}

bool GotstSpeechRuntime::load_tts_code_generator(const Dictionary &config) {
    gotst::TtsModelPaths paths;
    paths.talker_gguf_path = String(config.get("talker_gguf_path", "")).utf8().get_data();
    paths.predictor_gguf_path = String(config.get("predictor_gguf_path", "")).utf8().get_data();
    paths.codec_embedding_onnx_path = String(config.get("codec_embedding_onnx_path", "")).utf8().get_data();
    paths.predictor_embedding_onnx_path = String(config.get("predictor_embedding_onnx_path", "")).utf8().get_data();

    gotst::TtsSessionConfig session_config;
    session_config.talker_n_ctx = static_cast<int32_t>(config.get("talker_n_ctx", 1024));
    session_config.talker_n_batch = static_cast<int32_t>(config.get("talker_n_batch", 1024));
    session_config.predictor_n_ctx = static_cast<int32_t>(config.get("predictor_n_ctx", 128));
    session_config.predictor_n_batch = static_cast<int32_t>(config.get("predictor_n_batch", 128));
    session_config.n_threads = static_cast<int32_t>(config.get("n_threads", -1));
    session_config.n_gpu_layers = static_cast<int32_t>(config.get("n_gpu_layers", 0));
    session_config.use_mmap = static_cast<bool>(config.get("use_mmap", true));
    session_config.use_mlock = static_cast<bool>(config.get("use_mlock", false));
    session_config.flash_attn_type = static_cast<int32_t>(config.get("flash_attn_type", -1));
    session_config.type_k = static_cast<int32_t>(config.get("type_k", -1));
    session_config.type_v = static_cast<int32_t>(config.get("type_v", -1));
    session_config.talker_position_components = static_cast<int32_t>(config.get("talker_position_components", 4));
    session_config.predictor_position_components = static_cast<int32_t>(config.get("predictor_position_components", 1));

    tts_code_generator_ = std::make_unique<gotst::TtsCodeGenerator>();
    auto result = tts_code_generator_->load(paths, session_config);
    if (!result.is_ok()) {
        ERR_PRINT(String("TTS code generator load failed: ") + String(result.error_message().c_str()));
        tts_code_generator_.reset();
        return false;
    }
    return true;
}

bool GotstSpeechRuntime::is_tts_code_generator_loaded() const {
    return tts_code_generator_ && tts_code_generator_->is_loaded();
}

Dictionary GotstSpeechRuntime::generate_tts_codes(const Dictionary &params) {
    Dictionary output;
    if (!tts_code_generator_ || !tts_code_generator_->is_loaded()) {
        output["error"] = "TTS code generator is not loaded.";
        return output;
    }

    const PackedFloat32Array initial_input = params.get("initial_language_input", PackedFloat32Array());
    const int32_t initial_length = static_cast<int32_t>(params.get("initial_sequence_length", 0));
    const PackedFloat32Array trailing_hidden = params.get("trailing_text_hidden", PackedFloat32Array());
    const int32_t trailing_length = static_cast<int32_t>(params.get("trailing_text_length", 0));
    const PackedFloat32Array pad_embedding = params.get("tts_pad_embedding", PackedFloat32Array());

    gotst::TtsSamplingConfig sampling;
    sampling.codebook_size = static_cast<int32_t>(params.get("codebook_size", 2048));
    sampling.residual_groups = static_cast<int32_t>(params.get("residual_groups", 15));
    sampling.target_frames = static_cast<int32_t>(params.get("target_frames", 96));
    sampling.min_frames_before_eos = static_cast<int32_t>(params.get("min_frames_before_eos", 8));
    sampling.hidden_size = static_cast<int32_t>(params.get("hidden_size", 1024));
    sampling.eos_token_id = static_cast<int32_t>(params.get("eos_token_id", 2150));
    sampling.eos_logit_margin = static_cast<float>(params.get("eos_logit_margin", 0.0));
    sampling.do_sample = static_cast<bool>(params.get("do_sample", true));
    sampling.top_k = static_cast<int32_t>(params.get("top_k", 50));
    sampling.top_p = static_cast<float>(params.get("top_p", 1.0));
    sampling.temperature = static_cast<float>(params.get("temperature", 0.9));
    sampling.sub_do_sample = static_cast<bool>(params.get("sub_do_sample", true));
    sampling.sub_top_k = static_cast<int32_t>(params.get("sub_top_k", 50));
    sampling.sub_top_p = static_cast<float>(params.get("sub_top_p", 1.0));
    sampling.sub_temperature = static_cast<float>(params.get("sub_temperature", 0.9));
    sampling.repetition_penalty = static_cast<float>(params.get("repetition_penalty", 1.05));
    sampling.rng_seed = static_cast<int64_t>(params.get("rng_seed", 1));

    auto result = tts_code_generator_->generate(
        {initial_input.ptr(), static_cast<size_t>(initial_input.size())},
        initial_length,
        {trailing_hidden.ptr(), static_cast<size_t>(trailing_hidden.size())},
        trailing_length,
        {pad_embedding.ptr(), static_cast<size_t>(pad_embedding.size())},
        sampling
    );

    if (!result.is_ok()) {
        output["error"] = String(result.error_message().c_str());
        return output;
    }

    const auto &gen = result.value();
    PackedInt64Array codes;
    codes.resize(static_cast<int64_t>(gen.codes.size()));
    if (!gen.codes.empty()) {
        memcpy(codes.ptrw(), gen.codes.data(), gen.codes.size() * sizeof(int64_t));
    }

    output["codes"] = codes;
    output["frame_count"] = gen.frame_count;
    output["codes_per_frame"] = gen.codes_per_frame;
    return output;
}

Dictionary GotstSpeechRuntime::generate_tts_codes_streaming(const Dictionary &params, int64_t request_id, int64_t chunk_frames) {
    Dictionary output;
    if (!tts_code_generator_ || !tts_code_generator_->is_loaded()) {
        output["error"] = "TTS code generator is not loaded.";
        return output;
    }

    stream_cancel_.reset();
    stream_active_.store(true, std::memory_order_release);

    const PackedFloat32Array initial_input = params.get("initial_language_input", PackedFloat32Array());
    const int32_t initial_length = static_cast<int32_t>(params.get("initial_sequence_length", 0));
    const PackedFloat32Array trailing_hidden = params.get("trailing_text_hidden", PackedFloat32Array());
    const int32_t trailing_length = static_cast<int32_t>(params.get("trailing_text_length", 0));
    const PackedFloat32Array pad_embedding = params.get("tts_pad_embedding", PackedFloat32Array());

    gotst::TtsSamplingConfig sampling;
    sampling.codebook_size = static_cast<int32_t>(params.get("codebook_size", 2048));
    sampling.residual_groups = static_cast<int32_t>(params.get("residual_groups", 15));
    sampling.target_frames = static_cast<int32_t>(params.get("target_frames", 96));
    sampling.min_frames_before_eos = static_cast<int32_t>(params.get("min_frames_before_eos", 8));
    sampling.hidden_size = static_cast<int32_t>(params.get("hidden_size", 1024));
    sampling.eos_token_id = static_cast<int32_t>(params.get("eos_token_id", 2150));
    sampling.eos_logit_margin = static_cast<float>(params.get("eos_logit_margin", 0.0));
    sampling.do_sample = static_cast<bool>(params.get("do_sample", true));
    sampling.top_k = static_cast<int32_t>(params.get("top_k", 50));
    sampling.top_p = static_cast<float>(params.get("top_p", 1.0));
    sampling.temperature = static_cast<float>(params.get("temperature", 0.9));
    sampling.sub_do_sample = static_cast<bool>(params.get("sub_do_sample", true));
    sampling.sub_top_k = static_cast<int32_t>(params.get("sub_top_k", 50));
    sampling.sub_top_p = static_cast<float>(params.get("sub_top_p", 1.0));
    sampling.sub_temperature = static_cast<float>(params.get("sub_temperature", 0.9));
    sampling.repetition_penalty = static_cast<float>(params.get("repetition_penalty", 1.05));
    sampling.rng_seed = static_cast<int64_t>(params.get("rng_seed", 1));

    auto on_chunk = [this, request_id](gotst::TtsFrameChunk chunk) {
        StreamEvent event;
        event.request_id = request_id;
        event.frame_count = chunk.frame_count;
        event.codes_per_frame = chunk.codes_per_frame;
        event.is_final = chunk.is_final;
        event.codes.resize(static_cast<int64_t>(chunk.codes.size()));
        if (!chunk.codes.empty()) {
            memcpy(event.codes.ptrw(), chunk.codes.data(), chunk.codes.size() * sizeof(int64_t));
        }
        std::lock_guard<std::mutex> lock(stream_mutex_);
        stream_queue_.push(std::move(event));
    };

    auto result = tts_code_generator_->generate_streaming(
        {initial_input.ptr(), static_cast<size_t>(initial_input.size())},
        initial_length,
        {trailing_hidden.ptr(), static_cast<size_t>(trailing_hidden.size())},
        trailing_length,
        {pad_embedding.ptr(), static_cast<size_t>(pad_embedding.size())},
        sampling,
        static_cast<int32_t>(chunk_frames),
        on_chunk,
        &stream_cancel_
    );

    stream_active_.store(false, std::memory_order_release);

    if (!result.is_ok()) {
        StreamEvent err_event;
        err_event.request_id = request_id;
        err_event.is_error = true;
        err_event.error_message = String(result.error_message().c_str());
        std::lock_guard<std::mutex> lock(stream_mutex_);
        stream_queue_.push(std::move(err_event));

        output["error"] = String(result.error_message().c_str());
        return output;
    }

    const auto &gen = result.value();
    output["frame_count"] = gen.frame_count;
    output["codes_per_frame"] = gen.codes_per_frame;
    return output;
}

Array GotstSpeechRuntime::poll_tts_stream() {
    Array events;
    std::lock_guard<std::mutex> lock(stream_mutex_);
    while (!stream_queue_.empty()) {
        StreamEvent &ev = stream_queue_.front();
        Dictionary d;
        d["request_id"] = ev.request_id;
        d["codes"] = ev.codes;
        d["frame_count"] = ev.frame_count;
        d["codes_per_frame"] = ev.codes_per_frame;
        d["is_final"] = ev.is_final;
        if (ev.is_error) {
            d["error"] = ev.error_message;
        }
        events.push_back(d);
        stream_queue_.pop();
    }
    return events;
}

void GotstSpeechRuntime::cancel_tts_stream() {
    stream_cancel_.cancel();
    stream_active_.store(false, std::memory_order_release);
}

bool GotstSpeechRuntime::load_tts_waveform_decoder(const Dictionary &config) {
    gotst::TtsWaveformDecoderConfig decoder_config;
    decoder_config.decoder_onnx_path = String(config.get("decoder_onnx_path", "")).utf8().get_data();
    decoder_config.provider = String(config.get("provider", "CPU")).utf8().get_data();
    decoder_config.intra_op_threads = static_cast<int32_t>(config.get("intra_op_threads", 0));
    decoder_config.inter_op_threads = static_cast<int32_t>(config.get("inter_op_threads", 0));
    decoder_config.optimization_level = static_cast<int32_t>(config.get("optimization_level", 99));
    decoder_config.optimized_model_path = String(config.get("optimized_model_path", "")).utf8().get_data();
    decoder_config.sample_rate = static_cast<int32_t>(config.get("sample_rate", 24000));
    decoder_config.normalize_waveform = static_cast<bool>(config.get("normalize_waveform", false));
    decoder_config.waveform_gain = static_cast<float>(config.get("waveform_gain", 1.0));

    tts_waveform_decoder_ = std::make_unique<gotst::TtsWaveformDecoder>();
    auto result = tts_waveform_decoder_->load(decoder_config);
    if(!result.is_ok()) {
        ERR_PRINT(String("TTS waveform decoder load failed: ") + String(result.error_message().c_str()));
        tts_waveform_decoder_.reset();
        return false;
    }
    return true;
}

bool GotstSpeechRuntime::is_tts_waveform_decoder_loaded() const {
    return tts_waveform_decoder_ && tts_waveform_decoder_->is_loaded();
}

Dictionary GotstSpeechRuntime::decode_tts_codes_to_waveform(
    const PackedInt64Array &audio_codes,
    int64_t frame_count
) const {
    Dictionary output;
    if(!tts_waveform_decoder_ || !tts_waveform_decoder_->is_loaded()) {
        output["error"] = "TTS waveform decoder is not loaded.";
        return output;
    }

    auto result = tts_waveform_decoder_->decode(
        std::span<const int64_t>(audio_codes.ptr(), static_cast<size_t>(audio_codes.size())),
        static_cast<int32_t>(frame_count)
    );
    if(!result.is_ok()) {
        output["error"] = String(result.error_message().c_str());
        return output;
    }

    const auto &decoded = result.value();
    output["waveform"] = pack_float_array(decoded.waveform);
    output["backend"] = String(decoded.backend.c_str());
    output["elapsed_ms"] = static_cast<int64_t>(std::llround(decoded.elapsed_ms));
    output["inference_ms"] = static_cast<int64_t>(std::llround(decoded.inference_ms));
    output["postprocess_ms"] = static_cast<int64_t>(std::llround(decoded.postprocess_ms));
    output["sample_count"] = decoded.sample_count;
    output["frame_count"] = decoded.frame_count;
    output["codes_per_frame"] = decoded.codes_per_frame;
    return output;
}

bool GotstSpeechRuntime::load_asr_token_decoder(const Dictionary &config) {
    gotst::AsrModelPaths paths;
    paths.thinker_gguf_path = String(config.get("thinker_gguf_path", "")).utf8().get_data();
    paths.embedding_onnx_path = String(config.get("embedding_onnx_path", "")).utf8().get_data();

    gotst::AsrSessionConfig session_config;
    session_config.n_ctx = static_cast<int32_t>(config.get("n_ctx", 1024));
    session_config.n_batch = static_cast<int32_t>(config.get("n_batch", 1024));
    session_config.n_threads = static_cast<int32_t>(config.get("n_threads", -1));
    session_config.n_gpu_layers = static_cast<int32_t>(config.get("n_gpu_layers", 0));
    session_config.use_mmap = static_cast<bool>(config.get("use_mmap", true));
    session_config.use_mlock = static_cast<bool>(config.get("use_mlock", false));
    session_config.flash_attn_type = static_cast<int32_t>(config.get("flash_attn_type", -1));
    session_config.type_k = static_cast<int32_t>(config.get("type_k", -1));
    session_config.type_v = static_cast<int32_t>(config.get("type_v", -1));
    session_config.position_components = static_cast<int32_t>(config.get("position_components", 3));

    asr_token_decoder_ = std::make_unique<gotst::AsrTokenDecoder>();
    auto result = asr_token_decoder_->load(paths, session_config);
    if (!result.is_ok()) {
        ERR_PRINT(String("ASR token decoder load failed: ") + String(result.error_message().c_str()));
        asr_token_decoder_.reset();
        return false;
    }
    return true;
}

bool GotstSpeechRuntime::is_asr_token_decoder_loaded() const {
    return asr_token_decoder_ && asr_token_decoder_->is_loaded();
}

Dictionary GotstSpeechRuntime::decode_asr_tokens(const Dictionary &params) {
    Dictionary output;
    if (!asr_token_decoder_ || !asr_token_decoder_->is_loaded()) {
        output["error"] = "ASR token decoder is not loaded.";
        return output;
    }

    const PackedFloat32Array prompt_embeddings = params.get("prompt_embeddings", PackedFloat32Array());
    const int32_t prompt_length = static_cast<int32_t>(params.get("prompt_length", 0));

    gotst::AsrDecodeParams decode_params;
    decode_params.max_tokens = static_cast<int32_t>(params.get("max_tokens", 64));
    decode_params.hidden_size = static_cast<int32_t>(params.get("hidden_size", 896));
    decode_params.vocab_size = static_cast<int32_t>(params.get("vocab_size", 152064));
    decode_params.eos_token_id = static_cast<int32_t>(params.get("eos_token_id", 151645));

    auto result = asr_token_decoder_->decode(
        {prompt_embeddings.ptr(), static_cast<size_t>(prompt_embeddings.size())},
        prompt_length,
        decode_params
    );

    if (!result.is_ok()) {
        output["error"] = String(result.error_message().c_str());
        return output;
    }

    const auto &decoded = result.value();
    PackedInt32Array token_ids;
    token_ids.resize(static_cast<int64_t>(decoded.token_ids.size()));
    if (!decoded.token_ids.empty()) {
        memcpy(token_ids.ptrw(), decoded.token_ids.data(), decoded.token_ids.size() * sizeof(int32_t));
    }

    output["token_ids"] = token_ids;
    return output;
}

bool GotstSpeechRuntime::load_ten_vad(const Dictionary &config) {
    gotst::TenVadConfig ten_vad_config;
    ten_vad_config.model_path = String(config.get("model_path", "")).utf8().get_data();
    ten_vad_config.model_sample_rate = static_cast<int32_t>(config.get("model_sample_rate", 16000));
    ten_vad_config.hop_size = static_cast<int32_t>(config.get("hop_size", 256));
    ten_vad_config.threshold = static_cast<float>(config.get("threshold", 0.5));
    ten_vad_config.reset_frame_count = static_cast<int32_t>(config.get("reset_frame_count", 1875));
    ten_vad_config.pitch_est_voiced_threshold = static_cast<float>(config.get("pitch_est_voiced_threshold", 0.4));

    ten_vad_ = std::make_unique<gotst::TenVad>();
    auto result = ten_vad_->load(ten_vad_config);
    if(!result.is_ok()) {
        ERR_PRINT(String("TEN-Vad load failed: ") + String(result.error_message().c_str()));
        ten_vad_.reset();
        return false;
    }
    return true;
}

bool GotstSpeechRuntime::is_ten_vad_loaded() const {
    return ten_vad_ && ten_vad_->is_loaded();
}

void GotstSpeechRuntime::reset_ten_vad() {
    if(!ten_vad_ || !ten_vad_->is_loaded()) {
        return;
    }
    auto result = ten_vad_->reset();
    if(!result.is_ok()) {
        ERR_PRINT(String("TEN-Vad reset failed: ") + String(result.error_message().c_str()));
    }
}

Dictionary GotstSpeechRuntime::process_ten_vad_samples(
    const PackedFloat32Array &samples,
    int64_t input_sample_rate
) {
    Dictionary output;
    if(!ten_vad_ || !ten_vad_->is_loaded()) {
        output["error"] = "TEN-Vad is not loaded.";
        return output;
    }

    auto result = ten_vad_->process(
        std::span<const float>(samples.ptr(), static_cast<size_t>(samples.size())),
        static_cast<int32_t>(input_sample_rate)
    );
    if(!result.is_ok()) {
        output["error"] = String(result.error_message().c_str());
        return output;
    }

    const auto &processed = result.value();
    PackedFloat32Array probabilities;
    probabilities.resize(static_cast<int64_t>(processed.frames.size()));
    PackedInt32Array flags;
    flags.resize(static_cast<int64_t>(processed.frames.size()));
    for(int64_t index = 0; index < static_cast<int64_t>(processed.frames.size()); ++index) {
        probabilities.set(index, processed.frames[static_cast<size_t>(index)].probability);
        flags.set(index, processed.frames[static_cast<size_t>(index)].is_voice ? 1 : 0);
    }

    output["probabilities"] = probabilities;
    output["flags"] = flags;
    output["last_probability"] = processed.last_probability;
    output["last_flag"] = processed.last_is_voice;
    output["voice_detected"] = processed.any_voice;
    output["processed_frame_count"] = processed.processed_frame_count;
    output["hop_size"] = processed.hop_size;
    output["model_sample_rate"] = processed.model_sample_rate;
    return output;
}

} // namespace godot
