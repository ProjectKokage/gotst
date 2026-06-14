#include "gotst/godot/speech_runtime.hpp"

#include "gotst/godot/speech_runtime_config.hpp"
#include "gotst/core/asr_frontend.hpp"
#include "core/onnx_embedding_utils.hpp"
#include "core/sampling_utils.hpp"
#include "gotst/core/speaker_mel.hpp"
#include "gotst/core/tts_prompt_assembly.hpp"

#include <gonx/core/provider.hpp>
#include <gonx/core/session.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <span>
#include <thread>
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

std::vector<float> copy_float_vector(const PackedFloat32Array &values) {
    std::vector<float> copied(static_cast<size_t>(values.size()));
    if(values.size() > 0) {
        std::memcpy(copied.data(), values.ptr(), static_cast<size_t>(values.size()) * sizeof(float));
    }
    return copied;
}

std::vector<int64_t> copy_int64_vector(const PackedInt64Array &values) {
    std::vector<int64_t> copied(static_cast<size_t>(values.size()));
    if(values.size() > 0) {
        std::memcpy(copied.data(), values.ptr(), static_cast<size_t>(values.size()) * sizeof(int64_t));
    }
    return copied;
}

std::vector<uint8_t> copy_bool_vector_from_int64(const PackedInt64Array &values) {
    std::vector<uint8_t> copied(static_cast<size_t>(values.size()));
    for(int64_t index = 0; index < values.size(); ++index) {
        copied[static_cast<size_t>(index)] = values[index] != 0 ? uint8_t{1} : uint8_t{0};
    }
    return copied;
}

std::vector<int64_t> dictionary_int64_array(const Dictionary &dict, const char *key) {
    const Variant value = dict.get(key, PackedInt64Array());
    if(value.get_type() == Variant::PACKED_INT64_ARRAY) {
        return copy_int64_vector(static_cast<PackedInt64Array>(value));
    }
    if(value.get_type() == Variant::ARRAY) {
        const Array array = static_cast<Array>(value);
        std::vector<int64_t> copied;
        copied.reserve(static_cast<size_t>(array.size()));
        for(int64_t index = 0; index < array.size(); ++index) {
            copied.push_back(static_cast<int64_t>(array[index]));
        }
        return copied;
    }
    return {};
}

std::vector<int32_t> dictionary_int32_array(const Dictionary &dict, const char *key) {
    const Variant value = dict.get(key, PackedInt32Array());
    if(value.get_type() == Variant::PACKED_INT32_ARRAY) {
        const PackedInt32Array packed = static_cast<PackedInt32Array>(value);
        std::vector<int32_t> copied(static_cast<size_t>(packed.size()));
        if(packed.size() > 0) {
            std::memcpy(copied.data(), packed.ptr(), static_cast<size_t>(packed.size()) * sizeof(int32_t));
        }
        return copied;
    }
    if(value.get_type() == Variant::PACKED_INT64_ARRAY) {
        const PackedInt64Array packed = static_cast<PackedInt64Array>(value);
        std::vector<int32_t> copied;
        copied.reserve(static_cast<size_t>(packed.size()));
        for(int64_t index = 0; index < packed.size(); ++index) {
            copied.push_back(static_cast<int32_t>(packed[index]));
        }
        return copied;
    }
    if(value.get_type() == Variant::ARRAY) {
        const Array array = static_cast<Array>(value);
        std::vector<int32_t> copied;
        copied.reserve(static_cast<size_t>(array.size()));
        for(int64_t index = 0; index < array.size(); ++index) {
            copied.push_back(static_cast<int32_t>(array[index]));
        }
        return copied;
    }
    return {};
}

std::vector<std::string> dictionary_string_array(const Dictionary &dict, const char *key) {
    const Variant value = dict.get(key, Array());
    std::vector<std::string> copied;
    if(value.get_type() == Variant::PACKED_STRING_ARRAY) {
        const PackedStringArray packed = static_cast<PackedStringArray>(value);
        copied.reserve(static_cast<size_t>(packed.size()));
        for(int64_t index = 0; index < packed.size(); ++index) {
            copied.emplace_back(String(packed[index]).utf8().get_data());
        }
        return copied;
    }
    if(value.get_type() == Variant::ARRAY) {
        const Array array = static_cast<Array>(value);
        copied.reserve(static_cast<size_t>(array.size()));
        for(int64_t index = 0; index < array.size(); ++index) {
            copied.emplace_back(String(array[index]).utf8().get_data());
        }
    }
    return copied;
}

std::vector<uint8_t> dictionary_bool_array(const Dictionary &dict, const char *key) {
    const Variant value = dict.get(key, PackedInt64Array());
    if(value.get_type() == Variant::PACKED_INT64_ARRAY) {
        return copy_bool_vector_from_int64(static_cast<PackedInt64Array>(value));
    }
    if(value.get_type() == Variant::ARRAY) {
        const Array array = static_cast<Array>(value);
        std::vector<uint8_t> copied;
        copied.reserve(static_cast<size_t>(array.size()));
        for(int64_t index = 0; index < array.size(); ++index) {
            copied.push_back(static_cast<bool>(array[index]) ? uint8_t{1} : uint8_t{0});
        }
        return copied;
    }
    return {};
}

std::vector<float> dictionary_float_array(const Dictionary &dict, const char *key) {
    const Variant value = dict.get(key, PackedFloat32Array());
    if(value.get_type() == Variant::PACKED_FLOAT32_ARRAY) {
        return copy_float_vector(static_cast<PackedFloat32Array>(value));
    }
    if(value.get_type() == Variant::ARRAY) {
        const Array array = static_cast<Array>(value);
        std::vector<float> copied;
        copied.reserve(static_cast<size_t>(array.size()));
        for(int64_t index = 0; index < array.size(); ++index) {
            copied.push_back(static_cast<float>(array[index]));
        }
        return copied;
    }
    return {};
}

gotst::TtsSamplingConfig tts_sampling_from_params(const Dictionary &params) {
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
    return sampling;
}

int64_t round_ms(double value) {
    return static_cast<int64_t>(std::llround(std::max(0.0, value)));
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

Dictionary pack_qwen_prompt_result(const gotst::QwenTtsPreparedPrompt &result) {
    Dictionary payload = pack_tts_prompt_result(result.prompt);
    payload["target_frames"] = result.target_frames;
    payload["min_frames_before_eos"] = result.sampling.min_frames_before_eos;
    payload["codebook_size"] = result.sampling.codebook_size;
    payload["residual_groups"] = result.sampling.residual_groups;
    payload["hidden_size"] = result.sampling.hidden_size;
    payload["eos_token_id"] = result.sampling.eos_token_id;
    payload["decode_prefix_frames"] = result.decode_prefix_frames;
    payload["codec_groups"] = result.codec_groups;
    return payload;
}

std::string dictionary_string(const Dictionary &dict, const char *key, const char *fallback = "") {
    return String(dict.get(key, fallback)).utf8().get_data();
}

bool dictionary_bool(const Dictionary &dict, const char *key, bool fallback = false) {
    return static_cast<bool>(dict.get(key, fallback));
}

int32_t dictionary_i32(const Dictionary &dict, const char *key, int32_t fallback = 0) {
    return static_cast<int32_t>(dict.get(key, fallback));
}

int64_t dictionary_i64(const Dictionary &dict, const char *key, int64_t fallback = 0) {
    return static_cast<int64_t>(dict.get(key, fallback));
}

float dictionary_f32(const Dictionary &dict, const char *key, float fallback = 0.0f) {
    return static_cast<float>(dict.get(key, fallback));
}

gotst::QwenTtsPipelineConfig qwen_tts_config_from_dictionary(const Dictionary &config) {
    gotst::QwenTtsPipelineConfig native;
    native.tokenizer_json_path = dictionary_string(config, "tokenizer_json_path", "");
    native.model_config_path = dictionary_string(config, "model_config_path", "");
    native.generation_config_path = dictionary_string(config, "generation_config_path", "");
    native.text_embedding_path = dictionary_string(config, "text_embedding_path", "");
    native.text_projection_path = dictionary_string(config, "text_projection_path", "");
    native.codec_embedding_path = dictionary_string(config, "codec_embedding_path", "");
    native.predictor_embedding_path = dictionary_string(config, "predictor_embedding_path", "");
    native.talker_gguf_path = dictionary_string(config, "talker_gguf_path", "");
    native.predictor_gguf_path = dictionary_string(config, "predictor_gguf_path", "");
    native.decoder_onnx_path = dictionary_string(config, "decoder_onnx_path", "");
    native.speaker_embedding_path = dictionary_string(config, "speaker_embedding_path", "");
    native.custom_voice_config_path = dictionary_string(config, "custom_voice_config_path", "");
    native.custom_voice_speaker_name = dictionary_string(config, "custom_voice_speaker_name", "");
    native.mode = dictionary_string(config, "mode", "base");
    native.provider = dictionary_string(config, "provider", "CPU");
    native.decoder_provider_requested = dictionary_string(config, "decoder_provider_requested", "CPU");
    native.decoder_provider = dictionary_string(config, "decoder_provider", "CPU");
    native.intra_op_threads = dictionary_i32(config, "intra_op_threads", 0);
    native.inter_op_threads = dictionary_i32(config, "inter_op_threads", 0);
    native.optimization_level = dictionary_i32(config, "optimization_level", 99);
    native.decoder_optimized_model_path = dictionary_string(config, "decoder_optimized_model_path", "");
    native.decoder_intra_op_threads = dictionary_i32(config, "decoder_intra_op_threads", 0);
    native.decoder_inter_op_threads = dictionary_i32(config, "decoder_inter_op_threads", 0);
    native.decoder_optimization_level = dictionary_i32(config, "decoder_optimization_level", 99);
    native.sample_rate = dictionary_i32(config, "sample_rate", 24000);
    native.max_text_tokens = dictionary_i32(config, "max_text_tokens", 512);
    native.target_frames_per_text_token = dictionary_f32(config, "target_frames_per_text_token", 5.0f);
    native.target_frame_padding = dictionary_i32(config, "target_frame_padding", 2);
    native.min_frames_before_eos = dictionary_i32(config, "min_frames_before_eos", 8);
    native.max_frames = dictionary_i32(config, "max_frames", 96);
    native.force_japanese_language = dictionary_bool(config, "force_japanese_language", true);
    native.use_style_instruction = dictionary_bool(config, "use_style_instruction", true);
    native.use_icl_voice_clone = dictionary_bool(config, "use_icl_voice_clone", false);
    native.icl_ref_text = dictionary_string(config, "icl_ref_text", "");
    native.normalize_waveform = dictionary_bool(config, "normalize_waveform", true);
    native.waveform_gain = dictionary_f32(config, "waveform_gain", 0.9f);
    native.stateful_chunk_frames = dictionary_i32(config, "stateful_chunk_frames", 12);
    native.talker_n_ctx = dictionary_i32(config, "talker_n_ctx", 1024);
    native.talker_n_batch = dictionary_i32(config, "talker_n_batch", 1024);
    native.predictor_n_ctx = dictionary_i32(config, "predictor_n_ctx", 128);
    native.predictor_n_batch = dictionary_i32(config, "predictor_n_batch", 128);
    native.n_threads = dictionary_i32(config, "n_threads", -1);
    native.n_gpu_layers = dictionary_i32(config, "n_gpu_layers", 0);
    native.predictor_n_gpu_layers = dictionary_i32(config, "predictor_n_gpu_layers", -2);
    native.use_mmap = dictionary_bool(config, "use_mmap", true);
    native.use_mlock = dictionary_bool(config, "use_mlock", false);
    native.flash_attn_type = dictionary_i32(config, "flash_attn_type", -1);
    native.type_k = dictionary_i32(config, "type_k", -1);
    native.type_v = dictionary_i32(config, "type_v", -1);
    return native;
}

gotst::QwenTtsPipelineRequest qwen_tts_request_from_dictionary(const Dictionary &params) {
    gotst::QwenTtsPipelineRequest request;
    request.text = dictionary_string(params, "text", "");
    request.mode = dictionary_string(params, "mode", "");
    request.style_instruction = dictionary_string(params, "style_instruction", "");
    request.voice_design = dictionary_string(params, "voice_design", "");
    request.seed = dictionary_i64(params, "seed", 1);
    return request;
}

std::string irodori_artifact_path(const Dictionary &config, const Dictionary &artifacts, const char *key) {
    if(artifacts.has(key)) {
        return dictionary_string(artifacts, key, "");
    }
    return dictionary_string(config, key, "");
}

std::map<std::string, std::string> dictionary_string_map(const Dictionary &dict) {
    std::map<std::string, std::string> output;
    const Array keys = dict.keys();
    for(int64_t index = 0; index < keys.size(); ++index) {
        const String key = keys[index];
        output[std::string(key.utf8().get_data())] =
            dictionary_string(dict, key.utf8().get_data(), "");
    }
    return output;
}

Dictionary nested_dictionary(const Dictionary &dict, const char *key) {
    const Variant value = dict.get(key, Dictionary());
    return value.get_type() == Variant::DICTIONARY ? static_cast<Dictionary>(value) : Dictionary();
}

gotst::IrodoriTtsProviderRoute irodori_provider_route_from_dictionary(
    const Dictionary &config,
    const Dictionary &routes,
    const char *stage
) {
    gotst::IrodoriTtsProviderRoute route;
    if(routes.has(stage)) {
        const Variant route_value = routes[stage];
        if(route_value.get_type() == Variant::DICTIONARY) {
            const Dictionary route_dict = static_cast<Dictionary>(route_value);
            route.provider_requested = dictionary_string(route_dict, "provider_requested", "");
            route.provider = dictionary_string(route_dict, "provider", "");
        }
    }

    const std::string prefix(stage);
    const std::string requested_key = prefix + "_provider_requested";
    const std::string provider_key = prefix + "_provider";
    if(config.has(requested_key.c_str())) {
        route.provider_requested = dictionary_string(config, requested_key.c_str(), route.provider_requested.c_str());
    }
    if(config.has(provider_key.c_str())) {
        route.provider = dictionary_string(config, provider_key.c_str(), route.provider.c_str());
    }
    return route;
}

gotst::IrodoriTtsSessionConfig irodori_config_from_dictionary(const Dictionary &config) {
    gotst::IrodoriTtsSessionConfig session_config;
    session_config.bundle_root = dictionary_string(config, "bundle_root", "");
    session_config.manifest_path = dictionary_string(config, "manifest_path", "");
    session_config.mode = gotst::parse_irodori_tts_mode(dictionary_string(config, "mode", ""));
    session_config.provider_profile = dictionary_string(config, "provider_profile", "cpu");
    session_config.provider_requested = dictionary_string(config, "provider_requested", "CPU");
    session_config.provider = dictionary_string(config, "provider", "CPU");
    session_config.strict_provider = dictionary_bool(config, "strict_provider", false);
    session_config.intra_op_threads = dictionary_i32(config, "intra_op_threads", 0);
    session_config.inter_op_threads = dictionary_i32(config, "inter_op_threads", 0);
    session_config.device_id = dictionary_i32(config, "device_id", 0);
    session_config.optimization_level = dictionary_i32(config, "optimization_level", 99);
    session_config.optimized_model_path = dictionary_string(config, "optimized_model_path", "");
    const Dictionary provider_options = config.has("provider_options") ?
        nested_dictionary(config, "provider_options") : nested_dictionary(config, "coreml_provider_options");
    session_config.provider_options = dictionary_string_map(provider_options);
    session_config.session_options = dictionary_string_map(nested_dictionary(config, "session_options"));
    session_config.ort_enable_profiling = dictionary_bool(config, "ort_enable_profiling", false);
    session_config.ort_profiling_prefix = dictionary_string(config, "ort_profiling_prefix", "");
    session_config.ort_log_severity_level = dictionary_i32(config, "ort_log_severity_level", -1);
    session_config.runtime_dispatch = dictionary_string(config, "runtime_dispatch", "force_cpu");
    session_config.rf_execution_mode = dictionary_string(config, "rf_execution_mode", "auto");
    session_config.dispatch_recommendation_path = dictionary_string(config, "dispatch_recommendation_path", "");
    session_config.print_provider_diagnostics = dictionary_bool(config, "print_provider_diagnostics", false);
    session_config.sample_rate = dictionary_i32(config, "sample_rate", 48000);
    session_config.latent_dim = dictionary_i32(config, "latent_dim", 32);
    session_config.latent_patch_size = dictionary_i32(config, "latent_patch_size", 1);
    session_config.speaker_patch_size = dictionary_i32(config, "speaker_patch_size", 1);
    session_config.codec_hop_length = dictionary_i32(config, "codec_hop_length", 640);
    session_config.duration_aux_dim = dictionary_i32(config, "duration_aux_dim", 14);
    session_config.max_text_tokens = dictionary_i32(config, "max_text_tokens", 256);
    session_config.max_caption_tokens = dictionary_i32(config, "max_caption_tokens", 512);
    session_config.max_ref_seconds = dictionary_i32(config, "max_ref_seconds", 30);
    session_config.default_num_steps = dictionary_i32(config, "default_num_steps", 8);
    session_config.default_t_schedule_mode = dictionary_string(config, "default_t_schedule_mode", "sway");
    session_config.default_sway_coeff = dictionary_f32(config, "default_sway_coeff", -1.0f);
    session_config.default_cfg_guidance_mode = dictionary_string(config, "default_cfg_guidance_mode", "");
    session_config.require_all_artifacts = dictionary_bool(config, "require_all_artifacts", true);
    session_config.enable_context_kv_cache = dictionary_bool(config, "enable_context_kv_cache", true);
    session_config.enable_ref_latent_cache = dictionary_bool(config, "enable_ref_latent_cache", true);
    session_config.enable_coreml_unrolled_rf_sampler =
        dictionary_bool(config, "enable_coreml_unrolled_rf_sampler", false);

    const Variant artifacts_value = config.get("artifacts", Dictionary());
    const Dictionary artifacts = artifacts_value.get_type() == Variant::DICTIONARY ?
        static_cast<Dictionary>(artifacts_value) : Dictionary();
    session_config.artifacts.tokenizer_json_path = irodori_artifact_path(config, artifacts, "tokenizer_json_path");
    session_config.artifacts.tokenizer_config_path = irodori_artifact_path(config, artifacts, "tokenizer_config_path");
    session_config.artifacts.model_config_json_path = irodori_artifact_path(config, artifacts, "model_config_json_path");
    session_config.artifacts.text_encoder_onnx_path = irodori_artifact_path(config, artifacts, "text_encoder_onnx_path");
    session_config.artifacts.caption_encoder_onnx_path = irodori_artifact_path(config, artifacts, "caption_encoder_onnx_path");
    session_config.artifacts.speaker_encoder_onnx_path = irodori_artifact_path(config, artifacts, "speaker_encoder_onnx_path");
    session_config.artifacts.duration_predictor_onnx_path = irodori_artifact_path(config, artifacts, "duration_predictor_onnx_path");
    session_config.artifacts.dit_step_onnx_path = irodori_artifact_path(config, artifacts, "dit_step_onnx_path");
    session_config.artifacts.dacvae_encoder_onnx_path = irodori_artifact_path(config, artifacts, "dacvae_encoder_onnx_path");
    session_config.artifacts.dacvae_decoder_onnx_path = irodori_artifact_path(config, artifacts, "dacvae_decoder_onnx_path");

    const Dictionary routes = nested_dictionary(config, "provider_routes");
    session_config.provider_routes.text_encoder = irodori_provider_route_from_dictionary(config, routes, "text_encoder");
    session_config.provider_routes.caption_encoder = irodori_provider_route_from_dictionary(config, routes, "caption_encoder");
    session_config.provider_routes.speaker_encoder = irodori_provider_route_from_dictionary(config, routes, "speaker_encoder");
    session_config.provider_routes.duration_predictor = irodori_provider_route_from_dictionary(config, routes, "duration_predictor");
    session_config.provider_routes.dit_step = irodori_provider_route_from_dictionary(config, routes, "dit_step");
    session_config.provider_routes.dacvae_encoder = irodori_provider_route_from_dictionary(config, routes, "dacvae_encoder");
    session_config.provider_routes.dacvae_decoder = irodori_provider_route_from_dictionary(config, routes, "dacvae_decoder");

    const Variant buckets_value = config.get("buckets", Array());
    if(buckets_value.get_type() == Variant::ARRAY) {
        const Array buckets = static_cast<Array>(buckets_value);
        for(int64_t index = 0; index < buckets.size(); ++index) {
            const Variant bucket_value = buckets[index];
            if(bucket_value.get_type() != Variant::DICTIONARY) {
                continue;
            }
            const Dictionary bucket_dict = static_cast<Dictionary>(bucket_value);
            gotst::IrodoriTtsBucket bucket;
            bucket.latent_steps = dictionary_i32(bucket_dict, "latent_steps", 0);
            bucket.text_tokens = dictionary_i32(bucket_dict, "text_tokens", 0);
            bucket.caption_tokens = dictionary_i32(bucket_dict, "caption_tokens", 0);
            bucket.ref_steps = dictionary_i32(bucket_dict, "ref_steps", 0);
            if(bucket.latent_steps > 0 && bucket.text_tokens > 0) {
                session_config.buckets.push_back(bucket);
            }
        }
    }

    const Array static_values = config.has("coreml_static_artifacts") ?
        static_cast<Array>(config.get("coreml_static_artifacts", Array())) : Array();
    for(int64_t index = 0; index < static_values.size(); ++index) {
        const Variant artifact_value = static_values[index];
        if(artifact_value.get_type() != Variant::DICTIONARY) {
            continue;
        }
        const Dictionary artifact_dict = static_cast<Dictionary>(artifact_value);
        gotst::IrodoriTtsStaticArtifact artifact;
        artifact.bucket.latent_steps = dictionary_i32(artifact_dict, "latent_steps", 0);
        artifact.bucket.text_tokens = dictionary_i32(artifact_dict, "text_tokens", 0);
        artifact.bucket.caption_tokens = dictionary_i32(artifact_dict, "caption_tokens", 0);
        artifact.bucket.ref_steps = dictionary_i32(artifact_dict, "ref_steps", 0);
        artifact.dit_step_onnx_path = dictionary_string(artifact_dict, "dit_step_onnx_path", "");
        artifact.dacvae_decoder_onnx_path = dictionary_string(artifact_dict, "dacvae_decoder_onnx_path", "");
        artifact.rf_sampler_6_step_onnx_path = dictionary_string(artifact_dict, "rf_sampler_6_step_onnx_path", "");
        artifact.rf_sampler_8_step_onnx_path = dictionary_string(artifact_dict, "rf_sampler_8_step_onnx_path", "");
        if(artifact.bucket.latent_steps > 0 && artifact.bucket.text_tokens > 0 &&
           !artifact.dit_step_onnx_path.empty()) {
            session_config.coreml_static_artifacts.push_back(artifact);
        }
    }

    return session_config;
}

gotst::IrodoriTtsRequest irodori_request_from_dictionary(const Dictionary &params) {
    gotst::IrodoriTtsRequest request;
    request.text = dictionary_string(params, "text", "");
    request.caption = dictionary_string(params, "caption", "");
    request.text_token_ids = dictionary_int64_array(params, "text_token_ids");
    request.text_token_mask = dictionary_bool_array(params, "text_token_mask");
    request.caption_token_ids = dictionary_int64_array(params, "caption_token_ids");
    request.caption_token_mask = dictionary_bool_array(params, "caption_token_mask");
    request.ref_wav_path = dictionary_string(params, "ref_wav_path", "");
    request.ref_latent_path = dictionary_string(params, "ref_latent_path", "");
    request.ref_latent = dictionary_float_array(params, "ref_latent");
    request.ref_latent_steps = dictionary_i32(params, "ref_latent_steps", 0);
    request.no_ref = dictionary_bool(params, "no_ref", false);
    request.seed = dictionary_i64(params, "seed", -1);
    request.num_steps = dictionary_i32(params, "num_steps", 8);
    request.duration_scale = dictionary_f32(params, "duration_scale", 1.0f);
    if(params.has("seconds")) {
        request.seconds = dictionary_f32(params, "seconds", 0.0f);
    }
    request.cfg_scale_text = dictionary_f32(params, "cfg_scale_text", 3.0f);
    request.cfg_scale_caption = dictionary_f32(params, "cfg_scale_caption", 3.0f);
    request.cfg_scale_speaker = dictionary_f32(params, "cfg_scale_speaker", 5.0f);
    if(params.has("cfg_scale")) {
        request.cfg_scale = dictionary_f32(params, "cfg_scale", 0.0f);
    }
    request.cfg_min_t = dictionary_f32(params, "cfg_min_t", 0.5f);
    request.cfg_max_t = dictionary_f32(params, "cfg_max_t", 1.0f);
    request.cfg_guidance_mode = dictionary_string(params, "cfg_guidance_mode", "");
    request.t_schedule_mode = dictionary_string(params, "t_schedule_mode", "sway");
    request.sway_coeff = dictionary_f32(params, "sway_coeff", -1.0f);
    request.context_kv_cache = dictionary_bool(params, "context_kv_cache", true);
    request.ref_latent_cache = dictionary_bool(params, "ref_latent_cache", true);
    request.decode_mode = dictionary_string(params, "decode_mode", "sequential");
    return request;
}

std::string dictionary_path_alias(
    const Dictionary &dict,
    const char *primary,
    const char *fallback
) {
    const std::string value = dictionary_string(dict, primary, "");
    if(!value.empty()) {
        return value;
    }
    return dictionary_string(dict, fallback, "");
}

gotst::Qwen3ForcedAlignerModelPaths forced_aligner_paths_from_dictionary(const Dictionary &config) {
    gotst::Qwen3ForcedAlignerModelPaths paths;
    paths.audio_conv_onnx_path = dictionary_path_alias(
        config,
        "audio_conv_onnx_path",
        "thinker_audio_conv_path"
    );
    paths.audio_encoder_onnx_path = dictionary_path_alias(
        config,
        "audio_encoder_onnx_path",
        "thinker_audio_encoder_path"
    );
    paths.embedding_onnx_path = dictionary_path_alias(
        config,
        "embedding_onnx_path",
        "thinker_embedding_path"
    );
    paths.backbone_gguf_path = dictionary_path_alias(
        config,
        "backbone_gguf_path",
        "thinker_gguf_path"
    );
    paths.classifier_head_path = dictionary_string(config, "classifier_head_path", "");
    return paths;
}

gotst::Qwen3ForcedAlignerSessionConfig forced_aligner_config_from_dictionary(const Dictionary &config) {
    gotst::Qwen3ForcedAlignerSessionConfig session_config;
    session_config.sample_rate = dictionary_i32(config, "sample_rate", 16000);
    session_config.mel_bins = dictionary_i32(config, "mel_bins", 128);
    session_config.fft_size = dictionary_i32(config, "fft_size", 400);
    session_config.hop_length = dictionary_i32(config, "hop_length", 160);
    session_config.chunk_length_seconds = static_cast<double>(config.get("chunk_length_seconds", 300.0));
    session_config.audio_conv_chunk_frames = dictionary_i32(config, "audio_conv_chunk_frames", 100);
    session_config.timestamp_token_id = dictionary_i32(config, "timestamp_token_id", 151705);
    session_config.timestamp_segment_ms = dictionary_i32(config, "timestamp_segment_ms", 80);
    session_config.classify_num = dictionary_i32(config, "classify_num", 5000);
    session_config.n_ctx = dictionary_i32(config, "n_ctx", 8192);
    session_config.n_batch = dictionary_i32(config, "n_batch", 1024);
    session_config.n_threads = dictionary_i32(config, "n_threads", -1);
    session_config.n_gpu_layers = dictionary_i32(config, "n_gpu_layers", 0);
    session_config.use_mmap = dictionary_bool(config, "use_mmap", true);
    session_config.use_mlock = dictionary_bool(config, "use_mlock", false);
    session_config.flash_attn_type = dictionary_i32(config, "flash_attn_type", -1);
    session_config.type_k = dictionary_i32(config, "type_k", -1);
    session_config.type_v = dictionary_i32(config, "type_v", -1);
    session_config.position_components = dictionary_i32(config, "position_components", 3);
    session_config.onnx_provider = dictionary_string(config, "onnx_provider", "CPU");
    if(session_config.onnx_provider == "CPU" && config.has("provider")) {
        session_config.onnx_provider = dictionary_string(config, "provider", "CPU");
    }
    session_config.onnx_device_id = dictionary_i32(config, "onnx_device_id", 0);
    session_config.onnx_intra_op_threads = dictionary_i32(config, "onnx_intra_op_threads", 0);
    session_config.onnx_inter_op_threads = dictionary_i32(config, "onnx_inter_op_threads", 0);
    session_config.onnx_optimization_level = dictionary_i32(config, "onnx_optimization_level", 99);
    return session_config;
}

gotst::Qwen3ForcedAlignmentRequest forced_alignment_request_from_dictionary(const Dictionary &params) {
    gotst::Qwen3ForcedAlignmentRequest request;
    request.waveform = dictionary_float_array(params, "waveform");
    request.input_sample_rate = dictionary_i32(params, "input_sample_rate", 0);
    request.language = dictionary_string(params, "language", "");
    request.text_units = dictionary_string_array(params, "text_units");
    if(request.text_units.empty() && params.has("text")) {
        request.text_units = gotst::split_forced_alignment_units(
            dictionary_string(params, "text", ""),
            dictionary_string(params, "unit_mode", request.language.c_str())
        );
    }
    request.token_ids = dictionary_int64_array(params, "token_ids");
    request.timestamp_token_indices = dictionary_int32_array(params, "timestamp_token_indices");
    request.audio_placeholder_start = dictionary_i32(params, "audio_placeholder_start", -1);
    request.audio_placeholder_count = dictionary_i32(params, "audio_placeholder_count", 0);
    request.allow_audio_truncation = dictionary_bool(params, "allow_audio_truncation", false);
    request.max_duration_seconds = static_cast<double>(params.get("max_duration_seconds", 300.0));
    return request;
}

Array pack_forced_alignment_spans(const std::vector<gotst::Qwen3ForcedAlignmentSpan> &spans) {
    Array output;
    for(const gotst::Qwen3ForcedAlignmentSpan &span : spans) {
        Dictionary item;
        item["text"] = String(span.text.c_str());
        item["start_sec"] = span.start_sec;
        item["end_sec"] = span.end_sec;
        item["start_bin"] = span.start_bin;
        item["end_bin"] = span.end_bin;
        item["confidence"] = span.confidence;
        output.push_back(item);
    }
    return output;
}

Dictionary forced_alignment_timings_to_dictionary(const gotst::Qwen3ForcedAlignmentResult &result) {
    Dictionary timings;
    timings["elapsed_ms"] = round_ms(result.elapsed_ms);
    timings["frontend_ms"] = round_ms(result.frontend_ms);
    timings["audio_conv_ms"] = round_ms(result.audio_conv_ms);
    timings["audio_encoder_ms"] = round_ms(result.audio_encoder_ms);
    timings["embedding_ms"] = round_ms(result.embedding_ms);
    timings["backbone_ms"] = round_ms(result.backbone_ms);
    timings["classifier_ms"] = round_ms(result.classifier_ms);
    return timings;
}

Dictionary irodori_timings_to_dictionary(const std::map<std::string, double> &timings) {
    Dictionary output;
    for(const auto &[name, value] : timings) {
        output[String(name.c_str())] = round_ms(value);
    }
    return output;
}

Dictionary string_map_to_dictionary(const std::map<std::string, std::string> &values) {
    Dictionary output;
    for(const auto &[key, value] : values) {
        output[String(key.c_str())] = String(value.c_str());
    }
    return output;
}

} // namespace

GotstSpeechRuntime::~GotstSpeechRuntime() {
    stop_forced_alignment_worker();
    stop_irodori_stream_worker();
    stop_waveform_stream_workers();

    // Keep the GGUF-backed speech decoders alive until process exit. On macOS
    // headless shutdown, destroying their llama contexts during RefCounted
    // teardown can raise std::system_error("mutex lock failed") inside the
    // native runtime after the useful work is already complete.
    (void)tts_code_generator_.release();
    (void)asr_token_decoder_.release();
    (void)qwen3_forced_aligner_.release();
}

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
        D_METHOD(
            "resolve_custom_voice_speaker_embedding",
            "config_json_path",
            "speaker_name",
            "codec_embedding_onnx_path",
            "hidden_size"
        ),
        &GotstSpeechRuntime::resolve_custom_voice_speaker_embedding
    );
    ClassDB::bind_method(
        D_METHOD(
            "prepare_voice_clone_decoder_codes",
            "generated_codes",
            "generated_frame_count",
            "ref_codes",
            "ref_frame_count",
            "codec_groups"
        ),
        &GotstSpeechRuntime::prepare_voice_clone_decoder_codes
    );
    ClassDB::bind_method(
        D_METHOD("trim_voice_clone_waveform_prefix", "waveform", "trim_prefix_frames", "decode_frame_count"),
        &GotstSpeechRuntime::trim_voice_clone_waveform_prefix
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
        D_METHOD("start_tts_waveform_stream", "params", "request_id", "chunk_frames"),
        &GotstSpeechRuntime::start_tts_waveform_stream
    );
    ClassDB::bind_method(
        D_METHOD("poll_tts_waveform_stream"),
        &GotstSpeechRuntime::poll_tts_waveform_stream
    );
    ClassDB::bind_method(
        D_METHOD("cancel_tts_waveform_stream", "request_id"),
        &GotstSpeechRuntime::cancel_tts_waveform_stream
    );
    ClassDB::bind_method(
        D_METHOD("load_qwen_tts", "config"),
        &GotstSpeechRuntime::load_qwen_tts
    );
    ClassDB::bind_method(
        D_METHOD("is_qwen_tts_loaded"),
        &GotstSpeechRuntime::is_qwen_tts_loaded
    );
    ClassDB::bind_method(
        D_METHOD("prepare_qwen_tts_prompt", "params"),
        &GotstSpeechRuntime::prepare_qwen_tts_prompt
    );
    ClassDB::bind_method(
        D_METHOD("start_qwen_tts_stream", "params", "request_id", "chunk_frames"),
        &GotstSpeechRuntime::start_qwen_tts_stream
    );
    ClassDB::bind_method(
        D_METHOD("poll_qwen_tts_stream"),
        &GotstSpeechRuntime::poll_qwen_tts_stream
    );
    ClassDB::bind_method(
        D_METHOD("cancel_qwen_tts_stream", "request_id"),
        &GotstSpeechRuntime::cancel_qwen_tts_stream
    );
    ClassDB::bind_method(
        D_METHOD("get_qwen_custom_voice_speaker_names"),
        &GotstSpeechRuntime::get_qwen_custom_voice_speaker_names
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
        D_METHOD("load_irodori_tts", "config"),
        &GotstSpeechRuntime::load_irodori_tts
    );
    ClassDB::bind_method(
        D_METHOD("is_irodori_tts_loaded"),
        &GotstSpeechRuntime::is_irodori_tts_loaded
    );
    ClassDB::bind_method(
        D_METHOD("load_irodori_tokenizer", "tokenizer_json_path", "tokenizer_config_path"),
        &GotstSpeechRuntime::load_irodori_tokenizer
    );
    ClassDB::bind_method(
        D_METHOD("normalize_irodori_text", "text"),
        &GotstSpeechRuntime::normalize_irodori_text
    );
    ClassDB::bind_method(
        D_METHOD("tokenize_irodori_text", "text", "max_tokens", "force_empty_mask"),
        &GotstSpeechRuntime::tokenize_irodori_text
    );
    ClassDB::bind_method(
        D_METHOD("start_irodori_tts_stream", "params", "request_id"),
        &GotstSpeechRuntime::start_irodori_tts_stream
    );
    ClassDB::bind_method(
        D_METHOD("poll_irodori_tts_stream"),
        &GotstSpeechRuntime::poll_irodori_tts_stream
    );
    ClassDB::bind_method(
        D_METHOD("cancel_irodori_tts_stream", "request_id"),
        &GotstSpeechRuntime::cancel_irodori_tts_stream
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
        D_METHOD("load_qwen3_forced_aligner", "config"),
        &GotstSpeechRuntime::load_qwen3_forced_aligner
    );
    ClassDB::bind_method(
        D_METHOD("is_qwen3_forced_aligner_loaded"),
        &GotstSpeechRuntime::is_qwen3_forced_aligner_loaded
    );
    ClassDB::bind_method(
        D_METHOD("start_qwen3_forced_alignment", "params", "request_id"),
        &GotstSpeechRuntime::start_qwen3_forced_alignment
    );
    ClassDB::bind_method(
        D_METHOD("poll_qwen3_forced_alignment"),
        &GotstSpeechRuntime::poll_qwen3_forced_alignment
    );
    ClassDB::bind_method(
        D_METHOD("cancel_qwen3_forced_alignment", "request_id"),
        &GotstSpeechRuntime::cancel_qwen3_forced_alignment
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

PackedFloat32Array GotstSpeechRuntime::resolve_custom_voice_speaker_embedding(
    const String &config_json_path,
    const String &speaker_name,
    const String &codec_embedding_onnx_path,
    int64_t hidden_size
) {
    if(hidden_size <= 0) {
        ERR_PRINT("CustomVoice speaker embedding requires a positive hidden_size.");
        return PackedFloat32Array();
    }

    const std::string speaker = std::string(speaker_name.utf8().get_data());
    const std::string config_path = std::string(config_json_path.utf8().get_data());
    const std::string embedding_path = std::string(codec_embedding_onnx_path.utf8().get_data());
    if(speaker.empty() || config_path.empty() || embedding_path.empty()) {
        ERR_PRINT("CustomVoice speaker embedding requires config path, speaker name, and codec embedding path.");
        return PackedFloat32Array();
    }

    if(!load_custom_voice_config(config_json_path)) {
        ERR_PRINT(String("CustomVoice speaker config load failed: ") + config_json_path);
        return PackedFloat32Array();
    }

    auto speaker_it = custom_voice_speakers_.find(speaker);
    if(speaker_it == custom_voice_speakers_.end() || speaker_it->second.empty()) {
        ERR_PRINT(String("CustomVoice speaker not found: ") + speaker_name);
        return PackedFloat32Array();
    }

    gonx::InferenceSession embedding_session;
    gonx::SessionConfig session_config;
    session_config.providers = {gonx::ExecutionProvider::CPU};
    auto load_status = embedding_session.load(std::filesystem::path(embedding_path), session_config);
    if(load_status.has_error()) {
        ERR_PRINT(String("CustomVoice codec embedding load failed: ") + String(load_status.error().message.c_str()));
        return PackedFloat32Array();
    }

    std::vector<float> embedding_values;
    embedding_values.reserve(
        speaker_it->second.size() * static_cast<size_t>(hidden_size)
    );
    gotst::detail::SingleTokenEmbeddingRunScratch scratch;
    for(int64_t token_id : speaker_it->second) {
        auto token_embedding = gotst::detail::run_single_token_float_embedding(
            embedding_session,
            scratch,
            token_id
        );
        if(!token_embedding.is_ok()) {
            ERR_PRINT(String("CustomVoice codec embedding inference failed: ") +
                String(token_embedding.error_message().c_str()));
            return PackedFloat32Array();
        }
        const std::span<const float> values = token_embedding.value().values;
        if(values.size() < static_cast<size_t>(hidden_size)) {
            ERR_PRINT("CustomVoice codec embedding returned too few values.");
            return PackedFloat32Array();
        }
        embedding_values.insert(
            embedding_values.end(),
            values.begin(),
            values.begin() + static_cast<std::ptrdiff_t>(hidden_size)
        );
    }

    return pack_float_array(embedding_values);
}

Dictionary GotstSpeechRuntime::prepare_voice_clone_decoder_codes(
    const PackedInt64Array &generated_codes,
    int64_t generated_frame_count,
    const PackedInt64Array &ref_codes,
    int64_t ref_frame_count,
    int64_t codec_groups
) const {
    Dictionary output;
    const int64_t generated_count = generated_codes.size();
    if(generated_count <= 0 || generated_frame_count <= 0) {
        output["error"] = "generated voice-clone codes are empty.";
        return output;
    }
    if(ref_frame_count <= 0 || codec_groups <= 0 || ref_codes.size() != ref_frame_count * codec_groups) {
        output["audio_codes"] = generated_codes;
        output["decode_frame_count"] = generated_frame_count;
        output["trim_prefix_frames"] = 0;
        output["visible_code_count"] = generated_count;
        return output;
    }

    PackedInt64Array audio_codes;
    audio_codes.resize(ref_codes.size() + generated_count);
    if(ref_codes.size() > 0) {
        std::memcpy(audio_codes.ptrw(), ref_codes.ptr(), static_cast<size_t>(ref_codes.size()) * sizeof(int64_t));
    }
    if(generated_count > 0) {
        std::memcpy(
            audio_codes.ptrw() + ref_codes.size(),
            generated_codes.ptr(),
            static_cast<size_t>(generated_count) * sizeof(int64_t)
        );
    }

    output["audio_codes"] = audio_codes;
    output["decode_frame_count"] = ref_frame_count + generated_frame_count;
    output["trim_prefix_frames"] = ref_frame_count;
    output["visible_code_count"] = generated_count;
    return output;
}

PackedFloat32Array GotstSpeechRuntime::trim_voice_clone_waveform_prefix(
    const PackedFloat32Array &waveform,
    int64_t trim_prefix_frames,
    int64_t decode_frame_count
) const {
    if(waveform.is_empty() || trim_prefix_frames <= 0 || decode_frame_count <= 0) {
        return waveform;
    }

    const int64_t trim_samples = static_cast<int64_t>(
        (static_cast<double>(trim_prefix_frames) /
            static_cast<double>(std::max<int64_t>(decode_frame_count, 1))) *
        static_cast<double>(waveform.size())
    );
    if(trim_samples <= 0) {
        return waveform;
    }
    if(trim_samples >= waveform.size()) {
        return PackedFloat32Array();
    }

    PackedFloat32Array trimmed;
    const int64_t remaining = waveform.size() - trim_samples;
    trimmed.resize(remaining);
    std::memcpy(
        trimmed.ptrw(),
        waveform.ptr() + trim_samples,
        static_cast<size_t>(remaining) * sizeof(float)
    );
    return trimmed;
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
    session_config.predictor_n_gpu_layers = static_cast<int32_t>(
        config.get("predictor_n_gpu_layers", session_config.n_gpu_layers)
    );
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

bool GotstSpeechRuntime::load_qwen_tts(const Dictionary &config) {
    stop_waveform_stream_workers();
    clear_waveform_events();

    qwen_tts_pipeline_ = std::make_unique<gotst::QwenTtsPipeline>();
    auto result = qwen_tts_pipeline_->load(qwen_tts_config_from_dictionary(config));
    if(!result.is_ok()) {
        ERR_PRINT(String("Qwen TTS pipeline load failed: ") + String(result.error_message().c_str()));
        qwen_tts_pipeline_.reset();
        return false;
    }
    return true;
}

bool GotstSpeechRuntime::is_qwen_tts_loaded() const {
    return qwen_tts_pipeline_ && qwen_tts_pipeline_->is_loaded();
}

Dictionary GotstSpeechRuntime::prepare_qwen_tts_prompt(const Dictionary &params) {
    Dictionary output;
    if(!qwen_tts_pipeline_ || !qwen_tts_pipeline_->is_loaded()) {
        output["error"] = "Qwen TTS pipeline is not loaded.";
        return output;
    }
    auto result = qwen_tts_pipeline_->prepare_prompt(qwen_tts_request_from_dictionary(params));
    if(!result.is_ok()) {
        output["error"] = String(result.error_message().c_str());
        return output;
    }
    return pack_qwen_prompt_result(result.value());
}

Dictionary GotstSpeechRuntime::start_qwen_tts_stream(
    const Dictionary &params,
    int64_t request_id,
    int64_t chunk_frames
) {
    Dictionary output;
    if(!qwen_tts_pipeline_ || !qwen_tts_pipeline_->is_loaded()) {
        output["error"] = "Qwen TTS pipeline is not loaded.";
        return output;
    }

    ensure_waveform_stream_workers_started();
    if(!waveform_stream_active_.load(std::memory_order_acquire)) {
        clear_waveform_events();
    }

    WaveformStreamRequest request;
    request.request_id = request_id;
    request.chunk_frames = static_cast<int32_t>(std::max<int64_t>(1, chunk_frames));
    request.queued_at = std::chrono::steady_clock::now();
    request.cancel = std::make_shared<gotst::CancellationToken>();
    request.use_qwen_pipeline = true;
    request.qwen_request = qwen_tts_request_from_dictionary(params);
    {
        std::lock_guard<std::mutex> lock(waveform_request_mutex_);
        waveform_request_queue_.push_back(std::move(request));
        waveform_stream_active_.store(true, std::memory_order_release);
    }
    waveform_request_cv_.notify_one();

    output["started"] = true;
    output["request_id"] = request_id;
    return output;
}

Array GotstSpeechRuntime::poll_qwen_tts_stream() {
    return poll_tts_waveform_stream();
}

void GotstSpeechRuntime::cancel_qwen_tts_stream(int64_t request_id) {
    cancel_tts_waveform_stream(request_id);
}

Array GotstSpeechRuntime::get_qwen_custom_voice_speaker_names() const {
    Array names;
    if(!qwen_tts_pipeline_ || !qwen_tts_pipeline_->is_loaded()) {
        return names;
    }
    for(const std::string &name : qwen_tts_pipeline_->custom_voice_speaker_names()) {
        names.push_back(String(name.c_str()));
    }
    return names;
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
    output["elapsed_ms"] = static_cast<int64_t>(std::llround(gen.elapsed_ms));
    output["talker_prefill_ms"] = static_cast<int64_t>(std::llround(gen.talker_prefill_ms));
    output["talker_decode_ms"] = static_cast<int64_t>(std::llround(gen.talker_decode_ms));
    output["predictor_ms"] = static_cast<int64_t>(std::llround(gen.predictor_ms));
    output["onnx_embedding_ms"] = static_cast<int64_t>(std::llround(gen.onnx_embedding_ms));
    output["other_ms"] = static_cast<int64_t>(std::llround(gen.other_ms));
    return output;
}

Dictionary GotstSpeechRuntime::start_tts_waveform_stream(
    const Dictionary &params,
    int64_t request_id,
    int64_t chunk_frames
) {
    Dictionary output;
    if(!tts_code_generator_ || !tts_code_generator_->is_loaded()) {
        output["error"] = "TTS code generator is not loaded.";
        return output;
    }
    if(!tts_waveform_decoder_ || !tts_waveform_decoder_->is_loaded()) {
        output["error"] = "TTS waveform decoder is not loaded.";
        return output;
    }

    ensure_waveform_stream_workers_started();
    if(!waveform_stream_active_.load(std::memory_order_acquire)) {
        clear_waveform_events();
    }

    const PackedFloat32Array initial_input = params.get("initial_language_input", PackedFloat32Array());
    const PackedFloat32Array trailing_hidden = params.get("trailing_text_hidden", PackedFloat32Array());
    const PackedFloat32Array pad_embedding = params.get("tts_pad_embedding", PackedFloat32Array());

    WaveformStreamRequest request;
    request.request_id = request_id;
    request.initial_input = copy_float_vector(initial_input);
    request.initial_length = static_cast<int32_t>(params.get("initial_sequence_length", 0));
    request.trailing_hidden = copy_float_vector(trailing_hidden);
    request.trailing_length = static_cast<int32_t>(params.get("trailing_text_length", 0));
    request.pad_embedding = copy_float_vector(pad_embedding);
    request.sampling = tts_sampling_from_params(params);
    request.chunk_frames = static_cast<int32_t>(std::max<int64_t>(1, chunk_frames));
    request.queued_at = std::chrono::steady_clock::now();
    request.cancel = std::make_shared<gotst::CancellationToken>();
    {
        std::lock_guard<std::mutex> lock(waveform_request_mutex_);
        waveform_request_queue_.push_back(std::move(request));
        waveform_stream_active_.store(true, std::memory_order_release);
    }
    waveform_request_cv_.notify_one();

    output["started"] = true;
    output["request_id"] = request_id;
    return output;
}

void GotstSpeechRuntime::ensure_waveform_stream_workers_started() {
    waveform_workers_stop_.store(false, std::memory_order_release);
    if(!waveform_decoder_worker_.joinable()) {
        waveform_decoder_worker_ = std::thread(&GotstSpeechRuntime::waveform_decoder_worker_main, this);
    }
    if(!waveform_generation_worker_.joinable()) {
        waveform_generation_worker_ = std::thread(&GotstSpeechRuntime::waveform_generation_worker_main, this);
    }
}

void GotstSpeechRuntime::stop_waveform_stream_workers() {
    waveform_workers_stop_.store(true, std::memory_order_release);
    if(waveform_active_cancel_) {
        waveform_active_cancel_->cancel();
    }
    {
        std::lock_guard<std::mutex> lock(waveform_request_mutex_);
        for(auto &request : waveform_request_queue_) {
            if(request.cancel) {
                request.cancel->cancel();
            }
        }
        waveform_request_queue_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(waveform_decode_mutex_);
        for(auto &job : waveform_decode_queue_) {
            if(job.cancel) {
                job.cancel->cancel();
            }
        }
        waveform_decode_queue_.clear();
    }
    waveform_request_cv_.notify_all();
    waveform_decode_cv_.notify_all();
    if(waveform_generation_worker_.joinable()) {
        waveform_generation_worker_.join();
    }
    if(waveform_decoder_worker_.joinable()) {
        waveform_decoder_worker_.join();
    }
    waveform_stream_active_.store(false, std::memory_order_release);
}

void GotstSpeechRuntime::push_waveform_event(WaveformStreamEvent event) {
    std::lock_guard<std::mutex> lock(waveform_stream_mutex_);
    waveform_stream_queue_.push(std::move(event));
}

void GotstSpeechRuntime::clear_waveform_events() {
    std::lock_guard<std::mutex> lock(waveform_stream_mutex_);
    while(!waveform_stream_queue_.empty()) {
        waveform_stream_queue_.pop();
    }
}

void GotstSpeechRuntime::waveform_generation_worker_main() {
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    while(!waveform_workers_stop_.load(std::memory_order_acquire)) {
        WaveformStreamRequest request;
        {
            std::unique_lock<std::mutex> lock(waveform_request_mutex_);
            waveform_request_cv_.wait(lock, [&]() {
                return waveform_workers_stop_.load(std::memory_order_acquire) || !waveform_request_queue_.empty();
            });
            if(waveform_workers_stop_.load(std::memory_order_acquire)) {
                break;
            }
            request = std::move(waveform_request_queue_.front());
            waveform_request_queue_.pop_front();
            waveform_active_request_id_ = request.request_id;
            waveform_active_cancel_ = request.cancel;
            waveform_stream_active_.store(true, std::memory_order_release);
        }

        const auto stream_start = Clock::now();
        const double pipeline_queue_wait_ms = Ms(stream_start - request.queued_at).count();

        if(request.use_qwen_pipeline) {
            if(!qwen_tts_pipeline_ || !qwen_tts_pipeline_->is_loaded()) {
                WaveformStreamEvent event;
                event.request_id = request.request_id;
                event.is_error = true;
                event.error_message = "Qwen TTS pipeline is not loaded.";
                event.elapsed_ms = round_ms(Ms(Clock::now() - stream_start).count());
                event.pipeline_queue_wait_ms = round_ms(pipeline_queue_wait_ms);
                push_waveform_event(std::move(event));
                waveform_stream_active_.store(false, std::memory_order_release);
                continue;
            }

            auto result = qwen_tts_pipeline_->synthesize_streaming(
                request.qwen_request,
                request.chunk_frames,
                [this, &request, stream_start, pipeline_queue_wait_ms](const gotst::QwenTtsStreamChunk &chunk) {
                    if(request.cancel && request.cancel->is_cancelled()) {
                        return;
                    }
                    const auto pack_started = Clock::now();
                    PackedFloat32Array waveform = pack_float_array(chunk.waveform);
                    const double pack_ms = Ms(Clock::now() - pack_started).count();

                    WaveformStreamEvent event;
                    event.request_id = request.request_id;
                    event.waveform = std::move(waveform);
                    event.sample_rate = chunk.sample_rate;
                    event.frame_count = chunk.frame_count;
                    event.code_count = chunk.code_count;
                    event.chunk_index = chunk.chunk_index;
                    event.chunk_samples = static_cast<int32_t>(chunk.waveform.size());
                    event.is_final = chunk.is_final;
                    event.elapsed_ms = round_ms(Ms(Clock::now() - stream_start).count());
                    event.codegen_ms = round_ms(chunk.codegen_ms);
                    event.decoder_ms = round_ms(chunk.decoder_ms);
                    event.decoder_inference_ms = round_ms(chunk.decoder_inference_ms);
                    event.decoder_postprocess_ms = round_ms(chunk.decoder_postprocess_ms);
                    event.pipeline_queue_wait_ms = round_ms(pipeline_queue_wait_ms);
                    event.native_pack_ms = round_ms(pack_ms);
                    event.decoder_provider_requested = String(chunk.decoder_provider_requested.c_str());
                    event.decoder_provider_effective = String(chunk.decoder_provider_effective.c_str());
                    event.decoder_cpu_fallback_node_count = chunk.decoder_cpu_fallback_node_count;
                    event.decoder_fixed_shape = chunk.decoder_fixed_shape;
                    push_waveform_event(std::move(event));
                },
                request.cancel.get()
            );

            if(!result.is_ok()) {
                if(!request.cancel || !request.cancel->is_cancelled()) {
                    WaveformStreamEvent event;
                    event.request_id = request.request_id;
                    event.is_error = true;
                    event.error_message = String(result.error_message().c_str());
                    event.elapsed_ms = round_ms(Ms(Clock::now() - stream_start).count());
                    event.pipeline_queue_wait_ms = round_ms(pipeline_queue_wait_ms);
                    push_waveform_event(std::move(event));
                }
            } else {
                const auto &synth = result.value();
                WaveformStreamEvent event;
                event.request_id = request.request_id;
                event.is_stats = true;
                event.frame_count = synth.frame_count;
                event.code_count = synth.code_count;
                event.elapsed_ms = round_ms(synth.elapsed_ms);
                event.codegen_ms = round_ms(synth.codegen_ms);
                event.decoder_ms = round_ms(synth.decoder_ms);
                event.decoder_inference_ms = round_ms(synth.decoder_inference_ms);
                event.decoder_postprocess_ms = round_ms(synth.decoder_postprocess_ms);
                event.pipeline_queue_wait_ms = round_ms(pipeline_queue_wait_ms);
                event.talker_prefill_ms = round_ms(synth.talker_prefill_ms);
                event.talker_decode_ms = round_ms(synth.talker_decode_ms);
                event.predictor_ms = round_ms(synth.predictor_ms);
                event.onnx_embedding_ms = round_ms(synth.onnx_embedding_ms);
                event.codegen_other_ms = round_ms(synth.codegen_other_ms);
                push_waveform_event(std::move(event));
            }

            {
                std::lock_guard<std::mutex> lock(waveform_request_mutex_);
                if(waveform_active_request_id_ == request.request_id) {
                    waveform_active_request_id_ = 0;
                    waveform_active_cancel_.reset();
                }
            }
            if(!request.cancel || !request.cancel->is_cancelled()) {
                waveform_stream_active_.store(false, std::memory_order_release);
            }
            continue;
        }

        auto decoder_stream_result = tts_waveform_decoder_->create_stream();
        if(!decoder_stream_result.is_ok()) {
            WaveformStreamEvent event;
            event.request_id = request.request_id;
            event.is_error = true;
            event.error_message = String(decoder_stream_result.error_message().c_str());
            event.elapsed_ms = round_ms(Ms(Clock::now() - stream_start).count());
            event.pipeline_queue_wait_ms = round_ms(pipeline_queue_wait_ms);
            push_waveform_event(std::move(event));
            waveform_stream_active_.store(false, std::memory_order_release);
            continue;
        }

        std::shared_ptr<gotst::TtsWaveformDecoderStream> decoder_stream(
            std::move(decoder_stream_result.value())
        );

        int32_t chunk_index = 0;
        auto on_chunk = [
            this,
            &request,
            decoder_stream,
            stream_start,
            pipeline_queue_wait_ms,
            &chunk_index
        ](
            gotst::TtsFrameChunk chunk
        ) {
            if(request.cancel && request.cancel->is_cancelled()) {
                return;
            }

            WaveformDecodeJob job;
            job.request_id = request.request_id;
            job.decoder_stream = decoder_stream;
            job.cancel = request.cancel;
            job.codes = std::move(chunk.codes);
            job.frame_count = chunk.frame_count;
            job.codes_per_frame = chunk.codes_per_frame;
            job.chunk_index = ++chunk_index;
            job.is_final = chunk.is_final;
            job.codegen_ms = Ms(Clock::now() - stream_start).count();
            job.pipeline_queue_wait_ms = pipeline_queue_wait_ms;
            job.stream_start = stream_start;
            job.queued_at = Clock::now();
            {
                std::lock_guard<std::mutex> lock(waveform_decode_mutex_);
                waveform_decode_queue_.push_back(std::move(job));
            }
            waveform_decode_cv_.notify_one();
        };

        auto result = tts_code_generator_->generate_streaming(
            {request.initial_input.data(), request.initial_input.size()},
            request.initial_length,
            {request.trailing_hidden.data(), request.trailing_hidden.size()},
            request.trailing_length,
            {request.pad_embedding.data(), request.pad_embedding.size()},
            request.sampling,
            request.chunk_frames,
            on_chunk,
            request.cancel.get()
        );

        if(!result.is_ok()) {
            if(!request.cancel || !request.cancel->is_cancelled()) {
                WaveformStreamEvent event;
                event.request_id = request.request_id;
                event.is_error = true;
                event.error_message = String(result.error_message().c_str());
                event.elapsed_ms = round_ms(Ms(Clock::now() - stream_start).count());
                event.pipeline_queue_wait_ms = round_ms(pipeline_queue_wait_ms);
                push_waveform_event(std::move(event));
            }
        } else {
            const auto &gen = result.value();
            WaveformStreamEvent event;
            event.request_id = request.request_id;
            event.is_stats = true;
            event.frame_count = gen.frame_count;
            event.code_count = static_cast<int32_t>(gen.codes.size());
            event.elapsed_ms = round_ms(Ms(Clock::now() - stream_start).count());
            event.codegen_ms = round_ms(gen.elapsed_ms);
            event.pipeline_queue_wait_ms = round_ms(pipeline_queue_wait_ms);
            event.talker_prefill_ms = round_ms(gen.talker_prefill_ms);
            event.talker_decode_ms = round_ms(gen.talker_decode_ms);
            event.predictor_ms = round_ms(gen.predictor_ms);
            event.onnx_embedding_ms = round_ms(gen.onnx_embedding_ms);
            event.codegen_other_ms = round_ms(gen.other_ms);
            push_waveform_event(std::move(event));
        }

        {
            std::lock_guard<std::mutex> lock(waveform_request_mutex_);
            if(waveform_active_request_id_ == request.request_id) {
                waveform_active_request_id_ = 0;
                waveform_active_cancel_.reset();
            }
        }
        if(!request.cancel || !request.cancel->is_cancelled()) {
            waveform_stream_active_.store(false, std::memory_order_release);
        }
    }
}

void GotstSpeechRuntime::waveform_decoder_worker_main() {
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    while(!waveform_workers_stop_.load(std::memory_order_acquire)) {
        WaveformDecodeJob job;
        {
            std::unique_lock<std::mutex> lock(waveform_decode_mutex_);
            waveform_decode_cv_.wait(lock, [&]() {
                return waveform_workers_stop_.load(std::memory_order_acquire) || !waveform_decode_queue_.empty();
            });
            if(waveform_workers_stop_.load(std::memory_order_acquire)) {
                break;
            }
            job = std::move(waveform_decode_queue_.front());
            waveform_decode_queue_.pop_front();
        }

        if((job.cancel && job.cancel->is_cancelled()) || !job.decoder_stream) {
            continue;
        }

        const auto decode_started = Clock::now();
        const double queue_wait_ms = Ms(decode_started - job.queued_at).count();
        auto decoded_result = job.decoder_stream->decode(
            std::span<const int64_t>(job.codes.data(), job.codes.size()),
            job.frame_count,
            job.is_final
        );

        if(!decoded_result.is_ok()) {
            WaveformStreamEvent event;
            event.request_id = job.request_id;
            event.is_error = true;
            event.error_message = String(decoded_result.error_message().c_str());
            event.elapsed_ms = round_ms(Ms(Clock::now() - job.stream_start).count());
            event.codegen_ms = round_ms(job.codegen_ms);
            event.queue_wait_ms = round_ms(queue_wait_ms);
            event.pipeline_queue_wait_ms = round_ms(job.pipeline_queue_wait_ms);
            push_waveform_event(std::move(event));
            if(job.cancel) {
                job.cancel->cancel();
            }
            continue;
        }

        const auto &decoded = decoded_result.value();
        const auto pack_started = Clock::now();
        PackedFloat32Array waveform = pack_float_array(decoded.waveform);
        const double pack_ms = Ms(Clock::now() - pack_started).count();

        WaveformStreamEvent event;
        event.request_id = job.request_id;
        event.waveform = std::move(waveform);
        event.sample_rate = tts_waveform_decoder_sample_rate_;
        event.frame_count = job.frame_count;
        event.code_count = static_cast<int32_t>(job.codes.size());
        event.chunk_index = job.chunk_index;
        event.chunk_samples = static_cast<int32_t>(decoded.waveform.size());
        event.is_final = job.is_final;
        event.elapsed_ms = round_ms(Ms(Clock::now() - job.stream_start).count());
        event.codegen_ms = round_ms(job.codegen_ms);
        event.decoder_ms = round_ms(decoded.elapsed_ms);
        event.decoder_inference_ms = round_ms(decoded.inference_ms);
        event.decoder_postprocess_ms = round_ms(decoded.postprocess_ms);
        event.queue_wait_ms = round_ms(queue_wait_ms);
        event.pipeline_queue_wait_ms = round_ms(job.pipeline_queue_wait_ms);
        event.native_pack_ms = round_ms(pack_ms);
        event.decoder_provider_requested = String(decoded.provider_requested.c_str());
        event.decoder_provider_effective = String(decoded.provider_effective.c_str());
        event.decoder_cpu_fallback_node_count = decoded.cpu_fallback_node_count;
        event.decoder_fixed_shape = decoded.fixed_shape;
        push_waveform_event(std::move(event));
    }
}

Array GotstSpeechRuntime::poll_tts_waveform_stream() {
    Array events;
    std::lock_guard<std::mutex> lock(waveform_stream_mutex_);
    while(!waveform_stream_queue_.empty()) {
        WaveformStreamEvent &ev = waveform_stream_queue_.front();
        Dictionary d;
        d["request_id"] = ev.request_id;
        d["waveform"] = ev.waveform;
        d["sample_rate"] = ev.sample_rate;
        d["frame_count"] = ev.frame_count;
        d["code_count"] = ev.code_count;
        d["chunk_index"] = ev.chunk_index;
        d["chunk_samples"] = ev.chunk_samples;
        d["is_final"] = ev.is_final;
        d["is_stats"] = ev.is_stats;
        d["elapsed_ms"] = ev.elapsed_ms;
        d["codegen_ms"] = ev.codegen_ms;
        d["decoder_ms"] = ev.decoder_ms;
        d["decoder_inference_ms"] = ev.decoder_inference_ms;
        d["decoder_postprocess_ms"] = ev.decoder_postprocess_ms;
        d["queue_wait_ms"] = ev.queue_wait_ms;
        d["pipeline_queue_wait_ms"] = ev.pipeline_queue_wait_ms;
        d["native_pack_ms"] = ev.native_pack_ms;
        d["decoder_provider_requested"] = ev.decoder_provider_requested;
        d["decoder_provider_effective"] = ev.decoder_provider_effective;
        d["decoder_cpu_fallback_node_count"] = ev.decoder_cpu_fallback_node_count;
        d["decoder_fixed_shape"] = ev.decoder_fixed_shape;
        d["talker_prefill_ms"] = ev.talker_prefill_ms;
        d["talker_decode_ms"] = ev.talker_decode_ms;
        d["predictor_ms"] = ev.predictor_ms;
        d["onnx_embedding_ms"] = ev.onnx_embedding_ms;
        d["codegen_other_ms"] = ev.codegen_other_ms;
        if(ev.is_error) {
            d["error"] = ev.error_message;
        }
        events.push_back(d);
        waveform_stream_queue_.pop();
    }
    return events;
}

void GotstSpeechRuntime::cancel_tts_waveform_stream(int64_t request_id) {
    std::shared_ptr<gotst::CancellationToken> active_cancel;
    {
        std::lock_guard<std::mutex> lock(waveform_request_mutex_);
        if(request_id <= 0 || waveform_active_request_id_ == request_id) {
            active_cancel = waveform_active_cancel_;
        }

        for(auto it = waveform_request_queue_.begin(); it != waveform_request_queue_.end();) {
            if(request_id <= 0 || it->request_id == request_id) {
                if(it->cancel) {
                    it->cancel->cancel();
                }
                it = waveform_request_queue_.erase(it);
            } else {
                ++it;
            }
        }
    }
    if(active_cancel) {
        active_cancel->cancel();
    }
    {
        std::lock_guard<std::mutex> lock(waveform_decode_mutex_);
        for(auto &job : waveform_decode_queue_) {
            if(request_id <= 0 || job.request_id == request_id) {
                if(job.cancel) {
                    job.cancel->cancel();
                }
            }
        }
    }
    waveform_decode_cv_.notify_all();
    waveform_stream_active_.store(false, std::memory_order_release);
}

bool GotstSpeechRuntime::load_tts_waveform_decoder(const Dictionary &config) {
    stop_waveform_stream_workers();

    gotst::TtsWaveformDecoderConfig decoder_config;
    decoder_config.decoder_onnx_path = String(config.get("decoder_onnx_path", "")).utf8().get_data();
    decoder_config.provider_requested = String(config.get("provider_requested", "")).utf8().get_data();
    decoder_config.provider = String(config.get("provider", "CPU")).utf8().get_data();
    decoder_config.intra_op_threads = static_cast<int32_t>(config.get("intra_op_threads", 0));
    decoder_config.inter_op_threads = static_cast<int32_t>(config.get("inter_op_threads", 0));
    decoder_config.optimization_level = static_cast<int32_t>(config.get("optimization_level", 99));
    decoder_config.optimized_model_path = String(config.get("optimized_model_path", "")).utf8().get_data();
    decoder_config.sample_rate = static_cast<int32_t>(config.get("sample_rate", 24000));
    decoder_config.normalize_waveform = static_cast<bool>(config.get("normalize_waveform", false));
    decoder_config.waveform_gain = static_cast<float>(config.get("waveform_gain", 1.0));
    decoder_config.stateful_chunk_frames = static_cast<int32_t>(config.get("stateful_chunk_frames", 12));

    tts_waveform_decoder_ = std::make_unique<gotst::TtsWaveformDecoder>();
    auto result = tts_waveform_decoder_->load(decoder_config);
    if(!result.is_ok()) {
        ERR_PRINT(String("TTS waveform decoder load failed: ") + String(result.error_message().c_str()));
        tts_waveform_decoder_.reset();
        return false;
    }
    tts_waveform_decoder_sample_rate_ = decoder_config.sample_rate;
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
    output["provider_requested"] = String(decoded.provider_requested.c_str());
    output["provider_effective"] = String(decoded.provider_effective.c_str());
    output["cpu_fallback_node_count"] = decoded.cpu_fallback_node_count;
    output["fixed_shape"] = decoded.fixed_shape;
    output["elapsed_ms"] = static_cast<int64_t>(std::llround(decoded.elapsed_ms));
    output["inference_ms"] = static_cast<int64_t>(std::llround(decoded.inference_ms));
    output["postprocess_ms"] = static_cast<int64_t>(std::llround(decoded.postprocess_ms));
    output["sample_count"] = decoded.sample_count;
    output["frame_count"] = decoded.frame_count;
    output["codes_per_frame"] = decoded.codes_per_frame;
    return output;
}

bool GotstSpeechRuntime::load_irodori_tts(const Dictionary &config) {
    stop_irodori_stream_worker();

    gotst::IrodoriTtsSessionConfig session_config = irodori_config_from_dictionary(config);
    irodori_tts_session_ = std::make_unique<gotst::IrodoriTtsSession>();
    auto result = irodori_tts_session_->load(session_config);
    if(!result.is_ok()) {
        ERR_PRINT(String("Irodori TTS load failed: ") + String(result.error_message().c_str()));
        irodori_tts_session_.reset();
        return false;
    }

    return true;
}

bool GotstSpeechRuntime::is_irodori_tts_loaded() const {
    return irodori_tts_session_ && irodori_tts_session_->is_loaded();
}

bool GotstSpeechRuntime::load_irodori_tokenizer(
    const String &tokenizer_json_path,
    const String &tokenizer_config_path
) {
    irodori_text_tokenizer_ = std::make_unique<gotst::IrodoriTextTokenizer>();
    auto result = irodori_text_tokenizer_->load(
        std::string(tokenizer_json_path.utf8().get_data()),
        std::string(tokenizer_config_path.utf8().get_data())
    );
    if(!result.is_ok()) {
        ERR_PRINT(String("Irodori tokenizer load failed: ") + String(result.error_message().c_str()));
        irodori_text_tokenizer_.reset();
        return false;
    }
    return true;
}

String GotstSpeechRuntime::normalize_irodori_text(const String &text) const {
    const std::string normalized = gotst::normalize_irodori_v3_text(
        std::string(text.utf8().get_data())
    );
    return String::utf8(normalized.c_str());
}

Dictionary GotstSpeechRuntime::tokenize_irodori_text(
    const String &text,
    int64_t max_tokens,
    bool force_empty_mask
) const {
    Dictionary output;
    if(!irodori_text_tokenizer_ || !irodori_text_tokenizer_->is_loaded()) {
        output["error"] = "Irodori tokenizer is not loaded.";
        return output;
    }
    const std::string normalized = gotst::normalize_irodori_v3_text(
        std::string(text.utf8().get_data())
    );
    auto result = irodori_text_tokenizer_->encode(
        normalized,
        static_cast<int32_t>(std::max<int64_t>(0, max_tokens)),
        force_empty_mask
    );
    if(!result.is_ok()) {
        output["error"] = String(result.error_message().c_str());
        return output;
    }
    const gotst::IrodoriTokenizedText &tokens = result.value();
    PackedInt64Array token_ids;
    token_ids.resize(static_cast<int64_t>(tokens.token_ids.size()));
    if(!tokens.token_ids.empty()) {
        std::memcpy(token_ids.ptrw(), tokens.token_ids.data(), tokens.token_ids.size() * sizeof(int64_t));
    }
    PackedInt64Array token_mask;
    token_mask.resize(static_cast<int64_t>(tokens.token_mask.size()));
    for(size_t index = 0; index < tokens.token_mask.size(); ++index) {
        token_mask.set(static_cast<int64_t>(index), tokens.token_mask[index] != 0 ? 1 : 0);
    }
    output["normalized_text"] = String::utf8(normalized.c_str());
    output["token_ids"] = token_ids;
    output["token_mask"] = token_mask;
    return output;
}

void GotstSpeechRuntime::push_irodori_event(IrodoriStreamEvent event) {
    std::lock_guard<std::mutex> lock(irodori_stream_mutex_);
    irodori_stream_queue_.push(std::move(event));
}

void GotstSpeechRuntime::clear_irodori_events() {
    std::lock_guard<std::mutex> lock(irodori_stream_mutex_);
    while(!irodori_stream_queue_.empty()) {
        irodori_stream_queue_.pop();
    }
}

void GotstSpeechRuntime::stop_irodori_stream_worker() {
    if(irodori_active_cancel_) {
        irodori_active_cancel_->cancel();
    }
    if(irodori_worker_.joinable()) {
        irodori_worker_.join();
    }
    irodori_stream_active_.store(false, std::memory_order_release);
    irodori_active_request_id_ = 0;
    irodori_active_cancel_.reset();
    clear_irodori_events();
}

Dictionary GotstSpeechRuntime::start_irodori_tts_stream(const Dictionary &params, int64_t request_id) {
    Dictionary output;
    if(!irodori_tts_session_ || !irodori_tts_session_->is_loaded()) {
        output["error"] = "Irodori TTS is not loaded.";
        return output;
    }
    if(request_id <= 0) {
        output["error"] = "Irodori TTS request_id must be positive.";
        return output;
    }
    if(irodori_stream_active_.load(std::memory_order_acquire)) {
        output["error"] = "Irodori TTS stream is already active.";
        return output;
    }
    if(irodori_worker_.joinable()) {
        irodori_worker_.join();
    }

    clear_irodori_events();
    gotst::IrodoriTtsRequest request = irodori_request_from_dictionary(params);
    auto cancel = std::make_shared<gotst::CancellationToken>();
    irodori_active_request_id_ = request_id;
    irodori_active_cancel_ = cancel;
    irodori_stream_active_.store(true, std::memory_order_release);

    irodori_worker_ = std::thread([this, request, request_id, cancel]() {
        using Clock = std::chrono::steady_clock;
        using Ms = std::chrono::duration<double, std::milli>;

        const auto started = Clock::now();
        IrodoriStreamEvent event;
        event.request_id = request_id;
        event.sample_rate = 48000;

        if(!irodori_tts_session_) {
            event.is_error = true;
            event.error_message = "Irodori TTS session is not available.";
            event.elapsed_ms = round_ms(Ms(Clock::now() - started).count());
            push_irodori_event(std::move(event));
            irodori_stream_active_.store(false, std::memory_order_release);
            return;
        }

        auto result = irodori_tts_session_->synthesize(request, cancel.get());
        event.elapsed_ms = round_ms(Ms(Clock::now() - started).count());
        if(!result.is_ok()) {
            const bool cancelled = cancel && cancel->is_cancelled();
            event.is_cancelled = cancelled;
            event.is_error = !cancelled;
            event.error_message = String(result.error_message().c_str());
            event.mode = String(gotst::irodori_tts_mode_name(irodori_tts_session_->config().mode).c_str());
            event.provider_profile = String(irodori_tts_session_->config().provider_profile.c_str());
            event.provider_requested = String(irodori_tts_session_->config().provider_requested.c_str());
            event.provider_effective = String(irodori_tts_session_->config().provider.c_str());
            push_irodori_event(std::move(event));
            irodori_stream_active_.store(false, std::memory_order_release);
            return;
        }

        const auto &synth = result.value();
        event.waveform = pack_float_array(synth.waveform);
        event.sample_rate = synth.sample_rate;
        event.frame_count = synth.frame_count;
        event.is_final = true;
        event.mode = String(synth.mode.c_str());
        event.selected_bucket = String(synth.selected_bucket.c_str());
        event.selected_bucket_latent_steps = synth.selected_bucket_latent_steps;
        event.provider_profile = String(synth.provider_profile.c_str());
        event.provider_requested = String(synth.provider_requested.c_str());
        event.provider_effective = String(synth.provider_effective.c_str());
        event.provider_requested_by_stage = string_map_to_dictionary(synth.provider_requested_by_stage);
        event.provider_effective_by_stage = string_map_to_dictionary(synth.provider_effective_by_stage);
        event.cache_hit = synth.cache_hit;
        event.timings_ms = irodori_timings_to_dictionary(synth.timings_ms);
        event.instrumentation_ms = irodori_timings_to_dictionary(synth.instrumentation_ms);
        event.diagnostics = string_map_to_dictionary(synth.diagnostics);
        push_irodori_event(std::move(event));
        irodori_stream_active_.store(false, std::memory_order_release);
    });

    output["started"] = true;
    output["request_id"] = request_id;
    return output;
}

Array GotstSpeechRuntime::poll_irodori_tts_stream() {
    Array events;
    {
        std::lock_guard<std::mutex> lock(irodori_stream_mutex_);
        while(!irodori_stream_queue_.empty()) {
            IrodoriStreamEvent &ev = irodori_stream_queue_.front();
            Dictionary d;
            d["request_id"] = ev.request_id;
            d["waveform"] = ev.waveform;
            d["sample_rate"] = ev.sample_rate;
            d["frame_count"] = ev.frame_count;
            d["is_final"] = ev.is_final;
            d["is_cancelled"] = ev.is_cancelled;
            d["elapsed_ms"] = ev.elapsed_ms;
            d["mode"] = ev.mode;
            d["selected_bucket"] = ev.selected_bucket;
            d["selected_bucket_latent_steps"] = ev.selected_bucket_latent_steps;
            d["provider_profile"] = ev.provider_profile;
            d["provider_requested"] = ev.provider_requested;
            d["provider_effective"] = ev.provider_effective;
            d["provider_requested_by_stage"] = ev.provider_requested_by_stage;
            d["provider_effective_by_stage"] = ev.provider_effective_by_stage;
            d["cache_hit"] = ev.cache_hit;
            d["timings_ms"] = ev.timings_ms;
            d["instrumentation_ms"] = ev.instrumentation_ms;
            d["diagnostics"] = ev.diagnostics;
            if(ev.is_error) {
                d["error"] = ev.error_message;
            }
            if(ev.is_cancelled) {
                d["cancelled"] = true;
            }
            events.push_back(d);
            irodori_stream_queue_.pop();
        }
    }

    if(!irodori_stream_active_.load(std::memory_order_acquire) && irodori_worker_.joinable()) {
        irodori_worker_.join();
        irodori_active_request_id_ = 0;
        irodori_active_cancel_.reset();
    }

    return events;
}

void GotstSpeechRuntime::cancel_irodori_tts_stream(int64_t request_id) {
    if(request_id <= 0 || irodori_active_request_id_ == request_id) {
        if(irodori_active_cancel_) {
            irodori_active_cancel_->cancel();
        }
    }
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
    output["elapsed_ms"] = static_cast<int64_t>(std::llround(decoded.elapsed_ms));
    output["prefill_ms"] = static_cast<int64_t>(std::llround(decoded.prefill_ms));
    output["decode_ms"] = static_cast<int64_t>(std::llround(decoded.decode_ms));
    output["onnx_embedding_ms"] = static_cast<int64_t>(std::llround(decoded.onnx_embedding_ms));
    return output;
}

bool GotstSpeechRuntime::load_qwen3_forced_aligner(const Dictionary &config) {
    gotst::Qwen3ForcedAlignerModelPaths paths = forced_aligner_paths_from_dictionary(config);
    gotst::Qwen3ForcedAlignerSessionConfig session_config = forced_aligner_config_from_dictionary(config);

    qwen3_forced_aligner_ = std::make_unique<gotst::Qwen3ForcedAligner>();
    auto result = qwen3_forced_aligner_->load(paths, session_config);
    if(!result.is_ok()) {
        ERR_PRINT(String("Qwen3 forced-aligner load failed: ") + String(result.error_message().c_str()));
        qwen3_forced_aligner_.reset();
        return false;
    }
    return true;
}

bool GotstSpeechRuntime::is_qwen3_forced_aligner_loaded() const {
    return qwen3_forced_aligner_ && qwen3_forced_aligner_->is_loaded();
}

void GotstSpeechRuntime::push_forced_alignment_event(ForcedAlignmentEvent event) {
    std::lock_guard<std::mutex> lock(forced_alignment_mutex_);
    forced_alignment_queue_.push(std::move(event));
}

void GotstSpeechRuntime::clear_forced_alignment_events() {
    std::lock_guard<std::mutex> lock(forced_alignment_mutex_);
    while(!forced_alignment_queue_.empty()) {
        forced_alignment_queue_.pop();
    }
}

void GotstSpeechRuntime::stop_forced_alignment_worker() {
    if(forced_alignment_active_cancel_) {
        forced_alignment_active_cancel_->cancel();
    }
    if(forced_alignment_worker_.joinable()) {
        forced_alignment_worker_.join();
    }
    forced_alignment_active_.store(false, std::memory_order_release);
    forced_alignment_active_request_id_ = 0;
    forced_alignment_active_cancel_.reset();
    clear_forced_alignment_events();
}

Dictionary GotstSpeechRuntime::start_qwen3_forced_alignment(const Dictionary &params, int64_t request_id) {
    Dictionary output;
    if(!qwen3_forced_aligner_ || !qwen3_forced_aligner_->is_loaded()) {
        output["error"] = "Qwen3 forced aligner is not loaded.";
        return output;
    }
    if(request_id <= 0) {
        output["error"] = "Qwen3 forced alignment request_id must be positive.";
        return output;
    }
    if(forced_alignment_active_.load(std::memory_order_acquire)) {
        output["error"] = "Qwen3 forced alignment is already active.";
        return output;
    }
    if(forced_alignment_worker_.joinable()) {
        forced_alignment_worker_.join();
    }
    clear_forced_alignment_events();

    gotst::Qwen3ForcedAlignmentRequest request = forced_alignment_request_from_dictionary(params);
    auto validation = gotst::validate_qwen3_forced_alignment_request(
        request,
        qwen3_forced_aligner_->config()
    );
    if(!validation.is_ok()) {
        output["error"] = String(validation.error_message().c_str());
        return output;
    }

    auto cancel = std::make_shared<gotst::CancellationToken>();
    forced_alignment_active_cancel_ = cancel;
    forced_alignment_active_request_id_ = request_id;
    forced_alignment_active_.store(true, std::memory_order_release);
    forced_alignment_worker_ = std::thread([this, request_id, request = std::move(request), cancel]() mutable {
        ForcedAlignmentEvent event;
        event.request_id = request_id;
        auto result = qwen3_forced_aligner_->align(request, cancel.get());
        if(!result.is_ok()) {
            if(result.error_code() == gotst::ErrorCode::Cancelled) {
                event.is_cancelled = true;
            } else {
                event.is_error = true;
                event.error_message = String(result.error_message().c_str());
            }
            push_forced_alignment_event(std::move(event));
            forced_alignment_active_.store(false, std::memory_order_release);
            return;
        }

        event.is_completed = true;
        event.spans = pack_forced_alignment_spans(result.value().spans);
        event.timings_ms = forced_alignment_timings_to_dictionary(result.value());
        event.token_count = result.value().token_count;
        event.audio_token_count = result.value().audio_token_count;
        push_forced_alignment_event(std::move(event));
        forced_alignment_active_.store(false, std::memory_order_release);
    });

    output["started"] = true;
    output["request_id"] = request_id;
    return output;
}

Array GotstSpeechRuntime::poll_qwen3_forced_alignment() {
    Array events;
    {
        std::lock_guard<std::mutex> lock(forced_alignment_mutex_);
        while(!forced_alignment_queue_.empty()) {
            ForcedAlignmentEvent event = std::move(forced_alignment_queue_.front());
            forced_alignment_queue_.pop();

            Dictionary payload;
            payload["request_id"] = event.request_id;
            if(event.is_completed) {
                payload["type"] = "completed";
                payload["spans"] = event.spans;
                payload["timings_ms"] = event.timings_ms;
                payload["token_count"] = event.token_count;
                payload["audio_token_count"] = event.audio_token_count;
            } else if(event.is_cancelled) {
                payload["type"] = "cancelled";
            } else {
                payload["type"] = "error";
                payload["error"] = event.error_message;
            }
            events.push_back(payload);
        }
    }

    if(!forced_alignment_active_.load(std::memory_order_acquire) && forced_alignment_worker_.joinable()) {
        forced_alignment_worker_.join();
        forced_alignment_active_request_id_ = 0;
        forced_alignment_active_cancel_.reset();
    }
    return events;
}

void GotstSpeechRuntime::cancel_qwen3_forced_alignment(int64_t request_id) {
    if(request_id <= 0 || forced_alignment_active_request_id_ == request_id) {
        if(forced_alignment_active_cancel_) {
            forced_alignment_active_cancel_->cancel();
        }
    }
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
