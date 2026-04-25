#include "gotst/core/tts_waveform_decoder.hpp"

#include <gonx/core/session.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gotst {

namespace {

struct DecoderLayout {
    bool has_layout = false;
    int64_t time_steps = 0;
    int64_t time_stride = 0;
    int64_t channel_stride = 0;
    int64_t channel_count = 1;
};

DecoderLayout resolve_decoder_output_layout(std::span<const int64_t> shape, int64_t output_length) {
    if(shape.size() < 2 || output_length <= 0) {
        return {
            .has_layout = false,
            .time_steps = output_length,
            .time_stride = 0,
            .channel_stride = 0,
            .channel_count = 1,
        };
    }

    std::vector<int64_t> dims(shape.begin(), shape.end());
    std::vector<int64_t> strides(dims.size(), 1);
    int64_t running_stride = 1;
    for(int64_t index = static_cast<int64_t>(dims.size()) - 1; index >= 0; --index) {
        strides[static_cast<size_t>(index)] = running_stride;
        running_stride *= std::max<int64_t>(dims[static_cast<size_t>(index)], 1);
    }

    const int64_t time_axis = std::max<int64_t>(0, static_cast<int64_t>(dims.size()) - 1);
    const int64_t time_steps = std::max<int64_t>(1, dims[static_cast<size_t>(time_axis)]);
    int64_t channel_axis = -1;
    int64_t channel_count = 1;
    for(int64_t axis = 0; axis < time_axis; ++axis) {
        if(dims[static_cast<size_t>(axis)] > 1) {
            channel_axis = axis;
            channel_count = dims[static_cast<size_t>(axis)];
            break;
        }
    }

    return {
        .has_layout = true,
        .time_steps = time_steps,
        .time_stride = strides[static_cast<size_t>(time_axis)],
        .channel_stride = channel_axis >= 0 ? strides[static_cast<size_t>(channel_axis)] : 0,
        .channel_count = channel_count,
    };
}

std::vector<float> convert_decoder_output_to_waveform(
    std::span<const float> output,
    std::span<const int64_t> output_shape,
    int32_t sample_rate,
    bool normalize_waveform,
    float waveform_gain
) {
    if(output.empty()) {
        return {};
    }

    const DecoderLayout layout = resolve_decoder_output_layout(output_shape, static_cast<int64_t>(output.size()));
    const int64_t sample_count = layout.time_steps;
    if(sample_count <= 0) {
        return {};
    }

    std::vector<float> waveform(static_cast<size_t>(sample_count), 0.0f);
    double sum = 0.0;
    double peak = 0.0;

    if(!layout.has_layout) {
        for(int64_t index = 0; index < sample_count; ++index) {
            const double sample = std::clamp<double>(output[static_cast<size_t>(index)], -1.5, 1.5) * waveform_gain;
            waveform[static_cast<size_t>(index)] = static_cast<float>(sample);
            sum += sample;
            peak = std::max(peak, std::abs(sample));
        }
    } else {
        const bool downmix = layout.channel_count > 1 && layout.channel_stride > 0;
        for(int64_t time_index = 0; time_index < sample_count; ++time_index) {
            double raw = 0.0;
            const int64_t base_index = time_index * layout.time_stride;
            if(!downmix) {
                if(base_index >= 0 && base_index < static_cast<int64_t>(output.size())) {
                    raw = output[static_cast<size_t>(base_index)];
                }
            } else {
                for(int64_t channel_index = 0; channel_index < layout.channel_count; ++channel_index) {
                    const int64_t sample_index = base_index + (channel_index * layout.channel_stride);
                    if(sample_index >= 0 && sample_index < static_cast<int64_t>(output.size())) {
                        raw += output[static_cast<size_t>(sample_index)];
                    }
                }
                raw /= static_cast<double>(std::max<int64_t>(layout.channel_count, 1));
            }
            const double sample = std::clamp<double>(raw, -1.5, 1.5) * waveform_gain;
            waveform[static_cast<size_t>(time_index)] = static_cast<float>(sample);
            sum += sample;
            peak = std::max(peak, std::abs(sample));
        }
    }

    const double dc_offset = sum / static_cast<double>(std::max<int64_t>(sample_count, 1));
    for(float &sample : waveform) {
        sample -= static_cast<float>(dc_offset);
    }

    peak = 0.0;
    for(float sample : waveform) {
        peak = std::max(peak, std::abs(static_cast<double>(sample)));
    }

    if(normalize_waveform && peak > 0.00001) {
        const double normalize_scale = std::min(1.0, 0.95 / peak);
        for(float &sample : waveform) {
            sample = static_cast<float>(sample * normalize_scale);
        }
    }

    for(float &sample : waveform) {
        sample = std::clamp(sample, -1.0f, 1.0f);
    }

    const int64_t edge_fade = std::clamp<int64_t>(
        static_cast<int64_t>(std::llround(0.004 * static_cast<double>(sample_rate))),
        8,
        std::max<int64_t>(8, sample_count / 10)
    );
    const int64_t safe_edge_fade = std::min<int64_t>(edge_fade, sample_count / 2);
    for(int64_t index = 0; index < safe_edge_fade; ++index) {
        const double scale = static_cast<double>(index) / static_cast<double>(std::max<int64_t>(safe_edge_fade, 1));
        waveform[static_cast<size_t>(index)] =
            static_cast<float>(waveform[static_cast<size_t>(index)] * scale);
        const int64_t right_index = sample_count - 1 - index;
        waveform[static_cast<size_t>(right_index)] =
            static_cast<float>(waveform[static_cast<size_t>(right_index)] * scale);
    }

    return waveform;
}

} // namespace

struct StatefulTensor {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

struct StatefulDecodeState {
    StatefulTensor pre_conv_history;
    StatefulTensor latent_buffer;
    StatefulTensor conv_history;
    std::vector<StatefulTensor> keys;
    std::vector<StatefulTensor> values;
};

enum class StatefulInputKind {
    AudioCodes,
    PreConvHistory,
    LatentBuffer,
    ConvHistory,
    IsLast,
    PastKey,
    PastValue,
    Unsupported,
};

struct StatefulInputBinding {
    StatefulInputKind kind = StatefulInputKind::Unsupported;
    int32_t layer_index = -1;
    std::string name;
};

struct StatefulDecodeScratch {
    std::vector<int64_t> chunk_codes;
    std::vector<int64_t> chunk_shape = {1, 0, 0};
    std::vector<int64_t> is_last_shape = {1};
    std::vector<Ort::Value> inputs;
    std::vector<float> raw_waveform;
};

int32_t parse_stateful_index(const std::string &name, std::string_view prefix) {
    if(name.rfind(std::string(prefix), 0) != 0) {
        return -1;
    }
    try {
        return static_cast<int32_t>(std::stoi(name.substr(prefix.size())));
    } catch(...) {
        return -1;
    }
}

std::string copy_float_output(Ort::Value &value, StatefulTensor &target, std::string_view name) {
    auto info = value.GetTensorTypeAndShapeInfo();
    if(info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return "stateful decoder output '" + std::string(name) + "' was not FLOAT.";
    }
    const size_t element_count = info.GetElementCount();
    const float *data = value.GetTensorData<float>();
    target.shape = info.GetShape();
    target.data.resize(element_count);
    if(element_count > 0) {
        std::copy_n(data, element_count, target.data.begin());
    }
    return {};
}

void reset_stateful_decode_state(
    StatefulDecodeState &state,
    int32_t num_layers,
    int32_t num_heads,
    int32_t head_dim
) {
    state.pre_conv_history.data.clear();
    state.pre_conv_history.shape = {1, 512, 0};
    state.latent_buffer.data.clear();
    state.latent_buffer.shape = {1, 1024, 0};
    state.conv_history.data.clear();
    state.conv_history.shape = {1, 1024, 0};
    state.keys.resize(static_cast<size_t>(num_layers));
    state.values.resize(static_cast<size_t>(num_layers));
    for(int32_t layer_index = 0; layer_index < num_layers; ++layer_index) {
        StatefulTensor &key = state.keys[static_cast<size_t>(layer_index)];
        StatefulTensor &value = state.values[static_cast<size_t>(layer_index)];
        key.data.clear();
        value.data.clear();
        key.shape = {1, num_heads, 0, head_dim};
        value.shape = {1, num_heads, 0, head_dim};
    }
}

struct TtsWaveformDecoder::Impl {
    gonx::InferenceSession session;
    TtsWaveformDecoderConfig config;
    bool loaded = false;
    bool stateful_decoder = false;
    int32_t stateful_num_layers = 0;
    int32_t stateful_num_heads = 16;
    int32_t stateful_head_dim = 64;
    int64_t input_time_axis = 1;
    int64_t input_quantizer_axis = 2;
    int64_t input_quantizer_count = -1;
    std::vector<StatefulInputBinding> stateful_input_bindings;
    int32_t stateful_final_wav_output = -1;
    int32_t stateful_valid_samples_output = -1;
    int32_t stateful_next_pre_conv_output = -1;
    int32_t stateful_next_latent_output = -1;
    int32_t stateful_next_conv_output = -1;
    std::vector<int32_t> stateful_next_key_outputs;
    std::vector<int32_t> stateful_next_value_outputs;
};

TtsWaveformDecoder::TtsWaveformDecoder() : impl_(std::make_unique<Impl>()) {}
TtsWaveformDecoder::~TtsWaveformDecoder() = default;
TtsWaveformDecoder::TtsWaveformDecoder(TtsWaveformDecoder &&) noexcept = default;
TtsWaveformDecoder &TtsWaveformDecoder::operator=(TtsWaveformDecoder &&) noexcept = default;

Result<void> TtsWaveformDecoder::load(const TtsWaveformDecoderConfig &config) {
    impl_->loaded = false;
    impl_->config = config;

    if(config.decoder_onnx_path.empty()) {
        return Error::invalid_argument("TtsWaveformDecoder::load: empty model path");
    }
    if(config.sample_rate <= 0) {
        return Error::invalid_argument("TtsWaveformDecoder::load: invalid sample rate");
    }

    gonx::SessionConfig session_config;
    session_config.providers = {gonx::parse_provider(config.provider)};
    session_config.intra_op_num_threads = config.intra_op_threads;
    session_config.inter_op_num_threads = config.inter_op_threads;
    session_config.optimization_level = config.optimization_level;
    session_config.optimized_model_path = config.optimized_model_path;

    auto status = impl_->session.load(config.decoder_onnx_path, session_config);
    if(status.has_error()) {
        return Error::model_not_loaded(
            "TtsWaveformDecoder::load: failed to load model from " + config.decoder_onnx_path + ": " +
            status.error().message
        );
    }

    impl_->input_time_axis = 1;
    impl_->input_quantizer_axis = 2;
    impl_->input_quantizer_count = -1;
    const auto &input_specs = impl_->session.input_specs();
    if(!input_specs.empty()) {
        const auto &shape = input_specs.front().shape;
        if(shape.size() >= 3) {
            const int64_t dim1 = shape[1];
            const int64_t dim2 = shape[2];
            const bool dim1_static = dim1 > 0;
            const bool dim2_static = dim2 > 0;
            if(dim1_static && !dim2_static) {
                impl_->input_quantizer_axis = 1;
                impl_->input_time_axis = 2;
                impl_->input_quantizer_count = dim1;
            } else if(!dim1_static && dim2_static) {
                impl_->input_quantizer_axis = 2;
                impl_->input_time_axis = 1;
                impl_->input_quantizer_count = dim2;
            } else if(dim1_static && dim2_static) {
                if(dim1 <= dim2) {
                    impl_->input_quantizer_axis = 1;
                    impl_->input_time_axis = 2;
                    impl_->input_quantizer_count = dim1;
                } else {
                    impl_->input_quantizer_axis = 2;
                    impl_->input_time_axis = 1;
                    impl_->input_quantizer_count = dim2;
                }
            }
        }
    }

    bool has_audio_codes = false;
    bool has_pre_conv_history = false;
    bool has_latent_buffer = false;
    bool has_conv_history = false;
    bool has_is_last = false;
    int32_t past_key_count = 0;
    int32_t past_value_count = 0;
    impl_->stateful_num_heads = 16;
    impl_->stateful_head_dim = 64;
    impl_->stateful_input_bindings.clear();
    impl_->stateful_input_bindings.reserve(input_specs.size());
    for(const auto &spec : input_specs) {
        StatefulInputBinding binding;
        binding.name = spec.name;
        if(spec.name == "audio_codes") {
            has_audio_codes = true;
            binding.kind = StatefulInputKind::AudioCodes;
        } else if(spec.name == "pre_conv_history") {
            has_pre_conv_history = true;
            binding.kind = StatefulInputKind::PreConvHistory;
        } else if(spec.name == "latent_buffer") {
            has_latent_buffer = true;
            binding.kind = StatefulInputKind::LatentBuffer;
        } else if(spec.name == "conv_history") {
            has_conv_history = true;
            binding.kind = StatefulInputKind::ConvHistory;
        } else if(spec.name == "is_last") {
            has_is_last = true;
            binding.kind = StatefulInputKind::IsLast;
        } else if(spec.name.rfind("past_key_", 0) == 0) {
            past_key_count += 1;
            binding.kind = StatefulInputKind::PastKey;
            binding.layer_index = parse_stateful_index(spec.name, "past_key_");
            if(spec.shape.size() == 4) {
                if(spec.shape[1] > 0) {
                    impl_->stateful_num_heads = static_cast<int32_t>(spec.shape[1]);
                }
                if(spec.shape[3] > 0) {
                    impl_->stateful_head_dim = static_cast<int32_t>(spec.shape[3]);
                }
            }
        } else if(spec.name.rfind("past_value_", 0) == 0) {
            past_value_count += 1;
            binding.kind = StatefulInputKind::PastValue;
            binding.layer_index = parse_stateful_index(spec.name, "past_value_");
        }
        impl_->stateful_input_bindings.push_back(std::move(binding));
    }
    impl_->stateful_decoder =
        has_audio_codes &&
        has_pre_conv_history &&
        has_latent_buffer &&
        has_conv_history &&
        has_is_last &&
        past_key_count > 0 &&
        past_key_count == past_value_count;
    impl_->stateful_num_layers = impl_->stateful_decoder ? past_key_count : 0;
    impl_->stateful_final_wav_output = -1;
    impl_->stateful_valid_samples_output = -1;
    impl_->stateful_next_pre_conv_output = -1;
    impl_->stateful_next_latent_output = -1;
    impl_->stateful_next_conv_output = -1;
    impl_->stateful_next_key_outputs.assign(static_cast<size_t>(impl_->stateful_num_layers), -1);
    impl_->stateful_next_value_outputs.assign(static_cast<size_t>(impl_->stateful_num_layers), -1);
    if(impl_->stateful_decoder) {
        const auto &output_specs = impl_->session.output_specs();
        for(int32_t index = 0; index < static_cast<int32_t>(output_specs.size()); ++index) {
            const std::string &name = output_specs[static_cast<size_t>(index)].name;
            if(name == "final_wav") {
                impl_->stateful_final_wav_output = index;
            } else if(name == "valid_samples") {
                impl_->stateful_valid_samples_output = index;
            } else if(name == "next_pre_conv_history") {
                impl_->stateful_next_pre_conv_output = index;
            } else if(name == "next_latent_buffer") {
                impl_->stateful_next_latent_output = index;
            } else if(name == "next_conv_history") {
                impl_->stateful_next_conv_output = index;
            } else if(name.rfind("next_key_", 0) == 0) {
                const int32_t layer_index = parse_stateful_index(name, "next_key_");
                if(layer_index >= 0 && layer_index < impl_->stateful_num_layers) {
                    impl_->stateful_next_key_outputs[static_cast<size_t>(layer_index)] = index;
                }
            } else if(name.rfind("next_value_", 0) == 0) {
                const int32_t layer_index = parse_stateful_index(name, "next_value_");
                if(layer_index >= 0 && layer_index < impl_->stateful_num_layers) {
                    impl_->stateful_next_value_outputs[static_cast<size_t>(layer_index)] = index;
                }
            }
        }
    }

    impl_->loaded = true;
    return {};
}

bool TtsWaveformDecoder::is_loaded() const {
    return impl_ && impl_->loaded && impl_->session.is_loaded();
}

Result<TtsWaveformDecodeResult> TtsWaveformDecoder::decode(
    std::span<const int64_t> audio_codes,
    int32_t frame_count
) const {
    if(!is_loaded()) {
        return Error::invalid_state("TtsWaveformDecoder::decode: decoder is not loaded.");
    }
    if(audio_codes.empty()) {
        return Error::empty_input("TtsWaveformDecoder::decode: audio codes are empty.");
    }
    if(frame_count <= 0) {
        return Error::invalid_argument("TtsWaveformDecoder::decode: frame_count must be > 0.");
    }
    if(audio_codes.size() % static_cast<size_t>(frame_count) != 0) {
        return Error::shape_mismatch("TtsWaveformDecoder::decode: audio codes do not divide evenly by frame_count.");
    }

    const int32_t codes_per_frame = static_cast<int32_t>(audio_codes.size() / static_cast<size_t>(frame_count));
    if(codes_per_frame <= 0) {
        return Error::shape_mismatch("TtsWaveformDecoder::decode: resolved codes_per_frame was <= 0.");
    }

    const int32_t target_quantizers = impl_->input_quantizer_count > 0
        ? static_cast<int32_t>(impl_->input_quantizer_count)
        : codes_per_frame;
    if(target_quantizers <= 0) {
        return Error::shape_mismatch("TtsWaveformDecoder::decode: resolved target quantizer count was <= 0.");
    }

    const bool requires_relayout =
        impl_->input_quantizer_axis == 1 || target_quantizers != codes_per_frame;
    std::vector<int64_t> relaid_codes;
    std::span<const int64_t> decoder_codes = audio_codes;
    if(requires_relayout) {
        relaid_codes.assign(
            static_cast<size_t>(frame_count) * static_cast<size_t>(target_quantizers),
            int64_t {0}
        );
        const int32_t copy_quantizers = std::min(codes_per_frame, target_quantizers);
        for(int32_t frame_index = 0; frame_index < frame_count; ++frame_index) {
            for(int32_t quantizer_index = 0; quantizer_index < copy_quantizers; ++quantizer_index) {
                const size_t src_index =
                    static_cast<size_t>(frame_index) * static_cast<size_t>(codes_per_frame) +
                    static_cast<size_t>(quantizer_index);
                size_t dst_index = src_index;
                if(impl_->input_quantizer_axis == 1) {
                    dst_index =
                        static_cast<size_t>(quantizer_index) * static_cast<size_t>(frame_count) +
                        static_cast<size_t>(frame_index);
                } else {
                    dst_index =
                        static_cast<size_t>(frame_index) * static_cast<size_t>(target_quantizers) +
                        static_cast<size_t>(quantizer_index);
                }
                relaid_codes[dst_index] = audio_codes[src_index];
            }
        }
        decoder_codes = relaid_codes;
    }

    if(impl_->stateful_decoder) {
        using Clock = std::chrono::steady_clock;
        using Ms = std::chrono::duration<double, std::milli>;

        const auto decode_start = Clock::now();
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const int32_t chunk_frames_limit = std::max<int32_t>(1, impl_->config.stateful_chunk_frames);
        const int32_t num_layers = impl_->stateful_num_layers;
        const int32_t num_heads = impl_->stateful_num_heads;
        const int32_t head_dim = impl_->stateful_head_dim;

        StatefulDecodeState state;
        reset_stateful_decode_state(state, num_layers, num_heads, head_dim);

        StatefulDecodeScratch scratch;
        scratch.inputs.reserve(impl_->stateful_input_bindings.size());
        scratch.chunk_codes.reserve(
            static_cast<size_t>(chunk_frames_limit) * static_cast<size_t>(target_quantizers)
        );
        scratch.raw_waveform.reserve(static_cast<size_t>(frame_count) * 1600);
        double inference_ms = 0.0;
        float zero_float = 0.0f;

        for(int32_t frame_offset = 0; frame_offset < frame_count; frame_offset += chunk_frames_limit) {
            const int32_t chunk_frames = std::min(chunk_frames_limit, frame_count - frame_offset);
            const bool chunk_is_final = (frame_offset + chunk_frames) >= frame_count;
            scratch.chunk_codes.assign(
                static_cast<size_t>(chunk_frames) * static_cast<size_t>(target_quantizers),
                int64_t {0}
            );
            const int32_t copy_quantizers = std::min(codes_per_frame, target_quantizers);
            for(int32_t frame_index = 0; frame_index < chunk_frames; ++frame_index) {
                for(int32_t quantizer_index = 0; quantizer_index < copy_quantizers; ++quantizer_index) {
                    const size_t src_index =
                        static_cast<size_t>(frame_offset + frame_index) * static_cast<size_t>(codes_per_frame) +
                        static_cast<size_t>(quantizer_index);
                    size_t dst_index =
                        static_cast<size_t>(frame_index) * static_cast<size_t>(target_quantizers) +
                        static_cast<size_t>(quantizer_index);
                    if(impl_->input_quantizer_axis == 1) {
                        dst_index =
                            static_cast<size_t>(quantizer_index) * static_cast<size_t>(chunk_frames) +
                            static_cast<size_t>(frame_index);
                    }
                    scratch.chunk_codes[dst_index] = audio_codes[src_index];
                }
            }

            if(impl_->input_quantizer_axis == 1) {
                scratch.chunk_shape[0] = 1;
                scratch.chunk_shape[1] = target_quantizers;
                scratch.chunk_shape[2] = chunk_frames;
            } else {
                scratch.chunk_shape[0] = 1;
                scratch.chunk_shape[1] = chunk_frames;
                scratch.chunk_shape[2] = target_quantizers;
            }
            float is_last_value = chunk_is_final ? 1.0f : 0.0f;

            scratch.inputs.clear();
            for(const StatefulInputBinding &binding : impl_->stateful_input_bindings) {
                switch(binding.kind) {
                case StatefulInputKind::AudioCodes:
                    scratch.inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                        memory_info,
                        scratch.chunk_codes.data(),
                        scratch.chunk_codes.size(),
                        scratch.chunk_shape.data(),
                        scratch.chunk_shape.size()
                    ));
                    break;
                case StatefulInputKind::PreConvHistory: {
                    float *data = state.pre_conv_history.data.empty()
                        ? &zero_float
                        : state.pre_conv_history.data.data();
                    scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        data,
                        state.pre_conv_history.data.size(),
                        state.pre_conv_history.shape.data(),
                        state.pre_conv_history.shape.size()
                    ));
                    break;
                }
                case StatefulInputKind::LatentBuffer: {
                    float *data = state.latent_buffer.data.empty()
                        ? &zero_float
                        : state.latent_buffer.data.data();
                    scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        data,
                        state.latent_buffer.data.size(),
                        state.latent_buffer.shape.data(),
                        state.latent_buffer.shape.size()
                    ));
                    break;
                }
                case StatefulInputKind::ConvHistory: {
                    float *data = state.conv_history.data.empty()
                        ? &zero_float
                        : state.conv_history.data.data();
                    scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        data,
                        state.conv_history.data.size(),
                        state.conv_history.shape.data(),
                        state.conv_history.shape.size()
                    ));
                    break;
                }
                case StatefulInputKind::IsLast:
                    scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        &is_last_value,
                        1,
                        scratch.is_last_shape.data(),
                        scratch.is_last_shape.size()
                    ));
                    break;
                case StatefulInputKind::PastKey: {
                    const int32_t layer_index = binding.layer_index;
                    if(layer_index < 0 || layer_index >= num_layers) {
                        return Error::shape_mismatch("TtsWaveformDecoder::decode: invalid stateful key input " + binding.name);
                    }
                    StatefulTensor &tensor = state.keys[static_cast<size_t>(layer_index)];
                    float *data = tensor.data.empty() ? &zero_float : tensor.data.data();
                    scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        data,
                        tensor.data.size(),
                        tensor.shape.data(),
                        tensor.shape.size()
                    ));
                    break;
                }
                case StatefulInputKind::PastValue: {
                    const int32_t layer_index = binding.layer_index;
                    if(layer_index < 0 || layer_index >= num_layers) {
                        return Error::shape_mismatch("TtsWaveformDecoder::decode: invalid stateful value input " + binding.name);
                    }
                    StatefulTensor &tensor = state.values[static_cast<size_t>(layer_index)];
                    float *data = tensor.data.empty() ? &zero_float : tensor.data.data();
                    scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        data,
                        tensor.data.size(),
                        tensor.shape.data(),
                        tensor.shape.size()
                    ));
                    break;
                }
                case StatefulInputKind::Unsupported:
                default:
                    return Error::shape_mismatch("TtsWaveformDecoder::decode: unsupported stateful decoder input " + binding.name);
                }
            }

            const auto inference_start = Clock::now();
            auto run_result = impl_->session.run(scratch.inputs);
            inference_ms += Ms(Clock::now() - inference_start).count();
            if(run_result.has_error()) {
                return Error::inference_failed(
                    "TtsWaveformDecoder::decode: stateful ONNX run failed: " + run_result.error().message
                );
            }

            auto &outputs = run_result.value();
            const int32_t final_wav_index = impl_->stateful_final_wav_output;
            const int32_t valid_samples_index = impl_->stateful_valid_samples_output;
            if(final_wav_index < 0 || final_wav_index >= static_cast<int32_t>(outputs.size()) ||
               valid_samples_index < 0 || valid_samples_index >= static_cast<int32_t>(outputs.size())) {
                return Error::shape_mismatch("TtsWaveformDecoder::decode: stateful decoder outputs are incomplete.");
            }

            auto &wav_tensor = outputs[static_cast<size_t>(final_wav_index)];
            auto wav_info = wav_tensor.GetTensorTypeAndShapeInfo();
            if(wav_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                return Error::shape_mismatch("TtsWaveformDecoder::decode: stateful final_wav output was not FLOAT.");
            }
            const size_t wav_size = wav_info.GetElementCount();
            const float *wav_data = wav_tensor.GetTensorData<float>();

            int64_t valid_samples = 0;
            auto &valid_tensor = outputs[static_cast<size_t>(valid_samples_index)];
            auto valid_info = valid_tensor.GetTensorTypeAndShapeInfo();
            if(valid_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                valid_samples = valid_info.GetElementCount() > 0
                    ? valid_tensor.GetTensorData<int64_t>()[0]
                    : 0;
            } else if(valid_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                valid_samples = valid_info.GetElementCount() > 0
                    ? static_cast<int64_t>(std::llround(valid_tensor.GetTensorData<float>()[0]))
                    : 0;
            } else {
                return Error::shape_mismatch("TtsWaveformDecoder::decode: stateful valid_samples output had an unsupported type.");
            }

            const size_t append_count = chunk_is_final
                ? wav_size
                : static_cast<size_t>(std::clamp<int64_t>(
                    valid_samples,
                    0,
                    static_cast<int64_t>(wav_size)
                ));
            scratch.raw_waveform.insert(scratch.raw_waveform.end(), wav_data, wav_data + append_count);

            const int32_t next_pre_conv_index = impl_->stateful_next_pre_conv_output;
            const int32_t next_latent_index = impl_->stateful_next_latent_output;
            const int32_t next_conv_index = impl_->stateful_next_conv_output;
            if(next_pre_conv_index < 0 || next_pre_conv_index >= static_cast<int32_t>(outputs.size()) ||
               next_latent_index < 0 || next_latent_index >= static_cast<int32_t>(outputs.size()) ||
               next_conv_index < 0 || next_conv_index >= static_cast<int32_t>(outputs.size())) {
                return Error::shape_mismatch("TtsWaveformDecoder::decode: stateful history outputs are incomplete.");
            }
            std::string copy_error = copy_float_output(
                outputs[static_cast<size_t>(next_pre_conv_index)],
                state.pre_conv_history,
                "next_pre_conv_history"
            );
            if(!copy_error.empty()) {
                return Error::shape_mismatch("TtsWaveformDecoder::decode: " + copy_error);
            }
            copy_error = copy_float_output(
                outputs[static_cast<size_t>(next_latent_index)],
                state.latent_buffer,
                "next_latent_buffer"
            );
            if(!copy_error.empty()) {
                return Error::shape_mismatch("TtsWaveformDecoder::decode: " + copy_error);
            }
            copy_error = copy_float_output(
                outputs[static_cast<size_t>(next_conv_index)],
                state.conv_history,
                "next_conv_history"
            );
            if(!copy_error.empty()) {
                return Error::shape_mismatch("TtsWaveformDecoder::decode: " + copy_error);
            }

            for(int32_t layer_index = 0; layer_index < num_layers; ++layer_index) {
                const int32_t key_index = impl_->stateful_next_key_outputs[static_cast<size_t>(layer_index)];
                const int32_t value_index = impl_->stateful_next_value_outputs[static_cast<size_t>(layer_index)];
                if(key_index < 0 || key_index >= static_cast<int32_t>(outputs.size()) ||
                   value_index < 0 || value_index >= static_cast<int32_t>(outputs.size())) {
                    return Error::shape_mismatch("TtsWaveformDecoder::decode: stateful KV outputs are incomplete.");
                }
                copy_error = copy_float_output(
                    outputs[static_cast<size_t>(key_index)],
                    state.keys[static_cast<size_t>(layer_index)],
                    "next_key"
                );
                if(!copy_error.empty()) {
                    return Error::shape_mismatch("TtsWaveformDecoder::decode: " + copy_error);
                }
                copy_error = copy_float_output(
                    outputs[static_cast<size_t>(value_index)],
                    state.values[static_cast<size_t>(layer_index)],
                    "next_value"
                );
                if(!copy_error.empty()) {
                    return Error::shape_mismatch("TtsWaveformDecoder::decode: " + copy_error);
                }
            }
        }

        if(scratch.raw_waveform.empty()) {
            return Error::inference_failed("TtsWaveformDecoder::decode: stateful decoder produced no samples.");
        }

        const auto postprocess_start = Clock::now();
        std::vector<int64_t> raw_shape = {static_cast<int64_t>(scratch.raw_waveform.size())};
        std::vector<float> waveform = convert_decoder_output_to_waveform(
            std::span<const float>(scratch.raw_waveform.data(), scratch.raw_waveform.size()),
            std::span<const int64_t>(raw_shape.data(), raw_shape.size()),
            impl_->config.sample_rate,
            impl_->config.normalize_waveform,
            impl_->config.waveform_gain
        );
        const double postprocess_ms = Ms(Clock::now() - postprocess_start).count();
        const double elapsed_ms = Ms(Clock::now() - decode_start).count();
        if(waveform.empty()) {
            return Error::inference_failed("TtsWaveformDecoder::decode: stateful waveform conversion produced no samples.");
        }

        TtsWaveformDecodeResult result;
        result.waveform = std::move(waveform);
        result.frame_count = frame_count;
        result.codes_per_frame = codes_per_frame;
        result.sample_count = static_cast<int32_t>(result.waveform.size());
        result.elapsed_ms = elapsed_ms;
        result.inference_ms = inference_ms;
        result.postprocess_ms = postprocess_ms;
        result.backend = "gotst_native_stateful";
        return result;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape =
        impl_->input_quantizer_axis == 1
        ? std::vector<int64_t> {1, target_quantizers, frame_count}
        : std::vector<int64_t> {1, frame_count, target_quantizers};
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t *>(decoder_codes.data()),
        decoder_codes.size(),
        input_shape.data(),
        input_shape.size()
    );

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_tensor));

    const auto decode_start = Clock::now();
    const auto inference_start = Clock::now();
    auto run_result = impl_->session.run(inputs);
    const double inference_ms = Ms(Clock::now() - inference_start).count();
    if(run_result.has_error()) {
        return Error::inference_failed(
            "TtsWaveformDecoder::decode: ONNX run failed: " + run_result.error().message
        );
    }

    auto &outputs = run_result.value();
    if(outputs.empty()) {
        return Error::inference_failed("TtsWaveformDecoder::decode: empty ONNX output.");
    }

    auto &output_tensor = outputs[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    if(output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Error::shape_mismatch("TtsWaveformDecoder::decode: expected FLOAT output tensor.");
    }

    const size_t output_size = output_info.GetElementCount();
    if(output_size == 0) {
        return Error::inference_failed("TtsWaveformDecoder::decode: decoder output tensor was empty.");
    }

    const float *output_data = output_tensor.GetTensorData<float>();
    const std::vector<int64_t> output_shape = output_info.GetShape();

    const auto postprocess_start = Clock::now();
    std::vector<float> waveform = convert_decoder_output_to_waveform(
        std::span<const float>(output_data, output_size),
        std::span<const int64_t>(output_shape.data(), output_shape.size()),
        impl_->config.sample_rate,
        impl_->config.normalize_waveform,
        impl_->config.waveform_gain
    );
    const double postprocess_ms = Ms(Clock::now() - postprocess_start).count();
    const double elapsed_ms = Ms(Clock::now() - decode_start).count();

    if(waveform.empty()) {
        return Error::inference_failed("TtsWaveformDecoder::decode: waveform conversion produced no samples.");
    }

    TtsWaveformDecodeResult result;
    result.waveform = std::move(waveform);
    result.frame_count = frame_count;
    result.codes_per_frame = codes_per_frame;
    result.sample_count = static_cast<int32_t>(result.waveform.size());
    result.elapsed_ms = elapsed_ms;
    result.inference_ms = inference_ms;
    result.postprocess_ms = postprocess_ms;
    return result;
}

Result<std::unique_ptr<TtsWaveformDecoderStream>> TtsWaveformDecoder::create_stream() const {
    if(!is_loaded()) {
        return Error::invalid_state("TtsWaveformDecoder::create_stream: decoder is not loaded.");
    }
    return std::unique_ptr<TtsWaveformDecoderStream>(new TtsWaveformDecoderStream(*this));
}

struct TtsWaveformDecoderStream::Impl {
    const TtsWaveformDecoder *decoder = nullptr;
    StatefulDecodeState state;
    StatefulDecodeScratch scratch;
    bool state_initialized = false;
};

TtsWaveformDecoderStream::TtsWaveformDecoderStream(const TtsWaveformDecoder &decoder)
    : impl_(std::make_unique<Impl>()) {
    impl_->decoder = &decoder;
    reset();
}

TtsWaveformDecoderStream::~TtsWaveformDecoderStream() = default;
TtsWaveformDecoderStream::TtsWaveformDecoderStream(TtsWaveformDecoderStream &&) noexcept = default;
TtsWaveformDecoderStream &TtsWaveformDecoderStream::operator=(TtsWaveformDecoderStream &&) noexcept = default;

void TtsWaveformDecoderStream::reset() {
    if(!impl_ || impl_->decoder == nullptr || impl_->decoder->impl_ == nullptr) {
        return;
    }
    const auto &decoder_impl = *impl_->decoder->impl_;
    reset_stateful_decode_state(
        impl_->state,
        decoder_impl.stateful_num_layers,
        decoder_impl.stateful_num_heads,
        decoder_impl.stateful_head_dim
    );
    impl_->scratch.inputs.clear();
    impl_->scratch.inputs.reserve(decoder_impl.stateful_input_bindings.size());
    impl_->scratch.chunk_codes.clear();
    impl_->scratch.raw_waveform.clear();
    impl_->state_initialized = true;
}

Result<TtsWaveformDecodeResult> TtsWaveformDecoderStream::decode(
    std::span<const int64_t> audio_codes,
    int32_t frame_count,
    bool is_final
) {
    if(!impl_ || impl_->decoder == nullptr || !impl_->decoder->is_loaded()) {
        return Error::invalid_state("TtsWaveformDecoderStream::decode: decoder is not loaded.");
    }
    if(audio_codes.empty()) {
        return Error::empty_input("TtsWaveformDecoderStream::decode: audio codes are empty.");
    }
    if(frame_count <= 0) {
        return Error::invalid_argument("TtsWaveformDecoderStream::decode: frame_count must be > 0.");
    }
    if(audio_codes.size() % static_cast<size_t>(frame_count) != 0) {
        return Error::shape_mismatch("TtsWaveformDecoderStream::decode: audio codes do not divide evenly by frame_count.");
    }

    auto &decoder_impl = *impl_->decoder->impl_;
    if(!decoder_impl.stateful_decoder) {
        return impl_->decoder->decode(audio_codes, frame_count);
    }
    if(!impl_->state_initialized) {
        reset();
    }

    const int32_t codes_per_frame = static_cast<int32_t>(audio_codes.size() / static_cast<size_t>(frame_count));
    if(codes_per_frame <= 0) {
        return Error::shape_mismatch("TtsWaveformDecoderStream::decode: resolved codes_per_frame was <= 0.");
    }

    const int32_t target_quantizers = decoder_impl.input_quantizer_count > 0
        ? static_cast<int32_t>(decoder_impl.input_quantizer_count)
        : codes_per_frame;
    if(target_quantizers <= 0) {
        return Error::shape_mismatch("TtsWaveformDecoderStream::decode: resolved target quantizer count was <= 0.");
    }

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    const auto decode_start = Clock::now();
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int32_t chunk_frames_limit = std::max<int32_t>(1, decoder_impl.config.stateful_chunk_frames);
    const int32_t num_layers = decoder_impl.stateful_num_layers;
    StatefulDecodeState &state = impl_->state;
    StatefulDecodeScratch &scratch = impl_->scratch;
    scratch.inputs.reserve(decoder_impl.stateful_input_bindings.size());
    scratch.chunk_codes.reserve(
        static_cast<size_t>(chunk_frames_limit) * static_cast<size_t>(target_quantizers)
    );
    scratch.raw_waveform.clear();
    scratch.raw_waveform.reserve(static_cast<size_t>(frame_count) * 1600);

    double inference_ms = 0.0;
    float zero_float = 0.0f;
    for(int32_t frame_offset = 0; frame_offset < frame_count; frame_offset += chunk_frames_limit) {
        const int32_t chunk_frames = std::min(chunk_frames_limit, frame_count - frame_offset);
        const bool chunk_is_final = is_final && (frame_offset + chunk_frames) >= frame_count;
        scratch.chunk_codes.assign(
            static_cast<size_t>(chunk_frames) * static_cast<size_t>(target_quantizers),
            int64_t {0}
        );
        const int32_t copy_quantizers = std::min(codes_per_frame, target_quantizers);
        for(int32_t frame_index = 0; frame_index < chunk_frames; ++frame_index) {
            for(int32_t quantizer_index = 0; quantizer_index < copy_quantizers; ++quantizer_index) {
                const size_t src_index =
                    static_cast<size_t>(frame_offset + frame_index) * static_cast<size_t>(codes_per_frame) +
                    static_cast<size_t>(quantizer_index);
                size_t dst_index =
                    static_cast<size_t>(frame_index) * static_cast<size_t>(target_quantizers) +
                    static_cast<size_t>(quantizer_index);
                if(decoder_impl.input_quantizer_axis == 1) {
                    dst_index =
                        static_cast<size_t>(quantizer_index) * static_cast<size_t>(chunk_frames) +
                        static_cast<size_t>(frame_index);
                }
                scratch.chunk_codes[dst_index] = audio_codes[src_index];
            }
        }

        if(decoder_impl.input_quantizer_axis == 1) {
            scratch.chunk_shape[0] = 1;
            scratch.chunk_shape[1] = target_quantizers;
            scratch.chunk_shape[2] = chunk_frames;
        } else {
            scratch.chunk_shape[0] = 1;
            scratch.chunk_shape[1] = chunk_frames;
            scratch.chunk_shape[2] = target_quantizers;
        }
        float is_last_value = chunk_is_final ? 1.0f : 0.0f;

        scratch.inputs.clear();
        for(const StatefulInputBinding &binding : decoder_impl.stateful_input_bindings) {
            switch(binding.kind) {
            case StatefulInputKind::AudioCodes:
                scratch.inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                    memory_info,
                    scratch.chunk_codes.data(),
                    scratch.chunk_codes.size(),
                    scratch.chunk_shape.data(),
                    scratch.chunk_shape.size()
                ));
                break;
            case StatefulInputKind::PreConvHistory: {
                float *data = state.pre_conv_history.data.empty()
                    ? &zero_float
                    : state.pre_conv_history.data.data();
                scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    data,
                    state.pre_conv_history.data.size(),
                    state.pre_conv_history.shape.data(),
                    state.pre_conv_history.shape.size()
                ));
                break;
            }
            case StatefulInputKind::LatentBuffer: {
                float *data = state.latent_buffer.data.empty()
                    ? &zero_float
                    : state.latent_buffer.data.data();
                scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    data,
                    state.latent_buffer.data.size(),
                    state.latent_buffer.shape.data(),
                    state.latent_buffer.shape.size()
                ));
                break;
            }
            case StatefulInputKind::ConvHistory: {
                float *data = state.conv_history.data.empty()
                    ? &zero_float
                    : state.conv_history.data.data();
                scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    data,
                    state.conv_history.data.size(),
                    state.conv_history.shape.data(),
                    state.conv_history.shape.size()
                ));
                break;
            }
            case StatefulInputKind::IsLast:
                scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    &is_last_value,
                    1,
                    scratch.is_last_shape.data(),
                    scratch.is_last_shape.size()
                ));
                break;
            case StatefulInputKind::PastKey: {
                const int32_t layer_index = binding.layer_index;
                if(layer_index < 0 || layer_index >= num_layers) {
                    return Error::shape_mismatch("TtsWaveformDecoderStream::decode: invalid stateful key input " + binding.name);
                }
                StatefulTensor &tensor = state.keys[static_cast<size_t>(layer_index)];
                float *data = tensor.data.empty() ? &zero_float : tensor.data.data();
                scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    data,
                    tensor.data.size(),
                    tensor.shape.data(),
                    tensor.shape.size()
                ));
                break;
            }
            case StatefulInputKind::PastValue: {
                const int32_t layer_index = binding.layer_index;
                if(layer_index < 0 || layer_index >= num_layers) {
                    return Error::shape_mismatch("TtsWaveformDecoderStream::decode: invalid stateful value input " + binding.name);
                }
                StatefulTensor &tensor = state.values[static_cast<size_t>(layer_index)];
                float *data = tensor.data.empty() ? &zero_float : tensor.data.data();
                scratch.inputs.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    data,
                    tensor.data.size(),
                    tensor.shape.data(),
                    tensor.shape.size()
                ));
                break;
            }
            case StatefulInputKind::Unsupported:
            default:
                return Error::shape_mismatch("TtsWaveformDecoderStream::decode: unsupported stateful decoder input " + binding.name);
            }
        }

        const auto inference_start = Clock::now();
        auto run_result = decoder_impl.session.run(scratch.inputs);
        inference_ms += Ms(Clock::now() - inference_start).count();
        if(run_result.has_error()) {
            return Error::inference_failed(
                "TtsWaveformDecoderStream::decode: stateful ONNX run failed: " + run_result.error().message
            );
        }

        auto &outputs = run_result.value();
        const int32_t final_wav_index = decoder_impl.stateful_final_wav_output;
        const int32_t valid_samples_index = decoder_impl.stateful_valid_samples_output;
        if(final_wav_index < 0 || final_wav_index >= static_cast<int32_t>(outputs.size()) ||
           valid_samples_index < 0 || valid_samples_index >= static_cast<int32_t>(outputs.size())) {
            return Error::shape_mismatch("TtsWaveformDecoderStream::decode: stateful decoder outputs are incomplete.");
        }

        auto &wav_tensor = outputs[static_cast<size_t>(final_wav_index)];
        auto wav_info = wav_tensor.GetTensorTypeAndShapeInfo();
        if(wav_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return Error::shape_mismatch("TtsWaveformDecoderStream::decode: stateful final_wav output was not FLOAT.");
        }
        const size_t wav_size = wav_info.GetElementCount();
        const float *wav_data = wav_tensor.GetTensorData<float>();

        int64_t valid_samples = 0;
        auto &valid_tensor = outputs[static_cast<size_t>(valid_samples_index)];
        auto valid_info = valid_tensor.GetTensorTypeAndShapeInfo();
        if(valid_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            valid_samples = valid_info.GetElementCount() > 0
                ? valid_tensor.GetTensorData<int64_t>()[0]
                : 0;
        } else if(valid_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            valid_samples = valid_info.GetElementCount() > 0
                ? static_cast<int64_t>(std::llround(valid_tensor.GetTensorData<float>()[0]))
                : 0;
        } else {
            return Error::shape_mismatch("TtsWaveformDecoderStream::decode: stateful valid_samples output had an unsupported type.");
        }

        const size_t append_count = chunk_is_final
            ? wav_size
            : static_cast<size_t>(std::clamp<int64_t>(
                valid_samples,
                0,
                static_cast<int64_t>(wav_size)
            ));
        scratch.raw_waveform.insert(scratch.raw_waveform.end(), wav_data, wav_data + append_count);

        const int32_t next_pre_conv_index = decoder_impl.stateful_next_pre_conv_output;
        const int32_t next_latent_index = decoder_impl.stateful_next_latent_output;
        const int32_t next_conv_index = decoder_impl.stateful_next_conv_output;
        if(next_pre_conv_index < 0 || next_pre_conv_index >= static_cast<int32_t>(outputs.size()) ||
           next_latent_index < 0 || next_latent_index >= static_cast<int32_t>(outputs.size()) ||
           next_conv_index < 0 || next_conv_index >= static_cast<int32_t>(outputs.size())) {
            return Error::shape_mismatch("TtsWaveformDecoderStream::decode: stateful history outputs are incomplete.");
        }
        std::string copy_error = copy_float_output(
            outputs[static_cast<size_t>(next_pre_conv_index)],
            state.pre_conv_history,
            "next_pre_conv_history"
        );
        if(!copy_error.empty()) {
            return Error::shape_mismatch("TtsWaveformDecoderStream::decode: " + copy_error);
        }
        copy_error = copy_float_output(
            outputs[static_cast<size_t>(next_latent_index)],
            state.latent_buffer,
            "next_latent_buffer"
        );
        if(!copy_error.empty()) {
            return Error::shape_mismatch("TtsWaveformDecoderStream::decode: " + copy_error);
        }
        copy_error = copy_float_output(
            outputs[static_cast<size_t>(next_conv_index)],
            state.conv_history,
            "next_conv_history"
        );
        if(!copy_error.empty()) {
            return Error::shape_mismatch("TtsWaveformDecoderStream::decode: " + copy_error);
        }

        for(int32_t layer_index = 0; layer_index < num_layers; ++layer_index) {
            const int32_t key_index = decoder_impl.stateful_next_key_outputs[static_cast<size_t>(layer_index)];
            const int32_t value_index = decoder_impl.stateful_next_value_outputs[static_cast<size_t>(layer_index)];
            if(key_index < 0 || key_index >= static_cast<int32_t>(outputs.size()) ||
               value_index < 0 || value_index >= static_cast<int32_t>(outputs.size())) {
                return Error::shape_mismatch("TtsWaveformDecoderStream::decode: stateful KV outputs are incomplete.");
            }
            copy_error = copy_float_output(
                outputs[static_cast<size_t>(key_index)],
                state.keys[static_cast<size_t>(layer_index)],
                "next_key"
            );
            if(!copy_error.empty()) {
                return Error::shape_mismatch("TtsWaveformDecoderStream::decode: " + copy_error);
            }
            copy_error = copy_float_output(
                outputs[static_cast<size_t>(value_index)],
                state.values[static_cast<size_t>(layer_index)],
                "next_value"
            );
            if(!copy_error.empty()) {
                return Error::shape_mismatch("TtsWaveformDecoderStream::decode: " + copy_error);
            }
        }
    }

    if(scratch.raw_waveform.empty()) {
        return Error::inference_failed("TtsWaveformDecoderStream::decode: stateful decoder produced no samples.");
    }

    const auto postprocess_start = Clock::now();
    std::vector<int64_t> raw_shape = {static_cast<int64_t>(scratch.raw_waveform.size())};
    std::vector<float> waveform = convert_decoder_output_to_waveform(
        std::span<const float>(scratch.raw_waveform.data(), scratch.raw_waveform.size()),
        std::span<const int64_t>(raw_shape.data(), raw_shape.size()),
        decoder_impl.config.sample_rate,
        decoder_impl.config.normalize_waveform,
        decoder_impl.config.waveform_gain
    );
    const double postprocess_ms = Ms(Clock::now() - postprocess_start).count();
    const double elapsed_ms = Ms(Clock::now() - decode_start).count();
    if(waveform.empty()) {
        return Error::inference_failed("TtsWaveformDecoderStream::decode: stateful waveform conversion produced no samples.");
    }

    TtsWaveformDecodeResult result;
    result.waveform = std::move(waveform);
    result.frame_count = frame_count;
    result.codes_per_frame = codes_per_frame;
    result.sample_count = static_cast<int32_t>(result.waveform.size());
    result.elapsed_ms = elapsed_ms;
    result.inference_ms = inference_ms;
    result.postprocess_ms = postprocess_ms;
    result.backend = "gotst_native_stateful_stream";
    return result;
}

} // namespace gotst
