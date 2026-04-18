#include "gotst/core/tts_waveform_decoder.hpp"

#include <gonx/core/session.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
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

struct TtsWaveformDecoder::Impl {
    gonx::InferenceSession session;
    TtsWaveformDecoderConfig config;
    bool loaded = false;
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

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, frame_count, codes_per_frame};
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t *>(audio_codes.data()),
        audio_codes.size(),
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

} // namespace gotst
