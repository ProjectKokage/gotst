#include "gotst/core/irodori_tts_session.hpp"

#include <gonx/core/provider.hpp>
#include <gonx/core/session.hpp>
#include <gonx/core/type_conversion.hpp>

#include <algorithm>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gotst {

namespace {

constexpr float kIrodoriInitScale = 0.999f;
constexpr size_t kCoreMLStaticBucketCacheLimit = 2;

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool is_finite(float value) {
    return std::isfinite(static_cast<double>(value));
}

bool path_exists(const std::string &path) {
    return !path.empty() && std::filesystem::exists(std::filesystem::path(path));
}

std::string bucket_to_string(const IrodoriTtsBucket &bucket) {
    std::ostringstream stream;
    stream << "latent=" << bucket.latent_steps
           << ",text=" << bucket.text_tokens
           << ",caption=" << bucket.caption_tokens
           << ",ref=" << bucket.ref_steps;
    return stream.str();
}

std::string bucket_cache_key(const IrodoriTtsBucket &bucket) {
    std::ostringstream stream;
    stream << bucket.latent_steps << 'x'
           << bucket.text_tokens << 'x'
           << bucket.caption_tokens << 'x'
           << bucket.ref_steps;
    return stream.str();
}

int64_t bucket_cost(
    const IrodoriTtsBucket &bucket,
    int32_t latent_steps,
    int32_t text_tokens,
    int32_t caption_tokens,
    int32_t ref_steps
) {
    if(bucket.latent_steps < latent_steps ||
       bucket.text_tokens < text_tokens ||
       bucket.caption_tokens < caption_tokens ||
       bucket.ref_steps < ref_steps) {
        return std::numeric_limits<int64_t>::max();
    }
    return static_cast<int64_t>(bucket.latent_steps - latent_steps) * 1000000LL +
        static_cast<int64_t>(bucket.text_tokens - text_tokens) * 1000LL +
        static_cast<int64_t>(bucket.caption_tokens - caption_tokens) * 100LL +
        static_cast<int64_t>(bucket.ref_steps - ref_steps);
}

std::vector<IrodoriTtsBucket> default_buckets(IrodoriTtsMode mode) {
    const bool caption = irodori_tts_mode_uses_caption(mode);
    const bool speaker = irodori_tts_mode_uses_speaker(mode);
    return {
        {128, 128, caption ? 256 : 0, speaker ? 256 : 0},
        {256, 192, caption ? 384 : 0, speaker ? 384 : 0},
        {384, 256, caption ? 512 : 0, speaker ? 512 : 0},
        {512, 320, caption ? 640 : 0, speaker ? 640 : 0},
        {768, 384, caption ? 768 : 0, speaker ? 768 : 0},
    };
}

struct OptionalSession {
    std::string path;
    std::string provider_requested;
    std::string provider_effective;
    gonx::InferenceSession session;
    bool requested = false;
    bool loaded = false;
    double session_create_ms = 0.0;
    std::string profiling_file_path;
};

struct StaticDitSession {
    IrodoriTtsBucket bucket;
    OptionalSession dit_step;
    std::string rf_sampler_6_step_path;
    std::string rf_sampler_8_step_path;
    OptionalSession rf_sampler_6_step;
    OptionalSession rf_sampler_8_step;
};

struct FloatTensor {
    std::vector<float> values;
    std::vector<int64_t> shape;
};

struct Int64Tensor {
    std::vector<int64_t> values;
    std::vector<int64_t> shape;
};

struct BoolTensor {
    std::vector<uint8_t> values;
    std::vector<int64_t> shape;
};

using SteadyClock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

double elapsed_ms_since(const SteadyClock::time_point &started) {
    return Milliseconds(SteadyClock::now() - started).count();
}

void add_metric(std::map<std::string, double> *metrics, const std::string &name, double value) {
    if(metrics) {
        (*metrics)[name] += value;
    }
}

void set_metric(std::map<std::string, double> *metrics, const std::string &name, double value) {
    if(metrics) {
        (*metrics)[name] = value;
    }
}

void add_diagnostic(
    std::map<std::string, std::string> *diagnostics,
    const std::string &name,
    const std::string &value
) {
    if(diagnostics) {
        (*diagnostics)[name] = value;
    }
}

struct WavData {
    std::vector<float> mono;
    int32_t sample_rate = 0;
};

struct ConditionState {
    FloatTensor text_state;
    BoolTensor text_mask;
    FloatTensor speaker_state;
    BoolTensor speaker_mask;
    FloatTensor caption_state;
    BoolTensor caption_mask;
    bool has_speaker = false;
    bool has_caption = false;
};

struct CachedConditionState {
    ConditionState conditions;
    int32_t ref_steps = 0;
};

std::string shape_to_string(std::span<const int64_t> shape) {
    std::ostringstream stream;
    stream << '[';
    for(size_t index = 0; index < shape.size(); ++index) {
        if(index > 0) {
            stream << ',';
        }
        stream << shape[index];
    }
    stream << ']';
    return stream.str();
}

std::vector<uint8_t> ones_bool(size_t count) {
    return std::vector<uint8_t>(count, uint8_t{1});
}

std::vector<uint8_t> zeros_bool(size_t count) {
    return std::vector<uint8_t>(count, uint8_t{0});
}

FloatTensor zeros_like(const FloatTensor &input) {
    return {std::vector<float>(input.values.size(), 0.0f), input.shape};
}

BoolTensor zeros_like(const BoolTensor &input) {
    return {zeros_bool(input.values.size()), input.shape};
}

FloatTensor pad_sequence_tensor(const FloatTensor &input, int64_t target_steps) {
    if(input.shape.size() < 2 || target_steps <= input.shape[1]) {
        return input;
    }
    const int64_t batch = input.shape[0];
    const int64_t source_steps = input.shape[1];
    int64_t feature_count = 1;
    for(size_t index = 2; index < input.shape.size(); ++index) {
        feature_count *= std::max<int64_t>(1, input.shape[index]);
    }

    FloatTensor output;
    output.shape = input.shape;
    output.shape[1] = target_steps;
    output.values.assign(static_cast<size_t>(batch * target_steps * feature_count), 0.0f);
    for(int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        const size_t source_offset = static_cast<size_t>(batch_index * source_steps * feature_count);
        const size_t target_offset = static_cast<size_t>(batch_index * target_steps * feature_count);
        const size_t copy_count = static_cast<size_t>(source_steps * feature_count);
        std::copy_n(input.values.begin() + static_cast<std::ptrdiff_t>(source_offset),
                    copy_count,
                    output.values.begin() + static_cast<std::ptrdiff_t>(target_offset));
    }
    return output;
}

BoolTensor pad_sequence_tensor(const BoolTensor &input, int64_t target_steps) {
    if(input.shape.size() < 2 || target_steps <= input.shape[1]) {
        return input;
    }
    const int64_t batch = input.shape[0];
    const int64_t source_steps = input.shape[1];
    BoolTensor output;
    output.shape = input.shape;
    output.shape[1] = target_steps;
    output.values.assign(static_cast<size_t>(batch * target_steps), uint8_t{0});
    for(int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        const size_t source_offset = static_cast<size_t>(batch_index * source_steps);
        const size_t target_offset = static_cast<size_t>(batch_index * target_steps);
        const size_t copy_count = static_cast<size_t>(source_steps);
        std::copy_n(input.values.begin() + static_cast<std::ptrdiff_t>(source_offset),
                    copy_count,
                    output.values.begin() + static_cast<std::ptrdiff_t>(target_offset));
    }
    return output;
}

ConditionState pad_conditions_to_bucket(
    const ConditionState &input,
    const IrodoriTtsBucket &bucket,
    IrodoriTtsMode mode
) {
    ConditionState output = input;
    output.text_state = pad_sequence_tensor(input.text_state, bucket.text_tokens);
    output.text_mask = pad_sequence_tensor(input.text_mask, bucket.text_tokens);
    if(irodori_tts_mode_uses_speaker(mode) && input.has_speaker) {
        output.speaker_state = pad_sequence_tensor(input.speaker_state, bucket.ref_steps);
        output.speaker_mask = pad_sequence_tensor(input.speaker_mask, bucket.ref_steps);
    }
    if(irodori_tts_mode_uses_caption(mode) && input.has_caption) {
        output.caption_state = pad_sequence_tensor(input.caption_state, bucket.caption_tokens);
        output.caption_mask = pad_sequence_tensor(input.caption_mask, bucket.caption_tokens);
    }
    return output;
}

uint16_t read_le_u16(const std::vector<uint8_t> &bytes, size_t offset) {
    return static_cast<uint16_t>(
        static_cast<uint16_t>(bytes[offset]) |
        (static_cast<uint16_t>(bytes[offset + 1]) << 8)
    );
}

uint32_t read_le_u32(const std::vector<uint8_t> &bytes, size_t offset) {
    return static_cast<uint32_t>(bytes[offset]) |
        (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
        (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
        (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

int32_t read_le_i24(const uint8_t *data) {
    int32_t value = static_cast<int32_t>(data[0]) |
        (static_cast<int32_t>(data[1]) << 8) |
        (static_cast<int32_t>(data[2]) << 16);
    if(value & 0x00800000) {
        value |= ~0x00ffffff;
    }
    return value;
}

float clamp_sample(float value) {
    if(!is_finite(value)) {
        return 0.0f;
    }
    return std::clamp(value, -1.0f, 1.0f);
}

Result<WavData> load_wav_mono(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if(!file) {
        return Error::not_found("failed to open WAV file: " + path);
    }
    file.seekg(0, std::ios::end);
    const std::streamoff byte_count = file.tellg();
    file.seekg(0, std::ios::beg);
    if(byte_count < 44) {
        return Error::io_error("WAV file is too small: " + path);
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(byte_count));
    file.read(reinterpret_cast<char *>(bytes.data()), byte_count);
    if(!file) {
        return Error::io_error("failed to read WAV file: " + path);
    }
    if(std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        return Error::invalid_argument("unsupported WAV container; expected RIFF/WAVE: " + path);
    }

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    size_t data_offset = 0;
    size_t data_size = 0;

    size_t offset = 12;
    while(offset + 8 <= bytes.size()) {
        const char *chunk_id = reinterpret_cast<const char *>(bytes.data() + offset);
        const uint32_t chunk_size = read_le_u32(bytes, offset + 4);
        const size_t chunk_data = offset + 8;
        if(chunk_data + chunk_size > bytes.size()) {
            return Error::io_error("WAV chunk exceeds file size: " + path);
        }
        if(std::memcmp(chunk_id, "fmt ", 4) == 0) {
            if(chunk_size < 16) {
                return Error::invalid_argument("WAV fmt chunk is too small: " + path);
            }
            audio_format = read_le_u16(bytes, chunk_data);
            channels = read_le_u16(bytes, chunk_data + 2);
            sample_rate = read_le_u32(bytes, chunk_data + 4);
            bits_per_sample = read_le_u16(bytes, chunk_data + 14);
        } else if(std::memcmp(chunk_id, "data", 4) == 0) {
            data_offset = chunk_data;
            data_size = chunk_size;
        }
        offset = chunk_data + chunk_size + (chunk_size & 1U);
    }

    if(audio_format == 0 || channels == 0 || sample_rate == 0 || bits_per_sample == 0) {
        return Error::invalid_argument("WAV file is missing a usable fmt chunk: " + path);
    }
    if(data_offset == 0 || data_size == 0) {
        return Error::invalid_argument("WAV file is missing audio data: " + path);
    }
    if(audio_format != 1 && audio_format != 3) {
        return Error::invalid_argument("WAV format must be PCM or float32: " + path);
    }

    const size_t bytes_per_sample = static_cast<size_t>(bits_per_sample / 8);
    if(bytes_per_sample == 0 || bits_per_sample % 8 != 0) {
        return Error::invalid_argument("WAV bits_per_sample must be byte-aligned: " + path);
    }
    const size_t frame_bytes = bytes_per_sample * static_cast<size_t>(channels);
    if(frame_bytes == 0 || data_size < frame_bytes) {
        return Error::invalid_argument("WAV data chunk is empty: " + path);
    }
    const size_t frame_count = data_size / frame_bytes;
    WavData wav;
    wav.sample_rate = static_cast<int32_t>(sample_rate);
    wav.mono.resize(frame_count, 0.0f);

    for(size_t frame = 0; frame < frame_count; ++frame) {
        double mixed = 0.0;
        for(uint16_t channel = 0; channel < channels; ++channel) {
            const uint8_t *sample = bytes.data() + data_offset + frame * frame_bytes +
                static_cast<size_t>(channel) * bytes_per_sample;
            float value = 0.0f;
            if(audio_format == 3 && bits_per_sample == 32) {
                std::memcpy(&value, sample, sizeof(float));
            } else if(audio_format == 1 && bits_per_sample == 8) {
                value = (static_cast<float>(*sample) - 128.0f) / 128.0f;
            } else if(audio_format == 1 && bits_per_sample == 16) {
                int16_t raw = 0;
                std::memcpy(&raw, sample, sizeof(int16_t));
                value = static_cast<float>(raw) / 32768.0f;
            } else if(audio_format == 1 && bits_per_sample == 24) {
                value = static_cast<float>(read_le_i24(sample)) / 8388608.0f;
            } else if(audio_format == 1 && bits_per_sample == 32) {
                int32_t raw = 0;
                std::memcpy(&raw, sample, sizeof(int32_t));
                value = static_cast<float>(static_cast<double>(raw) / 2147483648.0);
            } else {
                return Error::invalid_argument("unsupported WAV sample format: " + path);
            }
            mixed += clamp_sample(value);
        }
        wav.mono[frame] = static_cast<float>(mixed / static_cast<double>(channels));
    }

    return wav;
}

std::vector<float> resample_linear(
    const std::vector<float> &input,
    int32_t source_rate,
    int32_t target_rate
) {
    if(input.empty() || source_rate <= 0 || target_rate <= 0 || source_rate == target_rate) {
        return input;
    }
    const double ratio = static_cast<double>(target_rate) / static_cast<double>(source_rate);
    const size_t output_count = std::max<size_t>(1, static_cast<size_t>(std::llround(
        static_cast<double>(input.size()) * ratio
    )));
    std::vector<float> output(output_count, 0.0f);
    const double inv_ratio = static_cast<double>(source_rate) / static_cast<double>(target_rate);
    for(size_t index = 0; index < output_count; ++index) {
        const double source_pos = static_cast<double>(index) * inv_ratio;
        const size_t left = std::min(input.size() - 1, static_cast<size_t>(std::floor(source_pos)));
        const size_t right = std::min(input.size() - 1, left + 1);
        const float frac = static_cast<float>(source_pos - static_cast<double>(left));
        output[index] = input[left] * (1.0f - frac) + input[right] * frac;
    }
    return output;
}

std::string file_cache_stamp(const std::string &path) {
    std::error_code ec;
    const auto file_size = std::filesystem::file_size(path, ec);
    const auto last_write = std::filesystem::last_write_time(path, ec);
    std::ostringstream stream;
    stream << path << "|size=" << (ec ? 0 : file_size);
    if(!ec) {
        const auto stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            last_write.time_since_epoch()
        ).count();
        stream << "|mtime=" << static_cast<long long>(stamp);
    }
    return stream.str();
}

template <typename T>
size_t hash_vector_bytes(const std::vector<T> &values) {
    if(values.empty()) {
        return 0;
    }
    const std::string_view bytes(
        reinterpret_cast<const char *>(values.data()),
        values.size() * sizeof(T)
    );
    return std::hash<std::string_view>{}(bytes);
}

std::string build_condition_cache_key_internal(
    const IrodoriTtsSessionConfig &config,
    const IrodoriTtsRequest &request
) {
    std::ostringstream stream;
    stream << build_irodori_condition_cache_key(
        config.mode,
        request.text,
        request.caption,
        request.ref_latent_path.empty() ? request.ref_wav_path : request.ref_latent_path,
        request.no_ref
    );
    stream << "\ntext_ids:" << request.text_token_ids.size() << ':' << hash_vector_bytes(request.text_token_ids);
    stream << "\ntext_mask:" << request.text_token_mask.size() << ':' << hash_vector_bytes(request.text_token_mask);
    stream << "\ncaption_ids:" << request.caption_token_ids.size() << ':' << hash_vector_bytes(request.caption_token_ids);
    stream << "\ncaption_mask:" << request.caption_token_mask.size() << ':' << hash_vector_bytes(request.caption_token_mask);
    stream << "\nref_inline:" << request.ref_latent_steps << ':' << request.ref_latent.size() << ':' << hash_vector_bytes(request.ref_latent);
    stream << "\nlatent_dim:" << config.latent_dim
           << "\nlatent_patch_size:" << config.latent_patch_size
           << "\nspeaker_patch_size:" << config.speaker_patch_size;
    return stream.str();
}

Int64Tensor make_ids_tensor(const std::vector<int64_t> &ids) {
    return {ids, {1, static_cast<int64_t>(ids.size())}};
}

BoolTensor make_mask_tensor(const std::vector<uint8_t> &mask, size_t fallback_count) {
    if(mask.empty()) {
        return {ones_bool(fallback_count), {1, static_cast<int64_t>(fallback_count)}};
    }
    return {mask, {1, static_cast<int64_t>(mask.size())}};
}

BoolTensor make_latent_mask(int32_t actual_steps, int32_t total_steps) {
    const int32_t total = std::max(1, total_steps);
    const int32_t actual = std::clamp(actual_steps, 0, total);
    std::vector<uint8_t> values(static_cast<size_t>(total), uint8_t{0});
    std::fill_n(values.begin(), static_cast<size_t>(actual), uint8_t{1});
    return {std::move(values), {1, static_cast<int64_t>(total)}};
}

Result<FloatTensor> tensor_to_float_tensor(const Ort::Value &value, std::string_view label) {
    if(!value.IsTensor()) {
        return Error::shape_mismatch(std::string(label) + " is not a tensor");
    }
    auto info = value.GetTensorTypeAndShapeInfo();
    if(info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Error::shape_mismatch(std::string(label) + " must be float32");
    }
    const size_t count = info.GetElementCount();
    const float *data = value.GetTensorData<float>();
    FloatTensor out;
    out.shape = info.GetShape();
    out.values.assign(data, data + count);
    return out;
}

Result<BoolTensor> tensor_to_bool_tensor(const Ort::Value &value, std::string_view label) {
    if(!value.IsTensor()) {
        return Error::shape_mismatch(std::string(label) + " is not a tensor");
    }
    auto info = value.GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    BoolTensor out;
    out.shape = info.GetShape();
    out.values.resize(count);
    switch(info.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            const bool *data = value.GetTensorData<bool>();
            for(size_t index = 0; index < count; ++index) {
                out.values[index] = data[index] ? uint8_t{1} : uint8_t{0};
            }
            return out;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            const int64_t *data = value.GetTensorData<int64_t>();
            for(size_t index = 0; index < count; ++index) {
                out.values[index] = data[index] != 0 ? uint8_t{1} : uint8_t{0};
            }
            return out;
        }
        default:
            return Error::shape_mismatch(std::string(label) + " must be bool or int64");
    }
}

Result<Ort::Value> make_float_input(const FloatTensor &tensor) {
    auto tensor_result = gonx::create_float_tensor(tensor.values, tensor.shape);
    if(tensor_result.has_error()) {
        return Error::shape_mismatch("float input shape mismatch: " + tensor_result.error().message);
    }
    return std::move(tensor_result.value());
}

Result<Ort::Value> make_int64_input(const Int64Tensor &tensor) {
    auto tensor_result = gonx::create_int64_tensor(tensor.values, tensor.shape);
    if(tensor_result.has_error()) {
        return Error::shape_mismatch("int64 input shape mismatch: " + tensor_result.error().message);
    }
    return std::move(tensor_result.value());
}

Result<Ort::Value> make_bool_input(const BoolTensor &tensor, gonx::ElementType expected_type) {
    if(expected_type == gonx::ElementType::Int64) {
        std::vector<int64_t> promoted(tensor.values.size(), 0);
        for(size_t index = 0; index < tensor.values.size(); ++index) {
            promoted[index] = tensor.values[index] ? 1 : 0;
        }
        Int64Tensor int_tensor{std::move(promoted), tensor.shape};
        return make_int64_input(int_tensor);
    }
    auto tensor_result = gonx::create_bool_tensor(tensor.values, tensor.shape);
    if(tensor_result.has_error()) {
        return Error::shape_mismatch("bool input shape mismatch: " + tensor_result.error().message);
    }
    return std::move(tensor_result.value());
}

bool input_name_is(const std::string &name, std::initializer_list<std::string_view> options) {
    const std::string normalized = lower_ascii(name);
    for(std::string_view option : options) {
        if(normalized == option) {
            return true;
        }
    }
    return false;
}

Result<std::vector<Ort::Value>> run_text_encoder(
    OptionalSession &encoder,
    const std::vector<int64_t> &token_ids,
    const std::vector<uint8_t> &token_mask,
    std::string_view label
) {
    if(!encoder.loaded) {
        return Error::model_not_loaded(std::string(label) + " is not loaded");
    }
    if(token_ids.empty()) {
        return Error::empty_input(std::string(label) + " received empty token ids");
    }

    const Int64Tensor ids = make_ids_tensor(token_ids);
    const BoolTensor mask = make_mask_tensor(token_mask, token_ids.size());
    std::vector<Ort::Value> inputs;
    inputs.reserve(encoder.session.input_specs().size());
    for(const gonx::TensorSpec &spec : encoder.session.input_specs()) {
        Result<Ort::Value> value = Error::invalid_argument("unmatched input");
        if(input_name_is(spec.name, {"input_ids", "text_input_ids", "caption_input_ids", "tokens", "token_ids"})) {
            value = make_int64_input(ids);
        } else if(input_name_is(spec.name, {"attention_mask", "text_mask", "caption_mask", "mask"})) {
            value = make_bool_input(mask, spec.element_type);
        } else {
            return Error::invalid_argument(
                std::string(label) + " has unsupported input '" + spec.name + "'"
            );
        }
        if(!value.is_ok()) {
            return value.get_error();
        }
        inputs.push_back(std::move(value.value()));
    }

    auto run_result = encoder.session.run(inputs);
    if(run_result.has_error()) {
        return Error::inference_failed(std::string(label) + " failed: " + run_result.error().message);
    }
    return std::move(run_result.value());
}

Result<FloatTensor> run_encoder_state(
    OptionalSession &encoder,
    const std::vector<int64_t> &token_ids,
    const std::vector<uint8_t> &token_mask,
    std::string_view label
) {
    auto outputs_result = run_text_encoder(encoder, token_ids, token_mask, label);
    if(!outputs_result.is_ok()) {
        return outputs_result.get_error();
    }
    auto outputs = std::move(outputs_result.value());
    if(outputs.empty()) {
        return Error::inference_failed(std::string(label) + " returned no outputs");
    }
    return tensor_to_float_tensor(outputs.front(), label);
}

Result<std::vector<float>> read_raw_f32_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if(!file) {
        return Error::not_found("failed to open raw float32 file: " + path);
    }
    file.seekg(0, std::ios::end);
    const std::streamoff byte_count = file.tellg();
    file.seekg(0, std::ios::beg);
    if(byte_count <= 0 || byte_count % static_cast<std::streamoff>(sizeof(float)) != 0) {
        return Error::io_error("raw float32 file has invalid byte length: " + path);
    }
    std::vector<float> values(static_cast<size_t>(byte_count / static_cast<std::streamoff>(sizeof(float))));
    file.read(reinterpret_cast<char *>(values.data()), byte_count);
    if(!file) {
        return Error::io_error("failed to read raw float32 file: " + path);
    }
    return values;
}

Result<FloatTensor> normalize_latent_output(FloatTensor latent, int32_t latent_dim, std::string_view label) {
    if(latent.shape.size() == 2) {
        if(latent.shape[1] != latent_dim) {
            return Error::shape_mismatch(std::string(label) + " output shape must be [T,D] or [1,T,D]");
        }
        latent.shape = {1, latent.shape[0], latent.shape[1]};
        return latent;
    }
    if(latent.shape.size() != 3 || latent.shape[0] != 1) {
        return Error::shape_mismatch(std::string(label) + " output shape must be [1,T,D] or [1,D,T]");
    }
    if(latent.shape[2] == latent_dim) {
        return latent;
    }
    if(latent.shape[1] == latent_dim) {
        const int64_t steps = latent.shape[2];
        std::vector<float> transposed(static_cast<size_t>(steps * latent_dim), 0.0f);
        for(int64_t step = 0; step < steps; ++step) {
            for(int32_t dim = 0; dim < latent_dim; ++dim) {
                transposed[static_cast<size_t>(step * latent_dim + dim)] =
                    latent.values[static_cast<size_t>(dim * steps + step)];
            }
        }
        latent.values = std::move(transposed);
        latent.shape = {1, steps, latent_dim};
        return latent;
    }
    return Error::shape_mismatch(std::string(label) + " output latent dimension does not match config");
}

Result<FloatTensor> run_dacvae_encoder(
    OptionalSession &encoder,
    const std::vector<float> &mono,
    int32_t latent_dim
) {
    if(!encoder.loaded) {
        return Error::model_not_loaded("DACVAE encoder is not loaded");
    }
    if(mono.empty()) {
        return Error::empty_input("reference WAV has no samples");
    }
    FloatTensor waveform{mono, {1, 1, static_cast<int64_t>(mono.size())}};

    std::vector<Ort::Value> inputs;
    inputs.reserve(encoder.session.input_specs().size());
    for(const gonx::TensorSpec &spec : encoder.session.input_specs()) {
        if(input_name_is(spec.name, {"waveform", "wav", "audio", "input"})) {
            auto value = make_float_input(waveform);
            if(!value.is_ok()) {
                return value.get_error();
            }
            inputs.push_back(std::move(value.value()));
        } else {
            return Error::invalid_argument("DACVAE encoder has unsupported input '" + spec.name + "'");
        }
    }

    auto run_result = encoder.session.run(inputs);
    if(run_result.has_error()) {
        return Error::inference_failed("DACVAE encoder failed: " + run_result.error().message);
    }
    auto outputs = std::move(run_result.value());
    if(outputs.empty()) {
        return Error::inference_failed("DACVAE encoder returned no outputs");
    }
    auto latent_result = tensor_to_float_tensor(outputs.front(), "reference_latent");
    if(!latent_result.is_ok()) {
        return latent_result.get_error();
    }
    return normalize_latent_output(std::move(latent_result.value()), latent_dim, "DACVAE encoder");
}

void trim_latent_to_max_ref(FloatTensor &latent, const IrodoriTtsSessionConfig &config) {
    if(latent.shape.size() != 3 || latent.shape[1] <= 0) {
        return;
    }
    if(config.max_ref_seconds <= 0 || config.codec_hop_length <= 0) {
        return;
    }
    const int32_t max_steps = std::max(
        1,
        static_cast<int32_t>(std::ceil(
            static_cast<double>(config.max_ref_seconds) *
            static_cast<double>(config.sample_rate) /
            static_cast<double>(config.codec_hop_length)
        ))
    );
    if(latent.shape[1] <= max_steps) {
        return;
    }
    latent.shape[1] = max_steps;
    latent.values.resize(static_cast<size_t>(max_steps * config.latent_dim));
}

Result<FloatTensor> prepare_reference_latent(
    const IrodoriTtsSessionConfig &config,
    const IrodoriTtsRequest &request,
    OptionalSession &dacvae_encoder,
    std::unordered_map<std::string, FloatTensor> &ref_latent_cache,
    std::mutex &cache_mutex,
    bool *cache_hit,
    int32_t fallback_ref_steps
) {
    const int32_t latent_dim = std::max(1, config.latent_dim);
    if(request.no_ref) {
        return FloatTensor{
            std::vector<float>(static_cast<size_t>(latent_dim), 0.0f),
            {1, 1, latent_dim},
        };
    }
    if(!request.ref_latent.empty()) {
        const int32_t steps = request.ref_latent_steps > 0 ?
            request.ref_latent_steps :
            static_cast<int32_t>(request.ref_latent.size() / static_cast<size_t>(latent_dim));
        if(steps <= 0 || request.ref_latent.size() != static_cast<size_t>(steps * latent_dim)) {
            return Error::shape_mismatch("ref_latent does not match ref_latent_steps * latent_dim");
        }
        FloatTensor latent{request.ref_latent, {1, steps, latent_dim}};
        trim_latent_to_max_ref(latent, config);
        return latent;
    }
    if(!request.ref_latent_path.empty()) {
        const bool use_cache = config.enable_ref_latent_cache && request.ref_latent_cache;
        const std::string cache_key = "f32:" + file_cache_stamp(request.ref_latent_path) +
            "|latent_dim=" + std::to_string(latent_dim) +
            "|max_ref_seconds=" + std::to_string(config.max_ref_seconds);
        if(use_cache) {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto cached = ref_latent_cache.find(cache_key);
            if(cached != ref_latent_cache.end()) {
                if(cache_hit) {
                    *cache_hit = true;
                }
                return cached->second;
            }
        }
        auto values_result = read_raw_f32_file(request.ref_latent_path);
        if(!values_result.is_ok()) {
            return values_result.get_error();
        }
        std::vector<float> values = std::move(values_result.value());
        if(values.size() % static_cast<size_t>(latent_dim) != 0) {
            return Error::shape_mismatch("ref_latent_path float count is not divisible by latent_dim");
        }
        int32_t steps = static_cast<int32_t>(values.size() / static_cast<size_t>(latent_dim));
        if(config.max_ref_seconds > 0 && config.codec_hop_length > 0) {
            const int32_t max_steps = std::max(
                1,
                static_cast<int32_t>(std::ceil(
                    static_cast<double>(config.max_ref_seconds) *
                    static_cast<double>(config.sample_rate) /
                    static_cast<double>(config.codec_hop_length)
                ))
            );
            if(steps > max_steps) {
                steps = max_steps;
                values.resize(static_cast<size_t>(steps * latent_dim));
            }
        }
        FloatTensor latent{std::move(values), {1, steps, latent_dim}};
        if(use_cache) {
            std::lock_guard<std::mutex> lock(cache_mutex);
            ref_latent_cache[cache_key] = latent;
        }
        return latent;
    }
    if(!request.ref_wav_path.empty()) {
        if(!dacvae_encoder.loaded) {
            return Error::model_not_loaded(
                "Irodori ref_wav_path requires dacvae_encoder_onnx_path in the runtime bundle"
            );
        }
        const bool use_cache = config.enable_ref_latent_cache && request.ref_latent_cache;
        const std::string cache_key = "wav:" + file_cache_stamp(request.ref_wav_path) +
            "|sample_rate=" + std::to_string(config.sample_rate) +
            "|latent_dim=" + std::to_string(latent_dim) +
            "|max_ref_seconds=" + std::to_string(config.max_ref_seconds);
        if(use_cache) {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto cached = ref_latent_cache.find(cache_key);
            if(cached != ref_latent_cache.end()) {
                if(cache_hit) {
                    *cache_hit = true;
                }
                return cached->second;
            }
        }
        auto wav_result = load_wav_mono(request.ref_wav_path);
        if(!wav_result.is_ok()) {
            return wav_result.get_error();
        }
        WavData wav = std::move(wav_result.value());
        if(config.max_ref_seconds > 0) {
            const size_t max_samples = static_cast<size_t>(std::max(
                1,
                static_cast<int32_t>(std::llround(
                    static_cast<double>(config.max_ref_seconds) * static_cast<double>(wav.sample_rate)
                ))
            ));
            if(wav.mono.size() > max_samples) {
                wav.mono.resize(max_samples);
            }
        }
        std::vector<float> mono = resample_linear(wav.mono, wav.sample_rate, config.sample_rate);
        float peak = 0.0f;
        for(float sample : mono) {
            peak = std::max(peak, std::abs(sample));
        }
        if(peak > 1.0f && is_finite(peak)) {
            const float scale = 1.0f / peak;
            for(float &sample : mono) {
                sample *= scale;
            }
        }
        auto latent_result = run_dacvae_encoder(dacvae_encoder, mono, latent_dim);
        if(!latent_result.is_ok()) {
            return latent_result.get_error();
        }
        FloatTensor latent = std::move(latent_result.value());
        trim_latent_to_max_ref(latent, config);
        if(use_cache) {
            std::lock_guard<std::mutex> lock(cache_mutex);
            ref_latent_cache[cache_key] = latent;
        }
        return latent;
    }
    if(fallback_ref_steps > 0) {
        return FloatTensor{
            std::vector<float>(static_cast<size_t>(fallback_ref_steps * latent_dim), 0.0f),
            {1, fallback_ref_steps, latent_dim},
        };
    }
    return Error::invalid_argument("Base Irodori TTS requires ref_latent_path, ref_latent, or no_ref=true");
}

Result<std::pair<FloatTensor, BoolTensor>> run_speaker_encoder(
    OptionalSession &encoder,
    const FloatTensor &ref_latent,
    bool has_reference
) {
    if(!encoder.loaded) {
        return Error::model_not_loaded("speaker encoder is not loaded");
    }
    const int64_t ref_steps = ref_latent.shape.size() >= 2 ? ref_latent.shape[1] : 0;
    BoolTensor ref_mask{
        has_reference ? ones_bool(static_cast<size_t>(std::max<int64_t>(1, ref_steps))) :
            zeros_bool(static_cast<size_t>(std::max<int64_t>(1, ref_steps))),
        {1, std::max<int64_t>(1, ref_steps)},
    };

    std::vector<Ort::Value> inputs;
    inputs.reserve(encoder.session.input_specs().size());
    for(const gonx::TensorSpec &spec : encoder.session.input_specs()) {
        Result<Ort::Value> value = Error::invalid_argument("unmatched speaker input");
        if(input_name_is(spec.name, {"ref_latent", "reference_latent", "speaker_latent"})) {
            value = make_float_input(ref_latent);
        } else if(input_name_is(spec.name, {"ref_mask", "reference_mask", "speaker_mask", "mask"})) {
            value = make_bool_input(ref_mask, spec.element_type);
        } else {
            return Error::invalid_argument("speaker encoder has unsupported input '" + spec.name + "'");
        }
        if(!value.is_ok()) {
            return value.get_error();
        }
        inputs.push_back(std::move(value.value()));
    }

    auto run_result = encoder.session.run(inputs);
    if(run_result.has_error()) {
        return Error::inference_failed("speaker encoder failed: " + run_result.error().message);
    }
    auto outputs = std::move(run_result.value());
    if(outputs.empty()) {
        return Error::inference_failed("speaker encoder returned no outputs");
    }
    auto state_result = tensor_to_float_tensor(outputs[0], "speaker_state");
    if(!state_result.is_ok()) {
        return state_result.get_error();
    }
    BoolTensor mask = ref_mask;
    if(outputs.size() > 1) {
        auto mask_result = tensor_to_bool_tensor(outputs[1], "speaker_mask");
        if(!mask_result.is_ok()) {
            return mask_result.get_error();
        }
        mask = std::move(mask_result.value());
    } else if(state_result.value().shape.size() >= 2) {
        mask.values = has_reference ? ones_bool(static_cast<size_t>(state_result.value().shape[1])) :
            zeros_bool(static_cast<size_t>(state_result.value().shape[1]));
        mask.shape = {1, state_result.value().shape[1]};
    }
    return std::make_pair(std::move(state_result.value()), std::move(mask));
}

std::vector<float> build_duration_features_native(
    const std::string &text,
    int64_t token_count,
    int32_t max_text_len,
    bool has_speaker
) {
    const int64_t char_count = std::max<int64_t>(1, static_cast<int64_t>(text.size()));
    auto log1p_cap = [](int64_t count, double cap) {
        return static_cast<float>(std::log1p(static_cast<double>(std::max<int64_t>(0, count))) / std::log1p(cap));
    };
    int64_t period = 0;
    int64_t comma = 0;
    int64_t long_vowel = 0;
    int64_t ellipsis = 0;
    int64_t exclamation = 0;
    int64_t question = 0;
    int64_t alnum = 0;
    for(unsigned char ch : text) {
        if(ch == '.' || ch == 0xE3) {
            // UTF-8 Japanese punctuation is not decoded here; exporter parity tests cover exact Python features.
        }
        if(ch == '.') ++period;
        if(ch == ',') ++comma;
        if(ch == '!') ++exclamation;
        if(ch == '?') ++question;
        if(std::isalnum(ch)) ++alnum;
    }
    (void)long_vowel;
    (void)ellipsis;

    return {
        static_cast<float>(std::clamp<double>(
            static_cast<double>(std::max<int64_t>(0, token_count)) / static_cast<double>(std::max(1, max_text_len)),
            0.0,
            1.0
        )),
        log1p_cap(char_count, 512.0),
        static_cast<float>(static_cast<double>(token_count) / static_cast<double>(char_count)),
        log1p_cap(period, 8.0),
        log1p_cap(comma, 16.0),
        0.0f,
        0.0f,
        log1p_cap(exclamation, 8.0),
        log1p_cap(question, 8.0),
        0.0f,
        0.0f,
        0.0f,
        static_cast<float>(static_cast<double>(alnum) / static_cast<double>(char_count)),
        has_speaker ? 1.0f : 0.0f,
    };
}

Result<int32_t> run_duration_predictor(
    OptionalSession &predictor,
    const IrodoriTtsSessionConfig &config,
    const IrodoriTtsRequest &request,
    const ConditionState &conditions,
    int64_t token_count,
    std::map<std::string, double> *metrics = nullptr
) {
    if(!predictor.loaded) {
        return Error::model_not_loaded("duration predictor is not loaded");
    }
    const std::vector<float> features = build_duration_features_native(
        request.text,
        token_count,
        config.max_text_tokens,
        conditions.has_speaker
    );
    FloatTensor duration_features{features, {1, static_cast<int64_t>(features.size())}};
    BoolTensor has_speaker{
        std::vector<uint8_t>{conditions.has_speaker ? uint8_t{1} : uint8_t{0}},
        {1},
    };

    std::vector<Ort::Value> inputs;
    inputs.reserve(predictor.session.input_specs().size());
    const auto input_started = SteadyClock::now();
    for(const gonx::TensorSpec &spec : predictor.session.input_specs()) {
        Result<Ort::Value> value = Error::invalid_argument("unmatched duration input");
        if(input_name_is(spec.name, {"text_state"})) {
            value = make_float_input(conditions.text_state);
        } else if(input_name_is(spec.name, {"text_mask"})) {
            value = make_bool_input(conditions.text_mask, spec.element_type);
        } else if(input_name_is(spec.name, {"speaker_state"})) {
            value = make_float_input(conditions.speaker_state);
        } else if(input_name_is(spec.name, {"speaker_mask"})) {
            value = make_bool_input(conditions.speaker_mask, spec.element_type);
        } else if(input_name_is(spec.name, {"duration_features", "features", "aux_features"})) {
            value = make_float_input(duration_features);
        } else if(input_name_is(spec.name, {"has_speaker", "duration_has_speaker"})) {
            value = make_bool_input(has_speaker, spec.element_type);
        } else {
            return Error::invalid_argument("duration predictor has unsupported input '" + spec.name + "'");
        }
        if(!value.is_ok()) {
            return value.get_error();
        }
        inputs.push_back(std::move(value.value()));
    }
    add_metric(metrics, "duration_predictor_input_tensor_ms", elapsed_ms_since(input_started));

    const auto run_started = SteadyClock::now();
    auto run_result = predictor.session.run(inputs);
    add_metric(metrics, "duration_predictor_ort_run_ms", elapsed_ms_since(run_started));
    if(run_result.has_error()) {
        return Error::inference_failed("duration predictor failed: " + run_result.error().message);
    }
    auto outputs = std::move(run_result.value());
    if(outputs.empty()) {
        return Error::inference_failed("duration predictor returned no outputs");
    }
    const auto output_started = SteadyClock::now();
    auto log_frames_result = tensor_to_float_tensor(outputs.front(), "duration_log_frames");
    add_metric(metrics, "duration_predictor_output_tensor_ms", elapsed_ms_since(output_started));
    if(!log_frames_result.is_ok()) {
        return log_frames_result.get_error();
    }
    const auto &values = log_frames_result.value().values;
    if(values.empty() || !is_finite(values.front())) {
        return Error::inference_failed("duration predictor returned an invalid frame estimate");
    }
    const float pred_frames = std::expm1(values.front());
    const float scaled = pred_frames * std::max(0.1f, request.duration_scale);
    return std::max(1, static_cast<int32_t>(std::lround(scaled)));
}

Result<FloatTensor> run_dit_step(
    OptionalSession &dit_step,
    const FloatTensor &x_t,
    float t,
    const ConditionState &conditions,
    const BoolTensor &latent_mask,
    std::map<std::string, double> *metrics = nullptr
) {
    if(!dit_step.loaded) {
        return Error::model_not_loaded("DiT step is not loaded");
    }
    FloatTensor t_tensor{{t}, {1}};

    std::vector<Ort::Value> inputs;
    inputs.reserve(dit_step.session.input_specs().size());
    const auto input_started = SteadyClock::now();
    for(const gonx::TensorSpec &spec : dit_step.session.input_specs()) {
        Result<Ort::Value> value = Error::invalid_argument("unmatched DiT input");
        if(input_name_is(spec.name, {"x_t", "latent", "z_t"})) {
            value = make_float_input(x_t);
        } else if(input_name_is(spec.name, {"t", "timestep", "timesteps"})) {
            value = make_float_input(t_tensor);
        } else if(input_name_is(spec.name, {"text_state"})) {
            value = make_float_input(conditions.text_state);
        } else if(input_name_is(spec.name, {"text_mask"})) {
            value = make_bool_input(conditions.text_mask, spec.element_type);
        } else if(input_name_is(spec.name, {"latent_mask", "x_mask", "self_mask"})) {
            value = make_bool_input(latent_mask, spec.element_type);
        } else if(input_name_is(spec.name, {"speaker_state"})) {
            value = make_float_input(conditions.speaker_state);
        } else if(input_name_is(spec.name, {"speaker_mask"})) {
            value = make_bool_input(conditions.speaker_mask, spec.element_type);
        } else if(input_name_is(spec.name, {"caption_state"})) {
            value = make_float_input(conditions.caption_state);
        } else if(input_name_is(spec.name, {"caption_mask"})) {
            value = make_bool_input(conditions.caption_mask, spec.element_type);
        } else {
            return Error::invalid_argument("DiT step has unsupported input '" + spec.name + "'");
        }
        if(!value.is_ok()) {
            return value.get_error();
        }
        inputs.push_back(std::move(value.value()));
    }
    add_metric(metrics, "dit_step_input_tensor_ms", elapsed_ms_since(input_started));
    add_metric(metrics, "dit_step_invocations", 1.0);

    const auto run_started = SteadyClock::now();
    auto run_result = dit_step.session.run(inputs);
    add_metric(metrics, "dit_step_ort_run_ms", elapsed_ms_since(run_started));
    if(run_result.has_error()) {
        return Error::inference_failed("DiT step failed: " + run_result.error().message);
    }
    auto outputs = std::move(run_result.value());
    if(outputs.empty()) {
        return Error::inference_failed("DiT step returned no outputs");
    }
    const auto output_started = SteadyClock::now();
    auto tensor_result = tensor_to_float_tensor(outputs.front(), "dit_v_pred");
    add_metric(metrics, "dit_step_output_tensor_ms", elapsed_ms_since(output_started));
    return tensor_result;
}

Result<FloatTensor> run_rf_sampler(
    OptionalSession &sampler,
    const FloatTensor &x_t,
    const ConditionState &conditions,
    const BoolTensor &latent_mask,
    std::map<std::string, double> *metrics = nullptr
) {
    if(!sampler.loaded) {
        return Error::model_not_loaded("RF sampler is not loaded");
    }

    std::vector<Ort::Value> inputs;
    inputs.reserve(sampler.session.input_specs().size());
    const auto input_started = SteadyClock::now();
    for(const gonx::TensorSpec &spec : sampler.session.input_specs()) {
        Result<Ort::Value> value = Error::invalid_argument("unmatched RF sampler input");
        if(input_name_is(spec.name, {"x_t", "latent", "z_t"})) {
            value = make_float_input(x_t);
        } else if(input_name_is(spec.name, {"text_state"})) {
            value = make_float_input(conditions.text_state);
        } else if(input_name_is(spec.name, {"text_mask"})) {
            value = make_bool_input(conditions.text_mask, spec.element_type);
        } else if(input_name_is(spec.name, {"latent_mask", "x_mask", "self_mask"})) {
            value = make_bool_input(latent_mask, spec.element_type);
        } else if(input_name_is(spec.name, {"speaker_state"})) {
            value = make_float_input(conditions.speaker_state);
        } else if(input_name_is(spec.name, {"speaker_mask"})) {
            value = make_bool_input(conditions.speaker_mask, spec.element_type);
        } else if(input_name_is(spec.name, {"caption_state"})) {
            value = make_float_input(conditions.caption_state);
        } else if(input_name_is(spec.name, {"caption_mask"})) {
            value = make_bool_input(conditions.caption_mask, spec.element_type);
        } else {
            return Error::invalid_argument("RF sampler has unsupported input '" + spec.name + "'");
        }
        if(!value.is_ok()) {
            return value.get_error();
        }
        inputs.push_back(std::move(value.value()));
    }
    add_metric(metrics, "rf_sampler_input_tensor_ms", elapsed_ms_since(input_started));
    add_metric(metrics, "rf_sampler_invocations", 1.0);

    const auto run_started = SteadyClock::now();
    auto run_result = sampler.session.run(inputs);
    add_metric(metrics, "rf_sampler_ort_run_ms", elapsed_ms_since(run_started));
    if(run_result.has_error()) {
        return Error::inference_failed("RF sampler failed: " + run_result.error().message);
    }
    auto outputs = std::move(run_result.value());
    if(outputs.empty()) {
        return Error::inference_failed("RF sampler returned no outputs");
    }
    const auto output_started = SteadyClock::now();
    auto tensor_result = tensor_to_float_tensor(outputs.front(), "rf_sampler_x_0");
    add_metric(metrics, "rf_sampler_output_tensor_ms", elapsed_ms_since(output_started));
    return tensor_result;
}

ConditionState drop_condition(const ConditionState &input, std::string_view name) {
    ConditionState output = input;
    if(name == "text" || name == "all") {
        output.text_state = zeros_like(input.text_state);
        output.text_mask = zeros_like(input.text_mask);
    }
    if((name == "speaker" || name == "all") && input.has_speaker) {
        output.speaker_state = zeros_like(input.speaker_state);
        output.speaker_mask = zeros_like(input.speaker_mask);
    }
    if((name == "caption" || name == "all") && input.has_caption) {
        output.caption_state = zeros_like(input.caption_state);
        output.caption_mask = zeros_like(input.caption_mask);
    }
    return output;
}

void add_guided_velocity(
    std::vector<float> &v,
    const std::vector<float> &cond,
    const std::vector<float> &uncond,
    float scale
) {
    if(v.empty()) {
        v = cond;
    }
    const size_t count = std::min(v.size(), std::min(cond.size(), uncond.size()));
    for(size_t index = 0; index < count; ++index) {
        v[index] += scale * (cond[index] - uncond[index]);
    }
}

Result<FloatTensor> sample_rf(
    OptionalSession &dit_step,
    const IrodoriTtsSessionConfig &config,
    const IrodoriTtsRequest &request,
    const ConditionState &conditions,
    int32_t actual_patched_steps,
    int32_t patched_steps,
    const std::vector<float> &schedule,
    const std::vector<std::string> &branches,
    CancellationToken *cancel,
    std::map<std::string, double> *metrics = nullptr,
    std::map<std::string, std::string> *diagnostics = nullptr
) {
    add_diagnostic(diagnostics, "rf_execution_mode", "cpu_vector");
    const int32_t patched_dim = std::max(1, config.latent_dim * config.latent_patch_size);
    FloatTensor x_t;
    x_t.shape = {1, patched_steps, patched_dim};
    x_t.values.resize(static_cast<size_t>(patched_steps * patched_dim));
    const uint64_t seed = request.seed >= 0 ?
        static_cast<uint64_t>(request.seed) :
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for(float &value : x_t.values) {
        value = normal(rng);
    }
    const BoolTensor latent_mask = make_latent_mask(actual_patched_steps, patched_steps);

    const std::string guidance = lower_ascii(
        request.cfg_guidance_mode.empty() ? config.default_cfg_guidance_mode : request.cfg_guidance_mode
    );
    std::unordered_map<std::string, ConditionState> dropped_conditions;
    auto dropped_condition_for = [&](const std::string &branch) -> const ConditionState & {
        auto found = dropped_conditions.find(branch);
        if(found != dropped_conditions.end()) {
            return found->second;
        }
        auto inserted = dropped_conditions.emplace(branch, drop_condition(conditions, branch));
        return inserted.first->second;
    };

    for(size_t step = 0; step + 1 < schedule.size(); ++step) {
        const auto step_started = SteadyClock::now();
        if(cancel && cancel->is_cancelled()) {
            return Error::cancelled("IrodoriTtsSession::synthesize: request cancelled");
        }
        const float t = schedule[step];
        const float t_next = schedule[step + 1];
        const bool use_cfg = !branches.empty() && branches.size() > 1 &&
            request.cfg_min_t <= t && t <= request.cfg_max_t;

        auto cond_result = run_dit_step(dit_step, x_t, t, conditions, latent_mask, metrics);
        if(!cond_result.is_ok()) {
            return cond_result.get_error();
        }
        FloatTensor v = cond_result.value();
        if(use_cfg) {
            if(guidance == "joint") {
                auto uncond_result = run_dit_step(
                    dit_step,
                    x_t,
                    t,
                    dropped_condition_for("all"),
                    latent_mask,
                    metrics
                );
                if(!uncond_result.is_ok()) {
                    return uncond_result.get_error();
                }
                const float scale = request.cfg_scale.has_value() ? *request.cfg_scale : request.cfg_scale_text;
                add_guided_velocity(v.values, cond_result.value().values, uncond_result.value().values, scale);
            } else if(guidance == "alternating") {
                const size_t branch_index = 1 + (step % (branches.size() - 1));
                const std::string branch = branches[branch_index].rfind("drop_", 0) == 0 ?
                    branches[branch_index].substr(5) : branches[branch_index];
                auto uncond_result = run_dit_step(
                    dit_step,
                    x_t,
                    t,
                    dropped_condition_for(branch),
                    latent_mask,
                    metrics
                );
                if(!uncond_result.is_ok()) {
                    return uncond_result.get_error();
                }
                float scale = request.cfg_scale_text;
                if(branch == "speaker") {
                    scale = request.cfg_scale_speaker;
                } else if(branch == "caption") {
                    scale = request.cfg_scale_caption;
                }
                if(request.cfg_scale.has_value()) {
                    scale = *request.cfg_scale;
                }
                add_guided_velocity(v.values, cond_result.value().values, uncond_result.value().values, scale);
            } else {
                for(size_t branch_index = 1; branch_index < branches.size(); ++branch_index) {
                    const std::string branch = branches[branch_index].rfind("drop_", 0) == 0 ?
                        branches[branch_index].substr(5) : branches[branch_index];
                    auto uncond_result = run_dit_step(
                        dit_step,
                        x_t,
                        t,
                        dropped_condition_for(branch),
                        latent_mask,
                        metrics
                    );
                    if(!uncond_result.is_ok()) {
                        return uncond_result.get_error();
                    }
                    float scale = request.cfg_scale_text;
                    if(branch == "speaker") {
                        scale = request.cfg_scale_speaker;
                    } else if(branch == "caption") {
                        scale = request.cfg_scale_caption;
                    }
                    if(request.cfg_scale.has_value()) {
                        scale = *request.cfg_scale;
                    }
                    add_guided_velocity(v.values, cond_result.value().values, uncond_result.value().values, scale);
                }
            }
        }

        if(v.values.size() != x_t.values.size()) {
            return Error::shape_mismatch(
                "DiT step output shape " + shape_to_string(v.shape) +
                " does not match latent shape " + shape_to_string(x_t.shape)
            );
        }
        const float dt = t_next - t;
        for(size_t index = 0; index < x_t.values.size(); ++index) {
            x_t.values[index] += v.values[index] * dt;
        }
        add_metric(metrics, "rf_step_total_ms", elapsed_ms_since(step_started));
        set_metric(metrics, "rf_step_" + std::to_string(step) + "_ms", elapsed_ms_since(step_started));
    }

    return x_t;
}

Result<FloatTensor> sample_rf_unrolled(
    OptionalSession &sampler,
    const IrodoriTtsSessionConfig &config,
    const IrodoriTtsRequest &request,
    const ConditionState &conditions,
    int32_t actual_patched_steps,
    int32_t patched_steps,
    CancellationToken *cancel,
    std::map<std::string, double> *metrics = nullptr,
    std::map<std::string, std::string> *diagnostics = nullptr
) {
    add_diagnostic(diagnostics, "rf_execution_mode", "unrolled_run");
    if(cancel && cancel->is_cancelled()) {
        return Error::cancelled("IrodoriTtsSession::synthesize: request cancelled");
    }

    const int32_t patched_dim = std::max(1, config.latent_dim * config.latent_patch_size);
    FloatTensor x_t;
    x_t.shape = {1, patched_steps, patched_dim};
    x_t.values.resize(static_cast<size_t>(patched_steps * patched_dim));
    const uint64_t seed = request.seed >= 0 ?
        static_cast<uint64_t>(request.seed) :
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for(float &value : x_t.values) {
        value = normal(rng);
    }

    const BoolTensor latent_mask = make_latent_mask(actual_patched_steps, patched_steps);
    return run_rf_sampler(sampler, x_t, conditions, latent_mask, metrics);
}

Result<FloatTensor> unpatchify_latent(
    const FloatTensor &patched,
    int32_t latent_dim,
    int32_t latent_patch_size,
    int32_t target_latent_steps
) {
    if(patched.shape.size() != 3 || patched.shape[0] != 1) {
        return Error::shape_mismatch("patched latent must have shape [1, S, D]");
    }
    const int64_t patched_steps = patched.shape[1];
    const int64_t patched_dim = patched.shape[2];
    const int64_t expected_dim = static_cast<int64_t>(latent_dim) * static_cast<int64_t>(latent_patch_size);
    if(patched_dim != expected_dim) {
        return Error::shape_mismatch("patched latent dim does not equal latent_dim * latent_patch_size");
    }
    const int64_t full_steps = patched_steps * static_cast<int64_t>(latent_patch_size);
    FloatTensor out;
    out.shape = {1, std::min<int64_t>(target_latent_steps, full_steps), latent_dim};
    out.values.resize(static_cast<size_t>(out.shape[1] * latent_dim));
    size_t write = 0;
    for(int64_t step = 0; step < patched_steps; ++step) {
        for(int32_t patch = 0; patch < latent_patch_size; ++patch) {
            const int64_t latent_step = step * latent_patch_size + patch;
            if(latent_step >= out.shape[1]) {
                break;
            }
            const size_t read = static_cast<size_t>(
                (step * patched_dim) + (static_cast<int64_t>(patch) * latent_dim)
            );
            std::copy_n(patched.values.begin() + static_cast<std::ptrdiff_t>(read), latent_dim, out.values.begin() + static_cast<std::ptrdiff_t>(write));
            write += static_cast<size_t>(latent_dim);
        }
    }
    return out;
}

Result<std::vector<float>> decode_latent(
    OptionalSession &decoder,
    const FloatTensor &latent,
    std::map<std::string, double> *metrics = nullptr
) {
    if(!decoder.loaded) {
        return Error::model_not_loaded("DACVAE decoder is not loaded");
    }
    std::vector<Ort::Value> inputs;
    inputs.reserve(decoder.session.input_specs().size());
    const auto input_started = SteadyClock::now();
    for(const gonx::TensorSpec &spec : decoder.session.input_specs()) {
        if(input_name_is(spec.name, {"z", "latent", "latents"})) {
            auto value = make_float_input(latent);
            if(!value.is_ok()) {
                return value.get_error();
            }
            inputs.push_back(std::move(value.value()));
        } else {
            return Error::invalid_argument("DACVAE decoder has unsupported input '" + spec.name + "'");
        }
    }
    add_metric(metrics, "dacvae_decode_input_tensor_ms", elapsed_ms_since(input_started));
    add_metric(metrics, "dacvae_decode_invocations", 1.0);
    const auto run_started = SteadyClock::now();
    auto run_result = decoder.session.run(inputs);
    add_metric(metrics, "dacvae_decode_ort_run_ms", elapsed_ms_since(run_started));
    if(run_result.has_error()) {
        return Error::inference_failed("DACVAE decoder failed: " + run_result.error().message);
    }
    auto outputs = std::move(run_result.value());
    if(outputs.empty()) {
        return Error::inference_failed("DACVAE decoder returned no outputs");
    }
    const auto output_started = SteadyClock::now();
    auto waveform_result = tensor_to_float_tensor(outputs.front(), "waveform");
    add_metric(metrics, "dacvae_decode_output_tensor_ms", elapsed_ms_since(output_started));
    if(!waveform_result.is_ok()) {
        return waveform_result.get_error();
    }
    return std::move(waveform_result.value().values);
}

Result<void> load_optional_session(
    OptionalSession &target,
    const std::string &path,
    const gonx::SessionConfig &config,
    bool required,
    const IrodoriTtsProviderRoute &route,
    std::string_view label
) {
    target.path = path;
    target.provider_requested = route.provider_requested;
    target.provider_effective = route.provider;
    target.requested = required || !path.empty();
    target.loaded = false;
    target.session_create_ms = 0.0;
    target.profiling_file_path.clear();

    if(path.empty()) {
        if(required) {
            return Error::invalid_argument("IrodoriTtsSession::load: missing " + std::string(label) + " path");
        }
        return Result<void>();
    }

    if(!path_exists(path)) {
        if(required) {
            return Error::not_found("IrodoriTtsSession::load: " + std::string(label) + " not found at " + path);
        }
        return Result<void>();
    }

    auto status = target.session.load(path, config);
    if(status.has_error()) {
        return Error::model_not_loaded(
            "IrodoriTtsSession::load: failed to load " + std::string(label) + " from " + path + ": " +
            status.error().message
        );
    }

    target.loaded = true;
    target.session_create_ms = target.session.session_create_ms();
    return Result<void>();
}

IrodoriTtsProviderRoute resolve_provider_route(
    const IrodoriTtsSessionConfig &config,
    const IrodoriTtsProviderRoute &route
) {
    IrodoriTtsProviderRoute resolved;
    resolved.provider_requested = route.provider_requested.empty() ?
        config.provider_requested : route.provider_requested;
    resolved.provider = route.provider.empty() ? config.provider : route.provider;
    if(resolved.provider_requested.empty()) {
        resolved.provider_requested = "CPU";
    }
    if(resolved.provider.empty()) {
        resolved.provider = "CPU";
    }
    return resolved;
}

bool provider_is_available(const std::string &provider) {
    const gonx::ExecutionProvider requested = gonx::parse_provider(provider);
    if(requested == gonx::ExecutionProvider::CPU) {
        return true;
    }
    const std::vector<gonx::ExecutionProvider> available = gonx::available_providers();
    return std::find(available.begin(), available.end(), requested) != available.end();
}

bool requested_provider_requires_non_cpu(const std::string &provider) {
    const std::string normalized = lower_ascii(provider);
    return normalized == "gpu" ||
        normalized == "auto_gpu" ||
        normalized == "coreml" ||
        normalized == "coremlexecutionprovider" ||
        normalized == "cuda" ||
        normalized == "cudaexecutionprovider" ||
        normalized == "tensorrt" ||
        normalized == "trt" ||
        normalized == "tensorrtexecutionprovider" ||
        normalized == "directml" ||
        normalized == "dmlexecutionprovider" ||
        normalized == "openvino" ||
        normalized == "openvinoexecutionprovider";
}

Result<void> validate_provider_route(
    const IrodoriTtsSessionConfig &config,
    const IrodoriTtsProviderRoute &route,
    std::string_view label
) {
    if(!config.strict_provider) {
        return Result<void>();
    }
    if(requested_provider_requires_non_cpu(route.provider_requested) &&
       gonx::parse_provider(route.provider) == gonx::ExecutionProvider::CPU) {
        return Error::model_not_loaded(
            "IrodoriTtsSession::load: strict provider requested non-CPU for " +
            std::string(label) + " but effective provider is CPU"
        );
    }
    if(!provider_is_available(route.provider)) {
        return Error::model_not_loaded(
            "IrodoriTtsSession::load: strict provider '" + route.provider +
            "' is unavailable for " + std::string(label)
        );
    }
    return Result<void>();
}

bool buckets_match(const IrodoriTtsBucket &lhs, const IrodoriTtsBucket &rhs) {
    return lhs.latent_steps == rhs.latent_steps &&
        lhs.text_tokens == rhs.text_tokens &&
        lhs.caption_tokens == rhs.caption_tokens &&
        lhs.ref_steps == rhs.ref_steps;
}

gonx::SessionConfig make_session_config(
    const IrodoriTtsSessionConfig &config,
    const IrodoriTtsProviderRoute &route
) {
    gonx::SessionConfig session_config;
    session_config.providers = {gonx::parse_provider(route.provider)};
    session_config.device_id = config.device_id;
    session_config.intra_op_num_threads = config.intra_op_threads;
    session_config.inter_op_num_threads = config.inter_op_threads;
    session_config.optimization_level = config.optimization_level;
    session_config.optimized_model_path = config.optimized_model_path;
    for(const auto &[key, value] : config.provider_options) {
        session_config.provider_options[key] = value;
    }
    for(const auto &[key, value] : config.session_options) {
        session_config.session_options[key] = value;
    }
    session_config.enable_profiling = config.ort_enable_profiling;
    session_config.profiling_file_prefix = config.ort_profiling_prefix.empty() ?
        std::string() :
        config.ort_profiling_prefix + "_" + std::string(route.provider.empty() ? "CPU" : route.provider);
    session_config.log_severity_level = config.ort_log_severity_level;
    return session_config;
}

bool scales_enabled(float scale) {
    return scale > 0.0f && is_finite(scale);
}

bool can_use_unrolled_rf_sampler(
    int32_t num_steps,
    const std::string &schedule_mode,
    float sway_coeff,
    const std::vector<std::string> &branches
) {
    if(num_steps != 6 && num_steps != 8) {
        return false;
    }
    if(lower_ascii(schedule_mode) != "sway") {
        return false;
    }
    if(std::abs(sway_coeff - (-1.0f)) > 0.0001f) {
        return false;
    }
    return branches.size() <= 1;
}

Result<void> ensure_static_rf_sampler_loaded(
    StaticDitSession &static_session,
    int32_t num_steps,
    const IrodoriTtsSessionConfig &config
) {
    OptionalSession *sampler = nullptr;
    std::string path;
    std::string label;
    if(num_steps == 6) {
        sampler = &static_session.rf_sampler_6_step;
        path = static_session.rf_sampler_6_step_path;
        label = "static CoreML RF sampler 6-step";
    } else if(num_steps == 8) {
        sampler = &static_session.rf_sampler_8_step;
        path = static_session.rf_sampler_8_step_path;
        label = "static CoreML RF sampler 8-step";
    } else {
        return Error::invalid_argument("IrodoriTtsSession::synthesize: RF sampler supports only 6 or 8 steps");
    }

    if(sampler->loaded || path.empty()) {
        return Result<void>();
    }

    const IrodoriTtsProviderRoute route = resolve_provider_route(config, config.provider_routes.dit_step);
    auto provider_check = validate_provider_route(config, route, label);
    if(!provider_check.is_ok()) {
        return provider_check;
    }
    return load_optional_session(
        *sampler,
        path,
        make_session_config(config, route),
        false,
        route,
        label
    );
}

Result<void> ensure_static_dit_loaded(
    StaticDitSession &static_session,
    const IrodoriTtsSessionConfig &config
) {
    if(static_session.dit_step.loaded) {
        return Result<void>();
    }
    const IrodoriTtsProviderRoute route = resolve_provider_route(config, config.provider_routes.dit_step);
    auto provider_check = validate_provider_route(config, route, "static CoreML DiT step");
    if(!provider_check.is_ok()) {
        return provider_check;
    }
    return load_optional_session(
        static_session.dit_step,
        static_session.dit_step.path,
        make_session_config(config, route),
        true,
        route,
        "static CoreML DiT step"
    );
}

void unload_static_session(StaticDitSession &static_session) {
    static_session.dit_step.session = gonx::InferenceSession();
    static_session.dit_step.loaded = false;
    static_session.rf_sampler_6_step.session = gonx::InferenceSession();
    static_session.rf_sampler_6_step.loaded = false;
    static_session.rf_sampler_8_step.session = gonx::InferenceSession();
    static_session.rf_sampler_8_step.loaded = false;
}

void touch_static_bucket_lru(
    std::vector<StaticDitSession> &sessions,
    std::vector<std::string> &lru,
    const IrodoriTtsBucket &bucket
) {
    const std::string key = bucket_cache_key(bucket);
    lru.erase(std::remove(lru.begin(), lru.end(), key), lru.end());
    lru.push_back(key);

    while(lru.size() > kCoreMLStaticBucketCacheLimit) {
        const std::string evict_key = lru.front();
        lru.erase(lru.begin());
        if(evict_key == key) {
            continue;
        }
        for(StaticDitSession &session : sessions) {
            if(bucket_cache_key(session.bucket) == evict_key) {
                unload_static_session(session);
                break;
            }
        }
    }
}

} // namespace

IrodoriTtsMode parse_irodori_tts_mode(const std::string &mode) {
    const std::string normalized = lower_ascii(mode);
    if(normalized == "base_v3" || normalized == "v3" || normalized == "base") {
        return IrodoriTtsMode::BaseV3;
    }
    if(normalized == "base_v2" || normalized == "v2") {
        return IrodoriTtsMode::BaseV2;
    }
    if(normalized == "voice_design_v2" || normalized == "voice_design" || normalized == "voicedesign") {
        return IrodoriTtsMode::VoiceDesignV2;
    }
    return IrodoriTtsMode::Unknown;
}

std::string irodori_tts_mode_name(IrodoriTtsMode mode) {
    switch(mode) {
        case IrodoriTtsMode::BaseV3:
            return "base_v3";
        case IrodoriTtsMode::BaseV2:
            return "base_v2";
        case IrodoriTtsMode::VoiceDesignV2:
            return "voice_design_v2";
        case IrodoriTtsMode::Unknown:
        default:
            return "unknown";
    }
}

bool irodori_tts_mode_uses_caption(IrodoriTtsMode mode) {
    return mode == IrodoriTtsMode::VoiceDesignV2;
}

bool irodori_tts_mode_uses_speaker(IrodoriTtsMode mode) {
    return mode == IrodoriTtsMode::BaseV2 || mode == IrodoriTtsMode::BaseV3;
}

Result<std::vector<float>> build_irodori_sway_schedule(
    int32_t num_steps,
    const std::string &mode,
    float sway_coeff
) {
    if(num_steps <= 0) {
        return Error::invalid_argument("build_irodori_sway_schedule: num_steps must be positive");
    }
    if(!is_finite(sway_coeff)) {
        return Error::invalid_argument("build_irodori_sway_schedule: sway_coeff must be finite");
    }

    const std::string normalized = lower_ascii(mode);
    if(normalized != "linear" && normalized != "sway") {
        return Error::invalid_argument("build_irodori_sway_schedule: mode must be linear or sway");
    }

    std::vector<float> schedule;
    schedule.reserve(static_cast<size_t>(num_steps + 1));
    for(int32_t index = 0; index <= num_steps; ++index) {
        float u = static_cast<float>(index) / static_cast<float>(num_steps);
        if(normalized == "sway") {
            u = u + sway_coeff * (std::cos(0.5f * std::numbers::pi_v<float> * u) + u - 1.0f);
            u = std::clamp(u, 0.0f, 1.0f);
        }
        schedule.push_back((1.0f - u) * kIrodoriInitScale);
    }

    for(size_t index = 0; index + 1 < schedule.size(); ++index) {
        if(!(schedule[index] > schedule[index + 1])) {
            return Error::invalid_argument(
                "build_irodori_sway_schedule: schedule is not strictly decreasing"
            );
        }
    }

    return schedule;
}

Result<IrodoriTtsBucket> select_irodori_bucket(
    const std::vector<IrodoriTtsBucket> &buckets,
    int32_t latent_steps,
    int32_t text_tokens,
    int32_t caption_tokens,
    int32_t ref_steps
) {
    if(latent_steps <= 0 || text_tokens <= 0 || caption_tokens < 0 || ref_steps < 0) {
        return Error::invalid_argument("select_irodori_bucket: invalid requested dimensions");
    }
    if(buckets.empty()) {
        return Error::invalid_argument("select_irodori_bucket: no buckets configured");
    }

    const IrodoriTtsBucket *best = nullptr;
    int64_t best_cost = std::numeric_limits<int64_t>::max();
    for(const IrodoriTtsBucket &bucket : buckets) {
        const int64_t cost = bucket_cost(bucket, latent_steps, text_tokens, caption_tokens, ref_steps);
        if(cost < best_cost) {
            best = &bucket;
            best_cost = cost;
        }
    }

    if(!best) {
        return Error::shape_mismatch("select_irodori_bucket: no bucket can fit the requested dimensions");
    }

    return *best;
}

Result<std::vector<std::string>> build_irodori_cfg_branches(
    IrodoriTtsMode mode,
    const std::string &guidance_mode,
    float cfg_scale_text,
    float cfg_scale_caption,
    float cfg_scale_speaker
) {
    const std::string normalized = lower_ascii(guidance_mode.empty() ? "independent" : guidance_mode);
    if(normalized != "independent" && normalized != "joint" && normalized != "alternating") {
        return Error::invalid_argument("build_irodori_cfg_branches: unsupported CFG guidance mode");
    }

    std::vector<std::string> enabled;
    if(scales_enabled(cfg_scale_text)) {
        enabled.push_back("text");
    }
    if(irodori_tts_mode_uses_speaker(mode) && scales_enabled(cfg_scale_speaker)) {
        enabled.push_back("speaker");
    }
    if(irodori_tts_mode_uses_caption(mode) && scales_enabled(cfg_scale_caption)) {
        enabled.push_back("caption");
    }

    std::vector<std::string> branches = {"conditional"};
    if(enabled.empty()) {
        return branches;
    }

    if(normalized == "joint") {
        for(size_t index = 1; index < enabled.size(); ++index) {
            const std::string &name = enabled[index];
            const std::string &first = enabled.front();
            const float lhs = first == "text" ? cfg_scale_text :
                (first == "speaker" ? cfg_scale_speaker : cfg_scale_caption);
            const float rhs = name == "text" ? cfg_scale_text :
                (name == "speaker" ? cfg_scale_speaker : cfg_scale_caption);
            if(std::abs(lhs - rhs) > 0.0001f) {
                return Error::invalid_argument(
                    "build_irodori_cfg_branches: joint CFG requires equal enabled guidance scales"
                );
            }
        }
        branches.push_back("drop_all");
        return branches;
    }

    for(const std::string &name : enabled) {
        branches.push_back("drop_" + name);
    }
    return branches;
}

std::string build_irodori_condition_cache_key(
    IrodoriTtsMode mode,
    const std::string &text,
    const std::string &caption,
    const std::string &ref_latent_path,
    bool no_ref
) {
    std::ostringstream stream;
    stream << irodori_tts_mode_name(mode) << '\n'
           << "text:" << text << '\n'
           << "caption:" << caption << '\n'
           << "ref:" << (no_ref ? "<none>" : ref_latent_path);
    return stream.str();
}

struct IrodoriTtsSession::Impl {
    IrodoriTtsSessionConfig config;
    bool loaded = false;
    bool execution_ready = false;
    OptionalSession text_encoder;
    OptionalSession caption_encoder;
    OptionalSession speaker_encoder;
    OptionalSession duration_predictor;
    OptionalSession dit_step;
    OptionalSession dacvae_encoder;
    OptionalSession dacvae_decoder;
    std::vector<StaticDitSession> coreml_static_dit_steps;
    std::vector<std::string> coreml_static_bucket_lru;
    std::mutex cache_mutex;
    std::unordered_map<std::string, FloatTensor> ref_latent_cache;
    std::unordered_map<std::string, CachedConditionState> condition_cache;
};

IrodoriTtsSession::IrodoriTtsSession() : impl_(std::make_unique<Impl>()) {}
IrodoriTtsSession::~IrodoriTtsSession() = default;
IrodoriTtsSession::IrodoriTtsSession(IrodoriTtsSession &&) noexcept = default;
IrodoriTtsSession &IrodoriTtsSession::operator=(IrodoriTtsSession &&) noexcept = default;

Result<void> IrodoriTtsSession::load(const IrodoriTtsSessionConfig &config) {
    impl_->loaded = false;
    impl_->execution_ready = false;
    impl_->config = config;

    if(config.mode == IrodoriTtsMode::Unknown) {
        return Error::invalid_argument("IrodoriTtsSession::load: mode must be base_v3, base_v2, or voice_design_v2");
    }
    if(config.sample_rate != 48000) {
        return Error::invalid_argument("IrodoriTtsSession::load: Irodori runtime expects 48000 Hz output");
    }
    if(config.latent_dim <= 0) {
        return Error::invalid_argument("IrodoriTtsSession::load: latent_dim must be positive");
    }
    if(config.latent_patch_size <= 0 || config.speaker_patch_size <= 0) {
        return Error::invalid_argument("IrodoriTtsSession::load: latent patch sizes must be positive");
    }
    if(config.codec_hop_length <= 0) {
        return Error::invalid_argument("IrodoriTtsSession::load: codec_hop_length must be positive");
    }
    if(config.default_num_steps <= 0) {
        return Error::invalid_argument("IrodoriTtsSession::load: default_num_steps must be positive");
    }
    auto schedule_result = build_irodori_sway_schedule(
        config.default_num_steps,
        config.default_t_schedule_mode,
        config.default_sway_coeff
    );
    if(!schedule_result.is_ok()) {
        return schedule_result.get_error();
    }
    if(!config.manifest_path.empty() && !path_exists(config.manifest_path) && config.require_all_artifacts) {
        return Error::not_found("IrodoriTtsSession::load: manifest not found at " + config.manifest_path);
    }

    impl_->config.buckets = config.buckets.empty() ? default_buckets(config.mode) : config.buckets;

    const bool require = config.require_all_artifacts;
    const bool uses_caption = irodori_tts_mode_uses_caption(config.mode);
    const bool uses_speaker = irodori_tts_mode_uses_speaker(config.mode);
    const std::string normalized_profile = lower_ascii(config.provider_profile);
    const bool static_profile = normalized_profile == "coreml_static" ||
        normalized_profile == "coreml_static_unrolled";
    const std::string load_dispatch_mode = lower_ascii(config.runtime_dispatch.empty() ?
        "force_cpu" : config.runtime_dispatch);
    const bool cpu_dispatch_required = load_dispatch_mode == "force_cpu" ||
        load_dispatch_mode == "auto_dispatch";

    auto load_stage = [&](OptionalSession &session,
                          const std::string &path,
                          bool required,
                          const IrodoriTtsProviderRoute &route,
                          std::string_view label) -> Result<void> {
        const IrodoriTtsProviderRoute resolved = resolve_provider_route(config, route);
        if(path.empty() && !required) {
            return load_optional_session(session, path, gonx::SessionConfig{}, required, resolved, label);
        }
        auto provider_check = validate_provider_route(config, resolved, label);
        if(!provider_check.is_ok()) {
            return provider_check;
        }
        const gonx::SessionConfig session_config = make_session_config(config, resolved);
        return load_optional_session(session, path, session_config, required, resolved, label);
    };

    auto load_result = load_stage(
        impl_->text_encoder,
        config.artifacts.text_encoder_onnx_path,
        require,
        config.provider_routes.text_encoder,
        "text encoder"
    );
    if(!load_result.is_ok()) {
        return load_result;
    }
    load_result = load_stage(
        impl_->caption_encoder,
        config.artifacts.caption_encoder_onnx_path,
        require && uses_caption,
        config.provider_routes.caption_encoder,
        "caption encoder"
    );
    if(!load_result.is_ok()) {
        return load_result;
    }
    load_result = load_stage(
        impl_->speaker_encoder,
        config.artifacts.speaker_encoder_onnx_path,
        require && uses_speaker,
        config.provider_routes.speaker_encoder,
        "speaker encoder"
    );
    if(!load_result.is_ok()) {
        return load_result;
    }
    load_result = load_stage(
        impl_->duration_predictor,
        config.artifacts.duration_predictor_onnx_path,
        require && config.mode == IrodoriTtsMode::BaseV3,
        config.provider_routes.duration_predictor,
        "duration predictor"
    );
    if(!load_result.is_ok()) {
        return load_result;
    }
    const IrodoriTtsProviderRoute cpu_dit_route{"CPU", "CPU"};
    load_result = load_stage(
        impl_->dit_step,
        config.artifacts.dit_step_onnx_path,
        require && (!static_profile || cpu_dispatch_required),
        static_profile ? cpu_dit_route : config.provider_routes.dit_step,
        "DiT step"
    );
    if(!load_result.is_ok()) {
        return load_result;
    }
    load_result = load_stage(
        impl_->dacvae_encoder,
        config.artifacts.dacvae_encoder_onnx_path,
        require && uses_speaker,
        config.provider_routes.dacvae_encoder,
        "DACVAE encoder"
    );
    if(!load_result.is_ok()) {
        return load_result;
    }
    load_result = load_stage(
        impl_->dacvae_decoder,
        config.artifacts.dacvae_decoder_onnx_path,
        require,
        config.provider_routes.dacvae_decoder,
        "DACVAE decoder"
    );
    if(!load_result.is_ok()) {
        return load_result;
    }

    impl_->coreml_static_dit_steps.clear();
    impl_->coreml_static_bucket_lru.clear();
    if(static_profile) {
        const IrodoriTtsProviderRoute dit_route = resolve_provider_route(config, config.provider_routes.dit_step);
        auto provider_check = validate_provider_route(config, dit_route, "static CoreML DiT step");
        if(!provider_check.is_ok()) {
            return provider_check;
        }
        for(const IrodoriTtsStaticArtifact &artifact : config.coreml_static_artifacts) {
            if(artifact.dit_step_onnx_path.empty()) {
                continue;
            }
            StaticDitSession static_session;
            static_session.bucket = artifact.bucket;
            static_session.dit_step.path = artifact.dit_step_onnx_path;
            static_session.dit_step.provider_requested = dit_route.provider_requested;
            static_session.dit_step.provider_effective = dit_route.provider;
            static_session.dit_step.requested = true;
            static_session.rf_sampler_6_step_path = artifact.rf_sampler_6_step_onnx_path;
            static_session.rf_sampler_8_step_path = artifact.rf_sampler_8_step_onnx_path;
            impl_->coreml_static_dit_steps.push_back(std::move(static_session));
        }
    }
    if(static_profile && config.strict_provider && impl_->coreml_static_dit_steps.empty()) {
        return Error::model_not_loaded(
            "IrodoriTtsSession::load: coreml_static profile requested but no static DiT artifacts loaded"
        );
    }

    impl_->execution_ready =
        impl_->text_encoder.loaded &&
        (impl_->dit_step.loaded ||
            (static_profile && !cpu_dispatch_required && !impl_->coreml_static_dit_steps.empty())) &&
        impl_->dacvae_decoder.loaded &&
        (!uses_caption || impl_->caption_encoder.loaded) &&
        (config.mode != IrodoriTtsMode::BaseV3 || impl_->duration_predictor.loaded);
    if(uses_speaker) {
        impl_->execution_ready = impl_->execution_ready &&
            impl_->speaker_encoder.loaded &&
            impl_->dacvae_encoder.loaded;
    }

    impl_->loaded = true;
    return Result<void>();
}

bool IrodoriTtsSession::is_loaded() const {
    return impl_->loaded;
}

bool IrodoriTtsSession::is_execution_ready() const {
    return impl_->execution_ready;
}

Result<IrodoriTtsSynthesisResult> IrodoriTtsSession::synthesize(
    const IrodoriTtsRequest &request,
    CancellationToken *cancel
) {
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    const auto started = Clock::now();
    if(cancel && cancel->is_cancelled()) {
        return Error::cancelled("IrodoriTtsSession::synthesize: request cancelled");
    }
    if(!is_loaded()) {
        return Error::model_not_loaded("IrodoriTtsSession::synthesize: session is not loaded");
    }
    if(request.text.empty()) {
        return Error::empty_input("IrodoriTtsSession::synthesize: text is empty");
    }

    const int32_t num_steps = request.num_steps > 0 ? request.num_steps : impl_->config.default_num_steps;
    const std::string schedule_mode = request.t_schedule_mode.empty() ?
        impl_->config.default_t_schedule_mode : request.t_schedule_mode;
    const float sway_coeff = is_finite(request.sway_coeff) ? request.sway_coeff : impl_->config.default_sway_coeff;
    auto schedule_result = build_irodori_sway_schedule(num_steps, schedule_mode, sway_coeff);
    if(!schedule_result.is_ok()) {
        return schedule_result.get_error();
    }

    float cfg_scale_text = request.cfg_scale_text;
    float cfg_scale_caption = request.cfg_scale_caption;
    float cfg_scale_speaker = request.cfg_scale_speaker;
    if(request.cfg_scale.has_value()) {
        cfg_scale_text = *request.cfg_scale;
        cfg_scale_caption = *request.cfg_scale;
        cfg_scale_speaker = *request.cfg_scale;
    }
    const std::string guidance_mode = request.cfg_guidance_mode.empty() ?
        impl_->config.default_cfg_guidance_mode : request.cfg_guidance_mode;
    auto cfg_result = build_irodori_cfg_branches(
        impl_->config.mode,
        guidance_mode.empty() ? "independent" : guidance_mode,
        cfg_scale_text,
        cfg_scale_caption,
        cfg_scale_speaker
    );
    if(!cfg_result.is_ok()) {
        return cfg_result.get_error();
    }

    if(cancel && cancel->is_cancelled()) {
        return Error::cancelled("IrodoriTtsSession::synthesize: request cancelled");
    }
    if(!impl_->execution_ready) {
        return Error::model_not_loaded(
            "IrodoriTtsSession::synthesize: exported Irodori ONNX artifacts are not all loaded"
        );
    }
    if(request.text_token_ids.empty()) {
        return Error::invalid_argument(
            "IrodoriTtsSession::synthesize: text_token_ids are required; tokenize text before calling the native runtime"
        );
    }
    if(irodori_tts_mode_uses_caption(impl_->config.mode) && request.caption_token_ids.empty()) {
        return Error::invalid_argument(
            "IrodoriTtsSession::synthesize: caption_token_ids are required for VoiceDesign mode"
        );
    }

    IrodoriTtsSynthesisResult output;
    output.sample_rate = impl_->config.sample_rate;
    output.mode = irodori_tts_mode_name(impl_->config.mode);
    output.provider_profile = impl_->config.provider_profile;
    output.provider_requested = impl_->config.provider_requested;
    output.provider_effective = impl_->config.provider;
    auto add_provider_stage = [&](const std::string &name, const OptionalSession &session) {
        output.provider_requested_by_stage[name] = session.provider_requested.empty() ?
            impl_->config.provider_requested : session.provider_requested;
        output.provider_effective_by_stage[name] = session.provider_effective.empty() ?
            impl_->config.provider : session.provider_effective;
        if(session.loaded) {
            output.instrumentation_ms[name + "_session_create_ms"] = session.session_create_ms;
        }
    };
    add_provider_stage("text_encoder", impl_->text_encoder);
    add_provider_stage("caption_encoder", impl_->caption_encoder);
    add_provider_stage("speaker_encoder", impl_->speaker_encoder);
    add_provider_stage("duration_predictor", impl_->duration_predictor);
    add_provider_stage("dit_step", impl_->dit_step);
    add_provider_stage("dacvae_encoder", impl_->dacvae_encoder);
    add_provider_stage("dacvae_decoder", impl_->dacvae_decoder);
    output.diagnostics["runtime_dispatch"] = impl_->config.runtime_dispatch.empty() ?
        "force_cpu" : impl_->config.runtime_dispatch;
    output.diagnostics["rf_execution_mode_requested"] = impl_->config.rf_execution_mode.empty() ?
        "auto" : impl_->config.rf_execution_mode;
    output.diagnostics["dispatch_recommendation_path"] = impl_->config.dispatch_recommendation_path;
    output.diagnostics["provider_diagnostics"] = impl_->config.print_provider_diagnostics ?
        "enabled" : "disabled";

    auto mark_stage = [&](const std::string &name, const Clock::time_point &stage_started) {
        output.timings_ms[name] = Ms(Clock::now() - stage_started).count();
    };

    output.timings_ms["tokenize"] = 0.0;

    ConditionState conditions;
    int32_t ref_steps = 0;
    const bool use_condition_cache = impl_->config.enable_context_kv_cache && request.context_kv_cache;
    const std::string condition_cache_key = use_condition_cache ?
        build_condition_cache_key_internal(impl_->config, request) : std::string();
    bool used_condition_cache = false;
    if(use_condition_cache) {
        std::lock_guard<std::mutex> lock(impl_->cache_mutex);
        auto cached = impl_->condition_cache.find(condition_cache_key);
        if(cached != impl_->condition_cache.end()) {
            conditions = cached->second.conditions;
            ref_steps = cached->second.ref_steps;
            used_condition_cache = true;
            output.cache_hit = true;
            output.timings_ms["condition_encode"] = 0.0;
            output.timings_ms["reference_prep"] = 0.0;
        }
    }

    auto stage_started = Clock::now();
    if(!used_condition_cache) {
        const BoolTensor text_mask = make_mask_tensor(request.text_token_mask, request.text_token_ids.size());
        auto text_state_result = run_encoder_state(
            impl_->text_encoder,
            request.text_token_ids,
            request.text_token_mask,
            "text encoder"
        );
        if(!text_state_result.is_ok()) {
            return text_state_result.get_error();
        }
        conditions.text_state = std::move(text_state_result.value());
        conditions.text_mask = text_mask;

        if(irodori_tts_mode_uses_caption(impl_->config.mode)) {
            const BoolTensor caption_mask = make_mask_tensor(request.caption_token_mask, request.caption_token_ids.size());
            auto caption_state_result = run_encoder_state(
                impl_->caption_encoder,
                request.caption_token_ids,
                request.caption_token_mask,
                "caption encoder"
            );
            if(!caption_state_result.is_ok()) {
                return caption_state_result.get_error();
            }
            conditions.caption_state = std::move(caption_state_result.value());
            conditions.caption_mask = caption_mask;
            conditions.has_caption = true;
        }
        mark_stage("condition_encode", stage_started);
    }

    stage_started = Clock::now();
    if(!used_condition_cache && irodori_tts_mode_uses_speaker(impl_->config.mode)) {
        auto ref_result = prepare_reference_latent(
            impl_->config,
            request,
            impl_->dacvae_encoder,
            impl_->ref_latent_cache,
            impl_->cache_mutex,
            &output.cache_hit,
            1
        );
        if(!ref_result.is_ok()) {
            return ref_result.get_error();
        }
        FloatTensor ref_latent = std::move(ref_result.value());
        ref_steps = ref_latent.shape.size() >= 2 ? static_cast<int32_t>(ref_latent.shape[1]) : 0;
        const bool has_reference = !request.no_ref &&
            (!request.ref_latent.empty() || !request.ref_latent_path.empty() || !request.ref_wav_path.empty());
        auto speaker_result = run_speaker_encoder(impl_->speaker_encoder, ref_latent, has_reference);
        if(!speaker_result.is_ok()) {
            return speaker_result.get_error();
        }
        conditions.speaker_state = std::move(speaker_result.value().first);
        conditions.speaker_mask = std::move(speaker_result.value().second);
        conditions.has_speaker = true;
    }
    if(!used_condition_cache) {
        mark_stage("reference_prep", stage_started);
        if(use_condition_cache) {
            std::lock_guard<std::mutex> lock(impl_->cache_mutex);
            impl_->condition_cache[condition_cache_key] = CachedConditionState{conditions, ref_steps};
        }
    }

    if(cancel && cancel->is_cancelled()) {
        return Error::cancelled("IrodoriTtsSession::synthesize: request cancelled");
    }

    stage_started = Clock::now();
    int32_t latent_steps = 0;
    if(request.seconds.has_value()) {
        latent_steps = std::max(
            1,
            static_cast<int32_t>(std::ceil(
                static_cast<double>(*request.seconds) *
                static_cast<double>(impl_->config.sample_rate) /
                static_cast<double>(impl_->config.codec_hop_length)
            ))
        );
    } else if(impl_->duration_predictor.loaded) {
        auto duration_result = run_duration_predictor(
            impl_->duration_predictor,
            impl_->config,
            request,
            conditions,
            static_cast<int64_t>(request.text_token_ids.size()),
            &output.instrumentation_ms
        );
        if(!duration_result.is_ok()) {
            return duration_result.get_error();
        }
        latent_steps = duration_result.value();
    } else {
        latent_steps = std::max(
            1,
            static_cast<int32_t>(std::ceil(
                static_cast<float>(request.text_token_ids.size()) * 8.0f *
                std::max(0.1f, request.duration_scale)
            ))
        );
    }
    mark_stage("duration_prediction", stage_started);

    auto bucket_result = select_irodori_bucket(
        impl_->config.buckets,
        latent_steps,
        static_cast<int32_t>(request.text_token_ids.size()),
        irodori_tts_mode_uses_caption(impl_->config.mode) ?
            static_cast<int32_t>(request.caption_token_ids.size()) : 0,
        ref_steps
    );
    if(!bucket_result.is_ok()) {
        return bucket_result.get_error();
    }
    const IrodoriTtsBucket selected_bucket = bucket_result.value();
    output.selected_bucket_latent_steps = selected_bucket.latent_steps;
    output.selected_bucket = bucket_to_string(selected_bucket);

    const int32_t patched_steps = std::max(
        1,
        static_cast<int32_t>(std::ceil(
            static_cast<double>(latent_steps) /
            static_cast<double>(std::max(1, impl_->config.latent_patch_size))
        ))
    );

    OptionalSession *dit_step_session = &impl_->dit_step;
    OptionalSession *rf_sampler_session = nullptr;
    ConditionState rf_conditions = conditions;
    int32_t rf_patched_steps = patched_steps;
    const std::string normalized_profile = lower_ascii(impl_->config.provider_profile);
    const std::string dispatch_mode = lower_ascii(impl_->config.runtime_dispatch.empty() ?
        "force_cpu" : impl_->config.runtime_dispatch);
    const bool allow_static_gpu =
        (normalized_profile == "coreml_static" || normalized_profile == "coreml_static_unrolled") &&
        dispatch_mode != "force_cpu" &&
        dispatch_mode != "auto_dispatch";
    output.diagnostics["dispatch_decision"] = allow_static_gpu ?
        "gpu_static_bucket" :
        (dispatch_mode == "auto_dispatch" ? "cpu_default_no_threshold_table" : "cpu_vector");
    if(allow_static_gpu) {
        StaticDitSession *matched_static_session = nullptr;
        for(StaticDitSession &static_session : impl_->coreml_static_dit_steps) {
            if(buckets_match(static_session.bucket, selected_bucket)) {
                matched_static_session = &static_session;
                break;
            }
        }
        if(matched_static_session) {
            Result<void> static_load;
            const bool was_loaded = matched_static_session->dit_step.loaded;
            {
                std::lock_guard<std::mutex> lock(impl_->cache_mutex);
                static_load = ensure_static_dit_loaded(*matched_static_session, impl_->config);
                if(static_load.is_ok() && matched_static_session->dit_step.loaded) {
                    touch_static_bucket_lru(
                        impl_->coreml_static_dit_steps,
                        impl_->coreml_static_bucket_lru,
                        selected_bucket
                    );
                }
            }
            if(static_load.is_ok() && matched_static_session->dit_step.loaded && !was_loaded) {
                output.instrumentation_ms["static_bucket_load_count"] += 1.0;
                output.instrumentation_ms["static_dit_step_session_create_ms"] =
                    matched_static_session->dit_step.session_create_ms;
            }
            if(!static_load.is_ok()) {
                if(impl_->config.strict_provider) {
                    return static_load.get_error();
                }
                matched_static_session = nullptr;
            } else if(!matched_static_session->dit_step.loaded) {
                if(impl_->config.strict_provider) {
                    return Error::model_not_loaded(
                        "IrodoriTtsSession::synthesize: static CoreML DiT step did not load for bucket " +
                        bucket_to_string(selected_bucket)
                    );
                }
                matched_static_session = nullptr;
            }
        }
        if(matched_static_session) {
            dit_step_session = &matched_static_session->dit_step;
            rf_conditions = pad_conditions_to_bucket(conditions, selected_bucket, impl_->config.mode);
            rf_patched_steps = std::max(
                1,
                static_cast<int32_t>(std::ceil(
                    static_cast<double>(selected_bucket.latent_steps) /
                    static_cast<double>(std::max(1, impl_->config.latent_patch_size))
                ))
            );
            output.provider_requested_by_stage["dit_step"] = matched_static_session->dit_step.provider_requested;
            output.provider_effective_by_stage["dit_step"] = matched_static_session->dit_step.provider_effective;
            output.diagnostics["static_bucket_selected"] = bucket_to_string(selected_bucket);
            if(impl_->config.enable_coreml_unrolled_rf_sampler &&
               can_use_unrolled_rf_sampler(num_steps, schedule_mode, sway_coeff, cfg_result.value())) {
                const bool sampler_path_available = num_steps == 6 ?
                    !matched_static_session->rf_sampler_6_step_path.empty() :
                    !matched_static_session->rf_sampler_8_step_path.empty();
                if(sampler_path_available) {
                    Result<void> sampler_load;
                    {
                        std::lock_guard<std::mutex> lock(impl_->cache_mutex);
                        sampler_load = ensure_static_rf_sampler_loaded(
                            *matched_static_session,
                            num_steps,
                            impl_->config
                        );
                    }
                    if(!sampler_load.is_ok()) {
                        if(impl_->config.strict_provider) {
                            return sampler_load.get_error();
                        }
                    } else {
                        OptionalSession &candidate_sampler = num_steps == 6 ?
                            matched_static_session->rf_sampler_6_step :
                            matched_static_session->rf_sampler_8_step;
                        if(candidate_sampler.loaded) {
                            rf_sampler_session = &candidate_sampler;
                            output.provider_requested_by_stage["rf_sampler"] = candidate_sampler.provider_requested;
                            output.provider_effective_by_stage["rf_sampler"] = candidate_sampler.provider_effective;
                            output.instrumentation_ms["rf_sampler_session_create_ms"] =
                                candidate_sampler.session_create_ms;
                        }
                    }
                }
            }
        } else if(impl_->config.strict_provider) {
            return Error::model_not_loaded(
                "IrodoriTtsSession::synthesize: coreml_static profile selected bucket " +
                bucket_to_string(selected_bucket) + " but no matching static DiT artifact is loaded"
            );
        }
    }

    stage_started = Clock::now();
    Result<FloatTensor> sampled_result = rf_sampler_session ?
        sample_rf_unrolled(
            *rf_sampler_session,
            impl_->config,
            request,
            rf_conditions,
            patched_steps,
            rf_patched_steps,
            cancel,
            &output.instrumentation_ms,
            &output.diagnostics
        ) :
        sample_rf(
            *dit_step_session,
            impl_->config,
            request,
            rf_conditions,
            patched_steps,
            rf_patched_steps,
            schedule_result.value(),
            cfg_result.value(),
            cancel,
            &output.instrumentation_ms,
            &output.diagnostics
        );
    if(!sampled_result.is_ok()) {
        return sampled_result.get_error();
    }
    mark_stage("rf_sampling", stage_started);

    stage_started = Clock::now();
    auto latent_result = unpatchify_latent(
        sampled_result.value(),
        impl_->config.latent_dim,
        impl_->config.latent_patch_size,
        latent_steps
    );
    if(!latent_result.is_ok()) {
        return latent_result.get_error();
    }
    mark_stage("unpatchify", stage_started);

    stage_started = Clock::now();
    auto waveform_result = decode_latent(
        impl_->dacvae_decoder,
        latent_result.value(),
        &output.instrumentation_ms
    );
    if(!waveform_result.is_ok()) {
        return waveform_result.get_error();
    }
    output.waveform = std::move(waveform_result.value());
    const int32_t target_samples = std::max(1, latent_steps * impl_->config.codec_hop_length);
    if(static_cast<int32_t>(output.waveform.size()) > target_samples) {
        output.waveform.resize(static_cast<size_t>(target_samples));
    }
    output.frame_count = static_cast<int32_t>(output.waveform.size());
    mark_stage("decode", stage_started);
    if(impl_->config.ort_enable_profiling) {
        auto collect_profile = [&](const std::string &name, OptionalSession &session) {
            if(!session.loaded) {
                return;
            }
            auto profile_result = session.session.end_profiling();
            if(!profile_result.has_error() && !profile_result.value().empty()) {
                output.diagnostics[name + "_ort_profile_path"] = profile_result.value();
            }
        };
        collect_profile("text_encoder", impl_->text_encoder);
        collect_profile("caption_encoder", impl_->caption_encoder);
        collect_profile("speaker_encoder", impl_->speaker_encoder);
        collect_profile("duration_predictor", impl_->duration_predictor);
        collect_profile("dit_step", *dit_step_session);
        if(rf_sampler_session) {
            collect_profile("rf_sampler", *rf_sampler_session);
        }
        collect_profile("dacvae_encoder", impl_->dacvae_encoder);
        collect_profile("dacvae_decoder", impl_->dacvae_decoder);
    }
    output.elapsed_ms = Ms(Clock::now() - started).count();
    return output;
}

const IrodoriTtsSessionConfig &IrodoriTtsSession::config() const {
    return impl_->config;
}

} // namespace gotst
