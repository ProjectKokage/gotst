#include "gotst/core/qwen3_forced_aligner.hpp"

#include "gotst/core/asr_frontend.hpp"

#include <godot_llama/llama_context_handle.hpp>
#include <godot_llama/llama_model_handle.hpp>
#include <godot_llama/llama_params.hpp>
#include <godot_llama/llama_position_layout.hpp>
#include <gonx/core/provider.hpp>
#include <gonx/core/session.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace gotst {

namespace {

constexpr char CLASSIFIER_MAGIC[8] = {'G', 'F', 'A', 'H', '0', '0', '0', '1'};
constexpr uint32_t CLASSIFIER_VERSION = 1;

struct ClassifierHeader {
    char magic[8];
    uint32_t version = CLASSIFIER_VERSION;
    int32_t classify_num = 0;
    int32_t hidden_size = 0;
    int32_t reserved = 0;
};

struct FloatTensor {
    std::vector<float> values;
    std::vector<int64_t> shape;
};

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

Ort::MemoryInfo &cpu_memory_info() {
    static Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return info;
}

gonx::SessionConfig build_onnx_config(const Qwen3ForcedAlignerSessionConfig &config) {
    gonx::SessionConfig session_config;
    session_config.device_id = config.onnx_device_id;
    session_config.intra_op_num_threads = config.onnx_intra_op_threads;
    session_config.inter_op_num_threads = config.onnx_inter_op_threads;
    session_config.optimization_level = config.onnx_optimization_level;

    const std::string provider_name = config.onnx_provider;
    if(provider_name == "AUTO" || provider_name == "auto" || provider_name.empty()) {
        session_config.providers = {gonx::ExecutionProvider::CPU};
        const std::vector<gonx::ExecutionProvider> available = gonx::available_providers();
        for(gonx::ExecutionProvider provider : available) {
            if(provider != gonx::ExecutionProvider::CPU) {
                session_config.providers.insert(session_config.providers.begin(), provider);
                break;
            }
        }
        return session_config;
    }

    session_config.providers = {
        gonx::parse_provider(provider_name),
        gonx::ExecutionProvider::CPU,
    };
    return session_config;
}

Result<FloatTensor> first_float_output(
    std::vector<Ort::Value> outputs,
    const std::string &label
) {
    if(outputs.empty()) {
        return Error::inference_failed(label + ": ONNX returned no outputs.");
    }

    Ort::Value &tensor = outputs.front();
    const auto info = tensor.GetTensorTypeAndShapeInfo();
    if(info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Error::shape_mismatch(label + ": expected float output.");
    }

    const size_t element_count = info.GetElementCount();
    if(element_count == 0) {
        return Error::shape_mismatch(label + ": output tensor is empty.");
    }

    const float *data = tensor.GetTensorData<float>();
    FloatTensor result;
    result.shape = info.GetShape();
    result.values.assign(data, data + element_count);
    return result;
}

Result<FloatTensor> run_audio_conv(
    gonx::InferenceSession &session,
    const std::vector<float> &features,
    int64_t mel_bins,
    int64_t frame_count
) {
    if(features.empty() || mel_bins <= 0 || frame_count <= 0) {
        return Error::invalid_argument("audio_conv: empty features or invalid shape.");
    }
    const int64_t expected = mel_bins * frame_count;
    if(static_cast<int64_t>(features.size()) != expected) {
        return Error::shape_mismatch("audio_conv: feature size does not match mel_bins * frame_count.");
    }

    std::array<int64_t, 3> shape = {1, mel_bins, frame_count};
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        cpu_memory_info(),
        const_cast<float *>(features.data()),
        features.size(),
        shape.data(),
        shape.size()
    ));

    auto run_result = session.run(inputs);
    if(run_result.has_error()) {
        return Error::inference_failed("audio_conv: " + run_result.error().message);
    }
    return first_float_output(std::move(run_result).value(), "audio_conv");
}

Result<FloatTensor> run_audio_encoder(
    gonx::InferenceSession &session,
    std::vector<float> &hidden_states,
    int64_t sequence_length,
    int64_t hidden_size
) {
    if(hidden_states.empty() || sequence_length <= 0 || hidden_size <= 0) {
        return Error::invalid_argument("audio_encoder: empty hidden states or invalid shape.");
    }
    const int64_t expected = sequence_length * hidden_size;
    if(static_cast<int64_t>(hidden_states.size()) != expected) {
        return Error::shape_mismatch("audio_encoder: hidden state size does not match sequence shape.");
    }

    std::array<int64_t, 2> hidden_shape = {sequence_length, hidden_size};
    std::array<int32_t, 2> cu_seqlens = {0, static_cast<int32_t>(sequence_length)};
    std::array<int64_t, 1> cu_shape = {2};
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        cpu_memory_info(),
        hidden_states.data(),
        hidden_states.size(),
        hidden_shape.data(),
        hidden_shape.size()
    ));
    inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(
        cpu_memory_info(),
        cu_seqlens.data(),
        cu_seqlens.size(),
        cu_shape.data(),
        cu_shape.size()
    ));

    auto run_result = session.run(inputs);
    if(run_result.has_error()) {
        return Error::inference_failed("audio_encoder: " + run_result.error().message);
    }
    return first_float_output(std::move(run_result).value(), "audio_encoder");
}

Result<FloatTensor> run_embedding(gonx::InferenceSession &session, std::vector<int64_t> &token_ids) {
    if(token_ids.empty()) {
        return Error::invalid_argument("embedding: token IDs are empty.");
    }

    std::array<int64_t, 2> shape = {1, static_cast<int64_t>(token_ids.size())};
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
        cpu_memory_info(),
        token_ids.data(),
        token_ids.size(),
        shape.data(),
        shape.size()
    ));

    auto run_result = session.run(inputs);
    if(run_result.has_error()) {
        return Error::inference_failed("embedding: " + run_result.error().message);
    }
    return first_float_output(std::move(run_result).value(), "embedding");
}

int32_t tensor_sequence_length(const FloatTensor &tensor) {
    if(tensor.shape.size() >= 2) {
        return static_cast<int32_t>(tensor.shape[tensor.shape.size() - 2]);
    }
    return 1;
}

int32_t tensor_hidden_size(const FloatTensor &tensor) {
    if(!tensor.shape.empty()) {
        return static_cast<int32_t>(tensor.shape.back());
    }
    return static_cast<int32_t>(tensor.values.size());
}

int32_t compute_audio_token_count(int32_t mel_frames) {
    const int32_t safe_frames = std::max(mel_frames, 1);
    const int32_t input_lengths_leave = safe_frames % 100;
    const int32_t feat_lengths = static_cast<int32_t>(std::floor((input_lengths_leave - 1) / 2.0f)) + 1;
    const int32_t reduced_feat_lengths = static_cast<int32_t>(std::floor((feat_lengths - 1) / 2.0f)) + 1;
    const int32_t output_lengths =
        static_cast<int32_t>(std::floor((reduced_feat_lengths - 1) / 2.0f)) + 1 + ((safe_frames / 100) * 13);
    return std::max(output_lengths, 1);
}

std::vector<int32_t> build_audio_conv_chunk_lengths(int32_t mel_frames, int32_t chunk_frames) {
    const int32_t safe_frames = std::max(mel_frames, 1);
    const int32_t safe_chunk = std::max(chunk_frames, 1);
    const int32_t full_chunk_count = safe_frames / safe_chunk;
    const int32_t remainder = safe_frames % safe_chunk;
    const int32_t chunk_count = full_chunk_count + (remainder > 0 ? 1 : 0);
    if(chunk_count <= 0) {
        return {safe_frames};
    }

    std::vector<int32_t> lengths(static_cast<size_t>(chunk_count), safe_chunk);
    if(remainder > 0) {
        lengths.back() = remainder;
    }
    return lengths;
}

std::vector<float> slice_mel_columns(
    const std::vector<float> &features,
    int32_t mel_bins,
    int32_t total_frames,
    int32_t start_frame,
    int32_t frame_count
) {
    std::vector<float> slice(static_cast<size_t>(mel_bins) * static_cast<size_t>(frame_count));
    for(int32_t mel = 0; mel < mel_bins; ++mel) {
        const int32_t source_offset = (mel * total_frames) + start_frame;
        const int32_t target_offset = mel * frame_count;
        std::copy_n(
            features.begin() + source_offset,
            frame_count,
            slice.begin() + target_offset
        );
    }
    return slice;
}

std::vector<float> build_audio_positional_embedding(int32_t rows, int32_t channels) {
    const int32_t safe_rows = std::max(rows, 1);
    const int32_t safe_channels = std::max(channels, 2);
    const int32_t half_channels = safe_channels / 2;
    std::vector<float> embedding(static_cast<size_t>(safe_rows) * static_cast<size_t>(safe_channels), 0.0f);
    if(half_channels <= 0) {
        return embedding;
    }

    const float log_timescale_increment = half_channels <= 1
        ? 0.0f
        : static_cast<float>(std::log(10000.0) / static_cast<double>(std::max(half_channels - 1, 1)));
    for(int32_t channel = 0; channel < half_channels; ++channel) {
        const float inverse_timescale = std::exp(-log_timescale_increment * static_cast<float>(channel));
        for(int32_t row = 0; row < safe_rows; ++row) {
            const float scaled_time = static_cast<float>(row) * inverse_timescale;
            embedding[static_cast<size_t>((row * safe_channels) + channel)] = std::sin(scaled_time);
            embedding[static_cast<size_t>((row * safe_channels) + half_channels + channel)] = std::cos(scaled_time);
        }
    }
    return embedding;
}

std::vector<float> apply_chunked_audio_positional_adjustment(
    const std::vector<float> &hidden_states,
    const std::vector<int32_t> &row_lengths,
    int32_t hidden_size
) {
    if(hidden_states.empty() || row_lengths.empty() || hidden_size <= 0) {
        return hidden_states;
    }

    const int32_t total_rows = static_cast<int32_t>(hidden_states.size() / static_cast<size_t>(hidden_size));
    if(total_rows <= 0) {
        return hidden_states;
    }

    const std::vector<float> positional = build_audio_positional_embedding(total_rows, hidden_size);
    std::vector<float> adjusted(hidden_states.size(), 0.0f);
    int32_t global_row = 0;
    for(int32_t requested_rows : row_lengths) {
        const int32_t chunk_rows = std::min(requested_rows, total_rows - global_row);
        if(chunk_rows <= 0) {
            break;
        }

        for(int32_t row = 0; row < chunk_rows; ++row) {
            const int32_t source_offset = (global_row + row) * hidden_size;
            const int32_t global_pos_offset = source_offset;
            const int32_t chunk_pos_offset = row * hidden_size;
            for(int32_t column = 0; column < hidden_size; ++column) {
                adjusted[static_cast<size_t>(source_offset + column)] =
                    hidden_states[static_cast<size_t>(source_offset + column)] +
                    positional[static_cast<size_t>(chunk_pos_offset + column)] -
                    positional[static_cast<size_t>(global_pos_offset + column)];
            }
        }
        global_row += chunk_rows;
    }

    if(global_row < total_rows) {
        const size_t offset = static_cast<size_t>(global_row) * static_cast<size_t>(hidden_size);
        std::copy(hidden_states.begin() + static_cast<std::ptrdiff_t>(offset), hidden_states.end(), adjusted.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    return adjusted;
}

bool is_utf8_continuation(unsigned char byte) {
    return (byte & 0xC0U) == 0x80U;
}

std::vector<std::string> split_utf8_codepoints(const std::string &text) {
    std::vector<std::string> units;
    size_t index = 0;
    while(index < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[index]);
        size_t length = 1;
        if((lead & 0x80U) == 0) {
            length = 1;
        } else if((lead & 0xE0U) == 0xC0U) {
            length = 2;
        } else if((lead & 0xF0U) == 0xE0U) {
            length = 3;
        } else if((lead & 0xF8U) == 0xF0U) {
            length = 4;
        }

        if(index + length > text.size()) {
            length = 1;
        }
        bool valid = true;
        for(size_t offset = 1; offset < length; ++offset) {
            if(!is_utf8_continuation(static_cast<unsigned char>(text[index + offset]))) {
                valid = false;
                break;
            }
        }
        if(!valid) {
            length = 1;
        }

        std::string unit = text.substr(index, length);
        const bool ascii_space = length == 1 && std::isspace(static_cast<unsigned char>(unit[0])) != 0;
        if(!ascii_space) {
            units.push_back(std::move(unit));
        }
        index += length;
    }
    return units;
}

std::vector<std::string> split_ascii_words(const std::string &text) {
    std::vector<std::string> units;
    std::string current;
    for(unsigned char ch : text) {
        if(std::isspace(ch) != 0) {
            if(!current.empty()) {
                units.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(static_cast<char>(ch));
        }
    }
    if(!current.empty()) {
        units.push_back(current);
    }
    return units;
}

Result<void> check_path(const std::string &path, const std::string &label) {
    if(path.empty()) {
        return Error::invalid_argument(label + " path is empty.");
    }
    if(!std::filesystem::is_regular_file(path)) {
        return Error::not_found(label + " was not found: " + path);
    }
    return {};
}

} // namespace

bool Qwen3ForcedAlignerClassifierHead::is_valid() const {
    return classify_num > 0 &&
           hidden_size > 0 &&
           weights.size() == static_cast<size_t>(classify_num) * static_cast<size_t>(hidden_size);
}

Result<void> Qwen3ForcedAlignerClassifierHead::load(const std::string &path) {
    auto path_check = check_path(path, "Qwen3 forced-aligner classifier head");
    if(!path_check.is_ok()) {
        return path_check;
    }

    std::ifstream stream(path, std::ios::binary);
    if(!stream) {
        return Error::io_error("Failed to open classifier head: " + path);
    }

    ClassifierHeader header;
    stream.read(reinterpret_cast<char *>(&header), sizeof(header));
    if(stream.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        return Error::io_error("Classifier head is truncated: " + path);
    }
    if(std::memcmp(header.magic, CLASSIFIER_MAGIC, sizeof(CLASSIFIER_MAGIC)) != 0) {
        return Error::shape_mismatch("Classifier head magic did not match GOTST forced-aligner v1.");
    }
    if(header.version != CLASSIFIER_VERSION) {
        return Error::shape_mismatch("Unsupported classifier head version.");
    }
    if(header.classify_num <= 0 || header.hidden_size <= 0) {
        return Error::shape_mismatch("Classifier head has invalid dimensions.");
    }

    const size_t value_count = static_cast<size_t>(header.classify_num) * static_cast<size_t>(header.hidden_size);
    std::vector<float> loaded(value_count);
    stream.read(reinterpret_cast<char *>(loaded.data()), static_cast<std::streamsize>(value_count * sizeof(float)));
    if(stream.gcount() != static_cast<std::streamsize>(value_count * sizeof(float))) {
        return Error::io_error("Classifier head weights are truncated: " + path);
    }

    classify_num = header.classify_num;
    hidden_size = header.hidden_size;
    weights = std::move(loaded);
    return {};
}

Result<Qwen3TimestampPrediction> Qwen3ForcedAlignerClassifierHead::classify(std::span<const float> hidden) const {
    if(!is_valid()) {
        return Error::invalid_state("Classifier head is not loaded.");
    }
    if(static_cast<int32_t>(hidden.size()) != hidden_size) {
        return Error::shape_mismatch("Classifier hidden vector size does not match classifier head.");
    }

    int32_t best_index = 0;
    double best_value = -std::numeric_limits<double>::infinity();
    std::vector<double> logits(static_cast<size_t>(classify_num), 0.0);
    for(int32_t row = 0; row < classify_num; ++row) {
        const float *weight = weights.data() + (static_cast<size_t>(row) * static_cast<size_t>(hidden_size));
        double sum = 0.0;
        for(int32_t column = 0; column < hidden_size; ++column) {
            sum += static_cast<double>(weight[column]) * static_cast<double>(hidden[static_cast<size_t>(column)]);
        }
        logits[static_cast<size_t>(row)] = sum;
        if(sum > best_value) {
            best_value = sum;
            best_index = row;
        }
    }

    double exp_sum = 0.0;
    for(double value : logits) {
        exp_sum += std::exp(value - best_value);
    }

    Qwen3TimestampPrediction prediction;
    prediction.bin = best_index;
    prediction.confidence = exp_sum > 0.0 ? 1.0 / exp_sum : 0.0;
    return prediction;
}

std::vector<int32_t> repair_monotonic_timestamp_bins(std::span<const int32_t> bins) {
    const int32_t n = static_cast<int32_t>(bins.size());
    if(n <= 0) {
        return {};
    }

    std::vector<int32_t> data(bins.begin(), bins.end());
    std::vector<int32_t> dp(static_cast<size_t>(n), 1);
    std::vector<int32_t> parent(static_cast<size_t>(n), -1);
    for(int32_t i = 1; i < n; ++i) {
        for(int32_t j = 0; j < i; ++j) {
            if(data[static_cast<size_t>(j)] <= data[static_cast<size_t>(i)] &&
               dp[static_cast<size_t>(j)] + 1 > dp[static_cast<size_t>(i)]) {
                dp[static_cast<size_t>(i)] = dp[static_cast<size_t>(j)] + 1;
                parent[static_cast<size_t>(i)] = j;
            }
        }
    }

    int32_t max_idx = 0;
    for(int32_t i = 1; i < n; ++i) {
        if(dp[static_cast<size_t>(i)] > dp[static_cast<size_t>(max_idx)]) {
            max_idx = i;
        }
    }

    std::vector<bool> normal(static_cast<size_t>(n), false);
    for(int32_t idx = max_idx; idx != -1; idx = parent[static_cast<size_t>(idx)]) {
        normal[static_cast<size_t>(idx)] = true;
    }

    std::vector<int32_t> result = data;
    int32_t i = 0;
    while(i < n) {
        if(normal[static_cast<size_t>(i)]) {
            ++i;
            continue;
        }

        int32_t j = i;
        while(j < n && !normal[static_cast<size_t>(j)]) {
            ++j;
        }
        const int32_t anomaly_count = j - i;

        bool has_left = false;
        bool has_right = false;
        int32_t left_value = 0;
        int32_t right_value = 0;
        for(int32_t k = i - 1; k >= 0; --k) {
            if(normal[static_cast<size_t>(k)]) {
                has_left = true;
                left_value = result[static_cast<size_t>(k)];
                break;
            }
        }
        for(int32_t k = j; k < n; ++k) {
            if(normal[static_cast<size_t>(k)]) {
                has_right = true;
                right_value = result[static_cast<size_t>(k)];
                break;
            }
        }

        if(anomaly_count <= 2) {
            for(int32_t k = i; k < j; ++k) {
                if(!has_left && has_right) {
                    result[static_cast<size_t>(k)] = right_value;
                } else if(has_left && !has_right) {
                    result[static_cast<size_t>(k)] = left_value;
                } else if(has_left && has_right) {
                    result[static_cast<size_t>(k)] =
                        (k - (i - 1)) <= (j - k) ? left_value : right_value;
                }
            }
        } else if(has_left && has_right) {
            const double step =
                static_cast<double>(right_value - left_value) / static_cast<double>(anomaly_count + 1);
            for(int32_t k = i; k < j; ++k) {
                result[static_cast<size_t>(k)] =
                    static_cast<int32_t>(left_value + (step * static_cast<double>(k - i + 1)));
            }
        } else if(has_left) {
            for(int32_t k = i; k < j; ++k) {
                result[static_cast<size_t>(k)] = left_value;
            }
        } else if(has_right) {
            for(int32_t k = i; k < j; ++k) {
                result[static_cast<size_t>(k)] = right_value;
            }
        }
        i = j;
    }

    return result;
}

std::vector<std::string> split_forced_alignment_units(const std::string &text, const std::string &mode) {
    const std::string normalized = mode.empty() ? "whitespace" : mode;
    if(normalized == "japanese" || normalized == "ja" ||
       normalized == "chinese" || normalized == "zh" ||
       normalized == "character" || normalized == "char") {
        return split_utf8_codepoints(text);
    }
    return split_ascii_words(text);
}

Result<void> validate_qwen3_forced_alignment_request(
    const Qwen3ForcedAlignmentRequest &request,
    const Qwen3ForcedAlignerSessionConfig &config
) {
    if(request.waveform.empty()) {
        return Error::empty_input("Forced alignment waveform is empty.");
    }
    if(request.input_sample_rate <= 0) {
        return Error::invalid_argument("Forced alignment input sample rate must be positive.");
    }
    if(request.max_duration_seconds > 0.0) {
        const double duration =
            static_cast<double>(request.waveform.size()) / static_cast<double>(request.input_sample_rate);
        if(duration > request.max_duration_seconds) {
            return Error::invalid_argument("Forced alignment audio exceeds max_duration_seconds.");
        }
    }
    if(request.text_units.empty()) {
        return Error::invalid_argument("Forced alignment text_units must not be empty.");
    }
    if(request.token_ids.empty()) {
        return Error::invalid_argument("Forced alignment token_ids must not be empty.");
    }
    if(request.timestamp_token_indices.empty() ||
       (request.timestamp_token_indices.size() % 2U) != 0U) {
        return Error::invalid_argument("Forced alignment requires an even number of timestamp token indices.");
    }
    if(request.timestamp_token_indices.size() != request.text_units.size() * 2U) {
        return Error::invalid_argument("Forced alignment requires exactly two timestamp slots per text unit.");
    }
    if(request.audio_placeholder_start < 0 || request.audio_placeholder_count <= 0) {
        return Error::invalid_argument("Forced alignment audio placeholder range is invalid.");
    }
    if(request.audio_placeholder_start + request.audio_placeholder_count >
       static_cast<int32_t>(request.token_ids.size())) {
        return Error::invalid_argument("Forced alignment audio placeholder range is outside token_ids.");
    }
    if(config.n_ctx > 0 && static_cast<int32_t>(request.token_ids.size()) > config.n_ctx) {
        return Error::invalid_argument("Forced alignment token_ids exceed n_ctx.");
    }
    for(int32_t index : request.timestamp_token_indices) {
        if(index < 0 || index >= static_cast<int32_t>(request.token_ids.size())) {
            return Error::invalid_argument("Forced alignment timestamp index is outside token_ids.");
        }
        if(request.token_ids[static_cast<size_t>(index)] != config.timestamp_token_id) {
            return Error::invalid_argument("Forced alignment timestamp index does not point at timestamp_token_id.");
        }
    }
    if(config.timestamp_segment_ms <= 0 || config.classify_num <= 0) {
        return Error::invalid_argument("Forced alignment timestamp/classifier config is invalid.");
    }
    return {};
}

struct Qwen3ForcedAligner::Impl {
    gonx::InferenceSession audio_conv;
    gonx::InferenceSession audio_encoder;
    gonx::InferenceSession embedding;
    std::shared_ptr<godot_llama::LlamaModelHandle> backbone_model;
    godot_llama::LlamaContextHandle backbone_ctx;
    Qwen3ForcedAlignerClassifierHead classifier;
    Qwen3ForcedAlignerSessionConfig config;
    bool loaded = false;
};

Qwen3ForcedAligner::Qwen3ForcedAligner() : impl_(std::make_unique<Impl>()) {}
Qwen3ForcedAligner::~Qwen3ForcedAligner() = default;
Qwen3ForcedAligner::Qwen3ForcedAligner(Qwen3ForcedAligner &&) noexcept = default;
Qwen3ForcedAligner &Qwen3ForcedAligner::operator=(Qwen3ForcedAligner &&) noexcept = default;

Result<void> Qwen3ForcedAligner::load(
    const Qwen3ForcedAlignerModelPaths &paths,
    const Qwen3ForcedAlignerSessionConfig &config
) {
    impl_->loaded = false;
    impl_->config = config;

    auto check = check_path(paths.audio_conv_onnx_path, "Qwen3 forced-aligner audio conv ONNX");
    if(!check.is_ok()) {
        return check;
    }
    check = check_path(paths.audio_encoder_onnx_path, "Qwen3 forced-aligner audio encoder ONNX");
    if(!check.is_ok()) {
        return check;
    }
    check = check_path(paths.embedding_onnx_path, "Qwen3 forced-aligner embedding ONNX");
    if(!check.is_ok()) {
        return check;
    }
    check = check_path(paths.backbone_gguf_path, "Qwen3 forced-aligner backbone GGUF");
    if(!check.is_ok()) {
        return check;
    }

    const gonx::SessionConfig onnx_config = build_onnx_config(config);
    auto status = impl_->audio_conv.load(paths.audio_conv_onnx_path, onnx_config);
    if(status.has_error()) {
        return Error::model_not_loaded("Failed to load forced-aligner audio conv ONNX: " + status.error().message);
    }
    status = impl_->audio_encoder.load(paths.audio_encoder_onnx_path, onnx_config);
    if(status.has_error()) {
        return Error::model_not_loaded("Failed to load forced-aligner audio encoder ONNX: " + status.error().message);
    }
    status = impl_->embedding.load(paths.embedding_onnx_path, onnx_config);
    if(status.has_error()) {
        return Error::model_not_loaded("Failed to load forced-aligner embedding ONNX: " + status.error().message);
    }

    godot_llama::ModelConfig model_config;
    model_config.model_path = paths.backbone_gguf_path;
    model_config.n_ctx = config.n_ctx;
    model_config.n_batch = config.n_batch;
    model_config.n_threads = config.n_threads;
    model_config.n_gpu_layers = config.n_gpu_layers;
    model_config.use_mmap = config.use_mmap;
    model_config.use_mlock = config.use_mlock;
    model_config.flash_attn_type = config.flash_attn_type;
    model_config.type_k = config.type_k;
    model_config.type_v = config.type_v;
    model_config.embeddings_enabled = true;

    auto model_err = godot_llama::LlamaModelHandle::load(model_config, impl_->backbone_model);
    if(!model_err.ok()) {
        return Error::model_not_loaded("Failed to load forced-aligner GGUF backbone: " + model_err.message);
    }

    auto ctx_err = godot_llama::LlamaContextHandle::create(
        impl_->backbone_model,
        model_config,
        impl_->backbone_ctx
    );
    if(!ctx_err.ok()) {
        return Error::model_not_loaded("Failed to create forced-aligner GGUF context: " + ctx_err.message);
    }

    auto classifier_result = impl_->classifier.load(paths.classifier_head_path);
    if(!classifier_result.is_ok()) {
        return classifier_result;
    }
    if(impl_->classifier.classify_num != config.classify_num) {
        return Error::shape_mismatch("Classifier classify_num does not match session config.");
    }
    if(impl_->classifier.hidden_size != impl_->backbone_model->n_embd()) {
        return Error::shape_mismatch("Classifier hidden_size does not match GGUF backbone hidden size.");
    }

    impl_->loaded = true;
    return {};
}

bool Qwen3ForcedAligner::is_loaded() const {
    return impl_ && impl_->loaded;
}

const Qwen3ForcedAlignerSessionConfig &Qwen3ForcedAligner::config() const {
    return impl_->config;
}

Result<Qwen3ForcedAlignmentResult> Qwen3ForcedAligner::align(
    const Qwen3ForcedAlignmentRequest &request,
    CancellationToken *cancel
) {
    if(!is_loaded()) {
        return Error::model_not_loaded("Qwen3 forced aligner is not loaded.");
    }

    auto validation = validate_qwen3_forced_alignment_request(request, impl_->config);
    if(!validation.is_ok()) {
        return validation.get_error();
    }

    const auto total_start = Clock::now();
    Qwen3ForcedAlignmentResult result;
    result.token_count = static_cast<int32_t>(request.token_ids.size());

    if(cancel && cancel->is_cancelled()) {
        return Error::cancelled("Qwen3 forced alignment cancelled before frontend.");
    }

    auto t0 = Clock::now();
    auto mel_result = build_asr_log_mel_features(
        request.waveform.data(),
        static_cast<int64_t>(request.waveform.size()),
        request.input_sample_rate,
        impl_->config.sample_rate,
        impl_->config.mel_bins,
        impl_->config.fft_size,
        impl_->config.hop_length,
        request.max_duration_seconds > 0.0 ? request.max_duration_seconds : impl_->config.chunk_length_seconds
    );
    result.frontend_ms = Ms(Clock::now() - t0).count();
    if(!mel_result.is_ok()) {
        return mel_result.get_error();
    }

    if(cancel && cancel->is_cancelled()) {
        return Error::cancelled("Qwen3 forced alignment cancelled after frontend.");
    }

    t0 = Clock::now();
    const int32_t mel_frames = static_cast<int32_t>(mel_result.value().valid_frame_count);
    const std::vector<int32_t> chunk_lengths =
        build_audio_conv_chunk_lengths(mel_frames, impl_->config.audio_conv_chunk_frames);
    std::vector<int32_t> output_lengths;
    std::vector<float> merged_hidden_states;
    int32_t hidden_size = 0;
    int32_t start_frame = 0;
    for(int32_t chunk_frames : chunk_lengths) {
        std::vector<float> chunk = slice_mel_columns(
            mel_result.value().features,
            impl_->config.mel_bins,
            static_cast<int32_t>(mel_result.value().frame_count),
            start_frame,
            chunk_frames
        );
        auto conv = run_audio_conv(impl_->audio_conv, chunk, impl_->config.mel_bins, chunk_frames);
        if(!conv.is_ok()) {
            return conv.get_error();
        }

        const int32_t chunk_hidden = tensor_hidden_size(conv.value());
        const int32_t chunk_sequence = tensor_sequence_length(conv.value());
        const int32_t expected_rows = std::min(compute_audio_token_count(chunk_frames), chunk_sequence);
        if(chunk_hidden <= 0 || expected_rows <= 0) {
            return Error::shape_mismatch("audio_conv: invalid output shape.");
        }
        if(hidden_size == 0) {
            hidden_size = chunk_hidden;
        } else if(hidden_size != chunk_hidden) {
            return Error::shape_mismatch("audio_conv: chunk hidden sizes do not match.");
        }

        const size_t copy_count = static_cast<size_t>(expected_rows) * static_cast<size_t>(hidden_size);
        merged_hidden_states.insert(
            merged_hidden_states.end(),
            conv.value().values.begin(),
            conv.value().values.begin() + static_cast<std::ptrdiff_t>(copy_count)
        );
        output_lengths.push_back(expected_rows);
        start_frame += chunk_frames;
    }
    result.audio_conv_ms = Ms(Clock::now() - t0).count();

    if(cancel && cancel->is_cancelled()) {
        return Error::cancelled("Qwen3 forced alignment cancelled after audio conv.");
    }

    const int32_t audio_conv_rows = hidden_size > 0
        ? static_cast<int32_t>(merged_hidden_states.size() / static_cast<size_t>(hidden_size))
        : 0;
    if(audio_conv_rows <= 0) {
        return Error::shape_mismatch("audio_conv: produced no hidden rows.");
    }
    std::vector<float> adjusted_hidden =
        apply_chunked_audio_positional_adjustment(merged_hidden_states, output_lengths, hidden_size);

    t0 = Clock::now();
    auto audio_encoded = run_audio_encoder(
        impl_->audio_encoder,
        adjusted_hidden,
        audio_conv_rows,
        hidden_size
    );
    result.audio_encoder_ms = Ms(Clock::now() - t0).count();
    if(!audio_encoded.is_ok()) {
        return audio_encoded.get_error();
    }

    const int32_t audio_hidden = tensor_hidden_size(audio_encoded.value());
    const int32_t audio_rows = tensor_sequence_length(audio_encoded.value());
    if(audio_hidden <= 0 || audio_rows <= 0) {
        return Error::shape_mismatch("audio_encoder: invalid output shape.");
    }
    result.audio_token_count = audio_rows;

    if(!request.allow_audio_truncation && audio_rows != request.audio_placeholder_count) {
        return Error::shape_mismatch(
            "Forced alignment audio placeholder count must match encoded audio token count."
        );
    }

    if(cancel && cancel->is_cancelled()) {
        return Error::cancelled("Qwen3 forced alignment cancelled after audio encoder.");
    }

    t0 = Clock::now();
    std::vector<int64_t> token_ids = request.token_ids;
    auto embedded = run_embedding(impl_->embedding, token_ids);
    result.embedding_ms = Ms(Clock::now() - t0).count();
    if(!embedded.is_ok()) {
        return embedded.get_error();
    }

    const int32_t prompt_length = tensor_sequence_length(embedded.value());
    const int32_t prompt_hidden = tensor_hidden_size(embedded.value());
    if(prompt_length != static_cast<int32_t>(request.token_ids.size())) {
        return Error::shape_mismatch("embedding: output sequence length does not match token_ids.");
    }
    if(prompt_hidden != audio_hidden || prompt_hidden != impl_->backbone_model->n_embd()) {
        return Error::shape_mismatch("embedding/audio/backbone hidden sizes do not match.");
    }

    const int32_t replacement_rows = std::min(audio_rows, request.audio_placeholder_count);
    std::vector<float> prompt_embeddings = std::move(embedded.value().values);
    for(int32_t row = 0; row < replacement_rows; ++row) {
        const size_t dst = static_cast<size_t>(request.audio_placeholder_start + row) * static_cast<size_t>(prompt_hidden);
        const size_t src = static_cast<size_t>(row) * static_cast<size_t>(audio_hidden);
        std::copy_n(
            audio_encoded.value().values.begin() + static_cast<std::ptrdiff_t>(src),
            prompt_hidden,
            prompt_embeddings.begin() + static_cast<std::ptrdiff_t>(dst)
        );
    }

    if(cancel && cancel->is_cancelled()) {
        return Error::cancelled("Qwen3 forced alignment cancelled after embeddings.");
    }

    t0 = Clock::now();
    impl_->backbone_ctx.clear_kv_cache();
    std::vector<int32_t> base_positions(static_cast<size_t>(prompt_length));
    std::iota(base_positions.begin(), base_positions.end(), 0);
    std::vector<int32_t> positions;
    positions.reserve(static_cast<size_t>(prompt_length) * static_cast<size_t>(impl_->config.position_components));
    auto position_err = godot_llama::normalize_position_layout(
        base_positions,
        prompt_length,
        impl_->config.position_components,
        positions
    );
    if(position_err) {
        return Error::inference_failed("Forced-aligner position layout failed: " + position_err.message);
    }

    auto decode_err = impl_->backbone_ctx.decode_embeddings(
        prompt_embeddings,
        prompt_length,
        prompt_hidden,
        positions,
        impl_->config.position_components
    );
    result.backbone_ms = Ms(Clock::now() - t0).count();
    if(!decode_err.ok()) {
        return Error::inference_failed("Forced-aligner backbone decode failed: " + decode_err.message);
    }

    if(cancel && cancel->is_cancelled()) {
        return Error::cancelled("Qwen3 forced alignment cancelled after backbone.");
    }

    t0 = Clock::now();
    std::vector<int32_t> raw_bins;
    std::vector<double> confidences;
    raw_bins.reserve(request.timestamp_token_indices.size());
    confidences.reserve(request.timestamp_token_indices.size());
    float *hidden_base = impl_->backbone_ctx.get_embeddings();
    for(int32_t timestamp_index : request.timestamp_token_indices) {
        float *hidden = nullptr;
        if(hidden_base != nullptr) {
            hidden = hidden_base + (static_cast<ptrdiff_t>(timestamp_index) * prompt_hidden);
        }
        if(hidden == nullptr) {
            hidden = impl_->backbone_ctx.get_embeddings_ith(timestamp_index);
        }
        if(hidden == nullptr) {
            return Error::inference_failed("Forced-aligner backbone did not expose timestamp hidden states.");
        }

        auto prediction = impl_->classifier.classify({hidden, static_cast<size_t>(prompt_hidden)});
        if(!prediction.is_ok()) {
            return prediction.get_error();
        }
        raw_bins.push_back(prediction.value().bin);
        confidences.push_back(prediction.value().confidence);
    }

    const std::vector<int32_t> fixed_bins = repair_monotonic_timestamp_bins(raw_bins);
    result.spans.reserve(request.text_units.size());
    const double bin_to_seconds = static_cast<double>(impl_->config.timestamp_segment_ms) / 1000.0;
    for(size_t unit_index = 0; unit_index < request.text_units.size(); ++unit_index) {
        const size_t start_slot = unit_index * 2U;
        const size_t end_slot = start_slot + 1U;
        Qwen3ForcedAlignmentSpan span;
        span.text = request.text_units[unit_index];
        span.start_bin = fixed_bins[start_slot];
        span.end_bin = fixed_bins[end_slot];
        span.start_sec = static_cast<double>(span.start_bin) * bin_to_seconds;
        span.end_sec = static_cast<double>(span.end_bin) * bin_to_seconds;
        span.confidence = std::min(confidences[start_slot], confidences[end_slot]);
        result.spans.push_back(std::move(span));
    }
    result.classifier_ms = Ms(Clock::now() - t0).count();
    result.elapsed_ms = Ms(Clock::now() - total_start).count();
    return result;
}

} // namespace gotst
