#include "gotst/core/qwen_tts_pipeline.hpp"

#include "core/onnx_embedding_utils.hpp"

#include <gonx/core/provider.hpp>
#include <gonx/core/session.hpp>

#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gotst {

namespace {

constexpr int32_t kCodebookSize = 2048;
constexpr int32_t kWrappedPrefixTokenCount = 3;
constexpr int32_t kWrappedSuffixTokenCount = 5;

struct JsonValue {
    enum class Type {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };

    Type type = Type::Null;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<JsonValue> array_values;
    std::map<std::string, JsonValue> object_values;

    [[nodiscard]] const JsonValue *get(const std::string &key) const {
        if(type != Type::Object) {
            return nullptr;
        }
        const auto found = object_values.find(key);
        return found == object_values.end() ? nullptr : &found->second;
    }

    [[nodiscard]] bool is_array() const {
        return type == Type::Array;
    }

    [[nodiscard]] bool is_object() const {
        return type == Type::Object;
    }

    [[nodiscard]] std::string as_string(const std::string &fallback = "") const {
        return type == Type::String ? string_value : fallback;
    }

    [[nodiscard]] int32_t as_int(int32_t fallback = 0) const {
        if(type != Type::Number || !std::isfinite(number_value)) {
            return fallback;
        }
        return static_cast<int32_t>(number_value);
    }

    [[nodiscard]] int64_t as_int64(int64_t fallback = 0) const {
        if(type != Type::Number || !std::isfinite(number_value)) {
            return fallback;
        }
        return static_cast<int64_t>(number_value);
    }

    [[nodiscard]] float as_float(float fallback = 0.0f) const {
        if(type != Type::Number || !std::isfinite(number_value)) {
            return fallback;
        }
        return static_cast<float>(number_value);
    }

    [[nodiscard]] bool as_bool(bool fallback = false) const {
        return type == Type::Bool ? bool_value : fallback;
    }
};

void append_utf8(std::string &out, uint32_t codepoint) {
    if(codepoint <= 0x7fu) {
        out.push_back(static_cast<char>(codepoint));
    } else if(codepoint <= 0x7ffu) {
        out.push_back(static_cast<char>(0xc0u | (codepoint >> 6u)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
    } else if(codepoint <= 0xffffu) {
        out.push_back(static_cast<char>(0xe0u | (codepoint >> 12u)));
        out.push_back(static_cast<char>(0x80u | ((codepoint >> 6u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
    } else {
        out.push_back(static_cast<char>(0xf0u | (codepoint >> 18u)));
        out.push_back(static_cast<char>(0x80u | ((codepoint >> 12u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | ((codepoint >> 6u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
    }
}

int hex_value(char ch) {
    if(ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if(ch >= 'a' && ch <= 'f') {
        return 10 + ch - 'a';
    }
    if(ch >= 'A' && ch <= 'F') {
        return 10 + ch - 'A';
    }
    return -1;
}

std::optional<double> parse_double_value(const std::string &value) {
    char *end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if(end == value.c_str() || end == nullptr || *end != '\0') {
        return std::nullopt;
    }
    return parsed;
}

class JsonParser {
public:
    JsonParser(std::string_view text, std::string source_name)
        : text_(text), source_name_(std::move(source_name)) {}

    Result<JsonValue> parse() {
        skip_ws();
        auto value = parse_value();
        if(!value.is_ok()) {
            return value.get_error();
        }
        skip_ws();
        if(cursor_ != text_.size()) {
            return fail("trailing content after JSON value");
        }
        return value.value();
    }

private:
    Result<JsonValue> parse_value() {
        skip_ws();
        if(cursor_ >= text_.size()) {
            return fail("unexpected end of input");
        }
        const char ch = text_[cursor_];
        if(ch == 'n') {
            return parse_literal("null", JsonValue{});
        }
        if(ch == 't') {
            JsonValue value;
            value.type = JsonValue::Type::Bool;
            value.bool_value = true;
            return parse_literal("true", value);
        }
        if(ch == 'f') {
            JsonValue value;
            value.type = JsonValue::Type::Bool;
            value.bool_value = false;
            return parse_literal("false", value);
        }
        if(ch == '"') {
            return parse_string_value();
        }
        if(ch == '[') {
            return parse_array();
        }
        if(ch == '{') {
            return parse_object();
        }
        if(ch == '-' || (ch >= '0' && ch <= '9')) {
            return parse_number();
        }
        return fail("unexpected character in JSON value");
    }

    Result<JsonValue> parse_literal(std::string_view literal, JsonValue value) {
        if(text_.substr(cursor_, literal.size()) != literal) {
            return fail("invalid JSON literal");
        }
        cursor_ += literal.size();
        return value;
    }

    Result<JsonValue> parse_string_value() {
        auto parsed = parse_string();
        if(!parsed.is_ok()) {
            return parsed.get_error();
        }
        JsonValue value;
        value.type = JsonValue::Type::String;
        value.string_value = std::move(parsed.value());
        return value;
    }

    Result<std::string> parse_string() {
        if(cursor_ >= text_.size() || text_[cursor_] != '"') {
            return fail("expected string").get_error();
        }
        ++cursor_;
        std::string out;
        while(cursor_ < text_.size()) {
            const char ch = text_[cursor_++];
            if(ch == '"') {
                return out;
            }
            if(ch != '\\') {
                out.push_back(ch);
                continue;
            }
            if(cursor_ >= text_.size()) {
                return fail("unterminated escape sequence").get_error();
            }
            const char escaped = text_[cursor_++];
            switch(escaped) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    auto codepoint = parse_u16_escape();
                    if(!codepoint.has_value()) {
                        return fail("invalid unicode escape").get_error();
                    }
                    uint32_t cp = *codepoint;
                    if(cp >= 0xd800u && cp <= 0xdbffu && cursor_ + 1 < text_.size() &&
                       text_[cursor_] == '\\' && text_[cursor_ + 1] == 'u') {
                        cursor_ += 2;
                        auto low = parse_u16_escape();
                        if(!low.has_value() || *low < 0xdc00u || *low > 0xdfffu) {
                            return fail("invalid unicode surrogate pair").get_error();
                        }
                        cp = 0x10000u + (((cp - 0xd800u) << 10u) | (*low - 0xdc00u));
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    return fail("unsupported escape sequence").get_error();
            }
        }
        return fail("unterminated string").get_error();
    }

    std::optional<uint32_t> parse_u16_escape() {
        if(cursor_ + 4 > text_.size()) {
            return std::nullopt;
        }
        uint32_t value = 0;
        for(int index = 0; index < 4; ++index) {
            const int hex = hex_value(text_[cursor_++]);
            if(hex < 0) {
                return std::nullopt;
            }
            value = (value << 4u) | static_cast<uint32_t>(hex);
        }
        return value;
    }

    Result<JsonValue> parse_number() {
        const size_t start = cursor_;
        if(text_[cursor_] == '-') {
            ++cursor_;
        }
        consume_digits();
        if(cursor_ < text_.size() && text_[cursor_] == '.') {
            ++cursor_;
            consume_digits();
        }
        if(cursor_ < text_.size() && (text_[cursor_] == 'e' || text_[cursor_] == 'E')) {
            ++cursor_;
            if(cursor_ < text_.size() && (text_[cursor_] == '+' || text_[cursor_] == '-')) {
                ++cursor_;
            }
            consume_digits();
        }
        const std::string number_text(text_.substr(start, cursor_ - start));
        auto parsed = parse_double_value(number_text);
        if(!parsed.has_value()) {
            return fail("invalid number");
        }
        JsonValue value;
        value.type = JsonValue::Type::Number;
        value.number_value = *parsed;
        return value;
    }

    Result<JsonValue> parse_array() {
        JsonValue value;
        value.type = JsonValue::Type::Array;
        ++cursor_;
        skip_ws();
        if(cursor_ < text_.size() && text_[cursor_] == ']') {
            ++cursor_;
            return value;
        }
        while(true) {
            auto item = parse_value();
            if(!item.is_ok()) {
                return item.get_error();
            }
            value.array_values.push_back(std::move(item.value()));
            skip_ws();
            if(cursor_ >= text_.size()) {
                return fail("unterminated array");
            }
            if(text_[cursor_] == ']') {
                ++cursor_;
                return value;
            }
            if(text_[cursor_] != ',') {
                return fail("expected ',' or ']' in array");
            }
            ++cursor_;
        }
    }

    Result<JsonValue> parse_object() {
        JsonValue value;
        value.type = JsonValue::Type::Object;
        ++cursor_;
        skip_ws();
        if(cursor_ < text_.size() && text_[cursor_] == '}') {
            ++cursor_;
            return value;
        }
        while(true) {
            skip_ws();
            auto key = parse_string();
            if(!key.is_ok()) {
                return key.get_error();
            }
            skip_ws();
            if(cursor_ >= text_.size() || text_[cursor_] != ':') {
                return fail("expected ':' in object");
            }
            ++cursor_;
            auto item = parse_value();
            if(!item.is_ok()) {
                return item.get_error();
            }
            value.object_values[std::move(key.value())] = std::move(item.value());
            skip_ws();
            if(cursor_ >= text_.size()) {
                return fail("unterminated object");
            }
            if(text_[cursor_] == '}') {
                ++cursor_;
                return value;
            }
            if(text_[cursor_] != ',') {
                return fail("expected ',' or '}' in object");
            }
            ++cursor_;
        }
    }

    void skip_ws() {
        while(cursor_ < text_.size()) {
            const char ch = text_[cursor_];
            if(ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') {
                break;
            }
            ++cursor_;
        }
    }

    void consume_digits() {
        while(cursor_ < text_.size() && text_[cursor_] >= '0' && text_[cursor_] <= '9') {
            ++cursor_;
        }
    }

    Result<JsonValue> fail(const std::string &message) const {
        std::ostringstream stream;
        stream << source_name_ << ':' << cursor_ << ": " << message;
        return Error::invalid_argument(stream.str());
    }

    std::string_view text_;
    std::string source_name_;
    size_t cursor_ = 0;
};

Result<std::string> read_text_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if(!input) {
        return Error::not_found("failed to open file: " + path);
    }
    const std::streamoff size = input.tellg();
    if(size < 0) {
        return Error::io_error("failed to inspect file size: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::string text(static_cast<size_t>(size), '\0');
    if(!text.empty() && !input.read(text.data(), size)) {
        return Error::io_error("failed to read file: " + path);
    }
    return text;
}

Result<JsonValue> read_json_file(const std::string &path) {
    auto text = read_text_file(path);
    if(!text.is_ok()) {
        return text.get_error();
    }
    JsonParser parser(text.value(), path);
    return parser.parse();
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

int32_t json_int(const JsonValue *object, const std::string &key, int32_t fallback) {
    const JsonValue *value = object == nullptr ? nullptr : object->get(key);
    return value == nullptr ? fallback : value->as_int(fallback);
}

float json_float(const JsonValue *object, const std::string &key, float fallback) {
    const JsonValue *value = object == nullptr ? nullptr : object->get(key);
    return value == nullptr ? fallback : value->as_float(fallback);
}

bool json_bool(const JsonValue *object, const std::string &key, bool fallback) {
    const JsonValue *value = object == nullptr ? nullptr : object->get(key);
    return value == nullptr ? fallback : value->as_bool(fallback);
}

std::string json_string(const JsonValue *object, const std::string &key, const std::string &fallback) {
    const JsonValue *value = object == nullptr ? nullptr : object->get(key);
    return value == nullptr ? fallback : value->as_string(fallback);
}

int32_t clamp_int32(int32_t value, int32_t min_value, int32_t max_value) {
    return std::max(min_value, std::min(value, max_value));
}

std::vector<float> trim_waveform_prefix(
    std::vector<float> waveform,
    int32_t trim_prefix_frames,
    int32_t decode_frame_count
) {
    if(waveform.empty() || trim_prefix_frames <= 0 || decode_frame_count <= 0) {
        return waveform;
    }
    const int64_t trim_samples = static_cast<int64_t>(
        (static_cast<double>(trim_prefix_frames) /
            static_cast<double>(std::max<int32_t>(decode_frame_count, 1))) *
        static_cast<double>(waveform.size())
    );
    if(trim_samples <= 0) {
        return waveform;
    }
    if(trim_samples >= static_cast<int64_t>(waveform.size())) {
        return {};
    }
    waveform.erase(waveform.begin(), waveform.begin() + trim_samples);
    return waveform;
}

struct FloatTensor {
    std::vector<float> values;
    std::vector<int64_t> shape;
};

struct FloatSequence {
    std::vector<float> values;
    int64_t sequence_length = 0;
    int64_t hidden_size = 0;
};

struct QwenModelConfig {
    int32_t hidden_size = 1024;
    int32_t codec_groups = 16;
    int32_t tts_bos_token_id = 151672;
    int32_t tts_eos_token_id = 151673;
    int32_t tts_pad_token_id = 151671;
    int32_t codec_bos_token_id = 2149;
    int32_t codec_eos_token_id = 2150;
    int32_t codec_pad_token_id = 2148;
    int32_t codec_think_token_id = 2154;
    int32_t codec_nothink_token_id = 2155;
    int32_t codec_think_bos_token_id = 2156;
    int32_t codec_think_eos_token_id = 2157;
    int32_t codec_japanese_language_token_id = 2058;
    std::string tts_model_type = "base";
};

struct QwenGenerationConfig {
    bool do_sample = true;
    int32_t top_k = 50;
    float top_p = 1.0f;
    float temperature = 0.9f;
    bool sub_do_sample = true;
    int32_t sub_top_k = 50;
    float sub_top_p = 1.0f;
    float sub_temperature = 0.9f;
    float repetition_penalty = 1.05f;
};

QwenModelConfig read_model_config(const JsonValue &root) {
    QwenModelConfig config;
    const JsonValue *talker = root.get("talker_config");
    config.tts_model_type = json_string(&root, "tts_model_type", config.tts_model_type);
    config.tts_bos_token_id = json_int(&root, "tts_bos_token_id", config.tts_bos_token_id);
    config.tts_eos_token_id = json_int(&root, "tts_eos_token_id", config.tts_eos_token_id);
    config.tts_pad_token_id = json_int(&root, "tts_pad_token_id", config.tts_pad_token_id);
    config.hidden_size = json_int(talker, "hidden_size", config.hidden_size);
    config.codec_groups = json_int(talker, "num_code_groups", config.codec_groups);
    config.codec_bos_token_id = json_int(talker, "codec_bos_id", config.codec_bos_token_id);
    config.codec_eos_token_id = json_int(talker, "codec_eos_token_id", config.codec_eos_token_id);
    config.codec_pad_token_id = json_int(talker, "codec_pad_id", config.codec_pad_token_id);
    config.codec_think_token_id = json_int(talker, "codec_think_id", config.codec_think_token_id);
    config.codec_nothink_token_id = json_int(talker, "codec_nothink_id", config.codec_nothink_token_id);
    config.codec_think_bos_token_id = json_int(talker, "codec_think_bos_id", config.codec_think_bos_token_id);
    config.codec_think_eos_token_id = json_int(talker, "codec_think_eos_id", config.codec_think_eos_token_id);
    const JsonValue *language_ids = talker == nullptr ? nullptr : talker->get("codec_language_id");
    config.codec_japanese_language_token_id =
        json_int(language_ids, "japanese", config.codec_japanese_language_token_id);
    return config;
}

QwenGenerationConfig read_generation_config(const JsonValue &root) {
    QwenGenerationConfig config;
    config.do_sample = json_bool(&root, "do_sample", config.do_sample);
    config.top_k = json_int(&root, "top_k", config.top_k);
    config.top_p = json_float(&root, "top_p", config.top_p);
    config.temperature = json_float(&root, "temperature", config.temperature);
    config.sub_do_sample = json_bool(&root, "subtalker_dosample", config.sub_do_sample);
    config.sub_top_k = json_int(&root, "subtalker_top_k", config.sub_top_k);
    config.sub_top_p = json_float(&root, "subtalker_top_p", config.sub_top_p);
    config.sub_temperature = json_float(&root, "subtalker_temperature", config.sub_temperature);
    config.repetition_penalty = json_float(&root, "repetition_penalty", config.repetition_penalty);
    return config;
}

gonx::SessionConfig make_session_config(
    const std::string &provider,
    int32_t intra_op_threads,
    int32_t inter_op_threads,
    int32_t optimization_level,
    const std::string &optimized_model_path = {}
) {
    gonx::SessionConfig config;
    config.providers = {gonx::parse_provider(provider)};
    config.intra_op_num_threads = intra_op_threads;
    config.inter_op_num_threads = inter_op_threads;
    config.optimization_level = optimization_level;
    config.optimized_model_path = optimized_model_path;
    return config;
}

Result<void> load_session(
    gonx::InferenceSession &session,
    const std::string &path,
    const std::string &provider,
    int32_t intra_op_threads,
    int32_t inter_op_threads,
    int32_t optimization_level
) {
    if(path.empty()) {
        return Error::invalid_argument("empty ONNX path");
    }
    auto status = session.load(
        path,
        make_session_config(provider, intra_op_threads, inter_op_threads, optimization_level)
    );
    if(status.has_error()) {
        return Error::model_not_loaded("failed to load ONNX model " + path + ": " + status.error().message);
    }
    return {};
}

Ort::MemoryInfo &cpu_memory_info() {
    static Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return memory_info;
}

Result<std::vector<int64_t>> infer_shape(std::span<const int64_t> spec_shape, size_t element_count) {
    if(spec_shape.empty()) {
        return std::vector<int64_t>{static_cast<int64_t>(element_count)};
    }

    std::vector<int64_t> shape(spec_shape.begin(), spec_shape.end());
    int dynamic_count = 0;
    int dynamic_index = -1;
    int64_t known = 1;
    for(size_t index = 0; index < shape.size(); ++index) {
        if(shape[index] < 0) {
            dynamic_count += 1;
            dynamic_index = static_cast<int>(index);
            continue;
        }
        known *= std::max<int64_t>(shape[index], 1);
    }
    if(dynamic_count == 0) {
        if(known != static_cast<int64_t>(element_count)) {
            return Error::shape_mismatch("static ONNX input shape does not match value count");
        }
        return shape;
    }
    if(dynamic_count != 1 || known <= 0) {
        return Error::shape_mismatch("cannot infer ONNX input shape with multiple dynamic dimensions");
    }
    if((static_cast<int64_t>(element_count) % known) != 0) {
        return Error::shape_mismatch("ONNX input value count is not divisible by known shape product");
    }
    shape[static_cast<size_t>(dynamic_index)] = static_cast<int64_t>(element_count) / known;
    return shape;
}

Result<FloatTensor> run_single_input_int64(
    gonx::InferenceSession &session,
    std::span<const int64_t> ids,
    std::string_view label
) {
    if(session.input_specs().size() != 1) {
        return Error::shape_mismatch(std::string(label) + " expected a one-input ONNX model");
    }
    auto shape = infer_shape(session.input_specs().front().shape, ids.size());
    if(!shape.is_ok()) {
        return shape.get_error();
    }
    std::vector<int64_t> mutable_ids(ids.begin(), ids.end());
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
        cpu_memory_info(),
        mutable_ids.data(),
        mutable_ids.size(),
        shape.value().data(),
        shape.value().size()
    ));
    auto run_result = session.run(inputs);
    if(run_result.has_error()) {
        return Error::inference_failed(std::string(label) + " inference failed: " + run_result.error().message);
    }
    auto outputs = std::move(run_result).value();
    if(outputs.empty() || !outputs.front().IsTensor()) {
        return Error::inference_failed(std::string(label) + " returned no tensor output");
    }
    const auto info = outputs.front().GetTensorTypeAndShapeInfo();
    if(info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Error::shape_mismatch(std::string(label) + " returned a non-float output");
    }
    const size_t count = info.GetElementCount();
    const float *data = outputs.front().GetTensorData<float>();
    FloatTensor tensor;
    tensor.shape = info.GetShape();
    tensor.values.assign(data, data + count);
    return tensor;
}

Result<FloatTensor> run_single_input_float(
    gonx::InferenceSession &session,
    std::span<const float> values,
    std::string_view label
) {
    if(session.input_specs().size() != 1) {
        return Error::shape_mismatch(std::string(label) + " expected a one-input ONNX model");
    }
    auto shape = infer_shape(session.input_specs().front().shape, values.size());
    if(!shape.is_ok()) {
        return shape.get_error();
    }
    std::vector<float> mutable_values(values.begin(), values.end());
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        cpu_memory_info(),
        mutable_values.data(),
        mutable_values.size(),
        shape.value().data(),
        shape.value().size()
    ));
    auto run_result = session.run(inputs);
    if(run_result.has_error()) {
        return Error::inference_failed(std::string(label) + " inference failed: " + run_result.error().message);
    }
    auto outputs = std::move(run_result).value();
    if(outputs.empty() || !outputs.front().IsTensor()) {
        return Error::inference_failed(std::string(label) + " returned no tensor output");
    }
    const auto info = outputs.front().GetTensorTypeAndShapeInfo();
    if(info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Error::shape_mismatch(std::string(label) + " returned a non-float output");
    }
    const size_t count = info.GetElementCount();
    const float *data = outputs.front().GetTensorData<float>();
    FloatTensor tensor;
    tensor.shape = info.GetShape();
    tensor.values.assign(data, data + count);
    return tensor;
}

Result<FloatSequence> extract_sequence(const FloatTensor &tensor, std::string_view label) {
    if(tensor.values.empty() || tensor.shape.empty()) {
        return Error::empty_input(std::string(label) + " tensor was empty");
    }
    const int64_t hidden_size = tensor.shape.back();
    if(hidden_size <= 0) {
        return Error::shape_mismatch(std::string(label) + " tensor has invalid hidden size");
    }
    if((tensor.values.size() % static_cast<size_t>(hidden_size)) != 0) {
        return Error::shape_mismatch(std::string(label) + " tensor values are not aligned to hidden size");
    }
    FloatSequence sequence;
    sequence.values = tensor.values;
    sequence.hidden_size = hidden_size;
    sequence.sequence_length = static_cast<int64_t>(tensor.values.size() / static_cast<size_t>(hidden_size));
    return sequence;
}

Result<FloatSequence> project_text_tokens(
    gonx::InferenceSession &text_embedding,
    gonx::InferenceSession &text_projection,
    std::span<const int64_t> token_ids,
    std::string_view label
) {
    auto embedded = run_single_input_int64(text_embedding, token_ids, label);
    if(!embedded.is_ok()) {
        return embedded.get_error();
    }
    auto embedded_sequence = extract_sequence(embedded.value(), label);
    if(!embedded_sequence.is_ok()) {
        return embedded_sequence.get_error();
    }
    auto projected = run_single_input_float(
        text_projection,
        embedded_sequence.value().values,
        std::string(label) + " projection"
    );
    if(!projected.is_ok()) {
        return projected.get_error();
    }
    return extract_sequence(projected.value(), std::string(label) + " projected");
}

Result<FloatSequence> embed_codec_tokens(
    gonx::InferenceSession &codec_embedding,
    std::span<const int64_t> token_ids,
    std::string_view label
) {
    auto embedded = run_single_input_int64(codec_embedding, token_ids, label);
    if(!embedded.is_ok()) {
        return embedded.get_error();
    }
    return extract_sequence(embedded.value(), label);
}

Result<std::unique_ptr<tokenizers::Tokenizer>> load_tokenizer(const std::string &path) {
    auto blob = read_text_file(path);
    if(!blob.is_ok()) {
        return blob.get_error();
    }
    std::unique_ptr<tokenizers::Tokenizer> tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob.value());
    if(!tokenizer) {
        return Error::invalid_argument("failed to create tokenizer from " + path);
    }
    return tokenizer;
}

std::vector<int64_t> encode_tokens(
    tokenizers::Tokenizer &tokenizer,
    const std::string &text,
    int32_t max_tokens
) {
    std::vector<int32_t> raw = tokenizer.Encode(text);
    const size_t count = max_tokens > 0 ? std::min(raw.size(), static_cast<size_t>(max_tokens)) : raw.size();
    std::vector<int64_t> ids;
    ids.reserve(count);
    for(size_t index = 0; index < count; ++index) {
        ids.push_back(raw[index]);
    }
    return ids;
}

std::string wrapped_assistant_text(const std::string &text) {
    return "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string wrapped_ref_assistant_text(const std::string &text) {
    return "<|im_start|>assistant\n" + text + "<|im_end|>\n";
}

std::string wrapped_instruction_text(const std::string &text) {
    return "<|im_start|>user\n" + text + "<|im_end|>\n";
}

std::vector<int64_t> build_codec_prefill_tokens(
    const QwenTtsPipelineConfig &config,
    const QwenModelConfig &model
) {
    std::vector<int64_t> tokens;
    if(config.force_japanese_language) {
        tokens.push_back(model.codec_think_token_id);
        tokens.push_back(model.codec_think_bos_token_id);
        tokens.push_back(model.codec_japanese_language_token_id);
        tokens.push_back(model.codec_think_eos_token_id);
    } else {
        tokens.push_back(model.codec_nothink_token_id);
        tokens.push_back(model.codec_think_bos_token_id);
        tokens.push_back(model.codec_think_eos_token_id);
    }
    tokens.push_back(model.codec_pad_token_id);
    tokens.push_back(model.codec_bos_token_id);
    return tokens;
}

void append_float_array(const JsonValue *value, std::vector<float> &out) {
    if(value == nullptr || !value->is_array()) {
        return;
    }
    out.reserve(out.size() + value->array_values.size());
    for(const JsonValue &entry : value->array_values) {
        out.push_back(entry.as_float(0.0f));
    }
}

std::vector<float> speaker_embedding_from_json(const JsonValue &root) {
    std::vector<float> embedding;
    if(root.is_array()) {
        append_float_array(&root, embedding);
        return embedding;
    }
    append_float_array(root.get("speaker_embedding"), embedding);
    return embedding;
}

const JsonValue *speaker_map_from_json(const JsonValue &root) {
    const JsonValue *speaker_map = root.get("spk_id");
    if(speaker_map != nullptr && speaker_map->is_object()) {
        return speaker_map;
    }
    const JsonValue *talker = root.get("talker_config");
    speaker_map = talker == nullptr ? nullptr : talker->get("spk_id");
    return speaker_map != nullptr && speaker_map->is_object() ? speaker_map : nullptr;
}

std::vector<std::string> custom_voice_speaker_names_from_json(const JsonValue &root) {
    const JsonValue *speaker_map = speaker_map_from_json(root);
    if(speaker_map == nullptr) {
        return {};
    }
    std::vector<std::string> names;
    names.reserve(speaker_map->object_values.size());
    for(const auto &entry : speaker_map->object_values) {
        names.push_back(entry.first);
    }
    return names;
}

Result<std::vector<int64_t>> custom_voice_speaker_ids_from_json(
    const JsonValue &root,
    const std::string &speaker_name
) {
    if(speaker_name.empty()) {
        return Error::invalid_argument("custom_voice requires a speaker name");
    }
    const JsonValue *speaker_map = speaker_map_from_json(root);
    if(speaker_map == nullptr) {
        return Error::not_found("custom voice config has no spk_id map");
    }
    const JsonValue *entry = speaker_map->get(speaker_name);
    if(entry == nullptr) {
        return Error::not_found("custom voice speaker not found: " + speaker_name);
    }
    std::vector<int64_t> ids;
    if(entry->is_array()) {
        ids.reserve(entry->array_values.size());
        for(const JsonValue &value : entry->array_values) {
            const int64_t id = value.as_int64(-1);
            if(id >= 0) {
                ids.push_back(id);
            }
        }
    } else {
        const int64_t id = entry->as_int64(-1);
        if(id >= 0) {
            ids.push_back(id);
        }
    }
    if(ids.empty()) {
        return Error::empty_input("custom voice speaker has no token ids: " + speaker_name);
    }
    return ids;
}

std::vector<int64_t> int64_array_from_json(const JsonValue *value) {
    if(value == nullptr || !value->is_array()) {
        return {};
    }
    std::vector<int64_t> ids;
    ids.reserve(value->array_values.size());
    for(const JsonValue &entry : value->array_values) {
        ids.push_back(entry.as_int64(0));
    }
    return ids;
}

struct RefCodeData {
    std::vector<int64_t> codes;
    int32_t frames = 0;
    int32_t groups = 0;
};

RefCodeData ref_codes_from_json(const JsonValue &root) {
    RefCodeData data;
    data.codes = int64_array_from_json(root.get("ref_codes"));
    std::vector<int64_t> shape = int64_array_from_json(root.get("ref_codes_shape"));
    if(shape.size() >= 2) {
        data.frames = static_cast<int32_t>(shape[0]);
        data.groups = static_cast<int32_t>(shape[1]);
    }
    if(data.frames <= 0 || data.groups <= 0 ||
       data.codes.size() != static_cast<size_t>(data.frames) * static_cast<size_t>(data.groups)) {
        data.codes.clear();
        data.frames = 0;
        data.groups = 0;
    }
    return data;
}

Result<std::vector<float>> embed_custom_voice_tokens(
    gonx::InferenceSession &codec_embedding,
    std::span<const int64_t> token_ids,
    int32_t hidden_size
) {
    if(hidden_size <= 0) {
        return Error::invalid_argument("custom voice embedding requires hidden_size > 0");
    }
    std::vector<float> embedding_values;
    embedding_values.reserve(token_ids.size() * static_cast<size_t>(hidden_size));
    detail::SingleTokenEmbeddingRunScratch scratch;
    for(int64_t token_id : token_ids) {
        auto token_embedding = detail::run_single_token_float_embedding(codec_embedding, scratch, token_id);
        if(!token_embedding.is_ok()) {
            return token_embedding.get_error();
        }
        const std::span<const float> values = token_embedding.value().values;
        if(values.size() < static_cast<size_t>(hidden_size)) {
            return Error::shape_mismatch("custom voice codec embedding returned too few values");
        }
        embedding_values.insert(
            embedding_values.end(),
            values.begin(),
            values.begin() + static_cast<std::ptrdiff_t>(hidden_size)
        );
    }
    return embedding_values;
}

Result<std::vector<float>> embed_ref_codes(
    gonx::InferenceSession &codec_embedding,
    gonx::InferenceSession &predictor_embedding,
    const RefCodeData &ref_codes,
    int32_t hidden_size
) {
    if(ref_codes.codes.empty() || ref_codes.frames <= 0 || ref_codes.groups <= 0) {
        return std::vector<float>();
    }
    if(hidden_size <= 0) {
        return Error::invalid_argument("reference code embedding requires hidden_size > 0");
    }
    std::vector<float> all_frame_embeddings(
        static_cast<size_t>(ref_codes.frames) * static_cast<size_t>(hidden_size),
        0.0f
    );
    detail::SingleTokenEmbeddingRunScratch scratch;
    for(int32_t frame = 0; frame < ref_codes.frames; ++frame) {
        std::vector<float> summed(static_cast<size_t>(hidden_size), 0.0f);
        for(int32_t group = 0; group < ref_codes.groups; ++group) {
            const size_t code_index =
                static_cast<size_t>(frame) * static_cast<size_t>(ref_codes.groups) +
                static_cast<size_t>(group);
            const int64_t token_id = ref_codes.codes[code_index];
            const int64_t generation_step = group - 1;
            auto embedded = group == 0
                ? detail::run_single_token_float_embedding(codec_embedding, scratch, token_id)
                : detail::run_single_token_float_embedding(
                    predictor_embedding,
                    scratch,
                    token_id,
                    &generation_step
                );
            if(!embedded.is_ok()) {
                return embedded.get_error();
            }
            const std::span<const float> values = embedded.value().values;
            if(values.size() < static_cast<size_t>(hidden_size)) {
                return Error::shape_mismatch("reference code embedding returned too few values");
            }
            for(int32_t index = 0; index < hidden_size; ++index) {
                summed[static_cast<size_t>(index)] += values[static_cast<size_t>(index)];
            }
        }
        const size_t dest = static_cast<size_t>(frame) * static_cast<size_t>(hidden_size);
        std::copy(summed.begin(), summed.end(), all_frame_embeddings.begin() + static_cast<std::ptrdiff_t>(dest));
    }
    return all_frame_embeddings;
}

std::vector<int64_t> prefixed_codes(
    std::span<const int64_t> prefix,
    std::span<const int64_t> generated
) {
    if(prefix.empty()) {
        return std::vector<int64_t>(generated.begin(), generated.end());
    }
    std::vector<int64_t> output;
    output.reserve(prefix.size() + generated.size());
    output.insert(output.end(), prefix.begin(), prefix.end());
    output.insert(output.end(), generated.begin(), generated.end());
    return output;
}

int32_t estimate_target_frames(
    const QwenTtsPipelineConfig &config,
    int32_t text_token_count
) {
    const int32_t body_tokens = std::max<int32_t>(
        1,
        text_token_count - (kWrappedPrefixTokenCount + kWrappedSuffixTokenCount)
    );
    const int32_t target = static_cast<int32_t>(
        std::ceil(static_cast<float>(body_tokens) * std::max(config.target_frames_per_text_token, 1.0f))
    ) + config.target_frame_padding;
    return clamp_int32(
        target,
        std::max<int32_t>(1, config.min_frames_before_eos + 1),
        std::max<int32_t>(1, config.max_frames)
    );
}

} // namespace

struct QwenTtsPipeline::Impl {
    QwenTtsPipelineConfig config;
    QwenModelConfig model;
    QwenGenerationConfig generation;
    JsonValue model_json;
    JsonValue custom_voice_json;
    std::vector<std::string> custom_voice_names;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    gonx::InferenceSession text_embedding;
    gonx::InferenceSession text_projection;
    gonx::InferenceSession codec_embedding;
    gonx::InferenceSession predictor_embedding;
    TtsCodeGenerator generator;
    TtsWaveformDecoder decoder;
    bool loaded = false;
    mutable std::mutex mutex;
};

QwenTtsPipeline::QwenTtsPipeline() : impl_(std::make_unique<Impl>()) {}
QwenTtsPipeline::~QwenTtsPipeline() = default;
QwenTtsPipeline::QwenTtsPipeline(QwenTtsPipeline &&) noexcept = default;
QwenTtsPipeline &QwenTtsPipeline::operator=(QwenTtsPipeline &&) noexcept = default;

Result<void> QwenTtsPipeline::load(const QwenTtsPipelineConfig &config) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->loaded = false;
    impl_->config = config;

    auto model_json = read_json_file(config.model_config_path);
    if(!model_json.is_ok()) {
        return model_json.get_error();
    }
    auto generation_json = read_json_file(config.generation_config_path);
    if(!generation_json.is_ok()) {
        return generation_json.get_error();
    }
    impl_->model_json = std::move(model_json.value());
    impl_->model = read_model_config(impl_->model_json);
    impl_->generation = read_generation_config(generation_json.value());

    auto tokenizer = load_tokenizer(config.tokenizer_json_path);
    if(!tokenizer.is_ok()) {
        return tokenizer.get_error();
    }
    impl_->tokenizer = std::move(tokenizer.value());

    auto loaded = load_session(
        impl_->text_embedding,
        config.text_embedding_path,
        config.provider,
        config.intra_op_threads,
        config.inter_op_threads,
        config.optimization_level
    );
    if(!loaded.is_ok()) {
        return loaded.get_error();
    }
    loaded = load_session(
        impl_->text_projection,
        config.text_projection_path,
        config.provider,
        config.intra_op_threads,
        config.inter_op_threads,
        config.optimization_level
    );
    if(!loaded.is_ok()) {
        return loaded.get_error();
    }
    loaded = load_session(
        impl_->codec_embedding,
        config.codec_embedding_path,
        config.provider,
        config.intra_op_threads,
        config.inter_op_threads,
        config.optimization_level
    );
    if(!loaded.is_ok()) {
        return loaded.get_error();
    }
    loaded = load_session(
        impl_->predictor_embedding,
        config.predictor_embedding_path,
        config.provider,
        config.intra_op_threads,
        config.inter_op_threads,
        config.optimization_level
    );
    if(!loaded.is_ok()) {
        return loaded.get_error();
    }

    TtsModelPaths model_paths;
    model_paths.talker_gguf_path = config.talker_gguf_path;
    model_paths.predictor_gguf_path = config.predictor_gguf_path;
    model_paths.codec_embedding_onnx_path = config.codec_embedding_path;
    model_paths.predictor_embedding_onnx_path = config.predictor_embedding_path;

    TtsSessionConfig session_config;
    session_config.talker_n_ctx = config.talker_n_ctx;
    session_config.talker_n_batch = config.talker_n_batch;
    session_config.predictor_n_ctx = config.predictor_n_ctx;
    session_config.predictor_n_batch = config.predictor_n_batch;
    session_config.n_threads = config.n_threads;
    session_config.n_gpu_layers = config.n_gpu_layers;
    session_config.predictor_n_gpu_layers =
        config.predictor_n_gpu_layers >= -1 ? config.predictor_n_gpu_layers : config.n_gpu_layers;
    session_config.use_mmap = config.use_mmap;
    session_config.use_mlock = config.use_mlock;
    session_config.flash_attn_type = config.flash_attn_type;
    session_config.type_k = config.type_k;
    session_config.type_v = config.type_v;
    session_config.talker_position_components = 4;
    session_config.predictor_position_components = 1;
    auto generator_loaded = impl_->generator.load(model_paths, session_config);
    if(!generator_loaded.is_ok()) {
        return generator_loaded.get_error();
    }

    TtsWaveformDecoderConfig decoder_config;
    decoder_config.decoder_onnx_path = config.decoder_onnx_path;
    decoder_config.provider_requested = config.decoder_provider_requested;
    decoder_config.provider = config.decoder_provider;
    decoder_config.intra_op_threads = config.decoder_intra_op_threads;
    decoder_config.inter_op_threads = config.decoder_inter_op_threads;
    decoder_config.optimization_level = config.decoder_optimization_level;
    decoder_config.optimized_model_path = config.decoder_optimized_model_path;
    decoder_config.sample_rate = config.sample_rate;
    decoder_config.normalize_waveform = config.normalize_waveform;
    decoder_config.waveform_gain = config.waveform_gain;
    decoder_config.stateful_chunk_frames = config.stateful_chunk_frames;
    auto decoder_loaded = impl_->decoder.load(decoder_config);
    if(!decoder_loaded.is_ok()) {
        return decoder_loaded.get_error();
    }

    const std::string custom_config_path =
        !config.custom_voice_config_path.empty() ? config.custom_voice_config_path : config.model_config_path;
    auto custom_json = read_json_file(custom_config_path);
    if(custom_json.is_ok()) {
        impl_->custom_voice_json = std::move(custom_json.value());
        impl_->custom_voice_names = custom_voice_speaker_names_from_json(impl_->custom_voice_json);
    } else {
        impl_->custom_voice_json = impl_->model_json;
        impl_->custom_voice_names = custom_voice_speaker_names_from_json(impl_->custom_voice_json);
    }

    impl_->loaded = true;
    return {};
}

bool QwenTtsPipeline::is_loaded() const {
    return impl_ && impl_->loaded && impl_->tokenizer && impl_->generator.is_loaded() && impl_->decoder.is_loaded();
}

Result<QwenTtsPreparedPrompt> QwenTtsPipeline::prepare_prompt(const QwenTtsPipelineRequest &request) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(!is_loaded()) {
        return Error::model_not_loaded("QwenTtsPipeline is not loaded");
    }
    if(request.text.empty()) {
        return Error::empty_input("QwenTtsPipeline request text is empty");
    }

    const std::string mode = lower_ascii(request.mode.empty() ? impl_->config.mode : request.mode);
    const std::vector<int64_t> text_ids = encode_tokens(
        *impl_->tokenizer,
        wrapped_assistant_text(request.text),
        impl_->config.max_text_tokens
    );
    if(text_ids.empty()) {
        return Error::empty_input("assistant text tokenization returned no tokens");
    }

    auto text_projected = project_text_tokens(
        impl_->text_embedding,
        impl_->text_projection,
        text_ids,
        "assistant text"
    );
    if(!text_projected.is_ok()) {
        return text_projected.get_error();
    }

    const std::vector<int64_t> special_ids = {
        impl_->model.tts_bos_token_id,
        impl_->model.tts_eos_token_id,
        impl_->model.tts_pad_token_id,
    };
    auto special_projected = project_text_tokens(
        impl_->text_embedding,
        impl_->text_projection,
        special_ids,
        "special tokens"
    );
    if(!special_projected.is_ok()) {
        return special_projected.get_error();
    }

    const std::vector<int64_t> codec_prefill_ids = build_codec_prefill_tokens(impl_->config, impl_->model);
    auto codec_prefill = embed_codec_tokens(
        impl_->codec_embedding,
        codec_prefill_ids,
        "codec prefill"
    );
    if(!codec_prefill.is_ok()) {
        return codec_prefill.get_error();
    }

    std::vector<float> leading_prompt;
    int64_t leading_prompt_length = 0;
    const bool voice_design_mode = mode == "voice_design";
    const std::string instruction_text = voice_design_mode ? request.voice_design : request.style_instruction;
    if((voice_design_mode || impl_->config.use_style_instruction) && !instruction_text.empty()) {
        const std::vector<int64_t> instruction_ids = encode_tokens(
            *impl_->tokenizer,
            wrapped_instruction_text(instruction_text),
            256
        );
        if(!instruction_ids.empty()) {
            auto instruction_projected = project_text_tokens(
                impl_->text_embedding,
                impl_->text_projection,
                instruction_ids,
                voice_design_mode ? "voice design" : "style instruction"
            );
            if(!instruction_projected.is_ok()) {
                return instruction_projected.get_error();
            }
            leading_prompt = std::move(instruction_projected.value().values);
            leading_prompt_length = instruction_projected.value().sequence_length;
        }
    }

    std::vector<float> codec_prompt_insert;
    JsonValue speaker_json;
    bool has_speaker_json = false;
    if(mode == "custom_voice") {
        auto ids = custom_voice_speaker_ids_from_json(
            impl_->custom_voice_json,
            impl_->config.custom_voice_speaker_name
        );
        if(!ids.is_ok()) {
            return ids.get_error();
        }
        auto embedding = embed_custom_voice_tokens(
            impl_->codec_embedding,
            ids.value(),
            impl_->model.hidden_size
        );
        if(!embedding.is_ok()) {
            return embedding.get_error();
        }
        codec_prompt_insert = std::move(embedding.value());
    } else if(mode != "voice_design" && !impl_->config.speaker_embedding_path.empty()) {
        auto loaded_speaker = read_json_file(impl_->config.speaker_embedding_path);
        if(!loaded_speaker.is_ok()) {
            return loaded_speaker.get_error();
        }
        speaker_json = std::move(loaded_speaker.value());
        has_speaker_json = true;
        codec_prompt_insert = speaker_embedding_from_json(speaker_json);
        if(codec_prompt_insert.empty()) {
            return Error::empty_input("base speaker embedding is empty");
        }
    }

    std::vector<float> icl_overlay_values;
    int64_t icl_length = 0;
    std::vector<int64_t> decoder_prefix_codes;
    int32_t decoder_prefix_frames = 0;
    if(mode == "base" && impl_->config.use_icl_voice_clone && has_speaker_json) {
        const RefCodeData ref_codes = ref_codes_from_json(speaker_json);
        const std::string ref_text_override = impl_->config.icl_ref_text;
        const std::string ref_text = !ref_text_override.empty()
            ? ref_text_override
            : json_string(&speaker_json, "ref_text", "");
        if(!ref_text.empty() && !ref_codes.codes.empty() && ref_codes.groups == impl_->model.codec_groups) {
            std::vector<int64_t> ref_text_ids = encode_tokens(
                *impl_->tokenizer,
                wrapped_ref_assistant_text(ref_text),
                512
            );
            if(ref_text_ids.size() > static_cast<size_t>(kWrappedPrefixTokenCount)) {
                ref_text_ids.erase(
                    ref_text_ids.begin(),
                    ref_text_ids.begin() + kWrappedPrefixTokenCount
                );
                auto ref_text_projected = project_text_tokens(
                    impl_->text_embedding,
                    impl_->text_projection,
                    ref_text_ids,
                    "ICL reference text"
                );
                if(!ref_text_projected.is_ok()) {
                    return ref_text_projected.get_error();
                }
                auto ref_codec_embeddings = embed_ref_codes(
                    impl_->codec_embedding,
                    impl_->predictor_embedding,
                    ref_codes,
                    impl_->model.hidden_size
                );
                if(!ref_codec_embeddings.is_ok()) {
                    return ref_codec_embeddings.get_error();
                }
                auto overlay = build_voice_clone_icl_overlay(
                    ref_text_projected.value().values.data(),
                    ref_text_projected.value().sequence_length,
                    ref_codec_embeddings.value().data(),
                    ref_codes.frames,
                    impl_->model.hidden_size
                );
                if(!overlay.is_ok()) {
                    return overlay.get_error();
                }
                icl_overlay_values = std::move(overlay.value().icl_overlay);
                icl_length = overlay.value().icl_length;
                decoder_prefix_codes = ref_codes.codes;
                decoder_prefix_frames = ref_codes.frames;
            }
        }
    }

    auto prompt = build_tts_prompt_assembly({
        .text_projected_states = text_projected.value().values,
        .text_sequence_length = text_projected.value().sequence_length,
        .special_projected_states = special_projected.value().values,
        .codec_prefill_embeddings = codec_prefill.value().values,
        .codec_prefill_length = codec_prefill.value().sequence_length,
        .codec_prompt_insert = codec_prompt_insert,
        .leading_prompt_states = leading_prompt,
        .leading_prompt_length = leading_prompt_length,
        .icl_overlay = icl_overlay_values,
        .icl_length = icl_length,
        .hidden_size = text_projected.value().hidden_size,
        .wrapped_prefix_token_count = kWrappedPrefixTokenCount,
        .wrapped_suffix_token_count = kWrappedSuffixTokenCount,
    });
    if(!prompt.is_ok()) {
        return prompt.get_error();
    }

    QwenTtsPreparedPrompt prepared;
    prepared.prompt = std::move(prompt.value());
    prepared.codec_groups = impl_->model.codec_groups;
    prepared.decode_prefix_frames = decoder_prefix_frames;
    prepared.decoder_prefix_codes = std::move(decoder_prefix_codes);
    prepared.target_frames = estimate_target_frames(
        impl_->config,
        static_cast<int32_t>(text_ids.size())
    );

    prepared.sampling.codebook_size = kCodebookSize;
    prepared.sampling.residual_groups = std::max(0, impl_->model.codec_groups - 1);
    prepared.sampling.target_frames = prepared.target_frames;
    prepared.sampling.min_frames_before_eos =
        std::min(impl_->config.min_frames_before_eos, std::max(0, prepared.target_frames - 1));
    prepared.sampling.hidden_size = static_cast<int32_t>(prepared.prompt.hidden_size);
    prepared.sampling.eos_token_id = impl_->model.codec_eos_token_id;
    prepared.sampling.do_sample = impl_->generation.do_sample;
    prepared.sampling.top_k = impl_->generation.top_k;
    prepared.sampling.top_p = impl_->generation.top_p;
    prepared.sampling.temperature = impl_->generation.temperature;
    prepared.sampling.sub_do_sample = impl_->generation.sub_do_sample;
    prepared.sampling.sub_top_k = impl_->generation.sub_top_k;
    prepared.sampling.sub_top_p = impl_->generation.sub_top_p;
    prepared.sampling.sub_temperature = impl_->generation.sub_temperature;
    prepared.sampling.repetition_penalty = impl_->generation.repetition_penalty;
    prepared.sampling.rng_seed = request.seed > 0 ? request.seed : 1;
    return prepared;
}

Result<QwenTtsSynthesisResult> QwenTtsPipeline::synthesize(
    const QwenTtsPipelineRequest &request,
    CancellationToken *cancel
) {
    auto prepared = prepare_prompt(request);
    if(!prepared.is_ok()) {
        return prepared.get_error();
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(!is_loaded()) {
        return Error::model_not_loaded("QwenTtsPipeline is not loaded");
    }
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    const auto started = Clock::now();

    auto generated = impl_->generator.generate(
        prepared.value().prompt.language_sequence,
        static_cast<int32_t>(prepared.value().prompt.language_sequence_length),
        prepared.value().prompt.trailing_text_hidden,
        static_cast<int32_t>(prepared.value().prompt.trailing_text_length),
        prepared.value().prompt.tts_pad_embedding,
        prepared.value().sampling,
        cancel
    );
    if(!generated.is_ok()) {
        return generated.get_error();
    }

    const std::vector<int64_t> decode_codes = prefixed_codes(
        prepared.value().decoder_prefix_codes,
        generated.value().codes
    );
    const int32_t decode_frames = generated.value().frame_count + prepared.value().decode_prefix_frames;
    auto decoded = impl_->decoder.decode(decode_codes, decode_frames);
    if(!decoded.is_ok()) {
        return decoded.get_error();
    }

    QwenTtsSynthesisResult result;
    result.waveform = trim_waveform_prefix(
        decoded.value().waveform,
        prepared.value().decode_prefix_frames,
        decode_frames
    );
    result.sample_rate = impl_->config.sample_rate;
    result.frame_count = generated.value().frame_count;
    result.code_count = static_cast<int32_t>(generated.value().codes.size());
    result.visible_frame_count = generated.value().frame_count;
    result.visible_code_count = static_cast<int32_t>(generated.value().codes.size());
    result.elapsed_ms = Ms(Clock::now() - started).count();
    result.codegen_ms = generated.value().elapsed_ms;
    result.talker_prefill_ms = generated.value().talker_prefill_ms;
    result.talker_decode_ms = generated.value().talker_decode_ms;
    result.predictor_ms = generated.value().predictor_ms;
    result.onnx_embedding_ms = generated.value().onnx_embedding_ms;
    result.codegen_other_ms = generated.value().other_ms;
    result.decoder_ms = decoded.value().elapsed_ms;
    result.decoder_inference_ms = decoded.value().inference_ms;
    result.decoder_postprocess_ms = decoded.value().postprocess_ms;
    return result;
}

Result<QwenTtsSynthesisResult> QwenTtsPipeline::synthesize_streaming(
    const QwenTtsPipelineRequest &request,
    int32_t chunk_frames,
    QwenTtsStreamCallback on_chunk,
    CancellationToken *cancel
) {
    auto prepared = prepare_prompt(request);
    if(!prepared.is_ok()) {
        return prepared.get_error();
    }

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(!is_loaded()) {
        return Error::model_not_loaded("QwenTtsPipeline is not loaded");
    }
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    const auto started = Clock::now();

    auto decoder_stream = impl_->decoder.create_stream();
    if(!decoder_stream.is_ok()) {
        return decoder_stream.get_error();
    }

    std::vector<float> full_waveform;
    std::optional<Error> stream_error;
    int32_t chunk_index = 0;
    bool prefix_sent = false;
    auto callback = [&](TtsFrameChunk chunk) {
        if(cancel != nullptr && cancel->is_cancelled()) {
            return;
        }
        const bool should_prefix = !prefix_sent && !prepared.value().decoder_prefix_codes.empty();
        std::vector<int64_t> codes = should_prefix
            ? prefixed_codes(prepared.value().decoder_prefix_codes, chunk.codes)
            : std::move(chunk.codes);
        const int32_t decode_frame_count =
            chunk.frame_count + (should_prefix ? prepared.value().decode_prefix_frames : 0);
        prefix_sent = prefix_sent || should_prefix;
        auto decoded = decoder_stream.value()->decode(codes, decode_frame_count, chunk.is_final);
        if(!decoded.is_ok()) {
            stream_error = decoded.get_error();
            if(cancel != nullptr) {
                cancel->cancel();
            }
            return;
        }
        std::vector<float> waveform = trim_waveform_prefix(
            decoded.value().waveform,
            should_prefix ? prepared.value().decode_prefix_frames : 0,
            decode_frame_count
        );
        full_waveform.insert(full_waveform.end(), waveform.begin(), waveform.end());

        QwenTtsStreamChunk out;
        out.waveform = std::move(waveform);
        out.sample_rate = impl_->config.sample_rate;
        out.frame_count = chunk.frame_count;
        out.code_count = chunk.codes_per_frame * chunk.frame_count;
        out.visible_frame_count = chunk.frame_count;
        out.visible_code_count = out.code_count;
        out.chunk_index = ++chunk_index;
        out.is_final = chunk.is_final;
        out.elapsed_ms = Ms(Clock::now() - started).count();
        out.decoder_ms = decoded.value().elapsed_ms;
        out.decoder_inference_ms = decoded.value().inference_ms;
        out.decoder_postprocess_ms = decoded.value().postprocess_ms;
        out.decoder_provider_requested = decoded.value().provider_requested;
        out.decoder_provider_effective = decoded.value().provider_effective;
        out.decoder_cpu_fallback_node_count = decoded.value().cpu_fallback_node_count;
        out.decoder_fixed_shape = decoded.value().fixed_shape;
        on_chunk(out);
    };

    auto generated = impl_->generator.generate_streaming(
        prepared.value().prompt.language_sequence,
        static_cast<int32_t>(prepared.value().prompt.language_sequence_length),
        prepared.value().prompt.trailing_text_hidden,
        static_cast<int32_t>(prepared.value().prompt.trailing_text_length),
        prepared.value().prompt.tts_pad_embedding,
        prepared.value().sampling,
        std::max<int32_t>(1, chunk_frames),
        callback,
        cancel
    );
    if(!generated.is_ok()) {
        if(stream_error.has_value()) {
            return *stream_error;
        }
        return generated.get_error();
    }
    if(stream_error.has_value()) {
        return *stream_error;
    }

    QwenTtsSynthesisResult result;
    result.waveform = std::move(full_waveform);
    result.sample_rate = impl_->config.sample_rate;
    result.frame_count = generated.value().frame_count;
    result.code_count = static_cast<int32_t>(generated.value().codes.size());
    result.visible_frame_count = generated.value().frame_count;
    result.visible_code_count = static_cast<int32_t>(generated.value().codes.size());
    result.elapsed_ms = Ms(Clock::now() - started).count();
    result.codegen_ms = generated.value().elapsed_ms;
    result.talker_prefill_ms = generated.value().talker_prefill_ms;
    result.talker_decode_ms = generated.value().talker_decode_ms;
    result.predictor_ms = generated.value().predictor_ms;
    result.onnx_embedding_ms = generated.value().onnx_embedding_ms;
    result.codegen_other_ms = generated.value().other_ms;
    return result;
}

std::vector<std::string> QwenTtsPipeline::custom_voice_speaker_names() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->custom_voice_names;
}

} // namespace gotst
