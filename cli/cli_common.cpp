#include "cli_common.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string_view>

namespace gotst_cli {

namespace {

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

std::optional<int64_t> parse_int64(std::string_view value) {
    int64_t parsed = 0;
    const char *begin = value.data();
    const char *end = value.data() + value.size();
    auto result = std::from_chars(begin, end, parsed);
    if(result.ec != std::errc() || result.ptr != end) {
        return std::nullopt;
    }
    return parsed;
}

std::optional<double> parse_double_value(const std::string &value) {
    char *end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if(end == value.c_str() || end == nullptr || *end != '\0') {
        return std::nullopt;
    }
    return parsed;
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

void append_u16_le(std::vector<uint8_t> &bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value & 0xffu));
    bytes.push_back(static_cast<uint8_t>((value >> 8u) & 0xffu));
}

void append_u32_le(std::vector<uint8_t> &bytes, uint32_t value) {
    bytes.push_back(static_cast<uint8_t>(value & 0xffu));
    bytes.push_back(static_cast<uint8_t>((value >> 8u) & 0xffu));
    bytes.push_back(static_cast<uint8_t>((value >> 16u) & 0xffu));
    bytes.push_back(static_cast<uint8_t>((value >> 24u) & 0xffu));
}

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
        return 10 + (ch - 'a');
    }
    if(ch >= 'A' && ch <= 'F') {
        return 10 + (ch - 'A');
    }
    return -1;
}

class JsonParser {
public:
    JsonParser(std::string_view text, std::string source_name)
        : text_(text), source_name_(std::move(source_name)) {}

    gotst::Result<JsonValue> parse() {
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
    gotst::Result<JsonValue> parse_value() {
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

    gotst::Result<JsonValue> parse_literal(std::string_view literal, JsonValue value) {
        if(text_.substr(cursor_, literal.size()) != literal) {
            return fail("invalid JSON literal");
        }
        cursor_ += literal.size();
        return value;
    }

    gotst::Result<JsonValue> parse_string_value() {
        auto parsed = parse_string();
        if(!parsed.is_ok()) {
            return parsed.get_error();
        }
        JsonValue value;
        value.type = JsonValue::Type::String;
        value.string_value = std::move(parsed.value());
        return value;
    }

    gotst::Result<std::string> parse_string() {
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

    gotst::Result<JsonValue> parse_number() {
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

    gotst::Result<JsonValue> parse_array() {
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

    gotst::Result<JsonValue> parse_object() {
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

    gotst::Result<JsonValue> fail(const std::string &message) const {
        std::ostringstream stream;
        stream << source_name_ << ':' << cursor_ << ": " << message;
        return gotst::Error::invalid_argument(stream.str());
    }

    std::string_view text_;
    std::string source_name_;
    size_t cursor_ = 0;
};

} // namespace

gotst::Result<ParsedArgs> ParsedArgs::parse(int argc, char **argv, int start_index) {
    ParsedArgs args;
    for(int index = start_index; index < argc; ++index) {
        std::string token(argv[index]);
        if(!starts_with(token, "--")) {
            args.positionals_.push_back(std::move(token));
            continue;
        }

        token.erase(0, 2);
        if(token.empty()) {
            return gotst::Error::invalid_argument("empty option name");
        }

        const size_t equals = token.find('=');
        if(equals != std::string::npos) {
            args.values_[token.substr(0, equals)] = token.substr(equals + 1);
            continue;
        }

        if(index + 1 < argc && !starts_with(argv[index + 1], "--")) {
            args.values_[token] = argv[++index];
        } else {
            args.flags_.insert(token);
        }
    }
    return args;
}

bool ParsedArgs::has(const std::string &name) const {
    return values_.find(name) != values_.end() || flags_.find(name) != flags_.end();
}

std::string ParsedArgs::value(const std::string &name, const std::string &fallback) const {
    const auto found = values_.find(name);
    return found == values_.end() ? fallback : found->second;
}

bool ParsedArgs::bool_value(const std::string &name, bool fallback) const {
    if(flags_.find(name) != flags_.end()) {
        return true;
    }
    const auto found = values_.find(name);
    if(found == values_.end()) {
        return fallback;
    }
    const std::string normalized = lower_ascii(found->second);
    if(normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") {
        return true;
    }
    if(normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") {
        return false;
    }
    return fallback;
}

int32_t ParsedArgs::int_value(const std::string &name, int32_t fallback) const {
    const auto parsed = parse_int64(value(name));
    if(!parsed.has_value() || *parsed < std::numeric_limits<int32_t>::min() ||
       *parsed > std::numeric_limits<int32_t>::max()) {
        return fallback;
    }
    return static_cast<int32_t>(*parsed);
}

int64_t ParsedArgs::int64_value(const std::string &name, int64_t fallback) const {
    const auto parsed = parse_int64(value(name));
    return parsed.value_or(fallback);
}

float ParsedArgs::float_value(const std::string &name, float fallback) const {
    const auto parsed = parse_double_value(value(name));
    return parsed.has_value() ? static_cast<float>(*parsed) : fallback;
}

double ParsedArgs::double_value(const std::string &name, double fallback) const {
    const auto parsed = parse_double_value(value(name));
    return parsed.value_or(fallback);
}

const std::vector<std::string> &ParsedArgs::positionals() const {
    return positionals_;
}

const JsonValue *JsonValue::get(const std::string &key) const {
    if(type != Type::Object) {
        return nullptr;
    }
    const auto found = object_values.find(key);
    return found == object_values.end() ? nullptr : &found->second;
}

bool JsonValue::is_object() const {
    return type == Type::Object;
}

bool JsonValue::is_array() const {
    return type == Type::Array;
}

std::string JsonValue::as_string(const std::string &fallback) const {
    if(type == Type::String) {
        return string_value;
    }
    return fallback;
}

int32_t JsonValue::as_int(int32_t fallback) const {
    if(type != Type::Number || !std::isfinite(number_value)) {
        return fallback;
    }
    return static_cast<int32_t>(number_value);
}

int64_t JsonValue::as_int64(int64_t fallback) const {
    if(type != Type::Number || !std::isfinite(number_value)) {
        return fallback;
    }
    return static_cast<int64_t>(number_value);
}

float JsonValue::as_float(float fallback) const {
    if(type != Type::Number || !std::isfinite(number_value)) {
        return fallback;
    }
    return static_cast<float>(number_value);
}

double JsonValue::as_double(double fallback) const {
    if(type != Type::Number || !std::isfinite(number_value)) {
        return fallback;
    }
    return number_value;
}

bool JsonValue::as_bool(bool fallback) const {
    if(type == Type::Bool) {
        return bool_value;
    }
    return fallback;
}

gotst::Result<std::string> read_text_file(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if(!input) {
        return gotst::Error::not_found("failed to open file: " + path.string());
    }
    const std::streamoff size = input.tellg();
    if(size < 0) {
        return gotst::Error::io_error("failed to inspect file size: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::string text(static_cast<size_t>(size), '\0');
    if(!text.empty() && !input.read(text.data(), size)) {
        return gotst::Error::io_error("failed to read file: " + path.string());
    }
    return text;
}

gotst::Result<JsonValue> parse_json_text(const std::string &text, const std::string &source_name) {
    JsonParser parser(text, source_name);
    return parser.parse();
}

gotst::Result<JsonValue> read_json_file(const std::filesystem::path &path) {
    auto text = read_text_file(path);
    if(!text.is_ok()) {
        return text.get_error();
    }
    return parse_json_text(text.value(), path.string());
}

gotst::Result<void> write_wav_mono_f32(
    const std::filesystem::path &path,
    const std::vector<float> &samples,
    int32_t sample_rate
) {
    if(sample_rate <= 0) {
        return gotst::Error::invalid_argument("WAV sample rate must be positive");
    }
    if(samples.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max() / 2u)) {
        return gotst::Error::invalid_argument("WAV output is too large");
    }
    const std::filesystem::path parent = path.parent_path();
    if(!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if(ec) {
            return gotst::Error::io_error("failed to create output directory: " + parent.string() + ": " + ec.message());
        }
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * channels * (bits_per_sample / 8u);
    const uint16_t block_align = channels * (bits_per_sample / 8u);

    std::vector<uint8_t> header;
    header.reserve(44);
    header.insert(header.end(), {'R', 'I', 'F', 'F'});
    append_u32_le(header, 36u + data_bytes);
    header.insert(header.end(), {'W', 'A', 'V', 'E'});
    header.insert(header.end(), {'f', 'm', 't', ' '});
    append_u32_le(header, 16u);
    append_u16_le(header, 1u);
    append_u16_le(header, channels);
    append_u32_le(header, static_cast<uint32_t>(sample_rate));
    append_u32_le(header, byte_rate);
    append_u16_le(header, block_align);
    append_u16_le(header, bits_per_sample);
    header.insert(header.end(), {'d', 'a', 't', 'a'});
    append_u32_le(header, data_bytes);

    std::ofstream output(path, std::ios::binary);
    if(!output) {
        return gotst::Error::io_error("failed to open output WAV: " + path.string());
    }
    output.write(reinterpret_cast<const char *>(header.data()), static_cast<std::streamsize>(header.size()));
    for(float sample : samples) {
        const float finite = std::isfinite(static_cast<double>(sample)) ? sample : 0.0f;
        const float clamped = std::clamp(finite, -1.0f, 1.0f);
        const auto pcm = static_cast<int16_t>(std::lrint(clamped * 32767.0f));
        uint8_t bytes[2] = {
            static_cast<uint8_t>(pcm & 0xff),
            static_cast<uint8_t>((static_cast<uint16_t>(pcm) >> 8u) & 0xffu),
        };
        output.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
    }
    if(!output) {
        return gotst::Error::io_error("failed to write output WAV: " + path.string());
    }
    return gotst::Result<void>();
}

std::filesystem::path resolve_path(const std::filesystem::path &root, const std::string &path) {
    if(path.empty()) {
        return {};
    }
    std::filesystem::path fs_path(path);
    if(fs_path.is_absolute() || root.empty()) {
        return fs_path.lexically_normal();
    }
    return (root / fs_path).lexically_normal();
}

std::string json_string(const JsonValue *object, const std::string &key, const std::string &fallback) {
    return object == nullptr ? fallback : object->get(key) == nullptr ? fallback : object->get(key)->as_string(fallback);
}

int32_t json_int(const JsonValue *object, const std::string &key, int32_t fallback) {
    return object == nullptr ? fallback : object->get(key) == nullptr ? fallback : object->get(key)->as_int(fallback);
}

float json_float(const JsonValue *object, const std::string &key, float fallback) {
    return object == nullptr ? fallback : object->get(key) == nullptr ? fallback : object->get(key)->as_float(fallback);
}

bool json_bool(const JsonValue *object, const std::string &key, bool fallback) {
    return object == nullptr ? fallback : object->get(key) == nullptr ? fallback : object->get(key)->as_bool(fallback);
}

void print_global_help() {
    std::cout
        << "gotst command-line tool\n\n"
        << "Usage:\n"
        << "  gotst <command> [options]\n\n"
        << "Commands:\n"
        << "  inspect       Inspect an Irodori manifest and resolved artifact paths\n"
        << "  irodori-tts   Synthesize text with an Irodori-TTS bundle\n"
        << "  qwen-tts      Synthesize text with a Qwen3-TTS bundle\n\n"
        << "Run gotst <command> --help for command-specific options.\n";
}

int print_error(const gotst::Error &error) {
    std::cerr << "error: " << error.message << '\n';
    return error.code == gotst::ErrorCode::InvalidArgument ? 2 : 1;
}

int print_error(const std::string &message, int exit_code) {
    std::cerr << "error: " << message << '\n';
    return exit_code;
}

} // namespace gotst_cli
