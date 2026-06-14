#pragma once

#include "gotst/core/result.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace gotst_cli {

class ParsedArgs {
public:
    static gotst::Result<ParsedArgs> parse(int argc, char **argv, int start_index);

    [[nodiscard]] bool has(const std::string &name) const;
    [[nodiscard]] std::string value(const std::string &name, const std::string &fallback = "") const;
    [[nodiscard]] bool bool_value(const std::string &name, bool fallback = false) const;
    [[nodiscard]] int32_t int_value(const std::string &name, int32_t fallback = 0) const;
    [[nodiscard]] int64_t int64_value(const std::string &name, int64_t fallback = 0) const;
    [[nodiscard]] float float_value(const std::string &name, float fallback = 0.0f) const;
    [[nodiscard]] double double_value(const std::string &name, double fallback = 0.0) const;
    [[nodiscard]] const std::vector<std::string> &positionals() const;

private:
    std::map<std::string, std::string> values_;
    std::set<std::string> flags_;
    std::vector<std::string> positionals_;
};

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

    [[nodiscard]] const JsonValue *get(const std::string &key) const;
    [[nodiscard]] bool is_object() const;
    [[nodiscard]] bool is_array() const;
    [[nodiscard]] std::string as_string(const std::string &fallback = "") const;
    [[nodiscard]] int32_t as_int(int32_t fallback = 0) const;
    [[nodiscard]] int64_t as_int64(int64_t fallback = 0) const;
    [[nodiscard]] float as_float(float fallback = 0.0f) const;
    [[nodiscard]] double as_double(double fallback = 0.0) const;
    [[nodiscard]] bool as_bool(bool fallback = false) const;
};

gotst::Result<std::string> read_text_file(const std::filesystem::path &path);
gotst::Result<JsonValue> parse_json_text(const std::string &text, const std::string &source_name);
gotst::Result<JsonValue> read_json_file(const std::filesystem::path &path);
gotst::Result<void> write_wav_mono_f32(
    const std::filesystem::path &path,
    const std::vector<float> &samples,
    int32_t sample_rate
);

std::filesystem::path resolve_path(const std::filesystem::path &root, const std::string &path);
std::string json_string(const JsonValue *object, const std::string &key, const std::string &fallback = "");
int32_t json_int(const JsonValue *object, const std::string &key, int32_t fallback = 0);
float json_float(const JsonValue *object, const std::string &key, float fallback = 0.0f);
bool json_bool(const JsonValue *object, const std::string &key, bool fallback = false);

void print_global_help();
int print_error(const gotst::Error &error);
int print_error(const std::string &message, int exit_code = 1);

int command_inspect(const ParsedArgs &args);
int command_irodori_tts(const ParsedArgs &args);
int command_qwen_tts(const ParsedArgs &args);

} // namespace gotst_cli
