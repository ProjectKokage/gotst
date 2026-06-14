#include "gotst/core/text_tokenization.hpp"

#include <tokenizers_cpp.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace gotst {
namespace {

struct TokenizerConfig {
    bool add_bos = true;
    std::string bos_token = "<s>";
};

Result<std::string> read_text_file(const std::string &path) {
    if(path.empty()) {
        return Error::invalid_argument("tokenizer path is empty");
    }
    std::ifstream input(path, std::ios::binary);
    if(!input) {
        return Error::not_found("tokenizer file not found: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if(!input.good() && !input.eof()) {
        return Error::io_error("failed to read tokenizer file: " + path);
    }
    return buffer.str();
}

bool contains_json_true(const std::string &json, const std::string &key, bool fallback) {
    const std::string quoted_key = "\"" + key + "\"";
    const size_t key_pos = json.find(quoted_key);
    if(key_pos == std::string::npos) {
        return fallback;
    }
    const size_t colon_pos = json.find(':', key_pos + quoted_key.size());
    if(colon_pos == std::string::npos) {
        return fallback;
    }
    size_t value_pos = colon_pos + 1;
    while(value_pos < json.size() && std::isspace(static_cast<unsigned char>(json[value_pos]))) {
        ++value_pos;
    }
    if(json.compare(value_pos, 4, "true") == 0) {
        return true;
    }
    if(json.compare(value_pos, 5, "false") == 0) {
        return false;
    }
    return fallback;
}

std::string unescape_json_string(std::string_view value) {
    std::string output;
    output.reserve(value.size());
    for(size_t index = 0; index < value.size(); ++index) {
        const char ch = value[index];
        if(ch != '\\' || index + 1 >= value.size()) {
            output.push_back(ch);
            continue;
        }
        const char escaped = value[++index];
        switch(escaped) {
            case '"':
            case '\\':
            case '/':
                output.push_back(escaped);
                break;
            case 'b':
                output.push_back('\b');
                break;
            case 'f':
                output.push_back('\f');
                break;
            case 'n':
                output.push_back('\n');
                break;
            case 'r':
                output.push_back('\r');
                break;
            case 't':
                output.push_back('\t');
                break;
            default:
                output.push_back(escaped);
                break;
        }
    }
    return output;
}

std::string json_string_value(const std::string &json, const std::string &key, const std::string &fallback) {
    const std::string quoted_key = "\"" + key + "\"";
    const size_t key_pos = json.find(quoted_key);
    if(key_pos == std::string::npos) {
        return fallback;
    }
    const size_t colon_pos = json.find(':', key_pos + quoted_key.size());
    if(colon_pos == std::string::npos) {
        return fallback;
    }
    size_t value_pos = colon_pos + 1;
    while(value_pos < json.size() && std::isspace(static_cast<unsigned char>(json[value_pos]))) {
        ++value_pos;
    }
    if(value_pos >= json.size() || json[value_pos] != '"') {
        return fallback;
    }
    ++value_pos;
    size_t end_pos = value_pos;
    bool escaped = false;
    while(end_pos < json.size()) {
        const char ch = json[end_pos];
        if(!escaped && ch == '"') {
            return unescape_json_string(std::string_view(json).substr(value_pos, end_pos - value_pos));
        }
        escaped = !escaped && ch == '\\';
        if(ch != '\\') {
            escaped = false;
        }
        ++end_pos;
    }
    return fallback;
}

TokenizerConfig read_tokenizer_config(const std::string &path) {
    TokenizerConfig config;
    if(path.empty() || !std::filesystem::exists(std::filesystem::path(path))) {
        return config;
    }
    auto contents = read_text_file(path);
    if(!contents.is_ok()) {
        return config;
    }
    config.add_bos = contains_json_true(contents.value(), "add_bos_token", config.add_bos);
    config.bos_token = json_string_value(contents.value(), "bos_token", config.bos_token);
    return config;
}

std::vector<char32_t> decode_utf8(std::string_view input) {
    std::vector<char32_t> output;
    output.reserve(input.size());
    for(size_t index = 0; index < input.size();) {
        const unsigned char first = static_cast<unsigned char>(input[index]);
        if(first < 0x80) {
            output.push_back(first);
            ++index;
            continue;
        }
        char32_t codepoint = 0;
        size_t length = 0;
        if((first & 0xE0) == 0xC0) {
            codepoint = first & 0x1F;
            length = 2;
        } else if((first & 0xF0) == 0xE0) {
            codepoint = first & 0x0F;
            length = 3;
        } else if((first & 0xF8) == 0xF0) {
            codepoint = first & 0x07;
            length = 4;
        } else {
            output.push_back(0xFFFD);
            ++index;
            continue;
        }
        if(index + length > input.size()) {
            output.push_back(0xFFFD);
            break;
        }
        bool valid = true;
        for(size_t offset = 1; offset < length; ++offset) {
            const unsigned char ch = static_cast<unsigned char>(input[index + offset]);
            if((ch & 0xC0) != 0x80) {
                valid = false;
                break;
            }
            codepoint = (codepoint << 6) | (ch & 0x3F);
        }
        if(!valid) {
            output.push_back(0xFFFD);
            ++index;
            continue;
        }
        output.push_back(codepoint);
        index += length;
    }
    return output;
}

void append_utf8(std::string &output, char32_t codepoint) {
    if(codepoint <= 0x7F) {
        output.push_back(static_cast<char>(codepoint));
    } else if(codepoint <= 0x7FF) {
        output.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if(codepoint <= 0xFFFF) {
        output.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        output.push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

std::string encode_utf8(const std::vector<char32_t> &codepoints) {
    std::string output;
    output.reserve(codepoints.size());
    for(char32_t codepoint : codepoints) {
        append_utf8(output, codepoint);
    }
    return output;
}

bool is_removed_irodori_char(char32_t ch) {
    switch(ch) {
        case U';':
        case U'▼':
        case U'♀':
        case U'♂':
        case U'《':
        case U'》':
        case U'≪':
        case U'≫':
        case U'\uE63E':
        case U'①':
        case U'②':
        case U'③':
        case U'④':
        case U'⑤':
        case U'⑥':
            return true;
        default:
            return false;
    }
}

bool is_removed_dash(char32_t ch) {
    return ch == U'\u02D7' ||
        (ch >= U'\u2010' && ch <= U'\u2015') ||
        ch == U'\u2043' ||
        ch == U'\u2212' ||
        ch == U'\u23AF' ||
        ch == U'\u23E4' ||
        ch == U'\u2500' ||
        ch == U'\u2501' ||
        ch == U'\u2E3A' ||
        ch == U'\u2E3B';
}

bool is_trim_space(char32_t ch) {
    return ch == U' ' || ch == U'\n' || ch == U'\r' || ch == U'\f' || ch == U'\v';
}

std::vector<char32_t> trim_spaces(std::vector<char32_t> input) {
    size_t begin = 0;
    size_t end = input.size();
    while(begin < end && is_trim_space(input[begin])) {
        ++begin;
    }
    while(end > begin && is_trim_space(input[end - 1])) {
        --end;
    }
    if(begin == 0 && end == input.size()) {
        return input;
    }
    return std::vector<char32_t>(input.begin() + static_cast<std::ptrdiff_t>(begin),
                                 input.begin() + static_cast<std::ptrdiff_t>(end));
}

bool is_outer_pair(char32_t left, char32_t right) {
    return (left == U'「' && right == U'」') ||
        (left == U'『' && right == U'』') ||
        (left == U'（' && right == U'）') ||
        (left == U'【' && right == U'】') ||
        (left == U'(' && right == U')');
}

std::vector<char32_t> strip_outer_brackets(std::vector<char32_t> input) {
    input = trim_spaces(std::move(input));
    bool changed = true;
    while(changed && input.size() >= 2) {
        changed = false;
        if(is_outer_pair(input.front(), input.back())) {
            input.erase(input.begin());
            input.pop_back();
            input = trim_spaces(std::move(input));
            changed = true;
        }
    }
    return input;
}

char32_t normalize_compatibility_char(char32_t ch) {
    if(ch >= U'\uFF01' && ch <= U'\uFF5E') {
        return ch - U'\uFEE0';
    }
    switch(ch) {
        case U'\u3000':
            return U' ';
        case U'\u2215':
            return U'/';
        case U'\u29F8':
            return U'/';
        default:
            return ch;
    }
}

std::vector<char32_t> collapse_ellipses(const std::vector<char32_t> &input) {
    std::vector<char32_t> output;
    output.reserve(input.size());
    for(size_t index = 0; index < input.size();) {
        if(input[index] != U'…') {
            output.push_back(input[index]);
            ++index;
            continue;
        }
        size_t end = index;
        while(end < input.size() && input[end] == U'…') {
            ++end;
        }
        const size_t count = end - index;
        output.push_back(U'…');
        if(count >= 3) {
            output.push_back(U'…');
        } else if(count == 2) {
            output.push_back(U'…');
        }
        index = end;
    }
    return output;
}

std::vector<char32_t> replace_ascii_dot_runs(const std::vector<char32_t> &input) {
    std::vector<char32_t> output;
    output.reserve(input.size());
    for(size_t index = 0; index < input.size();) {
        if(input[index] != U'.') {
            output.push_back(input[index]);
            ++index;
            continue;
        }
        size_t end = index;
        while(end < input.size() && input[end] == U'.') {
            ++end;
        }
        const size_t count = end - index;
        if(count >= 2) {
            output.push_back(U'…');
            if(count > 3) {
                output.push_back(U'…');
            }
        } else {
            output.push_back(U'.');
        }
        index = end;
    }
    return output;
}

void erase_all(std::string &value, const std::string &needle) {
    if(needle.empty()) {
        return;
    }
    size_t pos = value.find(needle);
    while(pos != std::string::npos) {
        value.erase(pos, needle.size());
        pos = value.find(needle, pos);
    }
}

} // namespace

struct IrodoriTextTokenizer::Impl {
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    TokenizerConfig config;
};

IrodoriTextTokenizer::IrodoriTextTokenizer() : impl_(std::make_unique<Impl>()) {}
IrodoriTextTokenizer::~IrodoriTextTokenizer() = default;
IrodoriTextTokenizer::IrodoriTextTokenizer(IrodoriTextTokenizer &&) noexcept = default;
IrodoriTextTokenizer &IrodoriTextTokenizer::operator=(IrodoriTextTokenizer &&) noexcept = default;

Result<void> IrodoriTextTokenizer::load(
    const std::string &tokenizer_json_path,
    const std::string &tokenizer_config_path
) {
    auto blob = read_text_file(tokenizer_json_path);
    if(!blob.is_ok()) {
        return blob.get_error();
    }
    std::unique_ptr<tokenizers::Tokenizer> tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob.value());
    if(!tokenizer) {
        return Error::invalid_argument("failed to create tokenizer from " + tokenizer_json_path);
    }
    impl_->tokenizer = std::move(tokenizer);
    impl_->config = read_tokenizer_config(tokenizer_config_path);
    return Result<void>();
}

bool IrodoriTextTokenizer::is_loaded() const {
    return impl_ && impl_->tokenizer != nullptr;
}

Result<IrodoriTokenizedText> IrodoriTextTokenizer::encode(
    const std::string &text,
    int32_t max_tokens,
    bool force_empty_mask
) const {
    if(!is_loaded()) {
        return Error::model_not_loaded("Irodori tokenizer is not loaded");
    }
    if(max_tokens < 0) {
        return Error::invalid_argument("max_tokens must be non-negative");
    }

    std::vector<int32_t> ids;
    if(!force_empty_mask) {
        ids = impl_->tokenizer->Encode(text);
    }

    const int32_t bos_id = impl_->config.add_bos ? impl_->tokenizer->TokenToId(impl_->config.bos_token) : -1;
    if(impl_->config.add_bos && bos_id >= 0 && (ids.empty() || ids.front() != bos_id)) {
        ids.insert(ids.begin(), bos_id);
    }

    const size_t limit = max_tokens > 0 ? std::min(ids.size(), static_cast<size_t>(max_tokens)) : ids.size();
    IrodoriTokenizedText output;
    output.token_ids.reserve(limit);
    output.token_mask.reserve(limit);
    for(size_t index = 0; index < limit; ++index) {
        output.token_ids.push_back(static_cast<int64_t>(ids[index]));
        output.token_mask.push_back(force_empty_mask ? uint8_t{0} : uint8_t{1});
    }
    return output;
}

std::string normalize_irodori_v3_text(std::string_view text) {
    std::string preprocessed(text);
    erase_all(preprocessed, "[n]");
    erase_all(preprocessed, "\\[n\\]");

    std::vector<char32_t> filtered;
    filtered.reserve(preprocessed.size());
    for(char32_t ch : decode_utf8(preprocessed)) {
        if(ch == U'\t' || ch == U'\u3000') {
            continue;
        }
        if(is_removed_irodori_char(ch) || is_removed_dash(ch)) {
            continue;
        }
        if(ch == U'？') {
            filtered.push_back(U'?');
        } else if(ch == U'！') {
            filtered.push_back(U'!');
        } else if(ch == U'♥') {
            filtered.push_back(U'♡');
        } else if(ch == U'●' || ch == U'◯' || ch == U'〇') {
            filtered.push_back(U'○');
        } else if(ch == U'\uFF5E' || ch == U'\u301C') {
            filtered.push_back(U'ー');
        } else {
            filtered.push_back(ch);
        }
    }

    std::vector<char32_t> bracket_stripped = strip_outer_brackets(collapse_ellipses(filtered));
    std::vector<char32_t> compatible;
    compatible.reserve(bracket_stripped.size());
    for(char32_t ch : bracket_stripped) {
        compatible.push_back(normalize_compatibility_char(ch));
    }
    return encode_utf8(trim_spaces(replace_ascii_dot_runs(compatible)));
}

} // namespace gotst
