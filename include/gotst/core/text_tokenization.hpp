#pragma once

#include "gotst/core/result.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gotst {

struct IrodoriTokenizedText {
    std::vector<int64_t> token_ids;
    std::vector<uint8_t> token_mask;
};

std::string normalize_irodori_v3_text(std::string_view text);

class IrodoriTextTokenizer {
public:
    IrodoriTextTokenizer();
    ~IrodoriTextTokenizer();

    IrodoriTextTokenizer(const IrodoriTextTokenizer &) = delete;
    IrodoriTextTokenizer &operator=(const IrodoriTextTokenizer &) = delete;
    IrodoriTextTokenizer(IrodoriTextTokenizer &&) noexcept;
    IrodoriTextTokenizer &operator=(IrodoriTextTokenizer &&) noexcept;

    Result<void> load(const std::string &tokenizer_json_path, const std::string &tokenizer_config_path);
    bool is_loaded() const;

    Result<IrodoriTokenizedText> encode(
        const std::string &text,
        int32_t max_tokens,
        bool force_empty_mask = false
    ) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gotst
