#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gotst {

struct LanguageEntry {
    std::string name;
    int64_t codec_language_token_id = -1;
    std::string codec_prefix_mode;
};

class LanguageConfig {
public:
    LanguageConfig();

    std::vector<std::string> get_supported_tts_language_names() const;
    const std::map<std::string, LanguageEntry> &get_tts_language_map() const;
    int64_t get_tts_language_token_id(const std::string &language_name) const;
    bool has_tts_language(const std::string &language_name) const;

    static std::vector<int64_t> build_codec_prefix_tokens(
        int64_t language_token_id,
        int64_t think_token_id,
        int64_t nothink_token_id,
        int64_t think_bos_token_id,
        int64_t think_eos_token_id,
        int64_t pad_token_id,
        int64_t bos_token_id
    );

private:
    std::map<std::string, LanguageEntry> tts_languages_;
};

} // namespace gotst
