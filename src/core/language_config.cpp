#include "gotst/core/language_config.hpp"

namespace gotst {

LanguageConfig::LanguageConfig() {
    tts_languages_ = {
        {"chinese",     {"Chinese",     2055, "think"}},
        {"english",     {"English",     2050, "think"}},
        {"french",      {"French",      2061, "think"}},
        {"german",      {"German",      2053, "think"}},
        {"italian",     {"Italian",     2070, "think"}},
        {"japanese",    {"Japanese",    2058, "think"}},
        {"korean",      {"Korean",      2064, "think"}},
        {"portuguese",  {"Portuguese",  2071, "think"}},
        {"russian",     {"Russian",     2069, "think"}},
        {"spanish",     {"Spanish",     2054, "think"}},
    };
}

std::vector<std::string> LanguageConfig::get_supported_tts_language_names() const {
    std::vector<std::string> names;
    names.reserve(tts_languages_.size());
    for (const auto &pair : tts_languages_) {
        names.push_back(pair.second.name);
    }
    return names;
}

const std::map<std::string, LanguageEntry> &LanguageConfig::get_tts_language_map() const {
    return tts_languages_;
}

int64_t LanguageConfig::get_tts_language_token_id(const std::string &language_name) const {
    auto it = tts_languages_.find(language_name);
    if (it != tts_languages_.end()) {
        return it->second.codec_language_token_id;
    }
    return -1;
}

bool LanguageConfig::has_tts_language(const std::string &language_name) const {
    return tts_languages_.find(language_name) != tts_languages_.end();
}

std::vector<int64_t> LanguageConfig::build_codec_prefix_tokens(
    int64_t language_token_id,
    int64_t think_token_id,
    int64_t nothink_token_id,
    int64_t think_bos_token_id,
    int64_t think_eos_token_id,
    int64_t pad_token_id,
    int64_t bos_token_id
) {
    if (language_token_id >= 0) {
        return {
            think_token_id,
            think_bos_token_id,
            language_token_id,
            think_eos_token_id,
            pad_token_id,
            bos_token_id
        };
    }
    return {
        nothink_token_id,
        think_bos_token_id,
        think_eos_token_id,
        pad_token_id,
        bos_token_id
    };
}

} // namespace gotst
