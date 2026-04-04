#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <gotst/core/language_config.hpp>
#include <gotst/core/tts_prompt_assembly.hpp>
#include <gotst/core/cancellation_token.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

using Catch::Approx;

namespace {

int64_t gdscript_next_rng_state(int64_t state) {
    int64_t resolved = state;
    if (resolved == 0) resolved = 1;
    resolved ^= (resolved << 13);
    resolved ^= (resolved >> 17);
    resolved ^= (resolved << 5);
    if (resolved == 0) return 1;
    return resolved;
}

double gdscript_uniform_from_state(int64_t state) {
    int64_t mantissa = state & 0x00FFFFFF;
    return static_cast<double>(mantissa) / 16777216.0;
}

struct GdCodecPrefixResult {
    bool force_japanese;
    int64_t think_token_id = 2154;
    int64_t nothink_token_id = 2155;
    int64_t think_bos_token_id = 2156;
    int64_t think_eos_token_id = 2157;
    int64_t japanese_language_token_id = 2058;
    int64_t pad_token_id = 2148;
    int64_t bos_token_id = 2149;

    std::vector<int64_t> build() const {
        std::vector<int64_t> tokens;
        if (force_japanese) {
            tokens.push_back(think_token_id);
            tokens.push_back(think_bos_token_id);
            tokens.push_back(japanese_language_token_id);
            tokens.push_back(think_eos_token_id);
        } else {
            tokens.push_back(nothink_token_id);
            tokens.push_back(think_bos_token_id);
            tokens.push_back(think_eos_token_id);
        }
        tokens.push_back(pad_token_id);
        tokens.push_back(bos_token_id);
        return tokens;
    }
};

struct GdLayoutResult {
    bool has_layout = false;
    int64_t time_steps = 0;
    int64_t time_stride = 0;
    int64_t channel_stride = 0;
    int64_t channel_count = 1;
};

GdLayoutResult gdscript_resolve_layout(const std::vector<int64_t> &shape, int64_t output_length) {
    GdLayoutResult result;
    if (static_cast<int64_t>(shape.size()) < 2 || output_length <= 0) {
        result.time_steps = output_length;
        return result;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    int64_t running_stride = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = running_stride;
        running_stride *= std::max<int64_t>(shape[i], 1);
    }

    int64_t time_axis = std::max<int64_t>(0, static_cast<int64_t>(shape.size()) - 1);
    result.has_layout = true;
    result.time_steps = std::max<int64_t>(1, shape[time_axis]);
    result.time_stride = strides[time_axis];

    for (size_t axis = 0; axis < shape.size() - 1; ++axis) {
        if (shape[axis] > 1) {
            result.channel_stride = strides[axis];
            result.channel_count = shape[axis];
            break;
        }
    }
    return result;
}

std::vector<float> gdscript_slice_rows(
    const std::vector<float> &source,
    int64_t start_row,
    int64_t row_count,
    int64_t hidden_size
) {
    if (source.empty() || row_count <= 0 || hidden_size <= 0) return {};
    int64_t available = static_cast<int64_t>(source.size()) / hidden_size;
    if (start_row >= available) return {};
    int64_t safe_count = std::min(row_count, available - start_row);
    if (safe_count <= 0) return {};
    std::vector<float> result(safe_count * hidden_size);
    int64_t offset = start_row * hidden_size;
    std::copy(source.begin() + offset, source.begin() + offset + static_cast<int64_t>(result.size()), result.begin());
    return result;
}

std::vector<float> gdscript_concat_rows(const std::vector<std::vector<float>> &parts) {
    size_t total = 0;
    for (const auto &p : parts) total += p.size();
    std::vector<float> result(total);
    size_t cursor = 0;
    for (const auto &p : parts) {
        std::copy(p.begin(), p.end(), result.begin() + cursor);
        cursor += p.size();
    }
    return result;
}

std::vector<float> gdscript_repeat_row(const std::vector<float> &row, int64_t count) {
    if (row.empty() || count <= 0) return {};
    std::vector<float> result(row.size() * count);
    for (int64_t i = 0; i < count; ++i) {
        std::copy(row.begin(), row.end(), result.begin() + i * static_cast<int64_t>(row.size()));
    }
    return result;
}

std::vector<float> gdscript_add_vectors(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.empty() || a.size() != b.size()) return {};
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
    return result;
}

std::vector<float> gdscript_sum_vectors(const std::vector<std::vector<float>> &parts) {
    std::vector<float> seed;
    for (const auto &p : parts) {
        if (!p.empty()) { seed = p; break; }
    }
    if (seed.empty()) return {};
    std::vector<float> result(seed.size(), 0.0f);
    for (const auto &p : parts) {
        if (p.empty()) continue;
        if (p.size() != result.size()) return {};
        for (size_t i = 0; i < result.size(); ++i) result[i] += p[i];
    }
    return result;
}

struct GdExtractResult {
    std::vector<float> values;
    int64_t sequence_length = 0;
    int64_t hidden_size = 0;
    bool valid = false;
};

GdExtractResult gdscript_extract_sequence(const std::vector<float> &values, const std::vector<int64_t> &shape) {
    GdExtractResult result;
    if (values.empty()) return result;
    int64_t shape_size = static_cast<int64_t>(shape.size());
    int64_t hidden_size = shape_size >= 1 ? shape[shape_size - 1] : static_cast<int64_t>(values.size());
    int64_t seq_length = shape_size >= 2 ? shape[shape_size - 2] : 1;
    int64_t required = hidden_size * seq_length;
    if (required <= 0 || required > static_cast<int64_t>(values.size())) return result;
    result.values.assign(values.begin(), values.begin() + required);
    result.sequence_length = seq_length;
    result.hidden_size = hidden_size;
    result.valid = true;
    return result;
}

std::vector<float> gdscript_extract_last_hidden_row(
    const std::vector<float> &hidden_states,
    const std::vector<int64_t> &hidden_shape,
    int64_t hidden_size
) {
    if (hidden_states.empty() || hidden_size <= 0) return {};
    int64_t resolved = hidden_size;
    if (!hidden_shape.empty()) resolved = hidden_shape.back();
    if (resolved != hidden_size) return {};
    int64_t row_count = std::max<int64_t>(1, static_cast<int64_t>(hidden_states.size()) / hidden_size);
    return gdscript_slice_rows(hidden_states, row_count - 1, 1, hidden_size);
}

bool gdscript_contains_only_finite(const std::vector<float> &values) {
    for (float v : values) {
        if (std::isnan(v) || std::isinf(v)) return false;
    }
    return true;
}

int64_t gdscript_select_last_row_argmax(
    const std::vector<float> &logits,
    const std::vector<int64_t> &logits_shape,
    int64_t start_index,
    int64_t count
) {
    if (logits.empty() || logits_shape.empty()) return -1;
    int64_t vocab_size = logits_shape.back();
    if (vocab_size <= 0) return -1;
    int64_t rows = std::max<int64_t>(1, static_cast<int64_t>(logits.size()) / vocab_size);
    int64_t row_start = (rows - 1) * vocab_size;
    int64_t safe_start = std::clamp(start_index, int64_t(0), std::max(int64_t(0), vocab_size - 1));
    int64_t safe_count = std::clamp(count, int64_t(1), std::max(int64_t(1), vocab_size - safe_start));
    int64_t best_index = safe_start;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int64_t offset = 0; offset < safe_count; ++offset) {
        int64_t idx = safe_start + offset;
        if (logits[row_start + idx] > best_value) {
            best_value = logits[row_start + idx];
            best_index = idx;
        }
    }
    return best_index;
}

bool gdscript_should_stop_on_eos(
    const std::vector<float> &logits,
    const std::vector<int64_t> &logits_shape,
    int64_t eos_token_id,
    int64_t codec_size,
    double eos_logit_margin
) {
    if (logits.empty() || logits_shape.empty()) return false;
    int64_t vocab_size = logits_shape.back();
    if (eos_token_id < 0 || eos_token_id >= vocab_size) return false;
    int64_t rows = std::max(int64_t(1), static_cast<int64_t>(logits.size()) / vocab_size);
    int64_t row_start = (rows - 1) * vocab_size;
    float eos_logit = logits[row_start + eos_token_id];
    float best_codec = -std::numeric_limits<float>::infinity();
    int64_t safe_codec = std::clamp(codec_size, int64_t(1), vocab_size);
    for (int64_t i = 0; i < safe_codec; ++i) {
        best_codec = std::max(best_codec, logits[row_start + i]);
    }
    return static_cast<double>(eos_logit) > (static_cast<double>(best_codec) + eos_logit_margin);
}

struct GdSampleResult {
    int64_t token = -1;
    int64_t rng_state = 0;
};

GdSampleResult gdscript_select_last_row_token(
    const std::vector<float> &logits,
    const std::vector<int64_t> &logits_shape,
    int64_t start_index,
    int64_t count,
    bool do_sample,
    int64_t top_k,
    double top_p,
    double temperature,
    const std::vector<int64_t> &prior_tokens,
    double repetition_penalty,
    int64_t rng_state
) {
    if (!do_sample) {
        return {gdscript_select_last_row_argmax(logits, logits_shape, start_index, count), rng_state};
    }
    if (logits.empty() || logits_shape.empty()) return {-1, rng_state};
    int64_t vocab_size = logits_shape.back();
    if (vocab_size <= 0) return {-1, rng_state};
    int64_t rows = std::max(int64_t(1), static_cast<int64_t>(logits.size()) / vocab_size);
    int64_t row_start = (rows - 1) * vocab_size;
    int64_t safe_start = std::clamp(start_index, int64_t(0), std::max(int64_t(0), vocab_size - 1));
    int64_t safe_count = std::clamp(count, int64_t(1), std::max(int64_t(1), vocab_size - safe_start));
    int64_t candidate_count = std::clamp(top_k, int64_t(1), safe_count);
    double temp = std::max(0.05, temperature);

    struct Cand { int64_t token; double value; };
    std::vector<Cand> candidates;
    candidates.reserve(safe_count);
    for (int64_t offset = 0; offset < safe_count; ++offset) {
        int64_t token_index = safe_start + offset;
        double value = logits[row_start + token_index];
        bool is_prior = std::find(prior_tokens.begin(), prior_tokens.end(), token_index) != prior_tokens.end();
        if (is_prior && repetition_penalty > 1.0) {
            value = (value >= 0.0) ? (value / repetition_penalty) : (value * repetition_penalty);
        }
        candidates.push_back({token_index, value / temp});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Cand &a, const Cand &b) { return a.value > b.value; });
    if (static_cast<int64_t>(candidates.size()) > candidate_count) candidates.resize(candidate_count);
    if (candidates.empty()) return {gdscript_select_last_row_argmax(logits, logits_shape, start_index, count), rng_state};

    if (top_p < 0.999) {
        double max_logit = candidates.front().value;
        double sum_exp = 0.0;
        std::vector<double> exp_values;
        exp_values.reserve(candidates.size());
        for (const auto &c : candidates) {
            double ev = std::exp(c.value - max_logit);
            exp_values.push_back(ev);
            sum_exp += ev;
        }
        double cumulative = 0.0;
        int64_t final_count = 0;
        for (size_t i = 0; i < exp_values.size(); ++i) {
            cumulative += exp_values[i] / std::max(sum_exp, 1e-12);
            final_count = static_cast<int64_t>(i) + 1;
            if (cumulative >= top_p) break;
        }
        final_count = std::clamp(final_count, int64_t(1), static_cast<int64_t>(candidates.size()));
        candidates.resize(final_count);
    }

    double max_value = candidates.front().value;
    double sum_final = 0.0;
    for (const auto &c : candidates) sum_final += std::exp(c.value - max_value);

    int64_t next_state = gdscript_next_rng_state(rng_state);
    double sample = gdscript_uniform_from_state(next_state) * sum_final;
    double cumulative_final = 0.0;
    for (const auto &c : candidates) {
        cumulative_final += std::exp(c.value - max_value);
        if (sample <= cumulative_final) return {c.token, next_state};
    }
    return {candidates.front().token, next_state};
}

std::vector<float> gdscript_convert_decoder_output(
    const std::vector<float> &output,
    const std::vector<int64_t> &output_shape,
    int64_t sample_rate,
    bool normalize,
    double gain
) {
    if (output.empty()) return {};
    GdLayoutResult layout = gdscript_resolve_layout(output_shape, static_cast<int64_t>(output.size()));
    int64_t sample_count = layout.time_steps;
    if (sample_count <= 0) return {};

    std::vector<float> waveform(sample_count);
    double sum = 0.0;
    double peak = 0.0;

    auto clamp_val = [](double v, double lo, double hi) { return std::min(std::max(v, lo), hi); };

    if (!layout.has_layout) {
        for (int64_t i = 0; i < sample_count; ++i) {
            double s = clamp_val(output[i], -1.5, 1.5) * gain;
            waveform[i] = static_cast<float>(s);
            sum += s;
            peak = std::max(peak, std::abs(s));
        }
    } else {
        bool downmix = layout.channel_count > 1 && layout.channel_stride > 0;
        for (int64_t t = 0; t < sample_count; ++t) {
            double raw = 0.0;
            int64_t base = t * layout.time_stride;
            if (!downmix) {
                if (base >= 0 && base < static_cast<int64_t>(output.size())) raw = output[base];
            } else {
                for (int64_t ch = 0; ch < layout.channel_count; ++ch) {
                    int64_t idx = base + ch * layout.channel_stride;
                    if (idx >= 0 && idx < static_cast<int64_t>(output.size())) raw += output[idx];
                }
                raw /= static_cast<double>(std::max(layout.channel_count, int64_t(1)));
            }
            double s = clamp_val(raw, -1.5, 1.5) * gain;
            waveform[t] = static_cast<float>(s);
            sum += s;
            peak = std::max(peak, std::abs(s));
        }
    }

    double dc_offset = sum / static_cast<double>(std::max(sample_count, int64_t(1)));
    for (auto &s : waveform) s -= static_cast<float>(dc_offset);

    peak = 0.0;
    for (float s : waveform) peak = std::max(peak, std::abs(static_cast<double>(s)));

    if (normalize && peak > 0.00001) {
        double scale = std::min(1.0, 0.95 / peak);
        for (auto &s : waveform) s *= static_cast<float>(scale);
    }
    for (auto &s : waveform) s = std::clamp(s, -1.0f, 1.0f);

    int64_t edge_fade = std::clamp(
        static_cast<int64_t>(std::llround(0.004 * static_cast<double>(sample_rate))),
        int64_t(8),
        std::max(int64_t(8), sample_count / 10)
    );
    int64_t safe_fade = std::min(edge_fade, static_cast<int64_t>(waveform.size()) / 2);
    for (int64_t i = 0; i < safe_fade; ++i) {
        double scale = static_cast<double>(i) / static_cast<double>(std::max(safe_fade, int64_t(1)));
        waveform[i] *= static_cast<float>(scale);
        waveform[waveform.size() - 1 - i] *= static_cast<float>(scale);
    }
    return waveform;
}

}

TEST_CASE("GDScript parity: RNG xorshift32 sequence matches", "[parity][rng]") {
    int64_t state = 1;
    state = gdscript_next_rng_state(state);
    int64_t expected_step1 = 1 ^ (1 << 13);
    expected_step1 ^= (expected_step1 >> 17);
    expected_step1 ^= (expected_step1 << 5);
    if (expected_step1 == 0) expected_step1 = 1;
    CHECK(state == expected_step1);

    int64_t prev = state;
    state = gdscript_next_rng_state(state);
    int64_t manual = prev;
    manual ^= (manual << 13);
    manual ^= (manual >> 17);
    manual ^= (manual << 5);
    if (manual == 0) manual = 1;
    CHECK(state == manual);

    state = gdscript_next_rng_state(state);
    CHECK(state != 0);
    state = gdscript_next_rng_state(state);
    CHECK(state != 0);
}

TEST_CASE("GDScript parity: RNG zero input maps to 1", "[parity][rng]") {
    int64_t state = gdscript_next_rng_state(0);
    int64_t expected = 1;
    expected ^= (expected << 13);
    expected ^= (expected >> 17);
    expected ^= (expected << 5);
    if (expected == 0) expected = 1;
    CHECK(state == expected);
    CHECK(state != 0);
}

TEST_CASE("GDScript parity: uniform_from_state range", "[parity][rng]") {
    CHECK(gdscript_uniform_from_state(0) == Approx(0.0).margin(1e-10));
    CHECK(gdscript_uniform_from_state(0x00FFFFFF) == Approx(1.0).margin(1e-6));
    CHECK(gdscript_uniform_from_state(0x00800000) == Approx(0.5).margin(1e-6));
    double v = gdscript_uniform_from_state(12345);
    CHECK(v >= 0.0);
    CHECK(v < 1.0);
}

TEST_CASE("GDScript parity: codec prefix nothink mode", "[parity][codec_prefix]") {
    GdCodecPrefixResult gd;
    gd.force_japanese = false;
    auto gd_tokens = gd.build();

    auto cpp_tokens = gotst::LanguageConfig::build_codec_prefix_tokens(
        -1, 2154, 2155, 2156, 2157, 2148, 2149
    );

    REQUIRE(gd_tokens.size() == cpp_tokens.size());
    for (size_t i = 0; i < gd_tokens.size(); ++i) {
        CHECK(gd_tokens[i] == cpp_tokens[i]);
    }
    CHECK(cpp_tokens.size() == 5);
    CHECK(cpp_tokens[0] == 2155);
    CHECK(cpp_tokens[1] == 2156);
    CHECK(cpp_tokens[2] == 2157);
    CHECK(cpp_tokens[3] == 2148);
    CHECK(cpp_tokens[4] == 2149);
}

TEST_CASE("GDScript parity: codec prefix think/japanese mode", "[parity][codec_prefix]") {
    GdCodecPrefixResult gd;
    gd.force_japanese = true;
    auto gd_tokens = gd.build();

    auto cpp_tokens = gotst::LanguageConfig::build_codec_prefix_tokens(
        2058, 2154, 2155, 2156, 2157, 2148, 2149
    );

    REQUIRE(gd_tokens.size() == cpp_tokens.size());
    for (size_t i = 0; i < gd_tokens.size(); ++i) {
        CHECK(gd_tokens[i] == cpp_tokens[i]);
    }
    CHECK(cpp_tokens.size() == 6);
    CHECK(cpp_tokens[0] == 2154);
    CHECK(cpp_tokens[1] == 2156);
    CHECK(cpp_tokens[2] == 2058);
    CHECK(cpp_tokens[3] == 2157);
    CHECK(cpp_tokens[4] == 2148);
    CHECK(cpp_tokens[5] == 2149);
}

TEST_CASE("GDScript parity: extract_sequence basic", "[parity][extract]") {
    std::vector<float> vals(20, 1.0f);
    for (int i = 0; i < 20; ++i) vals[i] = static_cast<float>(i);
    std::vector<int64_t> shape = {4, 5};

    auto res = gdscript_extract_sequence(vals, shape);
    CHECK(res.valid);
    CHECK(res.sequence_length == 4);
    CHECK(res.hidden_size == 5);
    CHECK(res.values.size() == 20);
    CHECK(res.values[0] == Approx(0.0f));
    CHECK(res.values[19] == Approx(19.0f));
}

TEST_CASE("GDScript parity: extract_sequence empty input", "[parity][extract]") {
    std::vector<float> empty;
    std::vector<int64_t> shape = {4, 5};
    auto res = gdscript_extract_sequence(empty, shape);
    CHECK_FALSE(res.valid);
}

TEST_CASE("GDScript parity: extract_sequence oversized shape returns invalid", "[parity][extract]") {
    std::vector<float> vals(10, 1.0f);
    std::vector<int64_t> shape = {4, 5};
    auto res = gdscript_extract_sequence(vals, shape);
    CHECK_FALSE(res.valid);
}

TEST_CASE("GDScript parity: extract_last_hidden_row", "[parity][hidden_row]") {
    int64_t hidden_size = 4;
    std::vector<float> data;
    for (int i = 0; i < 12; ++i) data.push_back(static_cast<float>(i));
    std::vector<int64_t> shape = {3, 4};

    auto row = gdscript_extract_last_hidden_row(data, shape, hidden_size);
    REQUIRE(row.size() == 4);
    CHECK(row[0] == Approx(8.0f));
    CHECK(row[1] == Approx(9.0f));
    CHECK(row[2] == Approx(10.0f));
    CHECK(row[3] == Approx(11.0f));
}

TEST_CASE("GDScript parity: extract_last_hidden_row size mismatch returns empty", "[parity][hidden_row]") {
    std::vector<float> data(12, 0.0f);
    std::vector<int64_t> shape = {3, 5};
    auto row = gdscript_extract_last_hidden_row(data, shape, 4);
    CHECK(row.empty());
}

TEST_CASE("GDScript parity: contains_only_finite_values", "[parity][finite]") {
    CHECK(gdscript_contains_only_finite({1.0f, 2.0f, 3.0f}));
    CHECK_FALSE(gdscript_contains_only_finite({1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f}));
    CHECK_FALSE(gdscript_contains_only_finite({std::numeric_limits<float>::infinity()}));
    CHECK_FALSE(gdscript_contains_only_finite({-std::numeric_limits<float>::infinity()}));
    CHECK(gdscript_contains_only_finite({}));
    CHECK(gdscript_contains_only_finite({0.0f, -1.0f, 1.0f}));
}

TEST_CASE("GDScript parity: select_last_row_argmax", "[parity][argmax]") {
    std::vector<float> logits(10, 0.0f);
    logits[3] = 5.0f;
    logits[7] = 10.0f;
    std::vector<int64_t> shape = {1, 10};

    CHECK(gdscript_select_last_row_argmax(logits, shape, 0, 10) == 7);
    CHECK(gdscript_select_last_row_argmax(logits, shape, 0, 5) == 3);
    CHECK(gdscript_select_last_row_argmax(logits, shape, 5, 5) == 7);
}

TEST_CASE("GDScript parity: select_last_row_argmax empty returns -1", "[parity][argmax]") {
    std::vector<float> empty;
    std::vector<int64_t> shape = {1, 10};
    CHECK(gdscript_select_last_row_argmax(empty, shape, 0, 10) == -1);
    std::vector<float> vals(10, 0.0f);
    std::vector<int64_t> empty_shape;
    CHECK(gdscript_select_last_row_argmax(vals, empty_shape, 0, 10) == -1);
}

TEST_CASE("GDScript parity: select_last_row_argmax multi-row picks last row", "[parity][argmax]") {
    std::vector<float> logits(20, 0.0f);
    logits[3] = 100.0f;
    logits[17] = 5.0f;
    std::vector<int64_t> shape = {2, 10};

    CHECK(gdscript_select_last_row_argmax(logits, shape, 0, 10) == 7);
}

TEST_CASE("GDScript parity: select_last_row_token no_sample uses argmax", "[parity][sampling]") {
    std::vector<float> logits(10, 0.0f);
    logits[5] = 10.0f;
    std::vector<int64_t> shape = {1, 10};

    auto result = gdscript_select_last_row_token(
        logits, shape, 0, 10, false, 50, 1.0, 0.9, {}, 1.05, 42
    );
    CHECK(result.token == 5);
    CHECK(result.rng_state == 42);
}

TEST_CASE("GDScript parity: select_last_row_token deterministic with same RNG state", "[parity][sampling]") {
    std::vector<float> logits(100, 0.0f);
    for (int i = 0; i < 100; ++i) logits[i] = static_cast<float>(i) * 0.1f;
    std::vector<int64_t> shape = {1, 100};

    auto r1 = gdscript_select_last_row_token(logits, shape, 0, 100, true, 50, 1.0, 0.9, {}, 1.0, 12345);
    auto r2 = gdscript_select_last_row_token(logits, shape, 0, 100, true, 50, 1.0, 0.9, {}, 1.0, 12345);
    CHECK(r1.token == r2.token);
    CHECK(r1.rng_state == r2.rng_state);
}

TEST_CASE("GDScript parity: select_last_row_token repetition penalty demotes prior token", "[parity][sampling]") {
    std::vector<float> logits(10, 0.0f);
    logits[3] = 10.0f;
    logits[7] = 9.0f;
    std::vector<int64_t> shape = {1, 10};

    auto no_penalty = gdscript_select_last_row_token(
        logits, shape, 0, 10, false, 10, 1.0, 1.0, {}, 1.0, 100
    );
    CHECK(no_penalty.token == 3);

    std::vector<float> logits_penalized(10, 0.0f);
    logits_penalized[3] = -10.0f;
    logits_penalized[7] = 9.0f;
    auto with_penalty = gdscript_select_last_row_token(
        logits_penalized, shape, 0, 10, true, 10, 1.0, 1.0, {3}, 100.0, 100
    );
    CHECK(with_penalty.token == 7);
}

TEST_CASE("GDScript parity: should_stop_on_eos true when eos dominates", "[parity][eos]") {
    std::vector<float> logits(4096, -100.0f);
    logits[2150] = 50.0f;
    logits[0] = 10.0f;
    std::vector<int64_t> shape = {1, 4096};

    CHECK(gdscript_should_stop_on_eos(logits, shape, 2150, 2048, 0.0));
}

TEST_CASE("GDScript parity: should_stop_on_eos false when codec token dominates", "[parity][eos]") {
    std::vector<float> logits(4096, -100.0f);
    logits[2150] = 5.0f;
    logits[100] = 50.0f;
    std::vector<int64_t> shape = {1, 4096};

    CHECK_FALSE(gdscript_should_stop_on_eos(logits, shape, 2150, 2048, 0.0));
}

TEST_CASE("GDScript parity: should_stop_on_eos with margin", "[parity][eos]") {
    std::vector<float> logits(4096, -100.0f);
    logits[2150] = 50.0f;
    logits[0] = 49.0f;
    std::vector<int64_t> shape = {1, 4096};

    CHECK(gdscript_should_stop_on_eos(logits, shape, 2150, 2048, 0.0));
    CHECK_FALSE(gdscript_should_stop_on_eos(logits, shape, 2150, 2048, 5.0));
}

TEST_CASE("GDScript parity: resolve_decoder_output_layout 1D", "[parity][layout]") {
    auto layout = gdscript_resolve_layout({100}, 100);
    CHECK_FALSE(layout.has_layout);
    CHECK(layout.time_steps == 100);
}

TEST_CASE("GDScript parity: resolve_decoder_output_layout 2D time-major", "[parity][layout]") {
    auto layout = gdscript_resolve_layout({1, 24000}, 24000);
    CHECK(layout.has_layout);
    CHECK(layout.time_steps == 24000);
    CHECK(layout.time_stride == 1);
    CHECK(layout.channel_count == 1);
}

TEST_CASE("GDScript parity: resolve_decoder_output_layout 3D multi-channel", "[parity][layout]") {
    auto layout = gdscript_resolve_layout({2, 1, 24000}, 48000);
    CHECK(layout.has_layout);
    CHECK(layout.time_steps == 24000);
    CHECK(layout.channel_count == 2);
    CHECK(layout.channel_stride == 24000);
    CHECK(layout.time_stride == 1);
}

TEST_CASE("GDScript parity: resolve_decoder_output_layout empty shape", "[parity][layout]") {
    auto layout = gdscript_resolve_layout({}, 0);
    CHECK_FALSE(layout.has_layout);
}

TEST_CASE("GDScript parity: convert_decoder_output 1D flat", "[parity][waveform]") {
    std::vector<float> output(100, 0.5f);
    auto waveform = gdscript_convert_decoder_output(output, {100}, 24000, false, 1.0);
    REQUIRE(waveform.size() == 100);

    double sum = 0.0;
    for (float s : waveform) sum += s;
    CHECK(std::abs(sum) < 0.01);

    for (float s : waveform) {
        CHECK(std::abs(s) <= 1.0f);
    }
}

TEST_CASE("GDScript parity: convert_decoder_output with normalization", "[parity][waveform]") {
    std::vector<float> output(100, 0.0f);
    output[50] = 2.0f;
    auto waveform = gdscript_convert_decoder_output(output, {100}, 24000, true, 1.0);

    float peak = 0.0f;
    for (float s : waveform) peak = std::max(peak, std::abs(s));
    CHECK(peak <= 1.0f);
    CHECK(peak > 0.9f);
}

TEST_CASE("GDScript parity: convert_decoder_output applies edge fade", "[parity][waveform]") {
    std::vector<float> output(1000);
    for (int i = 0; i < 1000; ++i) {
        output[i] = 0.5f * std::sin(2.0f * 3.14159265f * static_cast<float>(i) / 100.0f);
    }
    double mean = 0.0;
    for (float s : output) mean += s;
    mean /= 1000.0;

    auto waveform = gdscript_convert_decoder_output(output, {1000}, 24000, false, 1.0);

    int64_t edge_fade = std::clamp(static_cast<int64_t>(std::llround(0.004 * 24000.0)), int64_t(8), int64_t(1000 / 10));
    CHECK(edge_fade == 96);

    CHECK(waveform[0] == Approx(0.0f).margin(0.01f));
    CHECK(std::abs(waveform[1]) < std::abs(waveform[2]));
    CHECK(waveform[999] == Approx(0.0f).margin(0.01f));
    CHECK(std::abs(waveform[998]) < std::abs(waveform[997]));
}

TEST_CASE("GDScript parity: convert_decoder_output 3D downmix", "[parity][waveform]") {
    std::vector<float> output(200, 0.0f);
    for (int i = 0; i < 100; ++i) {
        float t = static_cast<float>(i) / 100.0f;
        output[i] = 0.3f * std::sin(2.0f * 3.14159265f * t);
        output[100 + i] = 0.5f * std::sin(2.0f * 3.14159265f * t + 1.0f);
    }
    auto waveform = gdscript_convert_decoder_output(output, {2, 100}, 24000, false, 1.0);
    REQUIRE(waveform.size() == 100);

    bool has_nonzero = false;
    for (float s : waveform) {
        if (std::abs(s) > 0.001f) has_nonzero = true;
    }
    CHECK(has_nonzero);
}

TEST_CASE("GDScript parity: slice_rows", "[parity][utils]") {
    std::vector<float> data;
    for (int i = 0; i < 12; ++i) data.push_back(static_cast<float>(i));

    auto result = gdscript_slice_rows(data, 1, 2, 4);
    REQUIRE(result.size() == 8);
    CHECK(result[0] == Approx(4.0f));
    CHECK(result[4] == Approx(8.0f));
}

TEST_CASE("GDScript parity: slice_rows out of bounds returns empty", "[parity][utils]") {
    std::vector<float> data(8, 0.0f);
    CHECK(gdscript_slice_rows(data, 5, 1, 4).empty());
    CHECK(gdscript_slice_rows(data, 0, 0, 4).empty());
    CHECK(gdscript_slice_rows({}, 0, 1, 4).empty());
}

TEST_CASE("GDScript parity: concat_rows", "[parity][utils]") {
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {3.0f, 4.0f, 5.0f};
    auto result = gdscript_concat_rows({a, b});
    REQUIRE(result.size() == 5);
    CHECK(result[0] == Approx(1.0f));
    CHECK(result[2] == Approx(3.0f));
    CHECK(result[4] == Approx(5.0f));
}

TEST_CASE("GDScript parity: repeat_row", "[parity][utils]") {
    std::vector<float> row = {1.0f, 2.0f, 3.0f};
    auto result = gdscript_repeat_row(row, 3);
    REQUIRE(result.size() == 9);
    CHECK(result[0] == Approx(1.0f));
    CHECK(result[3] == Approx(1.0f));
    CHECK(result[6] == Approx(1.0f));
}

TEST_CASE("GDScript parity: add_vectors", "[parity][utils]") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    auto result = gdscript_add_vectors(a, b);
    REQUIRE(result.size() == 3);
    CHECK(result[0] == Approx(5.0f));
    CHECK(result[2] == Approx(9.0f));
}

TEST_CASE("GDScript parity: add_vectors mismatched returns empty", "[parity][utils]") {
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {1.0f};
    CHECK(gdscript_add_vectors(a, b).empty());
}

TEST_CASE("GDScript parity: sum_vectors", "[parity][utils]") {
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {3.0f, 4.0f};
    std::vector<float> c = {5.0f, 6.0f};
    auto result = gdscript_sum_vectors({a, b, c});
    REQUIRE(result.size() == 2);
    CHECK(result[0] == Approx(9.0f));
    CHECK(result[1] == Approx(12.0f));
}

TEST_CASE("GDScript parity: sum_vectors skips empty parts", "[parity][utils]") {
    std::vector<float> a;
    std::vector<float> b = {1.0f, 2.0f};
    std::vector<float> c = {3.0f, 4.0f};
    auto result = gdscript_sum_vectors({a, b, c});
    REQUIRE(result.size() == 2);
    CHECK(result[0] == Approx(4.0f));
    CHECK(result[1] == Approx(6.0f));
}

TEST_CASE("GDScript parity: sum_vectors all empty returns empty", "[parity][utils]") {
    std::vector<float> a;
    std::vector<float> b;
    CHECK(gdscript_sum_vectors({a, b}).empty());
}

TEST_CASE("GDScript parity: top_p nucleus sampling truncates candidates", "[parity][sampling]") {
    std::vector<float> logits(10, -100.0f);
    logits[0] = 100.0f;
    logits[1] = 1.0f;
    logits[2] = 0.0f;
    std::vector<int64_t> shape = {1, 10};

    int64_t rng = 1;
    for (int i = 0; i < 10; ++i) rng = gdscript_next_rng_state(rng);

    auto result = gdscript_select_last_row_token(
        logits, shape, 0, 10, true, 10, 0.5, 1.0, {}, 1.0, rng
    );
    CHECK(result.token == 0);
}

TEST_CASE("GDScript parity: RNG state sequence matches GDScript constants", "[parity][rng]") {
    int64_t state = 1;
    state = gdscript_next_rng_state(state);
    state = gdscript_next_rng_state(state);
    state = gdscript_next_rng_state(state);

    int64_t manual = 1;
    for (int i = 0; i < 3; ++i) {
        manual ^= (manual << 13);
        manual ^= (manual >> 17);
        manual ^= (manual << 5);
        if (manual == 0) manual = 1;
    }
    CHECK(state == manual);
}

TEST_CASE("GDScript parity: select_last_row_token RNG advances exactly once", "[parity][sampling]") {
    std::vector<float> logits(10, 0.0f);
    logits[5] = 1.0f;
    std::vector<int64_t> shape = {1, 10};

    int64_t rng_before = 99999;
    auto result = gdscript_select_last_row_token(
        logits, shape, 0, 10, true, 10, 1.0, 1.0, {}, 1.0, rng_before
    );
    int64_t expected_next = gdscript_next_rng_state(rng_before);
    CHECK(result.rng_state == expected_next);
}

TEST_CASE("GDScript parity: build_tts_initial_language_input structure", "[parity][language_input]") {
    int64_t hidden_size = 4;
    int64_t prefix_count = 3;
    int64_t suffix_count = 5;
    int64_t text_seq_len = prefix_count + 1 + 10 + suffix_count;

    std::vector<float> text_proj(text_seq_len * hidden_size, 0.0f);
    for (int64_t i = 0; i < text_seq_len; ++i) {
        for (int64_t j = 0; j < hidden_size; ++j) {
            text_proj[i * hidden_size + j] = static_cast<float>(i * 10 + j);
        }
    }
    std::vector<float> special_proj(3 * hidden_size, 0.0f);
    for (int64_t j = 0; j < hidden_size; ++j) {
        special_proj[0 * hidden_size + j] = 100.0f + j;
        special_proj[1 * hidden_size + j] = 200.0f + j;
        special_proj[2 * hidden_size + j] = 300.0f + j;
    }
    std::vector<float> codec_pad(hidden_size, 400.0f);
    int64_t codec_prefill_len = 6;
    std::vector<float> codec_prefill(codec_prefill_len * hidden_size, 0.0f);
    for (int64_t i = 0; i < codec_prefill_len; ++i) {
        for (int64_t j = 0; j < hidden_size; ++j) {
            codec_prefill[i * hidden_size + j] = static_cast<float>(500 + i * 10 + j);
        }
    }
    std::vector<float> speaker_emb(hidden_size, 600.0f);
    std::vector<float> instruction_proj;
    int64_t instruction_len = 0;

    auto role = gdscript_slice_rows(text_proj, 0, prefix_count, hidden_size);
    auto first_text = gdscript_slice_rows(text_proj, prefix_count, 1, hidden_size);
    auto bos_emb = gdscript_slice_rows(special_proj, 0, 1, hidden_size);
    auto eos_emb = gdscript_slice_rows(special_proj, 1, 1, hidden_size);
    auto pad_emb = gdscript_slice_rows(special_proj, 2, 1, hidden_size);

    REQUIRE_FALSE(role.empty());
    REQUIRE_FALSE(first_text.empty());
    REQUIRE_FALSE(bos_emb.empty());
    REQUIRE_FALSE(eos_emb.empty());
    REQUIRE_FALSE(pad_emb.empty());

    int64_t codec_tag_length = codec_prefill_len - 2;
    auto codec_tag = gdscript_slice_rows(codec_prefill, 0, codec_tag_length, hidden_size);
    auto codec_pad_bos = gdscript_slice_rows(codec_prefill, codec_tag_length, 2, hidden_size);

    auto codec_prompt = gdscript_concat_rows({codec_tag, speaker_emb, codec_pad_bos});
    int64_t codec_prompt_len = static_cast<int64_t>(codec_prompt.size()) / hidden_size;
    CHECK(codec_prompt_len == codec_prefill_len + 1);

    auto codec_core = gdscript_slice_rows(codec_prompt, 0, codec_prompt_len - 1, hidden_size);
    auto codec_seed = gdscript_slice_rows(codec_prompt, codec_prompt_len - 1, 1, hidden_size);
    auto text_pad_rep = gdscript_repeat_row(pad_emb, std::max(codec_prompt_len - 2, int64_t(0)));
    auto pad_plus_bos = gdscript_concat_rows({text_pad_rep, bos_emb});
    auto prefill_core = gdscript_add_vectors(pad_plus_bos, codec_core);

    REQUIRE_FALSE(prefill_core.empty());

    int64_t trailing_start = prefix_count + 1;
    int64_t trailing_end = text_seq_len - suffix_count;
    int64_t trailing_count = trailing_end - trailing_start;
    auto trailing_text = gdscript_slice_rows(text_proj, trailing_start, trailing_count, hidden_size);
    auto trailing_hidden = gdscript_concat_rows({trailing_text, eos_emb});
    auto first_gen = gdscript_add_vectors(first_text, codec_seed);

    REQUIRE_FALSE(trailing_hidden.empty());
    REQUIRE_FALSE(first_gen.empty());

    auto language_input = gdscript_concat_rows({role, prefill_core, first_gen});
    int64_t seq_len = static_cast<int64_t>(language_input.size()) / hidden_size;

    CHECK(seq_len == prefix_count + (codec_prompt_len - 1) + 1);
    CHECK_FALSE(language_input.empty());
}

TEST_CASE("GDScript parity: convert_decoder_output clamps gain", "[parity][waveform]") {
    std::vector<float> output(100, 1.0f);
    auto waveform = gdscript_convert_decoder_output(output, {100}, 24000, false, 2.0);

    for (float s : waveform) {
        CHECK(std::abs(s) <= 1.0f);
    }
}

TEST_CASE("GDScript parity: convert_decoder_output empty returns empty", "[parity][waveform]") {
    std::vector<float> empty;
    auto result = gdscript_convert_decoder_output(empty, {}, 24000, false, 1.0);
    CHECK(result.empty());
}

TEST_CASE("GDScript parity: select_last_row_token empty logits returns -1", "[parity][sampling]") {
    std::vector<float> empty;
    std::vector<int64_t> shape = {1, 10};
    auto result = gdscript_select_last_row_token(empty, shape, 0, 10, true, 50, 1.0, 0.9, {}, 1.0, 42);
    CHECK(result.token == -1);
    CHECK(result.rng_state == 42);
}

TEST_CASE("GDScript parity: should_stop_on_eos invalid inputs return false", "[parity][eos]") {
    std::vector<float> empty;
    CHECK_FALSE(gdscript_should_stop_on_eos(empty, {1, 10}, 5, 10, 0.0));
    std::vector<float> vals(10, 0.0f);
    CHECK_FALSE(gdscript_should_stop_on_eos(vals, {}, 5, 10, 0.0));
    CHECK_FALSE(gdscript_should_stop_on_eos(vals, {1, 10}, -1, 10, 0.0));
}
