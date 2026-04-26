#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace gotst::detail {

inline int64_t next_rng_state(int64_t state) {
    int64_t resolved_state = state == 0 ? 1 : state;
    resolved_state ^= (resolved_state << 13);
    resolved_state ^= (resolved_state >> 17);
    resolved_state ^= (resolved_state << 5);
    return resolved_state == 0 ? 1 : resolved_state;
}

inline double uniform_from_state(int64_t state) {
    return static_cast<double>(state & 0x00FFFFFF) / 16777216.0;
}

inline bool fast_sampling_enabled() {
    static const bool enabled = []() {
        const char *value = std::getenv("GOTST_TTS_FAST_SAMPLING");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

struct SampledToken {
    int64_t token = -1;
    int64_t rng_state = 1;
};

struct SamplingCandidate {
    int64_t token = -1;
    double value = 0.0;
};

struct SamplingScratch {
    std::vector<SamplingCandidate> candidates;
    std::vector<double> exp_values;
    std::array<SamplingCandidate, 256> top_candidates;
};

inline SampledToken sample_argmax(const float *logits, int32_t start, int32_t count, int64_t rng_state) {
    if(!logits || count <= 0) {
        return {-1, rng_state};
    }

    int64_t best = start;
    float best_value = -std::numeric_limits<float>::infinity();
    for(int32_t index = 0; index < count; ++index) {
        const float value = logits[start + index];
        if(value > best_value) {
            best_value = value;
            best = start + index;
        }
    }

    return {best, rng_state};
}

template <typename HasPriorToken>
inline SampledToken sample_token_fast_topk(const float *logits,
                                           int32_t start,
                                           int32_t count,
                                           int32_t top_k,
                                           float temperature,
                                           HasPriorToken &&has_prior_token,
                                           float repetition_penalty,
                                           int64_t rng_state,
                                           SamplingScratch &scratch) {
    const int32_t k = std::clamp(top_k, 1, count);
    const double temp = std::max(0.05, static_cast<double>(temperature));
    auto min_heap = [](const SamplingCandidate &a, const SamplingCandidate &b) {
        return a.value > b.value;
    };

    int32_t heap_size = 0;
    for(int32_t index = 0; index < count; ++index) {
        const int64_t token_index = start + index;
        double value = logits[token_index];
        if(has_prior_token(token_index) && repetition_penalty > 1.0f) {
            value = value >= 0.0 ? value / repetition_penalty : value * repetition_penalty;
        }
        SamplingCandidate candidate {token_index, value / temp};
        if(heap_size < k) {
            scratch.top_candidates[static_cast<size_t>(heap_size)] = candidate;
            heap_size += 1;
            if(heap_size == k) {
                std::make_heap(
                    scratch.top_candidates.begin(),
                    scratch.top_candidates.begin() + heap_size,
                    min_heap
                );
            }
            continue;
        }

        if(candidate.value <= scratch.top_candidates.front().value) {
            continue;
        }

        std::pop_heap(
            scratch.top_candidates.begin(),
            scratch.top_candidates.begin() + heap_size,
            min_heap
        );
        scratch.top_candidates[static_cast<size_t>(heap_size - 1)] = candidate;
        std::push_heap(
            scratch.top_candidates.begin(),
            scratch.top_candidates.begin() + heap_size,
            min_heap
        );
    }

    if(heap_size <= 0) {
        return sample_argmax(logits, start, count, rng_state);
    }

    std::sort(
        scratch.top_candidates.begin(),
        scratch.top_candidates.begin() + heap_size,
        [](const SamplingCandidate &a, const SamplingCandidate &b) {
            return a.value > b.value;
        }
    );

    const double max_value = scratch.top_candidates.front().value;
    double sum_final = 0.0;
    for(int32_t index = 0; index < heap_size; ++index) {
        sum_final += std::exp(scratch.top_candidates[static_cast<size_t>(index)].value - max_value);
    }

    const int64_t next_state = next_rng_state(rng_state);
    const double sample = uniform_from_state(next_state) * sum_final;
    double cumulative_final = 0.0;
    for(int32_t index = 0; index < heap_size; ++index) {
        const SamplingCandidate &candidate = scratch.top_candidates[static_cast<size_t>(index)];
        cumulative_final += std::exp(candidate.value - max_value);
        if(sample <= cumulative_final) {
            return {candidate.token, next_state};
        }
    }

    return {scratch.top_candidates.front().token, next_state};
}

template <typename HasPriorToken>
inline SampledToken sample_token(const float *logits,
                                 int32_t start,
                                 int32_t count,
                                 bool do_sample,
                                 int32_t top_k,
                                 float top_p,
                                 float temperature,
                                 HasPriorToken &&has_prior_token,
                                 float repetition_penalty,
                                 int64_t rng_state,
                                 SamplingScratch &scratch) {
    if(!do_sample) {
        return sample_argmax(logits, start, count, rng_state);
    }
    if(!logits || count <= 0) {
        return {-1, rng_state};
    }

    const int32_t k = std::clamp(top_k, 1, count);
    const double temp = std::max(0.05, static_cast<double>(temperature));
    if(fast_sampling_enabled() &&
       top_p >= 0.999f &&
       k <= static_cast<int32_t>(scratch.top_candidates.size())) {
        return sample_token_fast_topk(
            logits,
            start,
            count,
            k,
            temperature,
            std::forward<HasPriorToken>(has_prior_token),
            repetition_penalty,
            rng_state,
            scratch
        );
    }

    scratch.candidates.resize(static_cast<size_t>(count));

    for(int32_t index = 0; index < count; ++index) {
        const int64_t token_index = start + index;
        double value = logits[token_index];
        if(has_prior_token(token_index) && repetition_penalty > 1.0f) {
            value = value >= 0.0 ? value / repetition_penalty : value * repetition_penalty;
        }
        scratch.candidates[static_cast<size_t>(index)] = {token_index, value / temp};
    }

    auto by_value_desc = [](const SamplingCandidate &a, const SamplingCandidate &b) {
        return a.value > b.value;
    };

    if(static_cast<int32_t>(scratch.candidates.size()) > k) {
        auto nth = scratch.candidates.begin() + k;
        std::nth_element(scratch.candidates.begin(), nth, scratch.candidates.end(), by_value_desc);
        scratch.candidates.resize(static_cast<size_t>(k));
    }

    std::sort(scratch.candidates.begin(), scratch.candidates.end(), by_value_desc);
    if(scratch.candidates.empty()) {
        return sample_argmax(logits, start, count, rng_state);
    }

    if(top_p < 0.999f) {
        scratch.exp_values.clear();
        if(scratch.exp_values.capacity() < scratch.candidates.size()) {
            scratch.exp_values.reserve(scratch.candidates.size());
        }

        const double max_logit = scratch.candidates.front().value;
        double sum_exp = 0.0;
        for(const SamplingCandidate &candidate : scratch.candidates) {
            const double exp_value = std::exp(candidate.value - max_logit);
            scratch.exp_values.push_back(exp_value);
            sum_exp += exp_value;
        }

        double cumulative = 0.0;
        int64_t final_count = 0;
        const double normalized_sum = std::max(sum_exp, 1e-12);
        for(size_t index = 0; index < scratch.exp_values.size(); ++index) {
            cumulative += scratch.exp_values[index] / normalized_sum;
            final_count = static_cast<int64_t>(index) + 1;
            if(cumulative >= top_p) {
                break;
            }
        }

        final_count = std::clamp<int64_t>(final_count, 1, static_cast<int64_t>(scratch.candidates.size()));
        scratch.candidates.resize(static_cast<size_t>(final_count));
    }

    const double max_value = scratch.candidates.front().value;
    double sum_final = 0.0;
    for(const SamplingCandidate &candidate : scratch.candidates) {
        sum_final += std::exp(candidate.value - max_value);
    }

    const int64_t next_state = next_rng_state(rng_state);
    const double sample = uniform_from_state(next_state) * sum_final;
    double cumulative_final = 0.0;
    for(const SamplingCandidate &candidate : scratch.candidates) {
        cumulative_final += std::exp(candidate.value - max_value);
        if(sample <= cumulative_final) {
            return {candidate.token, next_state};
        }
    }

    return {scratch.candidates.front().token, next_state};
}

} // namespace gotst::detail
