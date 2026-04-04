#include "gotst/core/tts_code_generator.hpp"

#include <godot_llama/llama_context_handle.hpp>
#include <godot_llama/llama_model_handle.hpp>
#include <godot_llama/llama_params.hpp>
#include <godot_llama/llama_position_layout.hpp>
#include <gonx/core/session.hpp>
#include <gonx/core/type_conversion.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace gotst {
namespace {

int64_t next_rng(int64_t state) {
    int64_t s = state == 0 ? 1 : state;
    s ^= (s << 13);
    s ^= (s >> 17);
    s ^= (s << 5);
    return s == 0 ? 1 : s;
}

double uniform_rng(int64_t state) {
    return static_cast<double>(state & 0x00FFFFFF) / 16777216.0;
}

struct SampledToken {
    int64_t token = -1;
    int64_t rng_state = 1;
};

SampledToken sample_argmax(const float *logits, int32_t start, int32_t count, int64_t rng_state) {
    int64_t best = start;
    float best_val = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < count; ++i) {
        float v = logits[start + i];
        if (v > best_val) {
            best_val = v;
            best = start + i;
        }
    }
    return {best, rng_state};
}

SampledToken sample_token(
    const float *logits,
    int32_t start,
    int32_t count,
    bool do_sample,
    int32_t top_k,
    float top_p,
    float temperature,
    const std::unordered_set<int64_t> &prior_tokens,
    float repetition_penalty,
    int64_t rng_state
) {
    if (!do_sample) {
        return sample_argmax(logits, start, count, rng_state);
    }

    const int32_t k = std::clamp(top_k, 1, count);
    const double temp = std::max(0.05, static_cast<double>(temperature));

    struct Candidate {
        int64_t token;
        double value;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(count);
    for (int32_t i = 0; i < count; ++i) {
        const int64_t token_index = start + i;
        double value = logits[token_index];
        if (prior_tokens.count(token_index) && repetition_penalty > 1.0f) {
            value = value >= 0.0 ? value / repetition_penalty : value * repetition_penalty;
        }
        candidates.push_back({token_index, value / temp});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
        return a.value > b.value;
    });
    if (static_cast<int32_t>(candidates.size()) > k) {
        candidates.resize(k);
    }
    if (candidates.empty()) {
        return sample_argmax(logits, start, count, rng_state);
    }

    if (top_p < 0.999f) {
        const double max_logit = candidates.front().value;
        double sum_exp = 0.0;
        std::vector<double> exp_values;
        exp_values.reserve(candidates.size());
        for (const auto &c : candidates) {
            double e = std::exp(c.value - max_logit);
            exp_values.push_back(e);
            sum_exp += e;
        }
        double cumulative = 0.0;
        int64_t final_count = 0;
        for (size_t i = 0; i < exp_values.size(); ++i) {
            cumulative += exp_values[i] / std::max(sum_exp, 1e-12);
            final_count = static_cast<int64_t>(i) + 1;
            if (cumulative >= top_p) break;
        }
        final_count = std::clamp<int64_t>(final_count, 1, static_cast<int64_t>(candidates.size()));
        candidates.resize(final_count);
    }

    const double max_value = candidates.front().value;
    double sum_final = 0.0;
    for (const auto &c : candidates) {
        sum_final += std::exp(c.value - max_value);
    }

    const int64_t next_state = next_rng(rng_state);
    const double sample = uniform_rng(next_state) * sum_final;
    double cumulative_final = 0.0;
    for (const auto &c : candidates) {
        cumulative_final += std::exp(c.value - max_value);
        if (sample <= cumulative_final) {
            return {c.token, next_state};
        }
    }
    return {candidates.front().token, next_state};
}

bool should_stop_eos(const float *logits, int32_t eos_id, int32_t codec_size, float margin) {
    if (eos_id < 0) return false;
    const float eos_logit = logits[eos_id];
    float best_codec = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < codec_size; ++i) {
        best_codec = std::max(best_codec, logits[i]);
    }
    return static_cast<double>(eos_logit) > (static_cast<double>(best_codec) + margin);
}

gotst::Result<std::vector<float>> run_onnx_embedding(
    gonx::InferenceSession &session,
    std::span<const int64_t> input_ids,
    const int64_t *generation_steps = nullptr,
    int64_t gen_step_count = 0
) {
    std::vector<int64_t> id_shape = {1, static_cast<int64_t>(input_ids.size())};
    auto id_tensor = gonx::create_int64_tensor(input_ids, id_shape);
    if (id_tensor.has_error()) {
        return gotst::Error::inference_failed("ONNX embedding: create id_tensor failed: " + id_tensor.error().message);
    }

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(id_tensor).value());

    if (generation_steps && gen_step_count > 0) {
        std::vector<int64_t> gs_shape = {gen_step_count};
        auto gs_tensor = gonx::create_int64_tensor({generation_steps, static_cast<size_t>(gen_step_count)}, gs_shape);
        if (gs_tensor.has_error()) {
            return gotst::Error::inference_failed("ONNX embedding: create gs_tensor failed: " + gs_tensor.error().message);
        }
        inputs.push_back(std::move(gs_tensor).value());
    }

    auto result = session.run(inputs);
    if (result.has_error()) {
        return gotst::Error::inference_failed("ONNX embedding: session.run failed: " + result.error().message);
    }

    auto &outputs = result.value();
    if (outputs.empty()) {
        return gotst::Error::inference_failed("ONNX embedding: empty outputs");
    }

    auto shape = gonx::get_tensor_shape(outputs[0]);
    int64_t total = 1;
    for (auto d : shape) total *= d;
    if (total <= 0) {
        return gotst::Error::inference_failed("ONNX embedding: zero-size output tensor");
    }

    const float *data = outputs[0].GetTensorData<float>();
    return std::vector<float>{data, data + total};
}

} // namespace

struct TtsCodeGenerator::Impl {
    std::shared_ptr<godot_llama::LlamaModelHandle> talker_model;
    godot_llama::LlamaContextHandle talker_ctx;
    std::shared_ptr<godot_llama::LlamaModelHandle> predictor_model;
    godot_llama::LlamaContextHandle predictor_ctx;
    gonx::InferenceSession codec_embedding;
    gonx::InferenceSession predictor_embedding;
    TtsSessionConfig session_config;
    bool loaded = false;
};

TtsCodeGenerator::TtsCodeGenerator() : impl_(std::make_unique<Impl>()) {}
TtsCodeGenerator::~TtsCodeGenerator() = default;
TtsCodeGenerator::TtsCodeGenerator(TtsCodeGenerator &&) noexcept = default;
TtsCodeGenerator &TtsCodeGenerator::operator=(TtsCodeGenerator &&) noexcept = default;

bool TtsCodeGenerator::is_loaded() const {
    return impl_ && impl_->loaded;
}

Result<void> TtsCodeGenerator::load(const TtsModelPaths &paths, const TtsSessionConfig &config) {
    impl_->loaded = false;
    impl_->session_config = config;

    // Load talker GGUF.
    godot_llama::ModelConfig talker_cfg;
    talker_cfg.model_path = paths.talker_gguf_path;
    talker_cfg.n_ctx = config.talker_n_ctx;
    talker_cfg.n_batch = config.talker_n_batch;
    talker_cfg.n_threads = config.n_threads;
    talker_cfg.n_gpu_layers = config.n_gpu_layers;
    talker_cfg.use_mmap = config.use_mmap;
    talker_cfg.use_mlock = config.use_mlock;
    talker_cfg.flash_attn_type = config.flash_attn_type;
    talker_cfg.type_k = config.type_k;
    talker_cfg.type_v = config.type_v;
    talker_cfg.embeddings_enabled = true;

    auto talker_load_err = godot_llama::LlamaModelHandle::load(talker_cfg, impl_->talker_model);
    if (!talker_load_err.ok()) {
        return Error::model_not_loaded("Failed to load talker GGUF: " + talker_load_err.message);
    }
    auto talker_ctx_err = godot_llama::LlamaContextHandle::create(impl_->talker_model, talker_cfg, impl_->talker_ctx);
    if (!talker_ctx_err.ok()) {
        return Error::model_not_loaded("Failed to create talker context: " + talker_ctx_err.message);
    }

    // Load predictor GGUF.
    godot_llama::ModelConfig predictor_cfg;
    predictor_cfg.model_path = paths.predictor_gguf_path;
    predictor_cfg.n_ctx = config.predictor_n_ctx;
    predictor_cfg.n_batch = config.predictor_n_batch;
    predictor_cfg.n_threads = config.n_threads;
    predictor_cfg.n_gpu_layers = config.n_gpu_layers;
    predictor_cfg.use_mmap = config.use_mmap;
    predictor_cfg.use_mlock = config.use_mlock;
    predictor_cfg.flash_attn_type = config.flash_attn_type;
    predictor_cfg.type_k = config.type_k;
    predictor_cfg.type_v = config.type_v;
    predictor_cfg.embeddings_enabled = false;

    auto pred_load_err = godot_llama::LlamaModelHandle::load(predictor_cfg, impl_->predictor_model);
    if (!pred_load_err.ok()) {
        return Error::model_not_loaded("Failed to load predictor GGUF: " + pred_load_err.message);
    }
    auto pred_ctx_err = godot_llama::LlamaContextHandle::create(impl_->predictor_model, predictor_cfg, impl_->predictor_ctx);
    if (!pred_ctx_err.ok()) {
        return Error::model_not_loaded("Failed to create predictor context: " + pred_ctx_err.message);
    }

    // Load ONNX sessions.
    gonx::SessionConfig onnx_cfg;
    onnx_cfg.optimization_level = 99;

    auto codec_status = impl_->codec_embedding.load(paths.codec_embedding_onnx_path, onnx_cfg);
    if (codec_status.has_error()) {
        return Error::model_not_loaded("Failed to load codec embedding ONNX: " + codec_status.error().message);
    }

    auto pred_emb_status = impl_->predictor_embedding.load(paths.predictor_embedding_onnx_path, onnx_cfg);
    if (pred_emb_status.has_error()) {
        return Error::model_not_loaded("Failed to load predictor embedding ONNX: " + pred_emb_status.error().message);
    }

    impl_->loaded = true;
    return {};
}

Result<TtsGenerateResult> TtsCodeGenerator::generate(
    std::span<const float> initial_language_input,
    int32_t initial_sequence_length,
    std::span<const float> trailing_text_hidden,
    int32_t trailing_text_length,
    std::span<const float> tts_pad_embedding,
    const TtsSamplingConfig &params,
    CancellationToken *cancel
) {
    if (!impl_->loaded) {
        return Error::invalid_state("TTS code generator is not loaded.");
    }

    const int32_t hidden_size = params.hidden_size;
    const int32_t codebook_size = params.codebook_size;
    const int32_t residual_groups = params.residual_groups;
    const int32_t talker_pos_components = impl_->session_config.talker_position_components;
    const int32_t predictor_pos_components = impl_->session_config.predictor_position_components;

    if (initial_language_input.empty() || initial_sequence_length <= 0 || hidden_size <= 0) {
        return Error::invalid_argument("Initial language input is empty.");
    }

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    const auto t_gen_start = Clock::now();
    double ms_talker_prefill = 0, ms_talker_decode = 0;
    double ms_predictor_total = 0, ms_onnx_embedding = 0;

    // Working buffers.
    // language_sequence accumulates the full talker input; we only feed new tokens
    // after the first prefill to reuse KV cache.
    std::vector<float> language_sequence(initial_language_input.begin(), initial_language_input.end());
    int32_t language_length = initial_sequence_length;

    TtsGenerateResult result;
    result.codes_per_frame = 1 + residual_groups;

    std::unordered_set<int64_t> sampled_primary_codes;
    std::unordered_set<int64_t> sampled_all_codes;
    int64_t rng_state = params.rng_seed;
    int32_t produced_frames = 0;
    const float *latest_logits = nullptr;
    int32_t latest_vocab_size = 0;

    // --- Talker: initial prefill ---
    impl_->talker_ctx.clear_kv_cache();

    // Build position IDs for the initial sequence (base positions, then normalize).
    std::vector<int32_t> base_positions(language_length);
    std::iota(base_positions.begin(), base_positions.end(), 0);
    std::vector<int32_t> positions;
    auto pos_err = godot_llama::normalize_position_layout(base_positions, language_length, talker_pos_components, positions);
    if (pos_err) {
        return Error::inference_failed("Talker position layout failed: " + pos_err.message);
    }

    auto t0 = Clock::now();
    auto talker_err = impl_->talker_ctx.decode_embeddings(
        language_sequence, language_length, hidden_size, positions,
        talker_pos_components
    );
    ms_talker_prefill = Ms(Clock::now() - t0).count();
    if (!talker_err.ok()) {
        return Error::inference_failed("Talker initial prefill failed: " + talker_err.message);
    }

    // --- Frame generation loop ---
    for (int32_t frame = 0; frame < params.target_frames; ++frame) {
        if (cancel && cancel->is_cancelled()) {
            return Error::cancelled("TTS generation cancelled.");
        }

        // Get talker outputs (logits + hidden state).
        float *talker_logits = impl_->talker_ctx.get_logits(-1);
        // Use negative index to get the last output row directly, bypassing the
        // output_ids lookup that requires specific batch-position tracking.
        float *talker_hidden = impl_->talker_ctx.get_embeddings_ith(-1);
        if (!talker_logits || !talker_hidden) {
            return Error::inference_failed("Talker produced null outputs at frame " + std::to_string(frame));
        }
        latest_logits = talker_logits;
        latest_vocab_size = codebook_size;

        // Verify hidden state is finite.
        for (int32_t i = 0; i < hidden_size; ++i) {
            if (!std::isfinite(talker_hidden[i])) {
                return Error::inference_failed("Talker produced non-finite hidden state at frame " + std::to_string(frame));
            }
        }

        // Sample primary code.
        auto primary = sample_token(
            talker_logits, 0, codebook_size,
            params.do_sample, params.top_k, params.top_p, params.temperature,
            sampled_primary_codes, params.repetition_penalty, rng_state
        );
        if (primary.token < 0) {
            return Error::inference_failed("Failed to sample primary code at frame " + std::to_string(frame));
        }
        rng_state = primary.rng_state;
        sampled_primary_codes.insert(primary.token);
        sampled_all_codes.insert(primary.token);

        // Get codec embedding for the primary code.
        std::vector<int64_t> primary_ids = {primary.token};
        auto t_onnx = Clock::now();
        auto primary_emb_result = run_onnx_embedding(impl_->codec_embedding, primary_ids);
        if (!primary_emb_result.is_ok()) {
            return Error::inference_failed("Codec embedding at frame " + std::to_string(frame) + ": " + primary_emb_result.error_message());
        }
        auto &primary_embedding = primary_emb_result.value();
        ms_onnx_embedding += Ms(Clock::now() - t_onnx).count();
        if (static_cast<int32_t>(primary_embedding.size()) < hidden_size) {
            return Error::inference_failed("Codec embedding too small at frame " + std::to_string(frame));
        }

        // Store frame codes: primary first.
        std::vector<int64_t> frame_codes;
        frame_codes.reserve(result.codes_per_frame);
        frame_codes.push_back(primary.token);

        // Copy past_hidden locally (talker buffer may be reused).
        std::vector<float> past_hidden(talker_hidden, talker_hidden + hidden_size);

        // Sum for the final step embedding: start with primary.
        std::vector<float> summed_embedding(primary_embedding.begin(), primary_embedding.begin() + hidden_size);

        // --- Predictor residual loop ---
        auto t_pred_start = Clock::now();
        if (residual_groups > 0) {
            // Build initial predictor input: [past_hidden, primary_embedding].
            std::vector<float> predictor_input;
            predictor_input.reserve(2 * hidden_size);
            predictor_input.insert(predictor_input.end(), past_hidden.begin(), past_hidden.end());
            predictor_input.insert(predictor_input.end(), primary_embedding.begin(), primary_embedding.begin() + hidden_size);

            // Predictor: clear KV cache once per frame, prefill with 2-token sequence.
            impl_->predictor_ctx.clear_kv_cache();
            std::vector<int32_t> pred_base_pos = {0, 1};
            std::vector<int32_t> pred_positions;
            godot_llama::normalize_position_layout(pred_base_pos, 2, predictor_pos_components, pred_positions);

            auto pred_err = impl_->predictor_ctx.decode_embeddings(
                predictor_input, 2, hidden_size, pred_positions,
                predictor_pos_components
            );
            if (!pred_err.ok()) {
                return Error::inference_failed("Predictor prefill failed at frame " + std::to_string(frame));
            }

            int32_t pred_length = 2;

            for (int32_t group = 0; group < residual_groups; ++group) {
                if (cancel && cancel->is_cancelled()) {
                    return Error::cancelled("TTS generation cancelled during predictor.");
                }

                // Get predictor logits.
                float *pred_logits = impl_->predictor_ctx.get_logits(-1);
                if (!pred_logits) {
                    return Error::inference_failed("Predictor null logits at frame " + std::to_string(frame) + " group " + std::to_string(group));
                }

                // Logits are offset by group * codebook_size in the vocab.
                // sample_token returns absolute indices; convert to relative codec indices.
                const int32_t logit_offset = group * codebook_size;
                auto residual = sample_token(
                    pred_logits, logit_offset, codebook_size,
                    params.sub_do_sample, params.sub_top_k, params.sub_top_p, params.sub_temperature,
                    sampled_all_codes, params.repetition_penalty, rng_state
                );
                rng_state = residual.rng_state;
                // Convert from absolute logit index to relative codec index.
                int64_t residual_code = residual.token >= logit_offset ? residual.token - logit_offset : residual.token;
                if (residual_code < 0 || residual_code >= codebook_size) {
                    residual_code = primary.token;
                }
                sampled_all_codes.insert(residual.token);
                frame_codes.push_back(residual_code);

                // If not the last group, extend predictor with the new embedding.
                if (group + 1 < residual_groups) {
                    int64_t residual_ids[1] = {residual_code};
                    int64_t gen_steps[1] = {static_cast<int64_t>(group)};
                    auto t_emb = Clock::now();
                    auto res_emb_result = run_onnx_embedding(
                        impl_->predictor_embedding,
                        {residual_ids, 1}, gen_steps, 1
                    );
                    ms_onnx_embedding += Ms(Clock::now() - t_emb).count();
                    if (!res_emb_result.is_ok()) {
                        return Error::inference_failed("Predictor embedding at group " + std::to_string(group) + ": " + res_emb_result.error_message());
                    }
                    auto &residual_embedding = res_emb_result.value();
                    if (static_cast<int32_t>(residual_embedding.size()) < hidden_size) {
                        return Error::inference_failed("Predictor embedding too small at group " + std::to_string(group));
                    }

                    // Sum for final embedding.
                    for (int32_t i = 0; i < hidden_size; ++i) {
                        summed_embedding[i] += residual_embedding[i];
                    }

                    // Incremental predictor decode: only new token, KV cache preserved.
                    std::vector<int32_t> pred_next_base = {pred_length};
                    std::vector<int32_t> next_pos;
                    godot_llama::normalize_position_layout(pred_next_base, 1, predictor_pos_components, next_pos);
                    auto incr_err = impl_->predictor_ctx.decode_embeddings(
                        {residual_embedding.data(), static_cast<size_t>(hidden_size)},
                        1, hidden_size, next_pos, predictor_pos_components
                    );
                    if (!incr_err.ok()) {
                        return Error::inference_failed("Predictor incremental decode failed at group " + std::to_string(group));
                    }
                    pred_length++;
                } else {
                    // Last group: still need to get the embedding for the sum.
                    int64_t residual_ids[1] = {residual_code};
                    int64_t gen_steps[1] = {static_cast<int64_t>(group)};
                    auto t_emb2 = Clock::now();
                    auto last_emb_result = run_onnx_embedding(
                        impl_->predictor_embedding,
                        {residual_ids, 1}, gen_steps, 1
                    );
                    ms_onnx_embedding += Ms(Clock::now() - t_emb2).count();
                    if (last_emb_result.is_ok() && static_cast<int32_t>(last_emb_result.value().size()) >= hidden_size) {
                        auto &residual_embedding = last_emb_result.value();
                        for (int32_t i = 0; i < hidden_size; ++i) {
                            summed_embedding[i] += residual_embedding[i];
                        }
                    }
                }
            }
        }
        ms_predictor_total += Ms(Clock::now() - t_pred_start).count();

        // Append frame codes to result.
        result.codes.insert(result.codes.end(), frame_codes.begin(), frame_codes.end());
        produced_frames++;
        result.frame_count = produced_frames;

        // Check EOS before continuing.
        bool should_stop = (produced_frames >= params.target_frames);
        if (!should_stop && produced_frames >= params.min_frames_before_eos && latest_logits) {
            should_stop = should_stop_eos(latest_logits, params.eos_token_id, codebook_size, params.eos_logit_margin);
        }
        if (should_stop) break;

        // Compute the embedding for the next language step.
        // next_embedding = summed_embedding + trailing_text_or_pad
        int32_t next_gen_step = produced_frames - 1;
        std::vector<float> next_embedding(hidden_size);
        if (next_gen_step >= 0 && next_gen_step < trailing_text_length) {
            const float *trailing_row = trailing_text_hidden.data() + static_cast<size_t>(next_gen_step) * hidden_size;
            for (int32_t i = 0; i < hidden_size; ++i) {
                next_embedding[i] = summed_embedding[i] + trailing_row[i];
            }
        } else if (!tts_pad_embedding.empty()) {
            for (int32_t i = 0; i < hidden_size; ++i) {
                next_embedding[i] = summed_embedding[i] + tts_pad_embedding[i];
            }
        } else {
            next_embedding = summed_embedding;
        }

        // Incremental talker decode: feed only the new 1-token embedding, KV cache preserved.
        std::vector<int32_t> next_base_pos = {language_length};
        std::vector<int32_t> next_pos;
        godot_llama::normalize_position_layout(next_base_pos, 1, talker_pos_components, next_pos);
        auto t_talker = Clock::now();
        auto incr_talker_err = impl_->talker_ctx.decode_embeddings(
            {next_embedding.data(), static_cast<size_t>(hidden_size)},
            1, hidden_size, next_pos, talker_pos_components
        );
        ms_talker_decode += Ms(Clock::now() - t_talker).count();
        if (!incr_talker_err.ok()) {
            return Error::inference_failed("Talker incremental decode failed at frame " + std::to_string(frame));
        }
        language_length++;
    }

    double ms_total = Ms(Clock::now() - t_gen_start).count();
    double ms_other = ms_total - ms_talker_prefill - ms_talker_decode - ms_predictor_total;
    std::fprintf(stderr,
        "[gotst-tts] frames=%d  total=%.0fms  talker_prefill=%.0fms  talker_decode=%.0fms"
        "  predictor=%.0fms  onnx_emb=%.0fms  other=%.0fms\n",
        produced_frames, ms_total, ms_talker_prefill, ms_talker_decode,
        ms_predictor_total, ms_onnx_embedding, ms_other);

    return result;
}

Result<TtsGenerateResult> TtsCodeGenerator::generate_streaming(
    std::span<const float> initial_language_input,
    int32_t initial_sequence_length,
    std::span<const float> trailing_text_hidden,
    int32_t trailing_text_length,
    std::span<const float> tts_pad_embedding,
    const TtsSamplingConfig &params,
    int32_t chunk_frames,
    FrameChunkCallback on_chunk,
    CancellationToken *cancel
) {
    if (!impl_->loaded) {
        return Error::invalid_state("TTS code generator is not loaded.");
    }

    const int32_t hidden_size = params.hidden_size;
    const int32_t codebook_size = params.codebook_size;
    const int32_t residual_groups = params.residual_groups;
    const int32_t talker_pos_components = impl_->session_config.talker_position_components;
    const int32_t predictor_pos_components = impl_->session_config.predictor_position_components;

    if (initial_language_input.empty() || initial_sequence_length <= 0 || hidden_size <= 0) {
        return Error::invalid_argument("Initial language input is empty.");
    }
    if (chunk_frames <= 0) chunk_frames = 6;

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    const auto t_gen_start = Clock::now();
    double ms_talker_prefill = 0, ms_talker_decode = 0;
    double ms_predictor_total = 0, ms_onnx_embedding = 0;

    std::vector<float> language_sequence(initial_language_input.begin(), initial_language_input.end());
    int32_t language_length = initial_sequence_length;

    TtsGenerateResult result;
    result.codes_per_frame = 1 + residual_groups;

    std::unordered_set<int64_t> sampled_primary_codes;
    std::unordered_set<int64_t> sampled_all_codes;
    int64_t rng_state = params.rng_seed;
    int32_t produced_frames = 0;
    const float *latest_logits = nullptr;
    int32_t latest_vocab_size = 0;

    // Chunk accumulator.
    std::vector<int64_t> chunk_codes;
    int32_t chunk_frame_count = 0;

    // --- Talker: initial prefill ---
    impl_->talker_ctx.clear_kv_cache();

    std::vector<int32_t> base_positions(language_length);
    std::iota(base_positions.begin(), base_positions.end(), 0);
    std::vector<int32_t> positions;
    auto pos_err = godot_llama::normalize_position_layout(base_positions, language_length, talker_pos_components, positions);
    if (pos_err) {
        return Error::inference_failed("Talker position layout failed: " + pos_err.message);
    }

    auto t0 = Clock::now();
    auto talker_err = impl_->talker_ctx.decode_embeddings(
        language_sequence, language_length, hidden_size, positions,
        talker_pos_components
    );
    ms_talker_prefill = Ms(Clock::now() - t0).count();
    if (!talker_err.ok()) {
        return Error::inference_failed("Talker initial prefill failed: " + talker_err.message);
    }

    // --- Frame generation loop ---
    for (int32_t frame = 0; frame < params.target_frames; ++frame) {
        if (cancel && cancel->is_cancelled()) {
            return Error::cancelled("TTS generation cancelled.");
        }

        float *talker_logits = impl_->talker_ctx.get_logits(-1);
        float *talker_hidden = impl_->talker_ctx.get_embeddings_ith(-1);
        if (!talker_logits || !talker_hidden) {
            return Error::inference_failed("Talker produced null outputs at frame " + std::to_string(frame));
        }
        latest_logits = talker_logits;
        latest_vocab_size = codebook_size;

        for (int32_t i = 0; i < hidden_size; ++i) {
            if (!std::isfinite(talker_hidden[i])) {
                return Error::inference_failed("Talker produced non-finite hidden state at frame " + std::to_string(frame));
            }
        }

        auto primary = sample_token(
            talker_logits, 0, codebook_size,
            params.do_sample, params.top_k, params.top_p, params.temperature,
            sampled_primary_codes, params.repetition_penalty, rng_state
        );
        if (primary.token < 0) {
            return Error::inference_failed("Failed to sample primary code at frame " + std::to_string(frame));
        }
        rng_state = primary.rng_state;
        sampled_primary_codes.insert(primary.token);
        sampled_all_codes.insert(primary.token);

        auto t_onnx = Clock::now();
        std::vector<int64_t> primary_ids = {primary.token};
        auto primary_emb_result = run_onnx_embedding(impl_->codec_embedding, primary_ids);
        if (!primary_emb_result.is_ok()) {
            return Error::inference_failed("Codec embedding at frame " + std::to_string(frame) + ": " + primary_emb_result.error_message());
        }
        auto &primary_embedding = primary_emb_result.value();
        ms_onnx_embedding += Ms(Clock::now() - t_onnx).count();
        if (static_cast<int32_t>(primary_embedding.size()) < hidden_size) {
            return Error::inference_failed("Codec embedding too small at frame " + std::to_string(frame));
        }

        std::vector<int64_t> frame_codes;
        frame_codes.reserve(result.codes_per_frame);
        frame_codes.push_back(primary.token);

        std::vector<float> past_hidden(talker_hidden, talker_hidden + hidden_size);
        std::vector<float> summed_embedding(primary_embedding.begin(), primary_embedding.begin() + hidden_size);

        // --- Predictor residual loop ---
        auto t_pred_start = Clock::now();
        if (residual_groups > 0) {
            std::vector<float> predictor_input;
            predictor_input.reserve(2 * hidden_size);
            predictor_input.insert(predictor_input.end(), past_hidden.begin(), past_hidden.end());
            predictor_input.insert(predictor_input.end(), primary_embedding.begin(), primary_embedding.begin() + hidden_size);

            impl_->predictor_ctx.clear_kv_cache();
            std::vector<int32_t> pred_base_pos = {0, 1};
            std::vector<int32_t> pred_positions;
            godot_llama::normalize_position_layout(pred_base_pos, 2, predictor_pos_components, pred_positions);

            auto pred_err = impl_->predictor_ctx.decode_embeddings(
                predictor_input, 2, hidden_size, pred_positions,
                predictor_pos_components
            );
            if (!pred_err.ok()) {
                return Error::inference_failed("Predictor prefill failed at frame " + std::to_string(frame));
            }

            int32_t pred_length = 2;

            for (int32_t group = 0; group < residual_groups; ++group) {
                if (cancel && cancel->is_cancelled()) {
                    return Error::cancelled("TTS generation cancelled during predictor.");
                }

                float *pred_logits = impl_->predictor_ctx.get_logits(-1);
                if (!pred_logits) {
                    return Error::inference_failed("Predictor null logits at frame " + std::to_string(frame) + " group " + std::to_string(group));
                }

                const int32_t logit_offset = group * codebook_size;
                auto residual = sample_token(
                    pred_logits, logit_offset, codebook_size,
                    params.sub_do_sample, params.sub_top_k, params.sub_top_p, params.sub_temperature,
                    sampled_all_codes, params.repetition_penalty, rng_state
                );
                rng_state = residual.rng_state;
                int64_t residual_code = residual.token >= logit_offset ? residual.token - logit_offset : residual.token;
                if (residual_code < 0 || residual_code >= codebook_size) {
                    residual_code = primary.token;
                }
                sampled_all_codes.insert(residual.token);
                frame_codes.push_back(residual_code);

                if (group + 1 < residual_groups) {
                    int64_t residual_ids[1] = {residual_code};
                    int64_t gen_steps[1] = {static_cast<int64_t>(group)};
                    auto t_emb = Clock::now();
                    auto res_emb_result = run_onnx_embedding(
                        impl_->predictor_embedding,
                        {residual_ids, 1}, gen_steps, 1
                    );
                    ms_onnx_embedding += Ms(Clock::now() - t_emb).count();
                    if (!res_emb_result.is_ok()) {
                        return Error::inference_failed("Predictor embedding at group " + std::to_string(group) + ": " + res_emb_result.error_message());
                    }
                    auto &residual_embedding = res_emb_result.value();
                    if (static_cast<int32_t>(residual_embedding.size()) < hidden_size) {
                        return Error::inference_failed("Predictor embedding too small at group " + std::to_string(group));
                    }

                    for (int32_t i = 0; i < hidden_size; ++i) {
                        summed_embedding[i] += residual_embedding[i];
                    }

                    std::vector<int32_t> pred_next_base = {pred_length};
                    std::vector<int32_t> next_pos;
                    godot_llama::normalize_position_layout(pred_next_base, 1, predictor_pos_components, next_pos);
                    auto incr_err = impl_->predictor_ctx.decode_embeddings(
                        {residual_embedding.data(), static_cast<size_t>(hidden_size)},
                        1, hidden_size, next_pos, predictor_pos_components
                    );
                    if (!incr_err.ok()) {
                        return Error::inference_failed("Predictor incremental decode failed at group " + std::to_string(group));
                    }
                    pred_length++;
                } else {
                    int64_t residual_ids[1] = {residual_code};
                    int64_t gen_steps[1] = {static_cast<int64_t>(group)};
                    auto t_emb2 = Clock::now();
                    auto last_emb_result = run_onnx_embedding(
                        impl_->predictor_embedding,
                        {residual_ids, 1}, gen_steps, 1
                    );
                    ms_onnx_embedding += Ms(Clock::now() - t_emb2).count();
                    if (last_emb_result.is_ok() && static_cast<int32_t>(last_emb_result.value().size()) >= hidden_size) {
                        auto &residual_embedding = last_emb_result.value();
                        for (int32_t i = 0; i < hidden_size; ++i) {
                            summed_embedding[i] += residual_embedding[i];
                        }
                    }
                }
            }
        }
        ms_predictor_total += Ms(Clock::now() - t_pred_start).count();

        // Append frame codes to result and chunk buffer.
        result.codes.insert(result.codes.end(), frame_codes.begin(), frame_codes.end());
        chunk_codes.insert(chunk_codes.end(), frame_codes.begin(), frame_codes.end());
        produced_frames++;
        chunk_frame_count++;
        result.frame_count = produced_frames;

        // Check EOS before continuing.
        bool should_stop = (produced_frames >= params.target_frames);
        if (!should_stop && produced_frames >= params.min_frames_before_eos && latest_logits) {
            should_stop = should_stop_eos(latest_logits, params.eos_token_id, codebook_size, params.eos_logit_margin);
        }

        // Emit chunk if we've accumulated enough frames or this is the last frame.
        if (chunk_frame_count >= chunk_frames || should_stop) {
            if (on_chunk) {
                TtsFrameChunk chunk;
                chunk.codes = std::move(chunk_codes);
                chunk.frame_count = chunk_frame_count;
                chunk.codes_per_frame = result.codes_per_frame;
                chunk.is_final = should_stop;
                on_chunk(std::move(chunk));
            }
            chunk_codes.clear();
            chunk_frame_count = 0;
        }

        if (should_stop) break;

        // Compute the embedding for the next language step.
        int32_t next_gen_step = produced_frames - 1;
        std::vector<float> next_embedding(hidden_size);
        if (next_gen_step >= 0 && next_gen_step < trailing_text_length) {
            const float *trailing_row = trailing_text_hidden.data() + static_cast<size_t>(next_gen_step) * hidden_size;
            for (int32_t i = 0; i < hidden_size; ++i) {
                next_embedding[i] = summed_embedding[i] + trailing_row[i];
            }
        } else if (!tts_pad_embedding.empty()) {
            for (int32_t i = 0; i < hidden_size; ++i) {
                next_embedding[i] = summed_embedding[i] + tts_pad_embedding[i];
            }
        } else {
            next_embedding = summed_embedding;
        }

        std::vector<int32_t> next_base_pos = {language_length};
        std::vector<int32_t> next_pos;
        godot_llama::normalize_position_layout(next_base_pos, 1, talker_pos_components, next_pos);
        auto t_talker = Clock::now();
        auto incr_talker_err = impl_->talker_ctx.decode_embeddings(
            {next_embedding.data(), static_cast<size_t>(hidden_size)},
            1, hidden_size, next_pos, talker_pos_components
        );
        ms_talker_decode += Ms(Clock::now() - t_talker).count();
        if (!incr_talker_err.ok()) {
            return Error::inference_failed("Talker incremental decode failed at frame " + std::to_string(frame));
        }
        language_length++;
    }

    // Emit any remaining buffered frames as a final chunk.
    if (chunk_frame_count > 0 && on_chunk) {
        TtsFrameChunk chunk;
        chunk.codes = std::move(chunk_codes);
        chunk.frame_count = chunk_frame_count;
        chunk.codes_per_frame = result.codes_per_frame;
        chunk.is_final = true;
        on_chunk(std::move(chunk));
    }

    double ms_total = Ms(Clock::now() - t_gen_start).count();
    double ms_other = ms_total - ms_talker_prefill - ms_talker_decode - ms_predictor_total;
    std::fprintf(stderr,
        "[gotst-tts-stream] frames=%d  total=%.0fms  talker_prefill=%.0fms  talker_decode=%.0fms"
        "  predictor=%.0fms  onnx_emb=%.0fms  other=%.0fms\n",
        produced_frames, ms_total, ms_talker_prefill, ms_talker_decode,
        ms_predictor_total, ms_onnx_embedding, ms_other);

    return result;
}

} // namespace gotst
