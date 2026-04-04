#include "gotst/core/asr_token_decoder.hpp"

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
#include <limits>
#include <numeric>
#include <vector>

namespace gotst {

struct AsrTokenDecoder::Impl {
    std::shared_ptr<godot_llama::LlamaModelHandle> thinker_model;
    godot_llama::LlamaContextHandle thinker_ctx;
    gonx::InferenceSession embedding;
    AsrSessionConfig session_config;
    bool loaded = false;
};

AsrTokenDecoder::AsrTokenDecoder() : impl_(std::make_unique<Impl>()) {}
AsrTokenDecoder::~AsrTokenDecoder() = default;
AsrTokenDecoder::AsrTokenDecoder(AsrTokenDecoder &&) noexcept = default;
AsrTokenDecoder &AsrTokenDecoder::operator=(AsrTokenDecoder &&) noexcept = default;

bool AsrTokenDecoder::is_loaded() const {
    return impl_ && impl_->loaded;
}

Result<void> AsrTokenDecoder::load(const AsrModelPaths &paths, const AsrSessionConfig &config) {
    impl_->loaded = false;
    impl_->session_config = config;

    godot_llama::ModelConfig model_cfg;
    model_cfg.model_path = paths.thinker_gguf_path;
    model_cfg.n_ctx = config.n_ctx;
    model_cfg.n_batch = config.n_batch;
    model_cfg.n_threads = config.n_threads;
    model_cfg.n_gpu_layers = config.n_gpu_layers;
    model_cfg.use_mmap = config.use_mmap;
    model_cfg.use_mlock = config.use_mlock;
    model_cfg.flash_attn_type = config.flash_attn_type;
    model_cfg.type_k = config.type_k;
    model_cfg.type_v = config.type_v;
    model_cfg.embeddings_enabled = false;

    auto load_err = godot_llama::LlamaModelHandle::load(model_cfg, impl_->thinker_model);
    if (!load_err.ok()) {
        return Error::model_not_loaded("Failed to load thinker GGUF: " + load_err.message);
    }

    auto ctx_err = godot_llama::LlamaContextHandle::create(impl_->thinker_model, model_cfg, impl_->thinker_ctx);
    if (!ctx_err.ok()) {
        return Error::model_not_loaded("Failed to create thinker context: " + ctx_err.message);
    }

    gonx::SessionConfig onnx_cfg;
    onnx_cfg.optimization_level = 99;
    auto emb_status = impl_->embedding.load(paths.embedding_onnx_path, onnx_cfg);
    if (emb_status.has_error()) {
        return Error::model_not_loaded("Failed to load ASR embedding ONNX: " + emb_status.error().message);
    }

    impl_->loaded = true;
    return {};
}

Result<AsrDecodeResult> AsrTokenDecoder::decode(
    std::span<const float> prompt_embeddings,
    int32_t prompt_length,
    const AsrDecodeParams &params,
    CancellationToken *cancel
) {
    if (!impl_->loaded) {
        return Error::invalid_state("ASR token decoder is not loaded.");
    }

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    const auto t_gen_start = Clock::now();
    double ms_prefill = 0, ms_decode = 0, ms_onnx = 0;

    const int32_t hidden_size = params.hidden_size;
    const int32_t pos_components = impl_->session_config.position_components;

    if (prompt_embeddings.empty() || prompt_length <= 0 || hidden_size <= 0) {
        return Error::invalid_argument("Prompt embeddings are empty.");
    }

    // Clear KV cache and prefill the entire prompt.
    impl_->thinker_ctx.clear_kv_cache();

    std::vector<int32_t> base_positions(prompt_length);
    std::iota(base_positions.begin(), base_positions.end(), 0);
    std::vector<int32_t> positions;
    auto pos_err = godot_llama::normalize_position_layout(base_positions, prompt_length, pos_components, positions);
    if (pos_err) {
        return Error::inference_failed("ASR position layout failed: " + pos_err.message);
    }

    auto t0 = Clock::now();
    auto prefill_err = impl_->thinker_ctx.decode_embeddings(
        prompt_embeddings, prompt_length, hidden_size, positions,
        pos_components
    );
    ms_prefill = Ms(Clock::now() - t0).count();
    if (!prefill_err.ok()) {
        return Error::inference_failed("Thinker prefill failed: " + prefill_err.message);
    }

    AsrDecodeResult result;
    int32_t current_length = prompt_length;

    for (int32_t step = 0; step < params.max_tokens; ++step) {
        if (cancel && cancel->is_cancelled()) {
            return Error::cancelled("ASR decoding cancelled.");
        }

        // Get logits from last position.
        float *logits = impl_->thinker_ctx.get_logits(-1);
        if (!logits) {
            return Error::inference_failed("Thinker produced null logits at step " + std::to_string(step));
        }

        // Argmax over full vocabulary.
        int32_t best_token = 0;
        float best_value = -std::numeric_limits<float>::infinity();
        for (int32_t i = 0; i < params.vocab_size; ++i) {
            if (logits[i] > best_value) {
                best_value = logits[i];
                best_token = i;
            }
        }

        // Check for EOS.
        if (best_token == params.eos_token_id) {
            result.token_ids.push_back(best_token);
            break;
        }

        result.token_ids.push_back(best_token);

        // Get embedding for the decoded token via ONNX.
        auto t_emb = Clock::now();
        std::vector<int64_t> token_ids = {static_cast<int64_t>(best_token)};
        std::vector<int64_t> id_shape = {1, 1};
        auto id_tensor = gonx::create_int64_tensor(token_ids, id_shape);
        if (id_tensor.has_error()) {
            return Error::inference_failed("Failed to create token tensor at step " + std::to_string(step));
        }

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(id_tensor).value());
        auto run_result = impl_->embedding.run(inputs);
        if (run_result.has_error()) {
            return Error::inference_failed("Embedding ONNX failed at step " + std::to_string(step));
        }

        auto &outputs = run_result.value();
        if (outputs.empty()) {
            return Error::inference_failed("Embedding ONNX returned empty at step " + std::to_string(step));
        }

        const float *emb_data = outputs[0].GetTensorData<float>();
        auto emb_shape = gonx::get_tensor_shape(outputs[0]);
        int64_t emb_total = 1;
        for (auto d : emb_shape) emb_total *= d;
        if (emb_total < hidden_size) {
            return Error::inference_failed("Embedding too small at step " + std::to_string(step));
        }

        ms_onnx += Ms(Clock::now() - t_emb).count();

        // Incremental thinker decode: feed only the new token embedding.
        auto t_dec = Clock::now();
        std::vector<int32_t> next_base_pos = {current_length};
        std::vector<int32_t> next_pos;
        godot_llama::normalize_position_layout(next_base_pos, 1, pos_components, next_pos);
        auto incr_err = impl_->thinker_ctx.decode_embeddings(
            {emb_data, static_cast<size_t>(hidden_size)},
            1, hidden_size, next_pos, pos_components
        );
        ms_decode += Ms(Clock::now() - t_dec).count();
        if (!incr_err.ok()) {
            return Error::inference_failed("Thinker incremental decode failed at step " + std::to_string(step));
        }
        current_length++;
    }

    double ms_total = Ms(Clock::now() - t_gen_start).count();
    int32_t n_tokens = static_cast<int32_t>(result.token_ids.size());
    std::fprintf(stderr,
        "[gotst-asr] tokens=%d  total=%.0fms  prefill=%.0fms  decode=%.0fms  onnx_emb=%.0fms\n",
        n_tokens, ms_total, ms_prefill, ms_decode, ms_onnx);

    return result;
}

} // namespace gotst
