#include "gotst/core/tts_code_generator.hpp"
#include "core/onnx_embedding_utils.hpp"
#include "core/sampling_utils.hpp"

#include <godot_llama/llama_context_handle.hpp>
#include <godot_llama/llama_model_handle.hpp>
#include <godot_llama/llama_params.hpp>
#include <godot_llama/llama_position_layout.hpp>
#include <gonx/core/session.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <array>
#include <span>
#include <string>
#include <vector>

namespace gotst {
namespace {

bool has_sampled_token(const std::vector<uint8_t> &sampled_tokens, int64_t token_index) {
    return token_index >= 0 &&
           token_index < static_cast<int64_t>(sampled_tokens.size()) &&
           sampled_tokens[static_cast<size_t>(token_index)] != 0;
}

void mark_sampled_token(std::vector<uint8_t> &sampled_tokens, int64_t token_index) {
    if(token_index >= 0 && token_index < static_cast<int64_t>(sampled_tokens.size())) {
        sampled_tokens[static_cast<size_t>(token_index)] = 1;
    }
}

bool should_log_tts_timing() {
    const char *value = std::getenv("GOTST_TTS_TIMING");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
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

struct F32EmbeddingTable {
    std::vector<float> values;
    std::string path;

    [[nodiscard]] bool is_valid() const {
        return !values.empty();
    }

    [[nodiscard]] const float *row(int64_t row_index, int32_t hidden_size) const {
        if(row_index < 0 || hidden_size <= 0 || values.empty()) {
            return nullptr;
        }
        const size_t width = static_cast<size_t>(hidden_size);
        if(values.size() % width != 0) {
            return nullptr;
        }
        const size_t row_count = values.size() / width;
        const size_t index = static_cast<size_t>(row_index);
        if(index >= row_count) {
            return nullptr;
        }
        return values.data() + index * width;
    }
};

std::string f32_table_path_for_onnx(const std::string &onnx_path) {
    constexpr const char *suffix = ".onnx";
    if(onnx_path.size() >= std::strlen(suffix) &&
       onnx_path.compare(onnx_path.size() - std::strlen(suffix), std::strlen(suffix), suffix) == 0) {
        return onnx_path.substr(0, onnx_path.size() - std::strlen(suffix)) + ".f32.bin";
    }
    return onnx_path + ".f32.bin";
}

Result<F32EmbeddingTable> load_f32_embedding_table(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if(!input) {
        return Error::not_found("Embedding table not found: " + path);
    }

    const std::streamoff size = input.tellg();
    if(size <= 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
        return Error::shape_mismatch("Embedding table has invalid byte size: " + path);
    }
    input.seekg(0, std::ios::beg);

    F32EmbeddingTable table;
    table.path = path;
    table.values.resize(static_cast<size_t>(size) / sizeof(float));
    if(!input.read(reinterpret_cast<char *>(table.values.data()), size)) {
        return Error::io_error("Failed to read embedding table: " + path);
    }
    return table;
}

Error copy_embedding_vector(const F32EmbeddingTable &table,
                            gonx::InferenceSession &session,
                            detail::SingleTokenEmbeddingRunScratch &scratch,
                            int64_t token_id,
                            const int64_t *generation_step,
                            int32_t codebook_size,
                            int32_t hidden_size,
                            std::vector<float> &destination,
                            double &ms_onnx_embedding) {
    if(hidden_size <= 0) {
        return Error::invalid_argument("Embedding hidden size must be positive.");
    }
    if(destination.size() < static_cast<size_t>(hidden_size)) {
        destination.resize(static_cast<size_t>(hidden_size));
    }

    if(table.is_valid()) {
        const int64_t row_index = generation_step != nullptr
            ? (*generation_step * static_cast<int64_t>(codebook_size)) + token_id
            : token_id;
        const float *row = table.row(row_index, hidden_size);
        if(row == nullptr) {
            return Error::shape_mismatch("Embedding table row is out of range.");
        }
        std::copy_n(row, static_cast<size_t>(hidden_size), destination.begin());
        return Error::ok();
    }

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    const auto t_onnx = Clock::now();
    auto run_result = detail::run_single_token_float_embedding(
        session,
        scratch,
        token_id,
        generation_step
    );
    ms_onnx_embedding += Ms(Clock::now() - t_onnx).count();
    if(!run_result.is_ok()) {
        return Error::inference_failed(run_result.error_message());
    }
    if(run_result.value().values.size() < static_cast<size_t>(hidden_size)) {
        return Error::shape_mismatch("ONNX embedding returned too few values.");
    }
    std::copy_n(run_result.value().values.data(), static_cast<size_t>(hidden_size), destination.begin());
    return Error::ok();
}

} // namespace

struct TtsCodeGenerator::Impl {
    std::shared_ptr<godot_llama::LlamaModelHandle> talker_model;
    godot_llama::LlamaContextHandle talker_ctx;
    std::shared_ptr<godot_llama::LlamaModelHandle> predictor_model;
    godot_llama::LlamaContextHandle predictor_ctx;
    gonx::InferenceSession codec_embedding;
    gonx::InferenceSession predictor_embedding;
    F32EmbeddingTable codec_embedding_table;
    F32EmbeddingTable predictor_embedding_table;
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
    predictor_cfg.n_gpu_layers =
        config.predictor_n_gpu_layers >= -1 ? config.predictor_n_gpu_layers : config.n_gpu_layers;
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

    // Load embedding tables when available; fall back to ONNX sessions for
    // portability when converter-side raw tables have not been generated.
    gonx::SessionConfig onnx_cfg;
    onnx_cfg.optimization_level = 99;

    impl_->codec_embedding_table = {};
    auto codec_table = load_f32_embedding_table(f32_table_path_for_onnx(paths.codec_embedding_onnx_path));
    if(codec_table.is_ok()) {
        impl_->codec_embedding_table = std::move(codec_table.value());
        std::fprintf(stderr, "[gotst-tts] loaded codec embedding table: %s\n",
                     impl_->codec_embedding_table.path.c_str());
    } else {
        if(codec_table.error_code() != ErrorCode::NotFound) {
            std::fprintf(stderr, "[gotst-tts] codec embedding table unavailable: %s\n",
                         codec_table.error_message().c_str());
        }
        auto codec_status = impl_->codec_embedding.load(paths.codec_embedding_onnx_path, onnx_cfg);
        if (codec_status.has_error()) {
            return Error::model_not_loaded("Failed to load codec embedding ONNX: " + codec_status.error().message);
        }
    }

    impl_->predictor_embedding_table = {};
    auto predictor_table = load_f32_embedding_table(f32_table_path_for_onnx(paths.predictor_embedding_onnx_path));
    if(predictor_table.is_ok()) {
        impl_->predictor_embedding_table = std::move(predictor_table.value());
        std::fprintf(stderr, "[gotst-tts] loaded predictor embedding table: %s\n",
                     impl_->predictor_embedding_table.path.c_str());
    } else {
        if(predictor_table.error_code() != ErrorCode::NotFound) {
            std::fprintf(stderr, "[gotst-tts] predictor embedding table unavailable: %s\n",
                         predictor_table.error_message().c_str());
        }
        auto pred_emb_status = impl_->predictor_embedding.load(paths.predictor_embedding_onnx_path, onnx_cfg);
        if (pred_emb_status.has_error()) {
            return Error::model_not_loaded("Failed to load predictor embedding ONNX: " + pred_emb_status.error().message);
        }
    }

    impl_->loaded = true;
    return {};
}

Result<TtsGenerateResult> TtsCodeGenerator::run_generation_impl(std::span<const float> initial_language_input,
                                                                int32_t initial_sequence_length,
                                                                std::span<const float> trailing_text_hidden,
                                                                int32_t trailing_text_length,
                                                                std::span<const float> tts_pad_embedding,
                                                                const TtsSamplingConfig &params,
                                                                int32_t chunk_frames,
                                                                FrameChunkCallback on_chunk,
                                                                CancellationToken *cancel,
                                                                const char *log_tag) {
    const int32_t hidden_size = params.hidden_size;
    const int32_t codebook_size = params.codebook_size;
    const int32_t residual_groups = params.residual_groups;
    const int32_t talker_pos_components = impl_->session_config.talker_position_components;
    const int32_t predictor_pos_components = impl_->session_config.predictor_position_components;

    if(initial_language_input.empty() || initial_sequence_length <= 0 || hidden_size <= 0) {
        return Error::invalid_argument("Initial language input is empty.");
    }
    if(codebook_size <= 0 || residual_groups < 0 || params.target_frames < 0) {
        return Error::invalid_argument("Invalid TTS sampling dimensions.");
    }

    const bool streaming = static_cast<bool>(on_chunk);
    if(streaming && chunk_frames <= 0) {
        chunk_frames = 6;
    }

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    const auto t_gen_start = Clock::now();
    double ms_talker_prefill = 0.0;
    double ms_talker_decode = 0.0;
    double ms_predictor_total = 0.0;
    double ms_predictor_clear = 0.0;
    double ms_predictor_prefill = 0.0;
    double ms_predictor_sample = 0.0;
    double ms_predictor_incremental = 0.0;
    double ms_onnx_embedding = 0.0;

    int32_t language_length = initial_sequence_length;

    TtsGenerateResult result;
    result.codes_per_frame = 1 + residual_groups;
    result.codes.reserve(
        static_cast<size_t>(std::max(params.target_frames, 0)) *
        static_cast<size_t>(std::max(result.codes_per_frame, 0))
    );

    std::vector<uint8_t> sampled_primary_codes(static_cast<size_t>(codebook_size), 0);
    detail::SamplingScratch sampling_scratch;
    detail::SingleTokenEmbeddingRunScratch onnx_scratch;

    std::vector<int64_t> chunk_codes;
    const size_t chunk_capacity =
        streaming ? static_cast<size_t>(chunk_frames) * static_cast<size_t>(result.codes_per_frame) : 0;
    if(streaming) {
        chunk_codes.reserve(chunk_capacity);
    }
    int32_t chunk_frame_count = 0;

    std::vector<float> predictor_input;
    if(residual_groups > 0) {
        predictor_input.resize(static_cast<size_t>(2 * hidden_size));
    }
    std::vector<float> primary_embedding_buffer(static_cast<size_t>(hidden_size));
    std::vector<float> residual_embedding_buffer(static_cast<size_t>(hidden_size));
    std::vector<float> summed_embedding(static_cast<size_t>(hidden_size));
    std::vector<float> next_embedding(static_cast<size_t>(hidden_size));

    std::vector<int32_t> talker_positions;
    talker_positions.reserve(static_cast<size_t>(language_length) * static_cast<size_t>(talker_pos_components));
    std::vector<int32_t> predictor_prefill_positions;
    predictor_prefill_positions.reserve(static_cast<size_t>(2 * std::max(predictor_pos_components, 0)));
    std::vector<int32_t> next_talker_positions;
    next_talker_positions.reserve(static_cast<size_t>(std::max(talker_pos_components, 0)));
    std::vector<int32_t> next_predictor_positions;
    next_predictor_positions.reserve(static_cast<size_t>(std::max(predictor_pos_components, 0)));
    std::array<int32_t, 1> next_position_base = {0};

    impl_->talker_ctx.clear_kv_cache();

    std::vector<int32_t> base_positions(static_cast<size_t>(language_length));
    std::iota(base_positions.begin(), base_positions.end(), 0);
    auto talker_pos_err = godot_llama::normalize_position_layout(
        base_positions,
        language_length,
        talker_pos_components,
        talker_positions
    );
    if(talker_pos_err) {
        return Error::inference_failed("Talker position layout failed: " + talker_pos_err.message);
    }

    auto t0 = Clock::now();
    auto talker_err = impl_->talker_ctx.decode_embeddings(
        initial_language_input,
        language_length,
        hidden_size,
        talker_positions,
        talker_pos_components
    );
    ms_talker_prefill = Ms(Clock::now() - t0).count();
    if(!talker_err.ok()) {
        return Error::inference_failed("Talker initial prefill failed: " + talker_err.message);
    }

    if(residual_groups > 0) {
        const std::array<int32_t, 2> predictor_prefill_base = {0, 1};
        auto predictor_pos_err = godot_llama::normalize_position_layout(
            predictor_prefill_base,
            2,
            predictor_pos_components,
            predictor_prefill_positions
        );
        if(predictor_pos_err) {
            return Error::inference_failed("Predictor position layout failed: " + predictor_pos_err.message);
        }
    }

    int64_t rng_state = params.rng_seed;
    int32_t produced_frames = 0;
    const float *latest_logits = nullptr;
    const size_t hidden_bytes = static_cast<size_t>(hidden_size) * sizeof(float);

    for(int32_t frame = 0; frame < params.target_frames; ++frame) {
        if(cancel && cancel->is_cancelled()) {
            return Error::cancelled("TTS generation cancelled.");
        }

        float *talker_logits = impl_->talker_ctx.get_logits(-1);
        float *talker_hidden = impl_->talker_ctx.get_embeddings_ith(-1);
        if(!talker_logits || !talker_hidden) {
            return Error::inference_failed("Talker produced null outputs at frame " + std::to_string(frame));
        }
        latest_logits = talker_logits;

        for(int32_t index = 0; index < hidden_size; ++index) {
            if(!std::isfinite(talker_hidden[index])) {
                return Error::inference_failed("Talker produced non-finite hidden state at frame " + std::to_string(frame));
            }
        }

        auto primary = detail::sample_token(
            talker_logits,
            0,
            codebook_size,
            params.do_sample,
            params.top_k,
            params.top_p,
            params.temperature,
            [&sampled_primary_codes](int64_t token_index) {
                return has_sampled_token(sampled_primary_codes, token_index);
            },
            params.repetition_penalty,
            rng_state,
            sampling_scratch
        );
        if(primary.token < 0) {
            return Error::inference_failed("Failed to sample primary code at frame " + std::to_string(frame));
        }
        rng_state = primary.rng_state;
        mark_sampled_token(sampled_primary_codes, primary.token);
        result.codes.push_back(primary.token);
        if(streaming) {
            chunk_codes.push_back(primary.token);
        }

        auto primary_embedding_err = copy_embedding_vector(
            impl_->codec_embedding_table,
            impl_->codec_embedding,
            onnx_scratch,
            primary.token,
            nullptr,
            codebook_size,
            hidden_size,
            primary_embedding_buffer,
            ms_onnx_embedding
        );
        if(!primary_embedding_err.is_ok()) {
            return Error::inference_failed(
                "Codec embedding at frame " + std::to_string(frame) + ": " + primary_embedding_err.message
            );
        }
        std::copy_n(primary_embedding_buffer.data(), static_cast<size_t>(hidden_size), summed_embedding.begin());

        auto t_pred_start = Clock::now();
        if(residual_groups > 0) {
            std::memcpy(predictor_input.data(), talker_hidden, hidden_bytes);
            std::memcpy(predictor_input.data() + hidden_size, primary_embedding_buffer.data(), hidden_bytes);

            auto t_pred_clear = Clock::now();
            impl_->predictor_ctx.clear_kv_cache();
            ms_predictor_clear += Ms(Clock::now() - t_pred_clear).count();

            auto t_pred_prefill = Clock::now();
            auto predictor_err = impl_->predictor_ctx.decode_embeddings(
                predictor_input,
                2,
                hidden_size,
                predictor_prefill_positions,
                predictor_pos_components
            );
            ms_predictor_prefill += Ms(Clock::now() - t_pred_prefill).count();
            if(!predictor_err.ok()) {
                return Error::inference_failed("Predictor prefill failed at frame " + std::to_string(frame));
            }

            int32_t pred_length = 2;
            for(int32_t group = 0; group < residual_groups; ++group) {
                if(cancel && cancel->is_cancelled()) {
                    return Error::cancelled("TTS generation cancelled during predictor.");
                }

                float *pred_logits = impl_->predictor_ctx.get_logits(-1);
                if(!pred_logits) {
                    return Error::inference_failed(
                        "Predictor null logits at frame " + std::to_string(frame) + " group " + std::to_string(group)
                    );
                }

                const int32_t logit_offset = group * codebook_size;
                auto t_sample = Clock::now();
                auto residual = detail::sample_token(
                    pred_logits,
                    logit_offset,
                    codebook_size,
                    params.sub_do_sample,
                    params.sub_top_k,
                    params.sub_top_p,
                    params.sub_temperature,
                    [](int64_t) {
                        return false;
                    },
                    1.0f,
                    rng_state,
                    sampling_scratch
                );
                ms_predictor_sample += Ms(Clock::now() - t_sample).count();
                rng_state = residual.rng_state;

                int64_t residual_code = residual.token >= logit_offset ? residual.token - logit_offset : residual.token;
                if(residual_code < 0 || residual_code >= codebook_size) {
                    residual_code = primary.token;
                }

                result.codes.push_back(residual_code);
                if(streaming) {
                    chunk_codes.push_back(residual_code);
                }

                const int64_t generation_step = static_cast<int64_t>(group);
                auto residual_embedding_err = copy_embedding_vector(
                    impl_->predictor_embedding_table,
                    impl_->predictor_embedding,
                    onnx_scratch,
                    residual_code,
                    &generation_step,
                    codebook_size,
                    hidden_size,
                    residual_embedding_buffer,
                    ms_onnx_embedding
                );
                if(!residual_embedding_err.is_ok()) {
                    return Error::inference_failed(
                        "Predictor embedding at group " + std::to_string(group) + ": " +
                        residual_embedding_err.message
                    );
                }

                if(group + 1 < residual_groups) {
                    for(int32_t index = 0; index < hidden_size; ++index) {
                        summed_embedding[static_cast<size_t>(index)] += residual_embedding_buffer[static_cast<size_t>(index)];
                    }

                    next_position_base[0] = pred_length;
                    auto next_pos_err = godot_llama::normalize_position_layout(
                        next_position_base,
                        1,
                        predictor_pos_components,
                        next_predictor_positions
                    );
                    if(next_pos_err) {
                        return Error::inference_failed(
                            "Predictor next position layout failed at group " + std::to_string(group) + ": " +
                            next_pos_err.message
                        );
                    }

                    auto t_incr = Clock::now();
                    auto incr_err = impl_->predictor_ctx.decode_embeddings(
                        residual_embedding_buffer,
                        1,
                        hidden_size,
                        next_predictor_positions,
                        predictor_pos_components
                    );
                    ms_predictor_incremental += Ms(Clock::now() - t_incr).count();
                    if(!incr_err.ok()) {
                        return Error::inference_failed(
                            "Predictor incremental decode failed at group " + std::to_string(group)
                        );
                    }
                    pred_length++;
                } else {
                    for(int32_t index = 0; index < hidden_size; ++index) {
                        summed_embedding[static_cast<size_t>(index)] += residual_embedding_buffer[static_cast<size_t>(index)];
                    }
                }
            }
        }
        ms_predictor_total += Ms(Clock::now() - t_pred_start).count();

        produced_frames++;
        result.frame_count = produced_frames;
        if(streaming) {
            chunk_frame_count++;
        }

        bool should_stop = produced_frames >= params.target_frames;
        if(!should_stop && produced_frames >= params.min_frames_before_eos && latest_logits) {
            should_stop = should_stop_eos(latest_logits, params.eos_token_id, codebook_size, params.eos_logit_margin);
        }

        if(streaming && (chunk_frame_count >= chunk_frames || should_stop)) {
            TtsFrameChunk chunk;
            chunk.codes = std::move(chunk_codes);
            chunk.frame_count = chunk_frame_count;
            chunk.codes_per_frame = result.codes_per_frame;
            chunk.is_final = should_stop;
            on_chunk(std::move(chunk));
            chunk_codes.clear();
            chunk_codes.reserve(chunk_capacity);
            chunk_frame_count = 0;
        }

        if(should_stop) {
            break;
        }

        const int32_t next_gen_step = produced_frames - 1;
        const float *next_embedding_data = summed_embedding.data();
        if(next_gen_step >= 0 && next_gen_step < trailing_text_length) {
            const float *trailing_row =
                trailing_text_hidden.data() + static_cast<size_t>(next_gen_step) * static_cast<size_t>(hidden_size);
            for(int32_t index = 0; index < hidden_size; ++index) {
                next_embedding[static_cast<size_t>(index)] =
                    summed_embedding[static_cast<size_t>(index)] + trailing_row[static_cast<size_t>(index)];
            }
            next_embedding_data = next_embedding.data();
        } else if(!tts_pad_embedding.empty()) {
            for(int32_t index = 0; index < hidden_size; ++index) {
                next_embedding[static_cast<size_t>(index)] =
                    summed_embedding[static_cast<size_t>(index)] + tts_pad_embedding[static_cast<size_t>(index)];
            }
            next_embedding_data = next_embedding.data();
        }

        next_position_base[0] = language_length;
        auto next_talker_pos_err = godot_llama::normalize_position_layout(
            next_position_base,
            1,
            talker_pos_components,
            next_talker_positions
        );
        if(next_talker_pos_err) {
            return Error::inference_failed(
                "Talker next position layout failed at frame " + std::to_string(frame) + ": " +
                next_talker_pos_err.message
            );
        }

        auto t_talker = Clock::now();
        auto incr_talker_err = impl_->talker_ctx.decode_embeddings(
            {next_embedding_data, static_cast<size_t>(hidden_size)},
            1,
            hidden_size,
            next_talker_positions,
            talker_pos_components
        );
        ms_talker_decode += Ms(Clock::now() - t_talker).count();
        if(!incr_talker_err.ok()) {
            return Error::inference_failed("Talker incremental decode failed at frame " + std::to_string(frame));
        }

        language_length++;
    }

    if(streaming && chunk_frame_count > 0) {
        TtsFrameChunk chunk;
        chunk.codes = std::move(chunk_codes);
        chunk.frame_count = chunk_frame_count;
        chunk.codes_per_frame = result.codes_per_frame;
        chunk.is_final = true;
        on_chunk(std::move(chunk));
    }

    const double ms_total = Ms(Clock::now() - t_gen_start).count();
    const double ms_other = ms_total - ms_talker_prefill - ms_talker_decode - ms_predictor_total;
    result.elapsed_ms = ms_total;
    result.talker_prefill_ms = ms_talker_prefill;
    result.talker_decode_ms = ms_talker_decode;
    result.predictor_ms = ms_predictor_total;
    result.onnx_embedding_ms = ms_onnx_embedding;
    result.other_ms = ms_other;
    if(should_log_tts_timing()) {
        std::fprintf(
            stderr,
            "%s frames=%d  total=%.0fms  talker_prefill=%.0fms  talker_decode=%.0fms"
            "  predictor=%.0fms  pred_clear=%.0fms  pred_prefill=%.0fms"
            "  pred_incr=%.0fms  pred_sample=%.0fms  onnx_emb=%.0fms  other=%.0fms\n",
            log_tag,
            produced_frames,
            ms_total,
            ms_talker_prefill,
            ms_talker_decode,
            ms_predictor_total,
            ms_predictor_clear,
            ms_predictor_prefill,
            ms_predictor_incremental,
            ms_predictor_sample,
            ms_onnx_embedding,
            ms_other
        );
    }

    return result;
}

Result<TtsGenerateResult> TtsCodeGenerator::generate(std::span<const float> initial_language_input,
                                                     int32_t initial_sequence_length,
                                                     std::span<const float> trailing_text_hidden,
                                                     int32_t trailing_text_length,
                                                     std::span<const float> tts_pad_embedding,
                                                     const TtsSamplingConfig &params,
                                                     CancellationToken *cancel) {
    if(!impl_->loaded) {
        return Error::invalid_state("TTS code generator is not loaded.");
    }

    return run_generation_impl(
        initial_language_input,
        initial_sequence_length,
        trailing_text_hidden,
        trailing_text_length,
        tts_pad_embedding,
        params,
        0,
        {},
        cancel,
        "[gotst-tts]"
    );
}

Result<TtsGenerateResult> TtsCodeGenerator::generate_streaming(std::span<const float> initial_language_input,
                                                               int32_t initial_sequence_length,
                                                               std::span<const float> trailing_text_hidden,
                                                               int32_t trailing_text_length,
                                                               std::span<const float> tts_pad_embedding,
                                                               const TtsSamplingConfig &params,
                                                               int32_t chunk_frames,
                                                               FrameChunkCallback on_chunk,
                                                               CancellationToken *cancel) {
    if(!impl_->loaded) {
        return Error::invalid_state("TTS code generator is not loaded.");
    }

    return run_generation_impl(
        initial_language_input,
        initial_sequence_length,
        trailing_text_hidden,
        trailing_text_length,
        tts_pad_embedding,
        params,
        chunk_frames,
        std::move(on_chunk),
        cancel,
        "[gotst-tts-stream]"
    );
}

} // namespace gotst
