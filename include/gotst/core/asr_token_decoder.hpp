#pragma once

#include "gotst/core/cancellation_token.hpp"
#include "gotst/core/result.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gotst {

struct AsrModelPaths {
    std::string thinker_gguf_path;
    std::string embedding_onnx_path;
};

struct AsrSessionConfig {
    int32_t n_ctx = 1024;
    int32_t n_batch = 1024;
    int32_t n_threads = -1;
    int32_t n_gpu_layers = 0;
    bool use_mmap = true;
    bool use_mlock = false;
    int32_t flash_attn_type = -1;
    int32_t type_k = -1;
    int32_t type_v = -1;
    int32_t position_components = 3;
};

struct AsrDecodeParams {
    int32_t max_tokens = 64;
    int32_t hidden_size = 896;
    int32_t vocab_size = 152064;
    int32_t eos_token_id = 151645;
};

struct AsrDecodeResult {
    std::vector<int32_t> token_ids;
    double elapsed_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double onnx_embedding_ms = 0.0;
};

class AsrTokenDecoder {
public:
    AsrTokenDecoder();
    ~AsrTokenDecoder();

    AsrTokenDecoder(const AsrTokenDecoder &) = delete;
    AsrTokenDecoder &operator=(const AsrTokenDecoder &) = delete;
    AsrTokenDecoder(AsrTokenDecoder &&) noexcept;
    AsrTokenDecoder &operator=(AsrTokenDecoder &&) noexcept;

    Result<void> load(const AsrModelPaths &paths, const AsrSessionConfig &config);
    bool is_loaded() const;

    Result<AsrDecodeResult> decode(
        std::span<const float> prompt_embeddings,
        int32_t prompt_length,
        const AsrDecodeParams &params,
        CancellationToken *cancel = nullptr
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gotst
