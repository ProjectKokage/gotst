#pragma once

#include "gotst/core/result.hpp"

#include <gonx/core/session.hpp>

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace gotst::detail {

struct SingleTokenEmbeddingRunScratch {
    std::array<int64_t, 1> token_ids = {0};
    std::array<int64_t, 2> token_shape = {1, 1};
    std::array<int64_t, 1> generation_steps = {0};
    std::array<int64_t, 1> generation_shape = {1};
    std::vector<Ort::Value> inputs;

    SingleTokenEmbeddingRunScratch() {
        inputs.reserve(2);
    }
};

struct FloatTensorRunResult {
    std::vector<Ort::Value> outputs;
    std::span<const float> values;
};

inline Ort::MemoryInfo &cpu_tensor_memory_info() {
    static Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return memory_info;
}

inline Result<FloatTensorRunResult> run_single_token_float_embedding(gonx::InferenceSession &session,
                                                                     SingleTokenEmbeddingRunScratch &scratch,
                                                                     int64_t token_id,
                                                                     const int64_t *generation_step = nullptr) {
    scratch.token_ids[0] = token_id;
    scratch.inputs.clear();
    scratch.inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
        cpu_tensor_memory_info(),
        scratch.token_ids.data(),
        scratch.token_ids.size(),
        scratch.token_shape.data(),
        scratch.token_shape.size()
    ));

    if(generation_step != nullptr) {
        scratch.generation_steps[0] = *generation_step;
        scratch.inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            cpu_tensor_memory_info(),
            scratch.generation_steps.data(),
            scratch.generation_steps.size(),
            scratch.generation_shape.data(),
            scratch.generation_shape.size()
        ));
    }

    auto run_result = session.run(scratch.inputs);
    if(run_result.has_error()) {
        return Error::inference_failed("ONNX embedding run failed: " + run_result.error().message);
    }

    auto outputs = std::move(run_result).value();
    if(outputs.empty()) {
        return Error::inference_failed("ONNX embedding returned empty outputs");
    }

    FloatTensorRunResult output;
    output.outputs = std::move(outputs);

    const auto &tensor = output.outputs.front();
    const auto info = tensor.GetTensorTypeAndShapeInfo();
    const size_t element_count = info.GetElementCount();
    if(element_count == 0) {
        return Error::inference_failed("ONNX embedding returned a zero-sized tensor");
    }
    if(info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Error::shape_mismatch("ONNX embedding returned a non-float tensor");
    }

    output.values = {tensor.GetTensorData<float>(), element_count};
    return output;
}

} // namespace gotst::detail
