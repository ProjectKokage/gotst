#include "gotst/core/speech_encoder_session.hpp"

#include <gonx/core/session.hpp>

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace gotst {

struct SpeakerEncoderSession::Impl {
    gonx::InferenceSession session;
};

SpeakerEncoderSession::SpeakerEncoderSession() : impl_(std::make_unique<Impl>()) {}

SpeakerEncoderSession::~SpeakerEncoderSession() = default;

Result<void> SpeakerEncoderSession::load(const std::string &model_path) {
    if (model_path.empty()) {
        return Error::invalid_argument("SpeakerEncoderSession::load: empty model path");
    }
    auto status = impl_->session.load(model_path);
    if (!status.has_value()) {
        return Error::io_error("SpeakerEncoderSession::load: failed to load model from " + model_path);
    }
    return Result<void>();
}

bool SpeakerEncoderSession::is_loaded() const {
    return impl_->session.is_loaded();
}

Result<std::vector<float>> SpeakerEncoderSession::extract_embedding(
    const float *mel_features,
    int64_t frames,
    int64_t mel_dim
) const {
    if (!is_loaded()) {
        return Error::model_not_loaded("SpeakerEncoderSession::extract_embedding: model not loaded");
    }
    if (!mel_features || frames <= 0 || mel_dim <= 0) {
        return Error::invalid_argument("SpeakerEncoderSession::extract_embedding: null mel_features or invalid dimensions");
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t total_elements = frames * mel_dim;

    std::vector<int64_t> input_shape = {1, frames, mel_dim};
    // ORT CreateTensor takes non-const T* but does not mutate input data.
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float *>(mel_features),
        static_cast<size_t>(total_elements),
        input_shape.data(),
        input_shape.size()
    );

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_tensor));

    auto result = impl_->session.run(inputs);
    if (result.has_error()) {
        return Error::inference_failed("SpeakerEncoderSession::extract_embedding: ONNX run failed");
    }

    auto &outputs = result.value();
    if (outputs.empty()) {
        return Error::inference_failed("SpeakerEncoderSession::extract_embedding: empty ONNX output");
    }

    auto &output_tensor = outputs[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t output_size = output_info.GetElementCount();
    const float *output_data = output_tensor.GetTensorData<float>();

    return std::vector<float>(output_data, output_data + output_size);
}

struct SpeechTokenizerEncoderSession::Impl {
    gonx::InferenceSession session;
};

SpeechTokenizerEncoderSession::SpeechTokenizerEncoderSession() : impl_(std::make_unique<Impl>()) {}

SpeechTokenizerEncoderSession::~SpeechTokenizerEncoderSession() = default;

Result<void> SpeechTokenizerEncoderSession::load(const std::string &model_path) {
    if (model_path.empty()) {
        return Error::invalid_argument("SpeechTokenizerEncoderSession::load: empty model path");
    }
    auto status = impl_->session.load(model_path);
    if (!status.has_value()) {
        return Error::io_error("SpeechTokenizerEncoderSession::load: failed to load model from " + model_path);
    }
    return Result<void>();
}

bool SpeechTokenizerEncoderSession::is_loaded() const {
    return impl_->session.is_loaded();
}

Result<SpeechTokenizerEncodeResult> SpeechTokenizerEncoderSession::encode(
    const float *audio,
    int64_t num_samples
) const {
    if (!is_loaded()) {
        return Error::model_not_loaded("SpeechTokenizerEncoderSession::encode: model not loaded");
    }
    if (!audio || num_samples <= 0) {
        return Error::invalid_argument("SpeechTokenizerEncoderSession::encode: null audio or zero samples");
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, 1, num_samples};
    // ORT CreateTensor takes non-const T* but does not mutate input data.
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float *>(audio),
        static_cast<size_t>(num_samples),
        input_shape.data(),
        input_shape.size()
    );

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_tensor));

    auto result = impl_->session.run(inputs);
    if (result.has_error()) {
        return Error::inference_failed("SpeechTokenizerEncoderSession::encode: ONNX run failed");
    }

    auto &outputs = result.value();
    if (outputs.empty()) {
        return Error::inference_failed("SpeechTokenizerEncoderSession::encode: empty ONNX output");
    }

    auto &output_tensor = outputs[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();

    if (output_shape.size() < 3) {
        return Error::shape_mismatch("SpeechTokenizerEncoderSession::encode: expected 3D output");
    }

    const int64_t codebooks = output_shape[1];
    const int64_t frames = output_shape[2];
    const int64_t total_codes = codebooks * frames;

    SpeechTokenizerEncodeResult encode_result;
    encode_result.codes.resize(static_cast<size_t>(total_codes));
    encode_result.frames = frames;
    encode_result.codebooks = codebooks;

    auto element_type = output_info.GetElementType();

    if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        const int32_t *code_data = output_tensor.GetTensorData<int32_t>();
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t cb = 0; cb < codebooks; ++cb) {
                encode_result.codes[static_cast<size_t>(frame * codebooks + cb)] =
                    code_data[cb * frames + frame];
            }
        }
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t *code_data = output_tensor.GetTensorData<int64_t>();
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t cb = 0; cb < codebooks; ++cb) {
                encode_result.codes[static_cast<size_t>(frame * codebooks + cb)] =
                    static_cast<int32_t>(code_data[cb * frames + frame]);
            }
        }
    } else {
        return Error::shape_mismatch("SpeechTokenizerEncoderSession::encode: unexpected output element type");
    }

    return encode_result;
}

} // namespace gotst
