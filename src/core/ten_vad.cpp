#include "gotst/core/ten_vad.hpp"

extern "C" {
#include "aed.h"
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace gotst {

namespace {

constexpr float kInt16Scale = 32768.0f;

std::vector<float> resample_linear(
    std::span<const float> source,
    int32_t from_rate,
    int32_t to_rate
) {
    if(source.empty()) {
        return {};
    }
    if(from_rate <= 0 || to_rate <= 0) {
        return {source.begin(), source.end()};
    }
    if(from_rate == to_rate) {
        return {source.begin(), source.end()};
    }

    const int64_t target_size = std::max<int64_t>(
        1,
        static_cast<int64_t>(std::llround(
            static_cast<double>(source.size()) * static_cast<double>(to_rate) / static_cast<double>(from_rate)
        ))
    );

    std::vector<float> output(static_cast<size_t>(target_size), 0.0f);
    const double ratio = static_cast<double>(from_rate) / static_cast<double>(to_rate);
    for(int64_t index = 0; index < target_size; ++index) {
        const double position = static_cast<double>(index) * ratio;
        const int64_t left = static_cast<int64_t>(std::floor(position));
        const int64_t right = std::min<int64_t>(left + 1, static_cast<int64_t>(source.size()) - 1);
        const double fraction = position - static_cast<double>(left);
        const double left_value = source[static_cast<size_t>(std::max<int64_t>(left, 0))];
        const double right_value = source[static_cast<size_t>(std::max<int64_t>(right, 0))];
        output[static_cast<size_t>(index)] = static_cast<float>(
            left_value + ((right_value - left_value) * fraction)
        );
    }
    return output;
}

std::vector<float> convert_to_ten_vad_scale(
    std::span<const float> source,
    int32_t input_sample_rate,
    int32_t target_sample_rate
) {
    std::vector<float> working = resample_linear(source, input_sample_rate, target_sample_rate);
    for(float &sample : working) {
        const float clamped = std::clamp(sample, -1.0f, 1.0f);
        sample = clamped * kInt16Scale;
    }
    return working;
}

} // namespace

struct TenVad::Impl {
    void *state = nullptr;
    TenVadConfig config;
    std::vector<float> pending_samples;
    bool loaded = false;

    ~Impl() {
        if(state != nullptr) {
            void *handle = state;
            AUP_Aed_destroy(&handle);
            state = nullptr;
        }
    }
};

TenVad::TenVad() : impl_(std::make_unique<Impl>()) {}
TenVad::~TenVad() = default;
TenVad::TenVad(TenVad &&) noexcept = default;
TenVad &TenVad::operator=(TenVad &&) noexcept = default;

Result<void> TenVad::load(const TenVadConfig &config) {
    impl_ = std::make_unique<Impl>();
    impl_->config = config;

    if(config.model_path.empty()) {
        return Error::invalid_argument("TenVad::load: empty model path.");
    }
    if(!std::filesystem::exists(std::filesystem::path(config.model_path))) {
        return Error::not_found("TenVad::load: model path does not exist: " + config.model_path);
    }
    if(config.model_sample_rate <= 0) {
        return Error::invalid_argument("TenVad::load: invalid model sample rate.");
    }
    if(config.hop_size <= 0) {
        return Error::invalid_argument("TenVad::load: invalid hop size.");
    }

    if(AUP_Aed_create(&impl_->state) != 0 || impl_->state == nullptr) {
        impl_->state = nullptr;
        return Error::invalid_state("TenVad::load: failed to create TEN-Vad state.");
    }

    Aed_StaticCfg static_cfg;
    static_cfg.enableFlag = 1;
    static_cfg.fftSz = 0;
    static_cfg.hopSz = static_cast<size_t>(config.hop_size);
    static_cfg.anaWindowSz = 0;
    static_cfg.frqInputAvailableFlag = 0;

    if(
        AUP_Aed_memAllocateWithModel(
            impl_->state,
            &static_cfg,
            config.model_path.c_str()
        ) != 0
    ) {
        return Error::model_not_loaded("TenVad::load: failed to allocate/load TEN-Vad model.");
    }

    Aed_DynamCfg dynamic_cfg;
    if(AUP_Aed_getDynamCfg(impl_->state, &dynamic_cfg) != 0) {
        return Error::invalid_state("TenVad::load: failed to read TEN-Vad dynamic config.");
    }
    dynamic_cfg.extVoiceThr = config.threshold;
    dynamic_cfg.resetFrameNum = static_cast<size_t>(std::max(config.reset_frame_count, 1));
    dynamic_cfg.pitchEstVoicedThr = config.pitch_est_voiced_threshold;

    if(AUP_Aed_setDynamCfg(impl_->state, &dynamic_cfg) != 0) {
        return Error::invalid_state("TenVad::load: failed to apply TEN-Vad dynamic config.");
    }
    if(AUP_Aed_init(impl_->state) != 0) {
        return Error::invalid_state("TenVad::load: failed to initialize TEN-Vad state.");
    }

    impl_->loaded = true;
    return {};
}

bool TenVad::is_loaded() const {
    return impl_ && impl_->loaded && impl_->state != nullptr;
}

Result<void> TenVad::reset() {
    if(!is_loaded()) {
        return Error::invalid_state("TenVad::reset: TEN-Vad is not loaded.");
    }
    impl_->pending_samples.clear();
    if(AUP_Aed_init(impl_->state) != 0) {
        return Error::invalid_state("TenVad::reset: failed to reinitialize TEN-Vad state.");
    }
    return {};
}

Result<TenVadChunkResult> TenVad::process(std::span<const float> samples, int32_t input_sample_rate) {
    if(!is_loaded()) {
        return Error::invalid_state("TenVad::process: TEN-Vad is not loaded.");
    }
    if(input_sample_rate <= 0) {
        return Error::invalid_argument("TenVad::process: input sample rate must be > 0.");
    }

    TenVadChunkResult result;
    result.hop_size = impl_->config.hop_size;
    result.model_sample_rate = impl_->config.model_sample_rate;

    if(samples.empty()) {
        return result;
    }

    std::vector<float> converted = convert_to_ten_vad_scale(
        samples,
        input_sample_rate,
        impl_->config.model_sample_rate
    );
    impl_->pending_samples.insert(impl_->pending_samples.end(), converted.begin(), converted.end());

    const size_t hop_size = static_cast<size_t>(impl_->config.hop_size);
    while(impl_->pending_samples.size() >= hop_size) {
        Aed_InputData input;
        input.binPower = nullptr;
        input.nBins = -1;
        input.timeSignal = impl_->pending_samples.data();
        input.hopSz = static_cast<int>(hop_size);

        Aed_OutputData output{};
        if(AUP_Aed_proc(impl_->state, &input, &output) != 0) {
            return Error::inference_failed("TenVad::process: TEN-Vad frame processing failed.");
        }

        TenVadFrameResult frame;
        frame.probability = output.voiceProb;
        frame.is_voice = output.vadRes > 0;
        frame.frame_rms = output.frameRms;
        frame.frame_energy = output.frameEnergy;
        frame.pitch_frequency_hz = output.pitchFreq;
        result.frames.push_back(frame);
        result.last_probability = frame.probability;
        result.last_is_voice = frame.is_voice;
        result.any_voice = result.any_voice || frame.is_voice;
        impl_->pending_samples.erase(impl_->pending_samples.begin(), impl_->pending_samples.begin() + static_cast<std::ptrdiff_t>(hop_size));
    }

    result.processed_frame_count = static_cast<int32_t>(result.frames.size());
    return result;
}

} // namespace gotst
