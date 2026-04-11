# gotst Feature Completeness Plan

> **Status: Complete (2026-04).** All phases described below have been implemented and tested.
> This document is preserved as historical design context. See `AGENTS.md` for the current
> contract and `README.md` for the current API surface.

## Purpose

Make `gotst` feature-complete for all Qwen3-ASR and Qwen3-TTS capabilities, with first-class
support for VoiceClone, VoiceDesign, and CustomVoice modes. This document defines what "feature
complete" means, what gaps exist today, and a phased implementation plan.

---

## Current State

`gotst` provides native C++ GDExtension kernels for the Qwen3-ASR and Qwen3-TTS hybrid runtime.
It owns speech-specific DSP, feature extraction, ONNX session management for speech models,
prompt assembly for all voice modes, streaming/cancellation, and multi-language support.
The GDScript layer (`qwen_asr_client.gd`, `qwen_tts_client.gd`) owns session orchestration
and the autoregressive generation loop.

### What is implemented

| Category | Capability | gotst Method |
|----------|-----------|--------------|
| ASR preprocessing | Log-mel spectrogram (16 kHz, Slaney, DFT-based) | `build_log_mel_features` |
| ASR position IDs | 3-component MRoPE triplicate | `build_triplicate_position_ids` |
| Tensor helpers | Row slice, concat, repeat, add, sum, extract | `slice_rows`, `concat_rows`, `repeat_row`, `add_vectors`, `sum_vectors`, `extract_sequence`, `extract_last_hidden_row` |
| Sampling | Argmax, top-k/top-p with repetition penalty and deterministic RNG | `select_last_row_argmax`, `select_last_row_token` |
| EOS detection | Margin-based codec vs EOS logit check | `should_stop_on_eos` |
| Waveform output | Decoder output to PCM with DC removal, normalization, edge fade | `convert_decoder_output_to_waveform` |
| TTS prompt | Initial language input assembly (custom_voice mode only) | `build_tts_initial_language_input` |
| Validation | NaN/Inf guard | `contains_only_finite_values` |
| Backend check | Config path validation and dependency probing | `inspect_backends` |

### Implementation status (updated 2026-04)

All categories from the original gap analysis have been implemented:

| Category | Status | Implementation |
|----------|--------|----------------|
| VoiceClone | Done | `speech_encoder_session.cpp`, `speaker_mel.cpp`, `tts_prompt_assembly.cpp` |
| VoiceDesign | Done | `tts_prompt_assembly.cpp` (`build_voice_design_language_input`) |
| CustomVoice | Done | `tts_prompt_assembly.cpp` (`load_custom_voice_config`, speaker lookup) |
| Streaming | Done | `tts_code_generator.cpp`, `cancellation_token.hpp` |
| Multi-language | Done | `language_config.cpp` (10 TTS languages, codec prefix construction) |
| Session ownership | Done | `speech_encoder_session.cpp` (SpeakerEncoder + SpeechTokenizerEncoder) |
| Tests | Done | 12 test files, 165 test cases (Catch2) |
| FFT | Done | `fft.cpp` (replaces O(N^2) DFT) |

### Voice mode support matrix

| Feature | Base Model | CustomVoice Model | VoiceDesign Model |
|---------|-----------|-------------------|-------------------|
| Voice cloning (ref audio + transcript) | Yes | No | No |
| Speaker encoder (ECAPA-TDNN) | Yes | No | No |
| Reference codec codes (ICL) | Yes | No | No |
| x-vector-only mode (no transcript) | Yes | No | No |
| 9 preset speakers | No | Yes | No |
| Instruct/style control | No | Yes (1.7B only) | Yes |
| Voice from text description | No | No | Yes |
| Speaker prompt embedding | Yes | No | No |
| Streaming | Yes | Yes | Yes |

---

## Target State

gotst becomes the **native speech runtime** that owns:

1. All speech-specific DSP and feature extraction (ASR mel + speaker mel)
2. All speech-specific ONNX session management (`speaker_encoder`, `speech_tokenizer_encoder`)
3. All prompt assembly for every voice mode (VoiceClone ICL, VoiceDesign, CustomVoice, Base)
4. All speech token/code helpers (encoding, decoding, codebook math)
5. All waveform post-processing
6. Streaming and cancellation primitives for speech generation

gotst does **not** own:

- Generic GGUF inference (stays in `godorama`)
- Generic ONNX session management (stays in `gonx`)
- Talker/predictor autoregressive loop orchestration (stays in `godorama`)
- Conversation orchestration, UI, scene logic (stays in `project-kokage`)

---

## Phase 0: Foundation and Infrastructure

### 0.1 Expand config for voice modes

**File:** `include/gotst/core/speech_runtime_core.hpp`, `src/godot/speech_runtime_config.cpp`

Add to `HybridTtsConfig`:

```
voice_mode                string   // "base", "custom_voice", "voice_design"
speaker_encoder_model_path  string   // path to speaker_encoder.onnx
speech_tokenizer_encoder_model_path  string   // path to speech_tokenizer_encoder.onnx
speech_tokenizer_encoder_sample_rate int      // 24000
speaker_mel_params         struct { sample_rate, n_fft, hop_length, n_mels, fmin, fmax }
```

Add corresponding `GotstSpeechRuntimeConfig` properties with Godot exports.

Add to `HybridAsrConfig`:

```
default_language          string   // default language hint
supported_languages       vector<string>
```

### 0.2 ONNX session ownership for speech-specific models

gotst currently delegates all ONNX inference to `gonx` sessions managed by the GDScript layer.
For VoiceClone, gotst needs to **own** ONNX sessions for:

- `speaker_encoder.onnx` -- ECAPA-TDNN speaker embedding extraction
- `speech_tokenizer_encoder.onnx` -- Mimi-based audio-to-codes encoding

These are **speech-specific** models that do not belong in the generic `gonx` session pool.

**File:** New `include/gotst/core/speech_encoder_session.hpp`

```
class SpeakerEncoderSession {
public:
    bool load(const std::string& model_path);
    bool is_loaded() const;
    std::vector<float> extract_embedding(const float* mel_features,
                                          int64_t frames, int64_t mel_dim) const;
private:
    gonx::InferenceSession session_;
};
```

```
class SpeechTokenizerEncoderSession {
public:
    bool load(const std::string& model_path);
    bool is_loaded() const;
    struct EncodeResult {
        std::vector<int32_t> codes;     // flat [frames * 16]
        int64_t frames;
        int64_t codebooks;              // always 16
    };
    EncodeResult encode(const float* audio, int64_t num_samples) const;
private:
    gonx::InferenceSession session_;
};
```

### 0.3 Expand testing infrastructure

- Add Google Test or Catch2 as a proper test dependency (currently uses raw `assert`)
- Create test fixture directory: `tests/fixtures/` with small PCM samples, expected mel outputs, expected speaker embeddings
- Add CTest integration for all new tests

---

## Phase 1: Speaker Encoder Pipeline

**Goal:** gotst can extract a speaker embedding vector from a reference audio clip at runtime.

### 1.1 Speaker mel feature extraction

The speaker encoder uses **different mel parameters** than the ASR frontend:

| Parameter | ASR mel | Speaker mel |
|-----------|---------|-------------|
| Sample rate | 16000 | 24000 |
| FFT size | 400 | 1024 |
| Hop length | 160 | 256 |
| Mel bins | 128 | 128 |
| fmin | 0 | 0 |
| fmax | 8000 (Nyquist) | 12000 |
| Window | Hann | Hann |
| Padding | Reflect | Reflect |
| Normalization | Whisper-style `(x+4)*0.25` | `log(clip(x, 1e-5, None))` |

**File:** `src/core/speech_encoder.cpp`

New method on `GotstSpeechRuntime`:

```
Dictionary build_speaker_mel_features(
    PackedFloat32Array waveform,
    int64_t input_sample_rate,
    int64_t target_sample_rate,   // 24000
    int64_t mel_bins,             // 128
    int64_t fft_size,             // 1024
    int64_t hop_length,           // 256
    double fmin,                  // 0.0
    double fmax                   // 12000.0
) const
```

Returns `{ "mel_features": PackedFloat32Array, "frames": int, "mel_dim": int }` with layout
`[1, frames, mel_dim]` matching the `speaker_encoder.onnx` input contract.

Implementation notes:

- Reuse the existing DFT infrastructure but parameterize normalization
- Use `log(max(x, 1e-5))` instead of Whisper normalization
- Output shape must be `[batch=1, time=frames, mel_dim]` (batch-first, not mel-first like ASR)

### 1.2 Speaker embedding extraction

**File:** `src/core/speech_encoder.cpp`

New methods on `GotstSpeechRuntime`:

```
bool load_speaker_encoder(String model_path)
bool is_speaker_encoder_loaded() const
PackedFloat32Array extract_speaker_embedding(
    const PackedFloat32Array& mel_features,
    int64_t frames,
    int64_t mel_dim
) const
```

The `extract_speaker_embedding` method:

1. Validates session is loaded
2. Creates ONNX input tensor from `mel_features` with shape `[1, frames, mel_dim]`
3. Runs `speaker_encoder.onnx` inference
4. Extracts and returns the flat embedding vector

### 1.3 End-to-end speaker embedding from audio

Convenience method combining 1.1 and 1.2:

```
PackedFloat32Array compute_speaker_embedding_from_audio(
    const PackedFloat32Array& waveform,
    int64_t input_sample_rate
) const
```

### 1.4 Tests

- Unit: mel feature extraction with known input, compare against Python fixture
- Unit: speaker encoder session load/unload lifecycle
- Unit: embedding extraction with small fixture audio, compare output dimensions and finiteness
- Integration: extract embedding from fixture WAV, compare against pre-computed JSON embedding

---

## Phase 2: Speech Tokenizer Encoder Pipeline

**Goal:** gotst can encode reference audio into discrete codec codes at runtime.

### 2.1 Speech tokenizer encoder session

**File:** `src/core/speech_encoder.cpp`

New methods on `GotstSpeechRuntime`:

```
bool load_speech_tokenizer_encoder(String model_path)
bool is_speech_tokenizer_encoder_loaded() const
Dictionary encode_speech_codes(
    const PackedFloat32Array& waveform,
    int64_t input_sample_rate
) const
```

The `encode_speech_codes` method:

1. Validates session is loaded and `input_sample_rate == 24000`
2. Creates ONNX input tensor `input_values` with shape `[1, 1, num_samples]` from float32 audio
3. Runs `speech_tokenizer_encoder.onnx` inference
4. Extracts `audio_codes` with expected shape `[1, 16, frames]`
5. Transposes to `[frames, 16]` layout
6. Returns `{ "codes": PackedInt32Array, "shape": [frames, 16] }`

### 2.2 Voice clone fixture loading

Support loading pre-computed voice clone fixtures to avoid runtime re-encoding:

```
Dictionary load_voice_clone_fixture(String json_path) const
```

Returns:
```
{
    "ref_audio": String,
    "ref_text": String,
    "ref_codes": PackedInt32Array,
    "ref_codes_shape": [int, int],
    "ref_speaker_embedding": PackedFloat32Array
}
```

### 2.3 Tests

- Unit: encode 1-second 24kHz sine wave, verify output shape `[frames, 16]` with valid code range `[0, 2047]`
- Unit: fixture load/parse with known fixture file
- Integration: encode fixture audio, compare codes against pre-computed fixture

---

## Phase 3: VoiceClone Prompt Assembly

**Goal:** gotst can construct the full VoiceClone ICL (in-context learning) prompt for the talker
language model.

### 3.1 ICL prompt assembly

VoiceClone ICL mode constructs a prompt where reference audio codec codes serve as few-shot
examples for the talker model:

```
ref_text_tokens -> text_projection -> text_embed
ref_codes -> codec_embedding -> codec_embed (summed across 16 codebooks per frame)
target_text_tokens -> text_projection -> text_embed

ICL input: overlay(text_embed, codec_embed) where min(text_len, codec_len) positions overlap
```

**File:** `src/core/tts_prompt_assembly.cpp`

New method on `GotstSpeechRuntime`:

```
Dictionary build_voice_clone_language_input(
    const PackedFloat32Array& text_projected_states,      // target text projected
    int64_t text_sequence_length,
    const PackedFloat32Array& ref_text_projected_states,   // reference text projected
    int64_t ref_text_sequence_length,
    const PackedFloat32Array& ref_codec_projected_states,  // ref codes embedded and summed
    int64_t ref_codec_frames,
    const PackedFloat32Array& special_projected_states,    // BOS/EOS/PAD embeddings
    const PackedFloat32Array& codec_pad_embedding,
    const PackedFloat32Array& codec_prefill_embeddings,    // codec prefix (think + language)
    int64_t codec_prefill_length,
    const PackedFloat32Array& speaker_prompt_embedding,    // speaker embedding vector
    const PackedFloat32Array& instruction_projected_states,
    int64_t instruction_sequence_length,
    int64_t hidden_size,
    int64_t wrapped_prefix_token_count,
    int64_t wrapped_suffix_token_count
) const
```

The ICL overlay logic:

1. Compute `min_len = min(ref_text_len, ref_codec_len)`
2. For overlapping positions: `icl_embed[i] = text_embed[i] + codec_embed[i]`
3. For remaining text: `icl_embed[i] = text_embed[i]`
4. For remaining codec: `icl_embed[i] = codec_embed[i]`
5. Prepend target text projected states after the ICL section
6. Insert speaker embedding between codec prefix and codec BOS (same as current `_speaker_prompt_embedding` insertion)

### 3.2 x-vector-only mode prompt

When `x_vector_only_mode = true` (no reference text or codes):

- Use the standard `build_tts_initial_language_input` with only the speaker embedding
- No ICL overlay, no ref_codes
- Lower quality but no transcript needed

### 3.3 Codec code embedding helper

New helper to embed and sum reference codec codes:

```
PackedFloat32Array embed_and_sum_codec_codes(
    const PackedInt32Array& ref_codes,  // flat [frames * 16]
    int64_t frames,
    int64_t codebooks,                  // 16
    int64_t hidden_size,
    const Callable& codec_embed_fn      // callback to codec embedding ONNX session
) const
```

This takes raw int32 codes, calls the codec embedding session for each codebook, and sums
the 16 embeddings per frame to produce `[frames, hidden_size]`.

### 3.4 Tests

- Unit: ICL overlay with known inputs, verify element-wise addition in overlap region
- Unit: speaker embedding insertion at correct position in codec prefix
- Unit: x-vector-only prompt matches existing `build_tts_initial_language_input` behavior
- Integration: build full voice clone prompt from fixture data, verify shape and finiteness

---

## Phase 4: CustomVoice Support

**Goal:** gotst can select and embed any of the 9 preset speakers for the CustomVoice model variant.

### 4.1 Speaker name to token ID mapping

**File:** `src/core/tts_prompt_assembly.cpp`

The CustomVoice model stores speaker identities as lists of codec token IDs in
`config.talker_config.spk_id`. gotst needs a hardcoded or config-driven mapping:

```
struct CustomVoiceSpeaker {
    std::string name;           // e.g., "Serena"
    std::vector<int64_t> token_ids;
};
```

New method:

```
Dictionary get_custom_voice_speaker_ids(String speaker_name) const
```

Returns `{ "token_ids": PackedInt64Array }` or empty if speaker not found.

9 preset speakers:

| Name | Native Language | Notes |
|------|----------------|-------|
| Vivian | Chinese | Bright, slightly edgy young female |
| Serena | Chinese | Warm, gentle young female |
| Uncle_Fu | Chinese | Seasoned male, low mellow timbre |
| Dylan | Chinese (Beijing) | Youthful Beijing male |
| Eric | Chinese (Sichuan) | Lively Chengdu male |
| Ryan | English | Dynamic male, strong rhythmic drive |
| Aiden | English | Sunny American male |
| Ono_Anna | Japanese | Playful Japanese female |
| Sohee | Korean | Warm Korean female |

Note: The actual token IDs come from the model's `config.json` (`talker_config.spk_id` field).
gotst should load these from the config rather than hardcoding them, since different model
versions may use different IDs.

### 4.2 Custom voice prompt assembly

New method:

```
Dictionary build_custom_voice_language_input(
    const PackedFloat32Array& text_projected_states,
    int64_t text_sequence_length,
    const PackedFloat32Array& special_projected_states,
    const PackedFloat32Array& codec_pad_embedding,
    const PackedFloat32Array& codec_prefill_embeddings,
    int64_t codec_prefill_length,
    const PackedFloat32Array& speaker_token_embedding,     // sum of speaker's token embeddings
    const PackedFloat32Array& instruction_projected_states,
    int64_t instruction_sequence_length,
    int64_t hidden_size,
    int64_t wrapped_prefix_token_count,
    int64_t wrapped_suffix_token_count
) const
```

This is structurally similar to the current `build_tts_initial_language_input` but:
- Speaker token embedding replaces speaker prompt embedding
- Supports dialect-specific language IDs for speakers like Dylan (Beijing) and Eric (Sichuan)
- Instruction/style text is required for 1.7B, optional for 0.6B

### 4.3 Speaker token embedding computation

```
PackedFloat32Array compute_custom_voice_speaker_embedding(
    const PackedInt64Array& speaker_token_ids,
    const Callable& codec_embed_fn      // callback to codec embedding ONNX session
) const
```

Looks up each token ID in the codec embedding table, sums the resulting vectors to produce
a single `hidden_size`-dim speaker embedding.

### 4.4 Tests

- Unit: all 9 speaker names resolve to non-empty token ID arrays
- Unit: speaker token embedding computation with mock embedding lookup
- Unit: custom voice prompt assembly with speaker embedding at correct position
- Unit: dialect speakers override language ID correctly

---

## Phase 5: VoiceDesign Support

**Goal:** gotst can construct a VoiceDesign prompt that conditions synthesis on a natural
language voice description.

### 5.1 Voice design prompt assembly

The VoiceDesign model uses the instruction text channel to specify voice characteristics:

```
<|im_start|>user\n{voice_description}<|im_end|>\n
```

This follows the same instruction embedding path as style instructions in CustomVoice mode,
but the voice description directly controls timbre, emotion, pitch, and prosody.

**File:** `src/core/tts_prompt_assembly.cpp`

New method:

```
Dictionary build_voice_design_language_input(
    const PackedFloat32Array& text_projected_states,
    int64_t text_sequence_length,
    const PackedFloat32Array& special_projected_states,
    const PackedFloat32Array& codec_pad_embedding,
    const PackedFloat32Array& codec_prefill_embeddings,
    int64_t codec_prefill_length,
    const PackedFloat32Array& voice_description_projected_states,
    int64_t voice_description_sequence_length,
    int64_t hidden_size,
    int64_t wrapped_prefix_token_count,
    int64_t wrapped_suffix_token_count
) const
```

Key differences from other modes:

- **No speaker embedding** -- voice identity comes entirely from the description text
- **No reference codes** -- no audio conditioning at all
- **voice_description_projected_states** is the primary voice control signal
- The voice description is placed in the instruction/embedding prefix position

### 5.2 Tests

- Unit: voice design prompt assembly with description embedding at correct position
- Unit: prompt shape matches expected dimensions
- Unit: no speaker embedding insertion when none provided
- Integration: build voice design prompt, compare against Python reference output

---

## Phase 6: Streaming and Cancellation

**Goal:** All speech generation APIs support incremental output and cancellation.

### 6.1 Cancellation primitives

**File:** `include/gotst/core/cancellation_token.hpp`

```
class CancellationToken {
public:
    void cancel();
    bool is_cancelled() const;
private:
    std::atomic<bool> cancelled_{false};
};
```

Every streaming API accepts a `CancellationToken` reference.

### 6.2 Streaming TTS helpers

Currently the TTS pipeline generates all frames before decoding. For streaming:

1. **Frame-by-frame codec emission**: As each talker frame is generated (primary + 15 residual codes), emit the codes before all frames are complete
2. **Chunked decoding**: Decode codec codes in small windows (e.g., 8-16 frames) rather than waiting for all frames
3. **Incremental waveform emission**: Emit decoded PCM chunks as they become available

New methods:

```
PackedFloat32Array decode_codec_frames(
    const PackedInt64Array& codec_codes,   // [frames, codebooks] flat
    int64_t frames,
    int64_t codebooks,
    int64_t sample_rate,
    bool normalize,
    double gain,
    const CancellationToken& token
) const
```

This wraps the existing decoder + waveform pipeline but works on partial codec frames and
checks cancellation between chunks.

### 6.3 Streaming ASR helpers

For partial transcription:

1. **Chunk-based processing**: Process audio in overlapping windows
2. **Partial decode emission**: Emit text after each autoregressive step
3. **Context carryover**: Maintain encoder state across chunks

New method:

```
struct AsrChunkResult {
    std::string partial_text;
    bool is_final;
    int64_t tokens_generated;
};

AsrChunkResult decode_asr_step(
    const PackedFloat32Array& logits,
    const PackedInt64Array& logits_shape,
    const PackedFloat32Array& hidden_states,
    const PackedInt64Array& hidden_shape,
    int64_t hidden_size,
    bool is_final_chunk,
    const CancellationToken& token
) const
```

### 6.4 Godot-facing streaming signals

New signal-emitting methods on `GotstSpeechRuntime`:

```
void _emit_partial_synthesis(int64_t request_id, PackedFloat32Array pcm_chunk, int64_t sample_rate)
void _emit_partial_transcription(int64_t request_id, String partial_text)
```

These are designed to be called from the GDScript orchestration layer, not as direct signals
from gotst (gotst must not push scene tree mutations from worker threads).

### 6.5 Tests

- Unit: cancellation token state transitions
- Unit: decode_codec_frames with cancellation after N frames
- Unit: decode_asr_step with partial logits
- Integration: streaming TTS generation with cancellation mid-stream

---

## Phase 7: Multi-Language Support

**Goal:** All voice modes support configurable target language, not just Japanese.

### 7.1 Language tag system

**File:** `src/core/language_config.cpp`

```
struct LanguageConfig {
    std::string name;               // "Japanese", "English", "Chinese", etc.
    int64_t codec_language_token_id; // e.g., 2058 for Japanese
    std::string codec_prefix_mode;  // "think" (known language) or "nothink" (auto)
};
```

New method:

```
Dictionary get_supported_languages() const
int64_t get_language_token_id(String language_name) const
```

Supported languages for TTS (10): Chinese, English, Japanese, Korean, German, French,
Russian, Portuguese, Spanish, Italian.

Supported languages for ASR (30 + 22 Chinese dialects): see Qwen3-ASR documentation.

### 7.2 Language-dependent codec prefix

When a known language is specified:
```
codec_prefix = [THINK, THINK_BOS, LANGUAGE_ID, THINK_EOS, PAD, BOS]
```

When no language is specified (auto-detect):
```
codec_prefix = [NOTHINK, THINK_BOS, THINK_EOS, PAD, BOS]
```

This is currently hardcoded to Japanese in the GDScript layer. gotst should parameterize it:

```
PackedInt64Array build_codec_prefix_tokens(
    int64_t language_token_id,      // -1 for auto-detect
    int64_t think_token_id,
    int64_t think_bos_token_id,
    int64_t think_eos_token_id,
    int64_t pad_token_id,
    int64_t bos_token_id
) const
```

### 7.3 ASR language identification

For ASR, the language can be auto-detected or forced:

```
PackedInt64Array build_asr_language_hint_tokens(
    String language_hint            // empty for auto-detect
) const
```

### 7.4 Tests

- Unit: all 10 TTS languages resolve to valid token IDs
- Unit: codec prefix construction for known language vs auto-detect
- Unit: ASR language hint token construction

---

## Phase 8: Performance and Polish

### 8.1 FFT optimization

Replace the O(N^2) DFT with a proper FFT for both mel frontends:

- ASR: FFT size 400 (padded to 512) -- ~2x speedup expected
- Speaker: FFT size 1024 (padded to 1024) -- ~4x speedup expected

Use a lightweight in-place FFT implementation (e.g., KissFFT or a hand-rolled Cooley-Tukey).
No external dependency larger than the FFT kernel itself.

### 8.2 Comprehensive kernel tests

Every method on `GotstSpeechRuntime` must have at least one deterministic test:

| Method | Test Type |
|--------|-----------|
| `build_log_mel_features` | Known sine wave input, compare against NumPy reference |
| `build_speaker_mel_features` | Known sine wave input, compare against Python fixture |
| `extract_speaker_embedding` | Small fixture audio, compare against pre-computed JSON |
| `encode_speech_codes` | Small fixture audio, compare against pre-computed fixture |
| `build_voice_clone_language_input` | Known inputs, verify shape and element values |
| `build_custom_voice_language_input` | Known inputs, verify shape and element values |
| `build_voice_design_language_input` | Known inputs, verify shape and element values |
| `slice_rows` / `concat_rows` / `repeat_row` | Known inputs, exact output match |
| `add_vectors` / `sum_vectors` | Known inputs, exact output match |
| `select_last_row_token` | Known logits, deterministic seed, verify token selection |
| `should_stop_on_eos` | Known logits, verify threshold behavior |
| `convert_decoder_output_to_waveform` | Known decoder output, verify normalization |
| `build_triplicate_position_ids` | Verify triplicate pattern |
| `contains_only_finite_values` | NaN, Inf, and normal inputs |

### 8.3 GDScript fallback parity tests

For every gotst method that has a GDScript fallback, add cross-validation tests that run both
paths with identical inputs and verify the results match within a tolerance (e.g., `1e-5` for
float operations).

### 8.4 Integration tests

- Full hybrid ASR roundtrip: WAV -> mel -> audio conv -> encoder -> thinker -> text
- Full hybrid TTS roundtrip: text -> embeddings -> talker -> predictor -> decoder -> WAV
- Voice clone roundtrip: ref audio -> speaker embedding + codes -> ICL prompt -> synthesis
- Custom voice roundtrip: speaker name -> token IDs -> embedding -> prompt -> synthesis
- Voice design roundtrip: description -> embedding -> prompt -> synthesis

All integration tests use committed fixture files under `tests/fixtures/`.

### 8.5 Error reporting

Replace sentinel-value returns with structured error results:

```
struct gotst::Result<T> {
    T value;
    bool ok;
    std::string error_code;
    std::string error_message;
};
```

All computational methods should return `Result<T>` in the core layer. The Godot-facing layer
translates these to `Dictionary` returns with `{ "ok": bool, "error": String, ... }`.

---

## API Surface Summary (Final)

After all phases, `GotstSpeechRuntime` exposes these method groups:

### Config and Session Management
```
set_config / get_config / clear_config
inspect_backends
load_speaker_encoder / is_speaker_encoder_loaded
load_speech_tokenizer_encoder / is_speech_tokenizer_encoder_loaded
```

### ASR Feature Extraction
```
build_log_mel_features
build_triplicate_position_ids
build_asr_language_hint_tokens
decode_asr_step
```

### TTS Feature Extraction
```
build_speaker_mel_features
extract_speaker_embedding
compute_speaker_embedding_from_audio
encode_speech_codes
```

### Voice Mode Prompt Assembly
```
build_tts_initial_language_input        // Base mode (current)
build_voice_clone_language_input         // VoiceClone ICL mode
build_custom_voice_language_input        // CustomVoice mode
build_voice_design_language_input        // VoiceDesign mode
```

### Voice Configuration
```
get_custom_voice_speaker_ids
compute_custom_voice_speaker_embedding
load_voice_clone_fixture
embed_and_sum_codec_codes
get_supported_languages
get_language_token_id
build_codec_prefix_tokens
```

### Tensor and Sampling Helpers
```
slice_rows / concat_rows / repeat_row
add_vectors / sum_vectors
extract_sequence / extract_last_hidden_row
select_last_row_argmax / select_last_row_token
should_stop_on_eos / contains_only_finite_values
```

### Output Processing
```
convert_decoder_output_to_waveform
decode_codec_frames
```

### Streaming and Cancellation
```
CancellationToken (core only, not exposed to GDScript)
```

---

## Implementation Order and Dependencies

```
Phase 0 (Foundation)
  0.1 Config expansion ─────────────────────┐
  0.2 ONNX session ownership ───────────────┤
  0.3 Test infrastructure ──────────────────┘
         │
Phase 1 (Speaker Encoder) ────────────────── no dependency on Phase 2
  1.1 Speaker mel features
  1.2 Speaker embedding extraction
  1.3 End-to-end from audio
  1.4 Tests
         │
Phase 2 (Speech Tokenizer Encoder) ───────── no dependency on Phase 1
  2.1 Speech tokenizer encoder session
  2.2 Voice clone fixture loading
  2.3 Tests
         │
Phase 3 (VoiceClone Prompt) ──────────────── depends on Phase 1 + Phase 2
  3.1 ICL prompt assembly
  3.2 x-vector-only mode
  3.3 Codec code embedding helper
  3.4 Tests
         │
Phase 4 (CustomVoice) ────────────────────── depends on Phase 0 only
  4.1 Speaker name mapping
  4.2 Custom voice prompt
  4.3 Speaker token embedding
  4.4 Tests
         │
Phase 5 (VoiceDesign) ────────────────────── depends on Phase 0 only
  5.1 Voice design prompt
  5.2 Tests
         │
Phase 6 (Streaming) ──────────────────────── depends on Phase 3, 4, or 5
  6.1 Cancellation primitives
  6.2 Streaming TTS
  6.3 Streaming ASR
  6.4 Godot streaming signals
  6.5 Tests
         │
Phase 7 (Multi-Language) ─────────────────── depends on Phase 0 only
  7.1 Language tag system
  7.2 Codec prefix construction
  7.3 ASR language identification
  7.4 Tests
         │
Phase 8 (Performance and Polish) ─────────── depends on all previous phases
  8.1 FFT optimization
  8.2 Comprehensive kernel tests
  8.3 GDScript fallback parity tests
  8.4 Integration tests
  8.5 Error reporting
```

Phases 1, 2, 4, 5, and 7 can proceed in parallel after Phase 0. Phase 3 depends on 1 and 2.
Phase 6 depends on at least one voice mode being complete. Phase 8 is last.

---

## File Structure (implemented)

```
include/gotst/
  core/
    speech_runtime_core.hpp          (existing - expand config)
    speech_encoder_session.hpp       (NEW - speaker encoder + tokenizer encoder sessions)
    tts_prompt_assembly.hpp          (NEW - voice mode prompt assembly contracts)
    cancellation_token.hpp           (NEW - streaming cancellation)
    language_config.hpp              (NEW - language tag system)
    result.hpp                       (NEW - structured error handling)
  godot/
    speech_runtime.hpp               (existing - expand with all new methods)
    speech_runtime_config.hpp        (existing - expand with voice mode properties)

src/
  core/
    speech_runtime_core.cpp          (existing - expand config and inspection)
    speech_encoder.cpp               (NEW - speaker encoder + tokenizer encoder impl)
    tts_prompt_assembly.cpp          (NEW - all voice mode prompt assembly)
    cancellation_token.cpp           (NEW - cancellation impl)
    language_config.cpp              (NEW - language config impl)
    mel_features.cpp                 (NEW - extracted mel computation, shared by ASR and speaker)
  godot/
    register_types.cpp               (existing - register new methods)
    speech_runtime.cpp               (existing - expand with all new Godot-facing methods)
    speech_runtime_config.cpp        (existing - expand properties)

tests/
  unit/
    CMakeLists.txt
    test_speech_runtime_core.cpp     (existing - expand)
    test_mel_features.cpp            (NEW)
    test_speaker_encoder.cpp         (NEW)
    test_speech_tokenizer_encoder.cpp (NEW)
    test_tts_prompt_assembly.cpp     (NEW)
    test_voice_clone_prompt.cpp      (NEW)
    test_custom_voice_prompt.cpp     (NEW)
    test_voice_design_prompt.cpp     (NEW)
    test_sampling.cpp                (NEW)
    test_waveform.cpp                (NEW)
    test_cancellation.cpp            (NEW)
    test_language_config.cpp         (NEW)
  integration/
    CMakeLists.txt
    test_asr_roundtrip.cpp           (NEW)
    test_tts_roundtrip.cpp           (NEW)
    test_voice_clone_roundtrip.cpp   (NEW)
    test_custom_voice_roundtrip.cpp  (NEW)
    test_voice_design_roundtrip.cpp  (NEW)
  fixtures/
    sine_1khz_16k.wav               (NEW - ASR test audio)
    sine_1khz_24k.wav               (NEW - speaker encoder test audio)
    sample_speech_24k.wav            (NEW - voice clone reference audio)
    expected_asr_mel.bin             (NEW - pre-computed mel features)
    expected_speaker_embedding.json  (NEW - pre-computed speaker embedding)
    expected_voice_clone_fixture.json (NEW - pre-computed voice clone fixture)
    sample_speaker_token_ids.json    (NEW - CustomVoice token IDs from config.json)
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Speaker encoder ONNX output varies across model versions | Pin exact model version in fixture tests; document expected embedding size |
| Speech tokenizer encoder requires Mimi-specific ONNX ops | Validate ops are supported by bundled ONNX Runtime during session load |
| ICL overlay math differs from Python reference | Cross-validate against Python `modeling_qwen3_tts.py` with identical inputs |
| CustomVoice token IDs change between model checkpoints | Load from `config.json` at runtime rather than hardcoding |
| Streaming adds thread safety complexity | CancellationToken uses `std::atomic`; session access is single-threaded |
| FFT change breaks existing mel feature parity | Keep DFT path as fallback; FFT path validated against same fixtures |
| gotst scope creep beyond speech-specific logic | Every new method must justify why it belongs in gotst vs gonx or GDScript |

---

## Success Criteria

gotst is feature-complete when:

1. **VoiceClone**: A reference audio clip can be encoded to speaker embedding + codec codes,
   the ICL prompt can be assembled, and synthesis produces audio in the cloned voice
2. **VoiceDesign**: A natural language voice description can be embedded and used to assemble
   a voice design prompt, producing audio matching the description
3. **CustomVoice**: Any of the 9 preset speakers can be selected by name, their token IDs
   resolved, their embedding computed, and synthesis produces audio in that speaker's voice
4. **Streaming**: TTS can emit audio incrementally and ASR can emit partial transcriptions,
   both with cancellation support
5. **Multi-language**: All supported languages work in both ASR and TTS without code changes
6. **Tests**: Every kernel has deterministic unit tests with fixture cross-validation
7. **Performance**: Mel feature extraction uses FFT; hot loops stay off the main thread
8. **Error handling**: All methods return structured errors, no silent sentinel values
