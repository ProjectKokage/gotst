# Roundtrip Verification Handover

## Purpose

This document hands off the missing model-backed roundtrip task for `gotst`.

Scope of the handoff:

- real `text -> TTS -> waveform -> ASR -> text` verification
- committed integration fixtures and a repeatable harness
- clear ownership boundaries between `gotst`, `godorama`, `gonx`, and the game layer

This report reflects the repository state on 2026-04-11.

## Current Status

The performance-focused native work is complete for the optimization pass that was requested:

- TTS sampling now reuses scratch buffers and avoids full candidate sorts.
- Single-token ONNX embedding calls now reuse input state instead of rebuilding tiny tensors every step.
- TTS prompt assembly moved out of the Godot boundary into the native core layer.
- ASR log-mel extraction moved into native core and now caches sparse mel plans.
- FFT setup now caches bit-reversal and twiddle state.

Verification completed for those changes:

- targeted test runs after each optimization pass
- final full native test suite run
- final result: `165/165` tests passed

What is still **not** in place is a true model-backed roundtrip harness. That is the remaining task.

## Why Roundtrip Was Not Completed Here

The repository does not currently contain:

- a committed speech model bundle
- a roundtrip fixture manifest that binds text fixtures to model paths and expected outputs
- an integration test target that loads the full TTS and ASR path end to end

Because of that, the requested `text -> TTS -> ASR` verification could not be executed in this workspace without external model provisioning.

## Existing Native Capabilities

### TTS path pieces already implemented

- Prompt assembly helpers:
  - `include/gotst/core/tts_prompt_assembly.hpp`
  - `src/core/tts_prompt_assembly.cpp`
- TTS autoregressive code generation:
  - `include/gotst/core/tts_code_generator.hpp`
  - `src/core/tts_code_generator.cpp`
- Godot wrapper entry points:
  - `GotstSpeechRuntime::build_tts_initial_language_input`
  - `GotstSpeechRuntime::build_voice_clone_language_input`
  - `GotstSpeechRuntime::build_custom_voice_language_input`
  - `GotstSpeechRuntime::build_voice_design_language_input`
  - `GotstSpeechRuntime::load_tts_code_generator`
  - `GotstSpeechRuntime::generate_tts_codes`
  - `GotstSpeechRuntime::start_tts_waveform_stream`
  - `GotstSpeechRuntime::poll_tts_waveform_stream`
  - `GotstSpeechRuntime::cancel_tts_waveform_stream`
- Waveform post-processing helper:
  - `GotstSpeechRuntime::convert_decoder_output_to_waveform`

### ASR path pieces already implemented

- ASR log-mel frontend:
  - `include/gotst/core/asr_frontend.hpp`
  - `src/core/asr_frontend.cpp`
- ASR token decoding:
  - `include/gotst/core/asr_token_decoder.hpp`
  - `src/core/asr_token_decoder.cpp`
- Godot wrapper entry points:
  - `GotstSpeechRuntime::build_log_mel_features`
  - `GotstSpeechRuntime::load_asr_token_decoder`
  - `GotstSpeechRuntime::decode_asr_tokens`

### Additional model-backed helpers already present

- speaker encoder session:
  - `include/gotst/core/speech_encoder_session.hpp`
  - `src/core/speech_encoder_session.cpp`
- speech tokenizer encoder session:
  - same files as above

These are useful for voice-clone coverage, but they do not close the full text-to-text roundtrip by themselves.

## Important Current Gaps

The next owner should treat these as the actual blockers.

### 1. No integrated roundtrip harness

Current tests are only unit and parity tests:

- see `tests/unit/CMakeLists.txt`
- no `tests/integration/` target exists
- no model-backed roundtrip tests are registered today

### 2. Some configured model paths are only inspected, not executed

`SpeechRuntimeCore::inspect` checks file existence for these paths:

- `asr_frontend_model_path`
- `asr_text_embedding_model_path`
- `asr_thinker_model_path`
- `tts_text_embedding_model_path`
- `tts_speaker_embedding_model_path`
- `tts_audio_decoder_model_path`
- `tts_talker_model_path`
- `tts_predictor_model_path`
- optional:
  - `tts_speaker_encoder_model_path`
  - `tts_speech_tokenizer_encoder_model_path`

But only part of that stack is actually loaded through native runtime classes today.

Concrete gap:

- full TTS still needs the model-backed steps that produce projected text/speaker states and run the audio decoder output before waveform conversion
- full ASR still needs the model-backed steps that turn waveform-derived features into `prompt_embeddings` for `decode_asr_tokens`

### 3. Boundary ownership is still partially split with the game layer

Per `README.md`:

- `project-kokage` still owns tokenizer loading, request lifecycle, and conversation orchestration
- `gotst` owns numerically heavy speech kernels

Before implementing roundtrip tests, decide whether the missing TTS/ASR model-backed steps belong:

- inside `gotst` core as speech-specific inference helpers, or
- in the consuming project with `gotst` only supplying kernels and decoder/generator primitives

That decision affects both the test harness shape and the amount of code that should land in this repo.

### 4. Backend inspection messaging is stale

`src/core/speech_runtime_core.cpp` still includes notes saying:

- `Scaffold only: gotst does not load models or run inference yet.`
- `The current API verifies dependency wiring and speech bundle configuration only.`

That is no longer fully accurate because the repo now includes working native generator/decoder/session code. The note should be corrected when the roundtrip task is picked up.

## Exact Runtime Inputs Already Exposed

These are the current entry points that a roundtrip harness can call.

### TTS generator load config keys

Used by `GotstSpeechRuntime::load_tts_code_generator`:

- `talker_gguf_path`
- `predictor_gguf_path`
- `codec_embedding_onnx_path`
- `predictor_embedding_onnx_path`
- optional session knobs:
  - `talker_n_ctx`
  - `talker_n_batch`
  - `predictor_n_ctx`
  - `predictor_n_batch`
  - `n_threads`
  - `n_gpu_layers`
  - `use_mmap`
  - `use_mlock`
  - `flash_attn_type`
  - `type_k`
  - `type_v`
  - `talker_position_components`
  - `predictor_position_components`

### TTS generation parameter keys

Used by `GotstSpeechRuntime::generate_tts_codes` and streaming variant:

- `initial_language_input`
- `initial_sequence_length`
- `trailing_text_hidden`
- `trailing_text_length`
- `tts_pad_embedding`
- `codebook_size`
- `residual_groups`
- `target_frames`
- `min_frames_before_eos`
- `hidden_size`
- `eos_token_id`
- `eos_logit_margin`
- `do_sample`
- `top_k`
- `top_p`
- `temperature`
- `sub_do_sample`
- `sub_top_k`
- `sub_top_p`
- `sub_temperature`
- `repetition_penalty`
- `rng_seed`

### ASR decoder load config keys

Used by `GotstSpeechRuntime::load_asr_token_decoder`:

- `thinker_gguf_path`
- `embedding_onnx_path`
- optional session knobs:
  - `n_ctx`
  - `n_batch`
  - `n_threads`
  - `n_gpu_layers`
  - `use_mmap`
  - `use_mlock`
  - `flash_attn_type`
  - `type_k`
  - `type_v`
  - `position_components`

### ASR decode parameter keys

Used by `GotstSpeechRuntime::decode_asr_tokens`:

- `prompt_embeddings`
- `prompt_length`
- `max_tokens`
- `hidden_size`
- `vocab_size`
- `eos_token_id`

### ASR frontend helper inputs

Used by `GotstSpeechRuntime::build_log_mel_features`:

- waveform samples
- `input_sample_rate`
- `sample_rate`
- `mel_bins`
- `fft_size`
- `hop_length`
- `chunk_length_seconds`

## Recommended Task Breakdown

This is the suggested execution plan for the next owner.

### Phase 1: Set the boundary

Decide which of these two models is correct:

1. `gotst` owns the full speech-specific hybrid runtime and should expose enough native helpers to run fixture-backed TTS and ASR completely inside this repo.
2. `gotst` remains a lower-level speech kernel library and the roundtrip harness should live in the consuming project, with this repo only hosting lower-level fixture tests.

Recommendation:

- choose option 1 for the speech-specific path pieces that are already naturally in this repo
- keep generic session abstractions in `gonx` and generic llama plumbing in `godorama`
- do not put generic wrapper responsibilities into `gotst`

### Phase 2: Add fixture and model provisioning rules

Create a clear contract for local execution:

- fixture text inputs under `tests/fixtures/roundtrip/`
- fixture reference audio under `tests/fixtures/roundtrip/`
- a model manifest or environment-variable contract for locating local models
- skip behavior when models are unavailable

Recommended environment variables:

- `GOTST_QWEN_TTS_TALKER_GGUF`
- `GOTST_QWEN_TTS_PREDICTOR_GGUF`
- `GOTST_QWEN_TTS_CODEC_EMBED_ONNX`
- `GOTST_QWEN_TTS_PREDICTOR_EMBED_ONNX`
- `GOTST_QWEN_TTS_TEXT_EMBED_ONNX`
- `GOTST_QWEN_TTS_SPEAKER_EMBED_ONNX`
- `GOTST_QWEN_TTS_AUDIO_DECODER_ONNX`
- `GOTST_QWEN_ASR_FRONTEND_ONNX`
- `GOTST_QWEN_ASR_TEXT_EMBED_ONNX`
- `GOTST_QWEN_ASR_THINKER_GGUF`
- `GOTST_QWEN_ASR_DECODER_EMBED_ONNX`

If the consuming project already has a model manifest format, reuse it instead of inventing a second one.

### Phase 3: Close the missing model-backed steps

TTS side still needs a repeatable way to produce:

- projected text states
- projected speaker or voice-condition states
- decoder outputs that can be fed into `convert_decoder_output_to_waveform`

ASR side still needs a repeatable way to produce:

- prompt embeddings for `decode_asr_tokens`

Implementation rule from `AGENTS.md`:

- keep heavy work in native core
- keep Godot-facing classes thin

That means any missing speech-specific inference helper should land in `src/core/` and `include/gotst/core/`, with `speech_runtime.cpp` acting only as a wrapper.

### Phase 4: Add dedicated integration tests

Recommended new target:

- `tests/integration/`

Recommended test files:

- `test_tts_roundtrip.cpp`
- `test_asr_roundtrip.cpp`
- `test_text_tts_asr_roundtrip.cpp`
- optional:
  - `test_voice_clone_roundtrip.cpp`
  - `test_custom_voice_roundtrip.cpp`
  - `test_voice_design_roundtrip.cpp`

Recommended execution behavior:

- tests are compiled by default
- model-backed cases self-skip when required models are absent
- a single explicit tag such as `[integration][roundtrip]` isolates them from fast unit runs

### Phase 5: Add deterministic or bounded-output assertions

Use acceptance checks that reflect model behavior realistically.

For TTS:

- waveform is finite
- sample rate matches expected decoder output
- duration falls within a bounded window
- code count and frame count are internally consistent
- no NaN/Inf anywhere in intermediate tensors

For ASR:

- decoded token sequence is non-empty
- EOS behavior is sane
- decoded text matches a committed expected string or bounded normalization rule

For text -> TTS -> ASR:

- input text normalizes to the same output text
- if exact match is not realistic, define an allowed normalization layer up front
- do not use hand-wavy subjective checks

## Minimum Acceptance Criteria

The task should not be considered complete until all of the following are true.

1. A documented local command exists for model-backed roundtrip verification.
2. The command either passes or self-skips with an explicit missing-model reason.
3. At least one fixture exercises `text -> TTS -> waveform -> ASR -> text`.
4. The fixture assertion is deterministic or bounded by a documented normalization rule.
5. New model-backed coverage is isolated from the fast unit suite.
6. The Godot-facing wrapper and native tests are updated together.
7. Any new inference helper added for roundtrip support lives in native core, not only in `speech_runtime.cpp`.

## Suggested Command Shape

For local development:

```bash
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Debug
cmake --build build/dev --parallel
ctest --test-dir build/dev --output-on-failure -L roundtrip
```

If labels are not used, use an explicit regex for the integration target instead.

## Risks and Failure Modes

- Model assets may be too large or redistribution-restricted to commit under `tests/fixtures/`.
- Exact string equality may be too brittle unless text normalization is clearly defined.
- The repo boundary may drift if generic session logic is reimplemented in `gotst`.
- A Godot-only harness would slow iteration and make CI harder; prefer native integration coverage first.
- The existing backend inspection notes may mislead new contributors unless corrected.

## Recommended First Deliverable

The fastest useful first milestone is:

1. add a model-path contract
2. add one native integration test for `text -> TTS code generation -> audio decode -> waveform`
3. add one native integration test for `waveform -> ASR prompt embeddings -> token decode`
4. then connect them into a single `text -> TTS -> ASR` roundtrip fixture

This sequence avoids building the full end-to-end test blind.

## Files Most Relevant To The Next Owner

- `README.md`
- `docs/feature_plan.md`
- `include/gotst/godot/speech_runtime.hpp`
- `include/gotst/core/speech_runtime_core.hpp`
- `include/gotst/core/tts_code_generator.hpp`
- `include/gotst/core/asr_token_decoder.hpp`
- `include/gotst/core/asr_frontend.hpp`
- `include/gotst/core/tts_prompt_assembly.hpp`
- `src/godot/speech_runtime.cpp`
- `src/core/speech_runtime_core.cpp`
- `src/core/tts_code_generator.cpp`
- `src/core/asr_token_decoder.cpp`
- `src/core/asr_frontend.cpp`
- `src/core/speech_encoder_session.cpp`
- `tests/unit/CMakeLists.txt`

## Final Handoff Summary

The repo now has the optimized native kernels needed for the heavy inner loops, and the native test suite is green. The remaining work is not another micro-optimization pass. It is a model-provisioned integration task: decide the repo boundary, wire the missing model-backed steps needed for full TTS and ASR execution, add fixture-backed integration tests, and make the combined roundtrip reproducible.
