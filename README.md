# gotst

`gotst` is a speech-focused Godot 4 GDExtension for local ASR/TTS runtimes.

It is intentionally separate from:

- `godorama`: generic `llama.cpp` integration
- `gonx`: generic ONNX Runtime integration

The goal is to keep speech-specific runtime logic in one native boundary while still allowing the game to use `godorama` and `gonx` directly for other features.

## Current scope

This repository currently provides:

- a reusable native core target, `gotst::core`
- a Godot-facing GDExtension target, `gotst::godot`
- a minimal `GotstSpeechRuntimeConfig` resource
- a `GotstSpeechRuntime` native helper API used from Godot-side Qwen adapters
- a superbuild-friendly CMake layout that embeds `godorama` and `gonx` as internal core dependencies
- native implementations of the current hot speech kernels:
  - Qwen3-ASR log-mel frontend
  - packed-array row slicing/concatenation helpers
  - last-row argmax and sampling helpers
  - Qwen3-TTS initial prompt assembly helpers
  - tokenizer-decoder waveform conversion and normalization

## Architecture

`gotst` is intended to own:

- speech-specific DSP and feature extraction
- speech-side tensor and packed-array math that is too heavy for GDScript
- hybrid GGUF + ONNX orchestration for speech models as it migrates out of the game layer
- speech token/code helpers
- waveform post-processing
- speech runtime contracts exposed to Godot

`gotst` should not own:

- generic llama.cpp wrapper APIs
- generic ONNX session APIs
- scene logic or UI orchestration

## Build

Default local layout assumes sibling repositories:

- `/home/x17/code_repo/gotst`
- `/home/x17/code_repo/godorama`
- `/home/x17/code_repo/gonx`

Configure and build:

```bash
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Debug
cmake --build build/dev --parallel
ctest --test-dir build/dev --output-on-failure
```

Override dependency locations if needed:

```bash
cmake -S . -B build/dev \
  -DGOTST_GODORAMA_SOURCE_DIR=/path/to/godorama \
  -DGOTST_GONX_SOURCE_DIR=/path/to/gonx \
  -DGOTST_GODOT_CPP_SOURCE_DIR=/path/to/godot-cpp
```

## Godot addon layout

The built shared library is emitted into `addons/gotst/bin/`, and the manifest lives at:

- `addons/gotst/gotst.gdextension`

This keeps the repo directly usable as an addon workspace.

## Current responsibility split

- `project-kokage` still owns tokenizer loading, request lifecycle, and conversation orchestration.
- `gotst` owns the numerically heavy speech kernels currently called from those adapters.
- `godorama` remains the generic GGUF bridge.
- `gonx` remains the generic ONNX bridge.

## Next implementation steps

1. Move more of the Qwen TTS autoregressive inner loop from GDScript into native code.
2. Add cancellable session APIs for incremental speech streaming.
3. Expand native tests beyond backend inspection into deterministic DSP/sampling coverage.
4. Add integration tests that validate a full hybrid speech roundtrip against fixtures.
