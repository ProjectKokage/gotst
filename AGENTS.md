# AGENTS.md

## Purpose

This repository contains a dedicated native speech runtime for Godot. It exists to hold speech-specific
performance-sensitive logic that should not live in GDScript, `godorama`, or `gonx`.

## Boundaries

- Keep `godorama` generic. Do not move generic llama.cpp wrapper responsibilities into this repo.
- Keep `gonx` generic. Do not move generic ONNX session APIs into this repo.
- Put speech-specific hybrid runtime logic here.
- Current shipped native scope includes the Qwen3-ASR frontend, packed-array/tensor helpers, Qwen3-TTS prompt assembly helpers, sampling helpers, and waveform post-processing.
- Keep Godot-facing classes thin. The heavy work belongs in the native core layer.

## Build rules

- The preferred local layout is this repo beside `godorama` and `gonx`.
- Reuse a shared `godot-cpp` target when present.
- Depend on `gonx-core` and `godot_llama::core` rather than on their GDExtension shared libraries.
- Preserve Flatpak-safe runtime packaging by keeping ONNX Runtime beside the built addon and using `$ORIGIN`-style RPATHs where applicable.

## Coding rules

- Keep the speech core free of Godot headers unless code is specifically part of the Godot boundary layer.
- Favor typed native contracts over loosely structured `Dictionary` payloads inside the core.
- Keep cancellation explicit in every future streaming API.
- Do not push scene or UI policy into this repo.

## Testing

- Native unit tests should validate path/config inspection, DSP helpers, tensor packing, sampling behavior, and cancellation behavior.
- Integration tests should cover hybrid GGUF + ONNX session flows with reproducible fixtures.
- A runtime path is not done until the Godot-facing wrapper and native tests are updated together.
