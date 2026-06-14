# gotst CLI

Standalone native tooling for speech runtime checks and synthesis.

## Build

From a standalone `gotst` checkout with sibling `godorama` and `gonx`
checkouts:

```sh
cmake -S . -B build/cli -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGOTST_BUILD_GDEXTENSION=OFF \
  -DGOTST_BUILD_TESTS=OFF \
  -DGOTST_BUILD_CLI=ON

cmake --build build/cli --parallel
```

The executable is written to:

```text
build/cli/cli/gotst
```

This build enables `tokenizers-cpp` through godorama and requires Rust/Cargo for
the HuggingFace tokenizer binding.

For non-sibling dependency layouts, pass `GOTST_GODORAMA_SOURCE_DIR` and
`GOTST_GONX_SOURCE_DIR` during CMake configure. See
[`../docs/setup_build.md`](../docs/setup_build.md).

## Irodori-TTS

```sh
build/cli/cli/gotst irodori-tts \
  --bundle /path/to/irodori_bundle \
  --text "こんにちは。" \
  --no-ref \
  --seed 42 \
  --num-steps 6 \
  --output speech.wav
```

`--manifest` defaults to `irodori_bundle.json` under the bundle root. Artifact
paths are resolved from the manifest's `artifacts` object unless overridden with
explicit path flags such as `--text-encoder`, `--dit-step`, or
`--dacvae-decoder`.

Irodori text normalization and HuggingFace tokenizer encoding are owned by
`gotst-core`. The CLI passes raw text/caption request fields through the same
session path used by the Godot binding; raw token arrays are still accepted by
the core session for compatibility tests and older callers.

For `Aratako/Irodori-TTS-600M-v3-VoiceDesign`, use `--mode voice_design_v3`
and provide `--caption`. Reference audio is optional; `--no-ref` exercises the
pure Text + Caption path, while `--ref-wav` adds speaker reference conditioning.

## Qwen3-TTS

```sh
build/cli/cli/gotst qwen-tts \
  --bundle /path/to/qwen3_tts_bundle \
  --text "こんにちは。" \
  --speaker-embedding /path/to/speaker_embedding.json \
  --output speech.wav
```

`qwen-tts` performs synthesis through the shared gotst `QwenTtsPipeline`.
Tokenizer loading, text projection, prompt assembly, GGUF code generation,
custom voice speaker-token embedding, ICL reference-code handling, and waveform
decode all use the same native path as the Godot binding. The CLI supports
`base`, `voice_design`, and `custom_voice` modes:

```sh
build/cli/cli/gotst qwen-tts \
  --bundle /path/to/qwen3_tts_bundle \
  --mode voice_design \
  --voice-design "落ち着いた声で、少し嬉しそうに読む。" \
  --text "準備できたよ。" \
  --output speech.wav
```

The CLI only maps flags into a pipeline config/request and writes the resulting
PCM to WAV.

An exported CustomVoice bundle from `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
can be checked with:

```sh
build/cli/cli/gotst qwen-tts \
  --bundle /path/to/qwen3_tts_custom_voice_bundle \
  --mode custom_voice \
  --speaker serena \
  --text "こんにちは。今日はいい天気です。" \
  --text-projection talker_text_projection.onnx \
  --predictor-gguf predictor.q8_0.gguf \
  --decoder /path/to/qwen3_tts_base_bundle/speech_tokenizer_decoder_stateful.onnx \
  --output speech.wav
```
