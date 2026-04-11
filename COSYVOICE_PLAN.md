# CosyVoice3 TTS Backend Integration Plan

## Context

gotst currently implements Qwen3-TTS as its only TTS backend using a llama.cpp + ONNX hybrid
architecture. This plan adds CosyVoice3 (Alibaba's latest TTS system) as a second backend using
the same hybrid pattern. CosyVoice3 produces higher-quality speech with a simpler token structure
(single codebook, no predictor) but adds two new pipeline stages: flow-matching mel generation
(DiT) and a HiFi-GAN vocoder. The end result is that gotst can output PCM audio directly instead
of discrete codes that need an external decoder.

---

## Architectural Fit Assessment

**CosyVoice3 is an excellent fit for the llama.cpp + ONNX hybrid pattern:**

| Component | Runtime | Why |
|-----------|---------|-----|
| Qwen2 LLM (token generation) | **llama.cpp (GGUF)** | Autoregressive with KV cache — llama.cpp's strength |
| DiT flow estimator (mel generation) | **ONNX** | Non-autoregressive, 10 fixed diffusion steps |
| HiFi-GAN vocoder (waveform) | **ONNX** | Single forward pass, standard for vocoders |
| CAMPPlus speaker encoder | **ONNX** | Single forward pass, already ships as ONNX |
| Speech tokenizer v3 | **ONNX** | Single forward pass, already ships as ONNX |
| Speech/text embeddings, output decoder | **ONNX** | Simple lookup/projection, stateless |
| Euler ODE solver, CFG, sampling | **C++** | Lightweight orchestration loops |

**Simplifications over Qwen3-TTS:**
- Single LLM model (no separate predictor)
- Single codebook (6561 FSQ tokens at 25 Hz, no 15 residual codes)
- No speaker embedding in LLM (speaker only conditions the flow stage)

**New complexity:**
- Flow matching: DiT + 10-step Euler ODE solver with classifier-free guidance
- HiFi-GAN vocoder: gotst gains a mel→waveform stage (currently the vocoder lives in GDScript)
- Different mel-spectrogram parameters for different stages

---

## CosyVoice3 Pipeline Overview

```
Text + (optional) Reference Audio
  │
  ├─► Text tokenizer (Qwen2 BPE, in GDScript layer)
  │     → token IDs
  │
  ├─► Speaker encoder (CAMPPlus ONNX, 80-mel fbank @ 16 kHz)
  │     → 192-dim speaker embedding
  │
  ├─► Speech tokenizer (speech_tokenizer_v3 ONNX, Whisper 128-mel @ 16 kHz)
  │     → FSQ token sequence (vocab 6561)
  │
  └─► Mel extractor (80-mel @ 24 kHz, n_fft=1920, hop=480)
        → prompt mel-spectrogram for flow conditioning
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: LLM Token Generation (llama.cpp)           │
│                                                     │
│ Prompt: [SOS, text_emb, TASK_ID, prompt_speech_emb] │
│ Model: Qwen2ForCausalLM (GGUF)                     │
│ Embeddings: speech_embedding.onnx (6761×896)        │
│ Output head: llm_decoder.onnx (896→6761)            │
│ Sampling: RAS (top-k=25, top-p=0.8, win=10)        │
│ Output: FSQ token sequence                          │
└─────────────────┬───────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: Flow Matching (ONNX + C++ Euler solver)    │
│                                                     │
│ Token embed: Embedding(6561, 80)                    │
│ Encoder: PreLookaheadLayer (Conv1d)                 │
│ Upsample: repeat_interleave(2) → mel frame rate     │
│ Speaker: L2-norm → Linear(192, 80)                  │
│ DiT: 22 transformer blocks (1024-dim, 16 heads)     │
│ Solver: 10-step Euler ODE, cosine schedule          │
│ CFG: rate=0.7, batch doubled [guided, unguided]     │
│ Output: 80-channel mel-spectrogram                  │
└─────────────────┬───────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: Vocoder (ONNX)                             │
│                                                     │
│ Model: CausalHiFTGenerator                          │
│ F0 predictor + NSF source + HiFi-GAN + ISTFT        │
│ Upsample: 8×5×3 = 120× (mel frame → samples)       │
│ Output: 24 kHz PCM waveform                         │
└─────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Separate pipeline, shared infrastructure

CosyVoice3 is a parallel code path — not a modification of the Qwen3-TTS generator. The output
types are fundamentally different (PCM waveform vs discrete codec codes). Shared infrastructure:

- `CancellationToken`, `Result<T>` — identical
- `build_speaker_mel_features()` in `speaker_mel.cpp` — already parameterized, call with CosyVoice3 params
- `SpeakerEncoderSession` / `SpeechTokenizerEncoderSession` — reusable (different ONNX models, same interface)
- `fft.cpp` — reused for mel extraction
- Sampling utilities (extract `run_onnx_embedding` from `tts_code_generator.cpp` to shared utility)
- Session lifecycle pattern (pimpl, load/is_loaded/Result<T>)

### 2. Euler solver in C++ calling DiT ONNX per step

The 10-step Euler ODE solver runs in C++ and calls the DiT ONNX session once per step. This
gives us: cancellation between steps, progress reporting, controllable step count, and CFG
batch management. The DiT ONNX session is exported with dynamic batch (2 for CFG) and dynamic
sequence length.

### 3. Embedding strategy

CosyVoice3 has two embedding spaces that must be handled separately:

- **Text tokens**: Qwen2's `embed_tokens` (~150k vocab, 896-dim). Extracted as ONNX or looked up
  via a binary embedding table. GDScript tokenizes text, gotst embeds via ONNX.
- **Speech tokens**: Separate `Embedding(6761, 896)` covering FSQ tokens (0–6560) + 200 special
  tokens (SOS=6561, EOS=6562, TASK_ID=6563, FILL=6564, etc).

All tokens are embedded externally and fed to llama.cpp via `decode_embeddings()`. This is the
same pattern as the existing Qwen3-TTS talker.

### 4. No speaker embedding in LLM

Critical difference from Qwen3-TTS: CosyVoice3's LLM does NOT receive a speaker embedding.
The 192-dim CAMPPlus speaker embedding only conditions the flow model (projected to 80-dim).
The LLM prompt is simply: `[SOS_emb, text_emb, TASK_ID_emb, prompt_speech_emb]`.

### 5. Output type

CosyVoice3 outputs PCM audio (float32, 24 kHz) directly from gotst, unlike Qwen3-TTS which
outputs discrete codes for an external vocoder in GDScript. The Godot-facing API exposes a new
method that returns `PackedFloat32Array` audio samples.

---

## Model File Inventory

### GGUF (1 model, via llama.cpp)

| Model | Source | Notes |
|-------|--------|-------|
| `qwen2_cosyvoice3.gguf` | Convert `Qwen2ForCausalLM` from CosyVoice3 checkpoint | Standard Qwen2 transformer; text embedding table included but we bypass it via `decode_embeddings()` |

### ONNX sessions (8 models, via gonx)

| Session | Source | Input → Output | Notes |
|---------|--------|----------------|-------|
| `campplus.onnx` | CosyVoice3 release (ships as ONNX) | [1, frames, 80] → [1, 192] | Speaker encoder |
| `speech_tokenizer_v3.onnx` | CosyVoice3 release (ships as ONNX) | [feat, feat_len] → [codes] | Speech tokenizer (FSQ) |
| `cv3_text_embedding.onnx` | Extract `embed_tokens` from Qwen2 | [1, N] int64 → [1, N, 896] | Text token embedding |
| `cv3_speech_embedding.onnx` | Extract from CosyVoice3 LLM | [1, N] int64 → [1, N, 896] | Speech token embedding (6761 entries) |
| `cv3_llm_decoder.onnx` | Extract from CosyVoice3 LLM | [1, 896] → [1, 6761] | Output projection (no bias) |
| `cv3_flow_frontend.onnx` | Extract token_embedding + PreLookaheadLayer + spk_projection from flow model | tokens + spk → [1, T, 80] | Combined flow frontend |
| `cv3_dit_estimator.onnx` | Export from flow model (export support exists) | [2, 80, T] × 6 inputs → [2, 80, T] | DiT velocity estimator |
| `cv3_hifigan.onnx` | Export from CosyVoice3 vocoder | [1, 80, T] → [1, T×120] | Mel → waveform |

---

## New Files

### Core headers (`include/gotst/core/`)

| File | Purpose |
|------|---------|
| `cosyvoice3_types.hpp` | Config structs, result types, callback typedefs |
| `cosyvoice3_token_generator.hpp` | LLM stage: Qwen2 GGUF + ONNX embeddings + sampling |
| `cosyvoice3_flow_matching.hpp` | Flow stage: token embed + DiT ONNX + Euler solver |
| `cosyvoice3_vocoder.hpp` | Vocoder stage: HiFi-GAN ONNX wrapper |
| `cosyvoice3_generator.hpp` | Full pipeline orchestrator (token → mel → audio) |
| `onnx_embedding_utils.hpp` | Shared `run_onnx_embedding()` extracted from `tts_code_generator.cpp` |

### Core sources (`src/core/`)

| File | Purpose |
|------|---------|
| `cosyvoice3_token_generator.cpp` | Autoregressive loop: prefill → decode → sample → emit tokens |
| `cosyvoice3_flow_matching.cpp` | Euler solver loop with CFG, cosine schedule, token→mel pipeline |
| `cosyvoice3_vocoder.cpp` | Simple ONNX forward pass wrapper |
| `cosyvoice3_generator.cpp` | Orchestrates all three stages, streaming support |
| `onnx_embedding_utils.cpp` | Shared embedding utility (refactored from tts_code_generator.cpp) |

### Tests (`tests/unit/`)

| File | Covers |
|------|--------|
| `test_cosyvoice3_sampling.cpp` | RAS sampling, nucleus sampling, repetition-aware fallback |
| `test_cosyvoice3_flow.cpp` | Euler solver math, cosine schedule, CFG logic |
| `test_onnx_embedding_utils.cpp` | Shared embedding utility |

### Modified files

| File | Changes |
|------|---------|
| `include/gotst/core/speech_runtime_core.hpp` | Add `CosyVoice3TtsConfig` struct, extend `RuntimeConfig` |
| `include/gotst/godot/speech_runtime.hpp` | Add CosyVoice3 generator ownership and bound methods |
| `include/gotst/godot/speech_runtime_config.hpp` | Add CosyVoice3 config properties |
| `src/core/speech_runtime_core.cpp` | Add CosyVoice3 path inspection |
| `src/core/tts_code_generator.cpp` | Replace inline `run_onnx_embedding` with shared utility |
| `src/godot/speech_runtime.cpp` | Add CosyVoice3 load/generate/stream methods + bindings |
| `src/godot/speech_runtime_config.cpp` | Add CosyVoice3 property getters/setters |
| `CMakeLists.txt` | Add new source files to `gotst-core` |
| `tests/unit/CMakeLists.txt` | Add new test files |

---

## Phased Implementation

### Phase 1: Shared infrastructure & types

**Goal**: Extract shared utilities and define all CosyVoice3 types.

1. Create `onnx_embedding_utils.hpp/.cpp`:
   - Extract `run_onnx_embedding()` from `tts_code_generator.cpp` (currently in anonymous namespace at ~line 145)
   - Signature: `Result<std::vector<float>> run_onnx_embedding(gonx::InferenceSession&, span<const int64_t> input_ids, ...)`
   - Update `tts_code_generator.cpp` to call the shared version

2. Create `cosyvoice3_types.hpp`:
   ```cpp
   struct CosyVoice3ModelPaths {
       std::string qwen2_gguf_path;
       std::string text_embedding_onnx_path;
       std::string speech_embedding_onnx_path;
       std::string llm_decoder_onnx_path;
       std::string flow_frontend_onnx_path;
       std::string dit_estimator_onnx_path;
       std::string hifigan_onnx_path;
       std::string campplus_onnx_path;
       std::string speech_tokenizer_onnx_path;
   };

   struct CosyVoice3SessionConfig {
       int32_t llm_n_ctx = 4096;
       int32_t llm_n_batch = 4096;
       int32_t n_threads = -1;
       int32_t n_gpu_layers = 0;
       bool use_mmap = true;
       bool use_mlock = false;
       int32_t flash_attn_type = -1;
       int32_t type_k = -1;
       int32_t type_v = -1;
   };

   struct CosyVoice3SamplingConfig {
       int32_t speech_token_size = 6561;
       int32_t hidden_size = 896;
       int32_t top_k = 25;
       float top_p = 0.8f;
       int32_t ras_win_size = 10;
       float ras_tau_r = 0.1f;
       float min_token_text_ratio = 2.0f;
       float max_token_text_ratio = 20.0f;
       int64_t rng_seed = 1;
   };

   struct CosyVoice3FlowConfig {
       int32_t n_timesteps = 10;
       float cfg_rate = 0.7f;
       int32_t mel_channels = 80;
       int32_t token_mel_ratio = 2;
   };

   struct CosyVoice3AudioChunk {
       std::vector<float> samples;
       int32_t sample_rate = 24000;
       bool is_final = false;
   };

   struct CosyVoice3GenerateResult {
       std::vector<float> audio_samples;
       int32_t sample_rate = 24000;
       int32_t token_count = 0;
   };

   using AudioChunkCallback = std::function<void(CosyVoice3AudioChunk)>;
   ```

3. Add `CosyVoice3TtsConfig` to `speech_runtime_core.hpp`:
   ```cpp
   struct CosyVoice3TtsConfig {
       CosyVoice3ModelPaths model_paths;
       CosyVoice3SessionConfig session_config;
       CosyVoice3SamplingConfig sampling_config;
       CosyVoice3FlowConfig flow_config;
   };
   ```
   Extend `RuntimeConfig` with `CosyVoice3TtsConfig cv3_tts;`

4. Implement RAS sampling utility (in `onnx_embedding_utils.hpp` or a new `sampling_utils.hpp`):
   ```cpp
   // Repetition Aware Sampling
   // 1. nucleus_sample(logits, top_k, top_p) → candidate token
   // 2. count occurrences of candidate in last win_size tokens
   // 3. if count >= win_size * tau_r → ban candidate, random_sample(logits)
   int64_t ras_sample(span<const float> logits, span<const int64_t> history,
                      int32_t top_k, float top_p, int32_t win_size, float tau_r, uint64_t& rng);
   ```

5. Write tests: `test_cosyvoice3_sampling.cpp`, `test_onnx_embedding_utils.cpp`

### Phase 2: LLM token generator

**Goal**: Autoregressive speech token generation via Qwen2 GGUF + ONNX embeddings.

1. Create `cosyvoice3_token_generator.hpp/.cpp`:
   - `load(paths, config)` → loads Qwen2 GGUF via `LlamaModelHandle`, creates context; loads 3 ONNX sessions (text_embedding, speech_embedding, llm_decoder)
   - `generate(text_token_ids, prompt_speech_token_ids, sampling_config, cancel)` → returns `vector<int64_t>` FSQ tokens
   - `generate_streaming(...)` with a token callback for chunked emission

2. Prompt assembly logic (in `generate()`):
   ```
   // Build embedding sequence
   sos_emb = speech_embedding.run([SOS_ID=6561])      // [1, 896]
   text_emb = text_embedding.run(text_token_ids)        // [N_text, 896]
   task_id_emb = speech_embedding.run([TASK_ID=6563])   // [1, 896]
   prompt_emb = speech_embedding.run(prompt_token_ids)   // [N_prompt, 896]

   // Concatenate: [sos, text, task_id, prompt_speech]
   initial_sequence = concat(sos_emb, text_emb, task_id_emb, prompt_emb)

   // Prefill via llama.cpp
   ctx.clear_kv_cache()
   ctx.decode_embeddings(initial_sequence, total_length, hidden_size, positions, 1)
   ```

3. Autoregressive loop:
   ```
   for i in 0..max_len:
       hidden = ctx.get_embeddings_ith(-1)              // [896]
       logits = llm_decoder.run(hidden)                  // [6761]
       log_softmax(logits)
       token = ras_sample(logits, history, top_k, top_p, win_size, tau_r, rng)
       if token >= speech_token_size:                    // EOS or special
           break
       emit(token)
       next_emb = speech_embedding.run([token])          // [1, 896]
       ctx.decode_embeddings(next_emb, 1, hidden_size, next_pos, 1)
   ```

4. Key detail: The llm_decoder ONNX model outputs 6761 logits. Tokens 6561+ are special
   (SOS, EOS, TASK_ID, FILL, etc). During generation, EOS check is `token >= 6561`.
   Before `min_len`, mask out all tokens >= 6561.

### Phase 3: Flow matching (DiT)

**Goal**: Convert speech tokens to mel-spectrogram via ONNX DiT + C++ Euler solver.

1. Create `cosyvoice3_flow_matching.hpp/.cpp`:
   - `load(paths)` → loads flow_frontend ONNX and dit_estimator ONNX
   - `generate_mel(tokens, prompt_tokens, prompt_mel, speaker_embedding, config, cancel)` → returns `vector<float>` mel [80 × T]

2. Flow frontend processing:
   ```
   // Token embedding + PreLookaheadLayer (combined ONNX)
   h = flow_frontend.run(concat(prompt_tokens, tokens), speaker_embedding)
   // h shape: [1, T_tokens, 80]

   // Upsample to mel frame rate (C++)
   h = repeat_interleave(h, token_mel_ratio=2)   // [1, T_mel, 80]
   h = transpose(h)                               // [1, 80, T_mel]

   // Build condition mel
   mel_len1 = prompt_mel.frames
   mel_len2 = h.frames - mel_len1
   cond = zeros(1, 80, mel_len1 + mel_len2)
   cond[:, :, :mel_len1] = prompt_mel
   ```

3. Euler ODE solver with CFG (C++ loop):
   ```
   // Cosine time schedule
   t_span[i] = 1.0 - cos(pi/2 * i/n_timesteps)   for i in 0..n_timesteps

   // Initialize noise
   z = deterministic_randn(1, 80, T_mel, seed)
   x = z

   for step in 1..n_timesteps:
       t = t_span[step-1]
       dt = t_span[step] - t_span[step-1]

       // Check cancellation
       if cancel && cancel->is_cancelled(): return error

       // CFG: double batch [guided, unguided]
       x_in = stack([x, x])                  // [2, 80, T]
       mu_in = stack([h, zeros_like(h)])      // [2, 80, T] (unguided has zeroed condition)
       spks_in = stack([spk, zeros(80)])      // [2, 80]
       cond_in = stack([cond, zeros_like(cond)])
       t_in = [t, t]                          // [2]

       // DiT forward pass (ONNX)
       velocity = dit_estimator.run(x_in, mask, mu_in, t_in, spks_in, cond_in)

       // Apply CFG
       v_guided = velocity[0]
       v_unguided = velocity[1]
       v = (1.0 + cfg_rate) * v_guided - cfg_rate * v_unguided

       // Euler step
       x = x + dt * v

   mel = x[:, :, mel_len1:]   // strip prompt portion
   ```

4. Write tests: `test_cosyvoice3_flow.cpp` — cosine schedule values, CFG math, solver step verification with known inputs.

### Phase 4: Vocoder

**Goal**: Convert mel-spectrogram to 24 kHz PCM waveform.

1. Create `cosyvoice3_vocoder.hpp/.cpp`:
   - `load(hifigan_onnx_path)` → loads HiFi-GAN ONNX session
   - `synthesize(mel, mel_frames, mel_channels)` → returns `vector<float>` PCM samples

2. Implementation is a straightforward ONNX session wrapper:
   ```
   input: mel tensor [1, 80, T_mel]
   output: audio tensor [1, T_audio]  where T_audio ≈ T_mel × 480 (hop_size at 24kHz)
   ```

3. Post-processing (in C++):
   - Clip to [-0.99, 0.99] (matching CosyVoice3's `audio_limit`)
   - Optional: reuse existing `convert_decoder_output_to_waveform` for DC removal, normalization, edge fade

### Phase 5: Full pipeline orchestrator

**Goal**: Wire all three stages together with non-streaming generation.

1. Create `cosyvoice3_generator.hpp/.cpp`:
   - `load(model_paths, session_config)` → loads all sub-components
   - `generate(text_token_ids, prompt_speech_token_ids, prompt_mel, speaker_embedding, sampling, flow_config, cancel)` → `CosyVoice3GenerateResult`

2. Pipeline:
   ```
   tokens = token_generator.generate(text_ids, prompt_speech_ids, sampling, cancel)
   mel = flow_matching.generate_mel(tokens, prompt_tokens, prompt_mel, spk_emb, flow_config, cancel)
   audio = vocoder.synthesize(mel)
   return CosyVoice3GenerateResult { audio, 24000, tokens.size() }
   ```

3. Add to `speech_runtime_core.hpp`: Extend `BackendSummary` with `cv3_tts_hybrid_ready`, `cv3_*_ready` fields.

4. Update `SpeechRuntimeCore::inspect()` for CosyVoice3 paths.

### Phase 6: Godot bindings

**Goal**: Expose CosyVoice3 to GDScript.

1. Update `speech_runtime_config.hpp/.cpp`:
   - Add all CosyVoice3 model path properties
   - Add sampling/flow config properties

2. Update `speech_runtime.hpp/.cpp`:
   - Add `std::unique_ptr<gotst::CosyVoice3Generator> cv3_generator_`
   - Bind methods:
     - `load_cosyvoice3_generator()` → `Result<void>`
     - `is_cosyvoice3_generator_loaded()` → `bool`
     - `generate_cosyvoice3_speech(text_token_ids: PackedInt64Array, prompt_speech_token_ids: PackedInt64Array, prompt_mel: PackedFloat32Array, prompt_mel_frames: int, speaker_embedding: PackedFloat32Array, params: Dictionary)` → `Dictionary` with `audio: PackedFloat32Array, sample_rate: int`
   - Register in `_bind_methods()`

3. Frontend helpers to expose at the Godot boundary:
   - `build_cosyvoice3_speaker_mel(waveform, sample_rate)` → 80-mel fbank at 16kHz (for CAMPPlus)
   - `build_cosyvoice3_flow_mel(waveform, sample_rate)` → 80-mel at 24kHz (n_fft=1920, hop=480, for flow conditioning)
   - `extract_cosyvoice3_speaker_embedding(mel)` → 192-dim (CAMPPlus ONNX)
   - `extract_cosyvoice3_speech_tokens(whisper_mel)` → FSQ token sequence

### Phase 7: Streaming support

**Goal**: Chunked token→mel→audio pipeline with callbacks.

1. Add `generate_streaming()` to `CosyVoice3Generator`:
   - LLM generates tokens, accumulates in buffer
   - Every `token_hop_len` (25) tokens + `pre_lookahead_len` (3) context: run flow matching on the chunk
   - Flow output mel fed to vocoder, audio emitted via `AudioChunkCallback`
   - `token_hop_len` grows from 25 to 100 (stream_scale_factor=2) for better quality in later chunks

2. Streaming flow details:
   - Uses CausalConditionalCFM with pre-computed deterministic noise (matching CosyVoice3's `self.rand_noise`)
   - Token offset tracking: only emit newly generated mel frames per chunk
   - Vocoder runs on accumulated mel (with offset tracking to avoid re-synthesizing)

3. Godot streaming methods:
   - `start_cosyvoice3_stream(...)` → starts generation in background
   - `poll_cosyvoice3_stream()` → returns queued audio chunks

---

## Mel-Spectrogram Parameter Reference

| Use case | Sample rate | Mel bins | n_fft | hop | fmin | fmax | Normalization |
|----------|------------|----------|-------|-----|------|------|---------------|
| CAMPPlus speaker encoder | 16 kHz | 80 | Kaldi default | Kaldi default | — | — | Mean subtraction |
| Speech tokenizer v3 | 16 kHz | 128 | Whisper defaults | Whisper defaults | — | — | Whisper log-mel |
| Flow conditioning mel | 24 kHz | 80 | 1920 | 480 | 0 | None (Nyquist) | Standard log-mel |
| Existing ASR mel (unchanged) | 16 kHz | 128 | 400 | 160 | 0 | 8000 | Whisper-style |
| Existing speaker mel (unchanged) | 24 kHz | 128 | 1024 | 256 | 0 | 12000 | log(clip(x, 1e-5)) |

The existing `build_speaker_mel_features()` is parameterized and handles all mel variants.
The CAMPPlus mel uses Kaldi-style fbank (mean-subtracted) which needs a small adapter.

---

## Verification Plan

1. **Unit tests** (Catch2, run via `ctest`):
   - RAS sampling with known logits and history
   - Cosine time schedule computation
   - CFG combination formula
   - Euler solver step with identity velocity (x + dt × 1 = x + dt)
   - Shared `run_onnx_embedding` with mock session

2. **Integration tests** (require model files):
   - Token generator: fixture prompt → expected token sequence (deterministic with seed)
   - Flow matching: fixture tokens → expected mel shape and value range
   - Vocoder: fixture mel → expected audio shape and sample range
   - Full pipeline: text → audio, verify sample rate and non-silence

3. **Manual testing** (Godot):
   - Load CosyVoice3 models in Godot project
   - Generate speech for test sentences in English and Chinese
   - Compare audio quality to Python reference implementation
   - Test streaming latency (first chunk time)
