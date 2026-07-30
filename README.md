# mlx-audio8-tts-swift

Swift-MLX port of **[Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)**
— a multilingual, zero-shot voice-cloning text-to-speech model — wrapped as a conformant
[MLXEngine](https://github.com/xocialize/mlx-engine-swift) `tts` `ModelPackage` for Apple Silicon.

Clone a voice from one reference clip plus its transcript, and speak in any of 11 languages.
44.1 kHz mono `.wav` out.

```
reference clip ──▶ arktts codec encode ──▶ semantic + residual codes ┐
                                                                     ├─▶ slow AR ─┬─▶ fast AR ─▶ codec decode ─▶ waveform
text + transcript ──▶ Qwen tokenizer ──▶ chat-format prompt ─────────┘   (1 tok/frame) (10 codebooks/frame)
```

## Highlights

- **Token-exact with the PyTorch reference.** Greedy decoding reproduces the original model's
  code sequence for **102/102 frames**, and codec encoding is 100% code-exact. The port is
  exact, not merely close — see the gate table below.
- **Multilingual**: Cantonese, Chinese, Dutch, English, French, German, Italian, Japanese,
  Korean, Polish, Spanish.
- **Zero-shot cloning** from a reference clip + transcript (ICL-style — the transcript must match
  what the clip says). `voice.auto` uses the model's own default voice.
- **Faster than realtime**: RTF **0.96** on a 9.2 s utterance (release build, bf16 LM + fp32 codec).
- **Single permissive weight layer** — Apache-2.0 upstream, so it ships under MLXEngine's default
  `.permissiveOnly` policy with no acknowledgement flow.
- **Measured footprint**, not estimated: 2434 MB resident, 4413 MB transient on a 9.2 s
  utterance ([MEASUREMENTS.md](MEASUREMENTS.md)).

## Architecture

Two modules (the fleet's core/wrapper split):

| Module | Role |
|---|---|
| **`Audio8TTSCore`** | The inference port — the DualAR transformer (24-layer slow AR emitting one semantic token per audio frame, 4-layer fast AR emitting that frame's 10 codec codebooks), the 44.1 kHz arktts codec (DAC-style encoder/decoder, split semantic + residual RVQ, windowed transformer pre/post modules), reference-exact sampling, and the prompt builder. Depends on MLX + swift-transformers only — **MLXToolKit-free**. |
| **`MLXAudio8TTS`** | The engine-facing wrapper — `Audio8Configuration` + `Audio8Package` (the `tts` surface), the license gate, `WeightSourcing` auto-materialization, and the cancellation seams. |

## Install

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/xocialize/mlx-audio8-tts-swift", from: "0.1.0"),
]
```

Products: `Audio8TTSCore` (the model), `MLXAudio8TTS` (the engine package). Requires **macOS 26+**
and Apple Silicon (Metal GPU).

## Usage (via MLXEngine)

The package is engine-owned: you register it with an `MLXServeEngine`, which constructs, loads,
drives, and evicts it.

```swift
import MLXAudio8TTS

try engine.register(Audio8Package.self, configuration: Audio8Configuration())
try await engine.prepare(.tts)

let response = try await engine.run(TTSRequest(
    text: "Bonjour, ceci est une voix clonée.",
    voice: VoiceSelector(.referenceAudio(referenceClip)),
    referenceTranscript: "The transcript of what the reference clip says.",
    metaData: ["seed": .int(42)]
)) as! TTSResponse

try response.audio.data.write(to: outputURL)   // .wav, 44.1 kHz mono
```

### Request options (`metaData`)

| key | type | default | meaning |
|---|---|---|---|
| `temperature` | double | 0.7 | sampling temperature |
| `topP` | double | 0.9 | nucleus threshold |
| `topK` | int | 50 | top-k cutoff |
| `seed` | int | random | reproducible sampling (not a no-op — the default path samples) |
| `greedy` | bool | false | deterministic argmax; the configuration the S2 gate proves token-exact |
| `maxFrames` | int | 512 | generation cap; one frame ≈ 46 ms of audio |

## Direct use (without the engine)

```swift
import Audio8TTSCore

let model = try Audio8TTS.load(directory: modelDirectory)
let result = try await model.generate(
    text: "Running directly against the core.",
    referenceAudio: referenceSamples44k,       // mono fp32 at 44.1 kHz
    referenceText: "Transcript of the reference clip.",
    params: SamplingParams(seed: 42))
```

## Parity gates

Run as CLI modes (`swift run -c release audio8-gates --s1`) rather than XCTest — the SPM test
product's metallib is unreliable for kernel work. Fixtures are fp32 dumps from the PyTorch
reference (45 tensors).

| Gate | What it proves | Result |
|---|---|---|
| `--s0` | key contract: module keys == checkpoint keys | 226 LM params, 0 missing / 0 unused |
| `--s1` | LM + codec unit parity vs the oracle | rope tables **max_abs 0.0**; all units ≤ 7.6e-6 |
| `--s2` | end-to-end vs the oracle | codec encode **100% code-exact**; greedy **102/102 frames token-exact**; waveform 4.6e-6 |
| `--s2b` | tokenizer + one real GPU generation | prompt ids exact; 127 frames @ −18.5 dBFS |
| `--validate` | the engine path + measured footprint | RTF 0.96, −18.8 dBFS, footprint table |
| `--cancel` | live MID-RUN cancellation (the offline gate only covers the entry checkpoint) | unwrapped `CancellationError` 30 ms after cancel |

Offline conformance (manifest, MAT-1..5, CAN-1..3, C14/INF) runs in `swift test` — 15 tests.

## License

Apache-2.0, matching the upstream model weights.
