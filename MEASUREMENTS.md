# mlx-audio8-tts-swift — measurements

Everything here is a recorded run of `swift run -c release audio8-gates --validate`, which
drives the real `Audio8Package` through construct → `load()` → `run()` → `unload()` (the engine's
lifecycle, headless) and reports the numbers the manifest declares. Numbers are not estimated;
if a value is missing it is because the run that would produce it has not happened.

## 2026-07-30 — bf16 LM + fp32 codec, release build

Model: `release/Audio8-TTS-Preview-0.6b-bf16` (the artifact published as
`mlx-community/Audio8-TTS-Preview-0.6b-bf16`). Reference clip: `roxy_d1_trim` (10.2 s, 24 kHz,
resampled to 44.1 kHz by the wrapper).

### Output + timing

| | |
|---|---|
| response | `.wav`, 44100 Hz, 1 ch, 405 504 samples |
| audio level | **−18.8 dBFS** RMS over 9.20 s (silence would read −∞) |
| load | 0.09–0.44 s (warm vs cold page cache) |
| run | 8.85 s → **RTF 0.96** (faster than realtime, release build) |

### Memory

| | MLX-accounted | process phys |
|---|---|---|
| baseline | — | 10 MB |
| **resident** (post-load, pre-run) | **2434 MB** | 2449 MB |
| **peak** (9.2 s utterance) | **6847 MB** | 17 621 MB |
| transient (peak − resident) | **4413 MB** | — |
| post-`unload()` | active **0 MB** · cache **0 MB** | 10 674 MB |

**Declared in the manifest:** `residentBytes` 2.60 GB, `peakActivationBytes` 5.00 GB.
`peakActivationBytes` is the *transient*, not the absolute high-water — the engine adds it to
the resident floor when admitting, so declaring the absolute peak would double-count the weights.

### Two things this run settled

1. **`unload()` genuinely reclaims.** MLX active and cache both return to 0 MB. The ~10 GB of
   process `phys_footprint` still resident afterwards is allocator/Metal page retention that the
   package does not control — *not* a package leak. This is the distinction that "memory never
   releases" reports usually miss, so it is measured rather than assumed.
2. **The transient is utterance-length-driven, and it dominates.** Same model, same reference:

   | utterance | resident | peak | transient |
   |---|---|---|---|
   | short line (≤60 frames) | 2434 MB | 4057 MB | **1623 MB** |
   | 9.2 s (400-frame cap) | 2434 MB | 6847 MB | **4413 MB** |

   Cause: the codec decodes the whole utterance through its deep conv stack in one pass. This is
   the same shape Gepard had before its streaming decode landed, and it is the obvious target for
   a future windowed-decode optimization (which would also unlock `StreamEmitting`).

### Not yet measured

- Quantized (int8/int4) LM tier — no quantized variant published yet.
- Cold-start prewarm effect on load time.
- Sustained multi-run memory behavior under an engine `cacheLimit` cap.
