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

### Live cancellation (`--cancel`)

The offline CAN gate cancels *before* `run()` starts, so it only exercises the entry checkpoint.
This probe lets a 600-frame generation get underway, cancels at t=1.500 s, and measures:

| | |
|---|---|
| error surfaced | `CancellationError`, **unwrapped** (never a package error type) |
| stopped at | **1.53 s** — ~30 ms after the cancel, roughly one frame of compute |

That 30 ms is the evidence the **per-frame** checkpoint fires; an uncancelled run of the same
request takes >20 s. Without this, "CAN-1..3 green" would only mean the entry checkpoint works.

## 2026-07-30 (second pass) — 18-run corpus sweep through the ENGINE

Run from the Audio8 Demo app (Release, sandboxed, engine-owned lifecycle) rather than the CLI
gate: 6 corpus voices × 3 prompt lengths, deterministic decoding. This is the pass that
established the real activation envelope — **and corrected a 27% under-declaration** the
single-utterance first pass had produced.

### Throughput

| | median | best | worst |
|---|---|---|---|
| RTF | **0.94** | 0.85 | 1.64 |

Mean 21.8 frames/s. The median matches the CLI gate's 0.96, so the engine/sandbox path costs
essentially nothing — an earlier apparent ~15% gap was **warm-up plus a single short sample**,
not overhead. The first run after a load measured RTF 1.80 against a steady-state 1.11 on the
same input; never quote a first run.

**RTF is length-dependent, and short utterances are slower than realtime:**

| prompt | audio | RTF range |
|---|---|---|
| Short | ~0.9 s | 1.05 – 1.64 |
| Medium | ~5.2 s | 0.86 – 0.94 |
| Long | 12.4 – 15.7 s | 0.85 – 0.94 |

Fixed per-request cost (reference encode + prefill) dominates a one-second utterance. The model
is faster than realtime only above roughly 5 s of output — worth knowing before promising
realtime behavior on short conversational turns.

### Activation envelope (the correction)

| audio | transient |
|---|---|
| 0.9 s | ~1.6 GB |
| 5.2 s | ~4.4 GB |
| 13.2 – 15.7 s | 5.4 – **6.35 GB** |

The first pass declared 5.00 GB from one 9.2 s utterance measuring 4413 MB. The sweep's longest
utterance (15.65 s, Mia/Long) measures **6352 MB** — 27% over the declaration. `MemoryGovernor`
reserves exactly `peakActivationBytes`, so the under-declaration silently mis-sized every
co-resident admissibility decision. Declaration raised to **7.20 GB**.

Generalizable: when activation scales with an input dimension, one sample does not establish the
envelope — sweep the range and declare the max.

### Output level (a finding, not a defect in the port)

Cloned output tracks the REFERENCE clip's level, so a hot reference yields a hot render:

| voice | rms | peak |
|---|---|---|
| iris | −10.9 … −13.4 dBFS | **−0.3 … −0.7 dBFS** |
| mia | −15.4 … −20.7 | −0.7 … −8.0 |
| clara | −26.2 … −28.7 | −11.8 … −15.2 |

Iris sits at digital full scale on every prompt (2 of 18 runs cross the −0.5 dBFS clipping
threshold outright). This is inherited from the corpus clip, not introduced by the port — but any
consumer normalizing or concatenating these voices needs headroom management.

### Not yet measured

- Quantized (int8/int4) LM tier — no quantized variant published yet.
- Cold-start prewarm effect on load time.
- Sustained multi-run memory behavior under an engine `cacheLimit` cap.
