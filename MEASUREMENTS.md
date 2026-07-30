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

## 2026-07-30 (third pass) — the activation MODEL, and long-form orchestration

Two more measured points (a 48 s one-shot at 1035 frames, and a 7-segment orchestrated render of
the same passage) turned the activation envelope from a list of samples into a fitted model.

### The transient is linear in generated frames

Five points spanning 50× (20 → 1035 frames), least-squares:

    transient_MB ≈ 1824 + 14.2 × frames          (one frame ≈ 46 ms of audio)

| frames | audio | predicted | measured |
|---|---|---|---|
| 337 | 15.6 s | 6.45 GB | **6.20 GB** |
| 512 | 23.8 s | 8.87 GB | — (the package's default `maxFrames` cap) |
| 1035 | 48.0 s | 16.12 GB | **16.19 GB** |
| 2048 | 95.0 s | 30.14 GB | — (the `maxFrames` ceiling) |

Within 4% across the range. The large constant (~1.8 GB) is why short utterances are so
inefficient on both axes — memory and RTF.

**This is what set the final declaration.** `peakActivationBytes` must cover the longest run the
package permits *by default* — 512 frames ⇒ 8.87 GB ⇒ declared **9.50 GB**. The two earlier
values (5.00, then 7.20) were both under for the same reason: the sample never reached the cap.
A caller raising `maxFrames` is opting out of the declared envelope, which is inherent to
whole-utterance decoding rather than a defect.

### Long-form: chunked vs one-shot (48 s passage, same voice)

| | orchestrated (7 segments) | one-shot |
|---|---|---|
| audio | 48.1 s | 47.7 s |
| RTF | **1.16** | 1.33 |
| peak transient | **5.97 GB** | **16.19 GB** |

Chunking via `TTSOrchestratorKit` cut peak activation by 10.2 GB (63%) **and ran faster** — the
expectation that chunking would trade throughput for memory is wrong here, because the giant
single decode is memory-bound. Chunking also *bounds* the peak: the one-shot figure grows without
limit in passage length, the chunked one does not.

Note the orchestrated RTF (1.16) is still above the single-utterance median (0.94): seven requests
carry seven lots of fixed cost, and the anchored-reference mode adds one extra reference encode.
The fixed cost is the same ~1.8 GB / ~1 s constant that makes short utterances inefficient.

### Streaming: the in-package fix (not yet built)

The codec decoder is **strictly causal** — left-padded convs, right-trimmed transpose convs, and a
causal windowed attention mask — which is the precondition `mlx-gepard-swift` relies on for its
exact windowed decode (`NanoCodec.leftReceptiveFieldFrames`, decode `[a−L, b)` and discard the
first `L` frames' samples, bit-identical for `L ≥` the stack's left receptive field). The same
argument applies here, with one caveat: Audio8's `post_module` is a windowed transformer with
`window_size = 128` **frames**, so `L ≥ 128` (~5.9 s) versus NanoCodec's ~26. Still a constant,
so it still bounds the transient — just a larger one. Implementing it would cap activation
regardless of utterance length and unlock `StreamEmitting`.

## 2026-07-30 (fourth pass) — windowed streaming decode

Built the in-package fix the third pass identified. The naive placement — window the whole
decode, as `mlx-gepard-swift` does for NanoCodec — **does not work here**, and finding out why
determined the design.

### Why the obvious placement fails, and where the seam actually is

`post_module` is 8 stacked layers of 128-wide causal attention, so its receptive field
**compounds**: 8 × 127 = **1016 frames (47 s)**. Windowing above it would need more left context
than a typical utterance contains. But it is also not where the memory goes — it is a transformer
over 1024 × T, while the conv stack below expands to 2048 samples per frame through
1536/768/384/192/96-channel intermediates.

So the stack is split at `post_module`'s output:

    codes ─▶ RVQ + post_module ─▶ latent ─▶ upsample + decoder ─▶ waveform
             └─ long context (1016), cheap   └─ short context (11), expensive

Derived left receptive field of the windowed half: **11 frames**, by backward extent propagation
(residual units `Σ(k−1)·d = 78` per block, transpose convs `ceil((e+k−1)/stride)`).

### Bit-identity (`--s5`)

| context | max_abs vs whole-utterance decode |
|---|---|
| 0 | 1.8e-1 |
| 4 | 8.9e-3 |
| 8 | 2.1e-5 |
| **10** | **0.0** |
| 11 (derived) | 0.0 |
| 32 | 0.0 |

Measured minimum is 10; the derivation says 11 — **sufficient with 1 frame of slack**, which is
the direction to err. The gate asserts sufficiency, not tightness.

**A separate floor exists on CHUNK size, and it is not a context effect:**

| chunk | 8 | 16 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|
| max_abs | 1.4e-3 | 1.4e-3 | 2.5e-4 | **0.0** | **0.0** | **0.0** |

Raising context from 11 → 128 changes these numbers *not at all*, which is what proves the drift
is not a receptive-field shortfall: MLX selects different conv kernels for small inputs and their
fp32 reduction order differs from the full-length path. Hence `minimumExactChunkFrames = 48` and a
default of 64. (1.4e-3 on a [−1,1] waveform is ≈ −57 dBFS and inaudible, but "bit-identical" is
worth keeping literally true.)

### What it bought (`--stream`, same request, through the engine package)

| | batch | streaming |
|---|---|---|
| wall clock | 10.33 s | 9.96 s |
| peak activation | +5423 MB | **+3482 MB** (−36%) |
| time to first audio | — | **2.41 s** |

Aggregated streaming response is byte-identical in size to `run()`'s.

The first implementation completed the whole AR rollout before decoding, giving a useless TTFA of
9.51 s out of 9.90 s. Interleaving decode with generation — legitimate because `post_module` is
causal, so running it over a prefix yields identical values for that prefix — brought it to
**2.41 s**. The memory bound is the durable win: it no longer grows with utterance length.

## 2026-07-30 (fifth pass) — the envelope became FLAT

Routing `generate` through the windowed decode removed the length dependence entirely, which
changes the footprint from a fitted line into a constant.

| generated frames | batch peak (before) | batch peak (after) | streaming peak |
|---|---|---|---|
| 64 | 3457 MB | **3482 MB** | 3482 MB |
| 128 | 4969 MB | **3482 MB** | 3482 MB |
| 224 | 4993 MB | **3482 MB** | 3482 MB |
| 1035 | ~16 190 MB | — | 3482 MB |

`peakActivationBytes` **9.50 → 4.20 GB**, and the reduction is the smaller half of the point:
the governor reserves this number, and it is now one the real envelope *cannot* exceed. Before,
every value was a bet on how long an utterance a caller would ask for — a bet lost three times
(5.00 → 7.20 → 9.50 GB), each time for the same reason.

`maxFrames` no longer affects memory at all, only duration. `--s5` still passes, so the audio is
unchanged: this is the same computation, windowed.

### Not yet measured

- Quantized (int8/int4) LM tier — no quantized variant published yet.
- Cold-start prewarm effect on load time.
- Sustained multi-run memory behavior under an engine `cacheLimit` cap.
