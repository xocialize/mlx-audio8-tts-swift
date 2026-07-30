# mlx-audio8-tts-swift — Stage 1 porting spec

Source of truth: the parity-locked Python-MLX port at
`../mlx-audio/mlx_audio/tts/models/arktts/{arktts.py,codec.py}` (100% greedy
token agreement vs the PyTorch reference) and the fp32 PT oracle fixtures at
`../oracle-capture/goldens/arktts_goldens.safetensors` (45 tensors; int fixtures
int32, floats fp32; channels-FIRST layouts as dumped from torch — transpose to
NLC at the gate site, suffix key list in `../oracle-capture/goldens/manifest.json`).

Published weights: `mlx-community/Audio8-TTS-Preview-0.6b-bf16` — PRE-sanitized
(model./codec. prefixes, weight-norm folded, MLX channels-last conv layouts,
Snake alphas (1,1,C)). The Swift loader consumes them directly: no folding, no
transposes, no key remapping beyond the dotted-path → module-tree walk.

## Phase table

| Phase | Gate | Status |
|---|---|---|
| S0 | key contract: module `parameters().flattened()` keys == published safetensors headers (model.safetensors ∪ codec.safetensors), 0 missing / 0 unused | PASSED 2026-07-30 (226 LM params, 0/0; codec probe walks encode+decode) |
| S1a | LM units vs oracle: rope tables (tol 0 vs bf16 fixture), RMSNorm, slow FFN/attn/block0, fast block0 (CPU stream, fp32) | PASSED 2026-07-30 (rope tables max_abs 0.0; units ≤7.6e-6) |
| S1b | codec units: encoder, resunit, pre_module, qdecode, full decode (CPU stream, fp32) | PASSED 2026-07-30 (encoder 3.2e-6, resunit 2.1e-6, pre_module 9.1e-4, qdecode 3.5e-5, full decode 8.7e-6) |
| S2 | e2e prefill: prompt fixture → logits_last / hidden_last ≤ 1e-2; codec encode of ref clip 100% code-exact; greedy 200-frame generation 100% token-exact; waveform ≤ 5e-3 | PASSED 2026-07-30 (encode 100% code-exact; greedy **102/102 frames token-exact**; logits 1.0e-4; waveform 4.6e-6) — RE-VERIFIED after the Stage 2 Core refactor (throwing onFrame + encodeReference split): byte-identical results, waveform 4.6230853e-06 both runs |
| S2b | prompt entry + GPU: tokenizer → prompt ids == proc.* fixtures; ONE real sampled generation on GPU (bf16 LM + fp32 codec), dBFS-quantified | PASSED 2026-07-30 (prompt ids exact 52+28; 127 frames / 5.90 s @ −18.5 dBFS, RTF 1.30) |
| S5 | (optional) KV-slice decode optimization gated bit-identical vs full-buffer path | not done — see "Open optimizations" |
| S6 | quantized LM variant (int8/int4) per-pass cosine on GPU CLI lane | — |
| S7 | engine wrap: manifest + MAT-1..5 + CAN-1..3 + C14/INF (15 offline tests) and `--validate` through the real package (load → run → unload, measured footprint) | PASSED 2026-07-30 (15/15 offline; validate RTF 0.96, −18.8 dBFS, resident 2434 MB / transient 4413 MB — MEASUREMENTS.md) |

## Numerics contract (from the Python port — verified traps)

- RoPE tables: compute freqs in **Double** (`1/base^(2i/headDim)`), phases outer
  product, cos/sin stack → **round to bf16** (reference stores bf16 buffers) →
  upcast fp32 at apply. Slow table (2048, 32, 2) base 1e6; fast table (10, 32, 2);
  codec tables per-forward, base 1e4, full arange(0,d,2)/d.
- RoPE apply: pairwise (…, d/2, 2) rotation in fp32, cast back to x dtype.
- RMSNorm: fp32 mean-square, `normalized.astype(x.dtype) * weight` — weight
  multiplies AFTER downcast. Codec eps 1e-5, LM eps 1e-6, ConvNeXt LayerNorm 1e-6.
- Attention: stacked QKV split (q | kv | kv), GQA 14/2 via SDPA (MLXFast), bool
  masks (true = attend). KV cache = full max_seq_len buffer + validity mask
  (isomorphic to reference; S5 may slice).
- Slow prefill uses position_ids = cumsum(mask)−1 clamped ≥0; decode uses
  physical position for cache slot and token position for rope.
- Fast AR: cache primed by step at position 0 with projected slow hidden;
  codebook 0 = clip(semantic − semanticBeginId); positions 1..9 sampled.
- Sampling: semantic filter (−inf outside [semanticBegin, semanticEnd] ∪ {eos})
  → legacy top-k/top-p (sort desc, cumsum softmax > p OR rank ≥ k removed, rank 0
  kept) → ÷ max(temperature, 1e-5) → exponential-race `argmax(p / −log U)`.
  RAS: second sample at (ras_top_p 0.9, T 1.0) replaces repeats within window 10.
- Codec: causal convs pad left (k−s) + right extra to frame; transpose convs crop
  right (k−s). Snake `x + (1/(α+1e-9))·sin²(αx)`. VQ decode = cosine-normalized
  argmax; decode path clamps semantic [0,4095], residual [0,1023].
- Prompt: left-padded, pad row 0 with pad_token_id 151643.
- `argmax` yields uint32 — cast before mixing with −1 sentinels.
- Tokenizer: Qwen tokenizer.json; the reference pins `fix_mistral_regex=False`.
- NAX split-K: not applicable (max K 4864 < 10240). No chunking.

## Fixture layout notes

Codec fixtures are channels-first from torch: transpose (0,2,1) to compare with
NLC Swift outputs. `unit.codes_in` / code fixtures are int32. `prompt.ids`
(1, 11, T) int32. Reference audio fixture `ref_audio_44100` is pre-resampled —
resampler parity is decoupled (wrapper uses its own windowed-sinc at runtime).


## Open optimizations (measured, not speculative)

1. **Windowed codec decode.** `--validate` measured the run transient at **4413 MB** for a 9.2 s
   utterance vs **1623 MB** for a short line — it scales with utterance length because the codec
   decodes the whole utterance through its conv stack in one pass. A windowed/incremental decode
   (the decoder is causal, as Gepard's streaming work established for its own codec) would flatten
   this AND unlock `StreamEmitting`. Biggest single win available.
2. **KV-cache slicing (S5).** The cache is a full `max_seq_len` (2048) buffer with a validity mask,
   isomorphic to the reference. Slicing to the valid prefix cuts per-step attention work; gate it
   bit-identical against the full-buffer path.
3. **Quantized LM tier (S6).** No quantized variant is published yet. Run the quant gate in the
   **GPU** CLI lane — a CPU-pinned quantized forward silently grinds for hours.
