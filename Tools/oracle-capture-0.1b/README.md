# oracle-capture-0.1b

PyTorch-oracle capture and parity harness for **Audio8-TTS-Preview-0.1b** (`arktts` with
`slow_backbone: falcon_h1`). It produces the goldens that both the Python-MLX port and this
package's `audio8-gates` are checked against, and it builds the publishable MLX repo.

The scripts live in this git repo because they are the reproducibility of a published
model. The data they operate on does not: reference weights, the release repo and the
goldens are large binaries kept outside version control and regenerable from here.

## Layout

Everything resolves against `WORK`, defaulting to
`/Volumes/Satechi/Development/mlxengine-audio/WIP/audio8-tts`. Override with `AUDIO8_WORK`.

```
$WORK/
  reference/Audio8-TTS-Preview-0.1b/     upstream checkpoint (hf download)
  mlx-audio/                             the mlx-audio checkout carrying the arktts model
  release/Audio8-TTS-Preview-0.1b-bf16/  built by build_release_repo.py
  oracle-capture-0.1b/goldens/           written by capture_goldens.py
```

## Order of operations

| | |
|---|---|
| `capture_goldens.py` | PyTorch reference, fp32 CPU, deterministic → 53 `.npy` goldens + `manifest.json` |
| `parity_slow_stack.py` | de-risking probe: does stock `mlx_lm` Falcon-H1 reproduce this checkpoint, and is the weight-folded embedding seam wrong? |
| `parity_mlx.py` | full Python-MLX parity: embedding seam, prefill, fast AR, codec, greedy generation |
| `build_release_repo.py` | assembles the publishable `-bf16` repo |
| `gen_smoke.py` | e2e sampled generation through the built release repo |
| `bench_rtf.py` | steady-state RTF / peak memory, 0.1b vs 0.6b (`python bench_rtf.py 0.1b`) |

## Two environments, deliberately

**PyTorch capture** needs `transformers>=4.57,<5`. The reference remote code imports
`FalconHybridMambaAttentionDynamicCache`, which **transformers 5.x removed** (folded into
the generic `DynamicCache`). Goldens were captured against the pinned version the model
card specifies rather than against a shim — a golden from a patched reference is a golden
of the patch. That venv is `$WORK/oracle-capture-0.1b/.venv` (py3.12, torch 2.13,
transformers 4.57.6).

**MLX parity** needs `mlx` + `mlx_lm`, which the 0.6b harness venv already has:
`PROD/Audio8/oracle-capture/.venv`. All the MLX scripts here are run with that one.

## What the goldens pin that is not obvious

`embed.*` exists because of a trap that produces plausible audio when you get it wrong.
`embedding_multiplier` (0.10888671875) applies to the **composite** slow input — text
embedding plus the ten codebook embeddings — not to the token lookup alone. Both
`mlx_lm`'s and `mlx-swift-lm`'s FalconH1 `sanitize()` fold it into `embed_tokens.weight`,
which is correct for a plain Falcon-H1 LM and wrong here.

`capture_goldens.py` therefore saves each half of the sum and **both** orderings, so the
divergence is measurable rather than argued: 77.4% relative on the embedding, 38.2% at the
final hidden state. See `AB-L-0074`.

Step 9 of the capture is a cross-model check rather than a capture: upstream ships a
byte-identical `codec.pth` in the 0.6b and 0.1b repos (same LFS oid), so the 0.6b's saved
unit **inputs** are run through the 0.1b's codec and compared against the 0.6b's saved
outputs. Result: `max_abs_diff = 0.0` on the encoder, the quantizer decode and the full
decode. That is what licenses `build_release_repo.py` to copy the codec tensors from the
Apache-2.0 0.6b release instead of re-deriving them, keeping the codec half permissive
while only the LM carries the community licence.

## Known limitation

There is no token-exact validation of any `arktts` **GPU** rollout. The codec's VQ
nearest-neighbour encode is not code-exact across devices (100% CPU vs ~95% GPU), so the
reference conditioning itself differs and every generated frame after that is conditioned
differently. Python-MLX on GPU scores 7.76% against the CPU oracle; the Swift port scores
12.06%. Closing this needs either a device-deterministic VQ encode or a GPU-captured
oracle. See `AB-L-0076`.
