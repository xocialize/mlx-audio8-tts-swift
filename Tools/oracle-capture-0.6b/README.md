# oracle-capture-0.6b

PyTorch-oracle capture and parity harness for **Audio8-TTS-Preview-0.6b** (`arktts`). It produces
the goldens the Python-MLX port and this package's `audio8-gates` are checked against, and it
builds the publishable MLX repo — [`mlx-community/Audio8-TTS-Preview-0.6b-bf16`](https://huggingface.co/mlx-community/Audio8-TTS-Preview-0.6b-bf16).

Sibling of [`oracle-capture-0.1b`](../oracle-capture-0.1b), same reasoning: the scripts live in
this git repo because they are the reproducibility of a **published** model. Losing them does not
lose the weights — it loses the ability to prove the weights are right, or to re-derive them after
an upstream change. The data they operate on stays out of version control: reference weights,
goldens, the converted codec and the release repo are large binaries, regenerable from here.

Brought under version control per **AB-D-0070** (AB-T-0112), which found this harness tracked by no
repository at all.

## Paths

Rooted by environment variable rather than relative to the script, because the harness now lives in
the package repo while its data stays on the working volume. **Two roots**, because the inputs and
the artifacts drifted apart historically — the scripts previously assumed a single
`oracle-capture/..` parent and no longer resolved:

| var | default | holds |
|---|---|---|
| `AUDIO8_WORK` | `…/mlxengine-audio/WIP/audio8-tts` | `reference/Audio8-TTS-Preview-0.6b`, `mlx-audio/` (shared with the 0.1b harness) |
| `AUDIO8_06B_DATA` | `…/mlxengine-audio/PROD/Audio8` | `oracle-capture/{goldens,converted}`, `release/` |
| `AUDIO8_06B_CHECKPOINT` | `Audio8-TTS-Preview-0.6b` | which `reference/<name>` snapshot to read; the release lands in `release/<name>-bf16`. `Audio8-TTS-Realistic-0.6b` (E28) is the same graph with new LM values — give it its own `AUDIO8_06B_DATA` (`WIP/audio8-tts/realistic-0.6b`) |

Consolidating the two under one root would be tidier and is deliberately **not** done here — that
is a data move, and this change is only about getting the scripts versioned and running again.

## Order of operations

| | |
|---|---|
| `convert_codec.py` | codec weights → `converted/codec_raw.safetensors` (fp32) |
| `capture_goldens.py` | PyTorch reference, fp32 CPU, deterministic (`do_sample=False`) → `.npy` goldens + `manifest.json`: e2e fixtures, hooked intermediates, seeded unit fixtures |
| `pack_goldens.py` | the `.npy` goldens → `goldens/arktts_goldens.safetensors` (ints int32, floats fp32), the bundle `audio8-gates` reads |
| `parity_mlx.py` | full Python-MLX parity against those goldens |
| `gen_smoke.py` | end-to-end MLX smoke render → `goldens/mlx_smoke_bf16.wav` |
| `build_release_repo.py` | assembles the publishable bf16 MLX repo |

## The oracle needs transformers < 5

`capture_goldens.py` must run under upstream's tested pair — torch 2.8.0 + torchaudio 2.8.0 + transformers 4.57.3
(the card pins `transformers>=4.57,<5`). Under transformers 5.x the reference's RoPE tables come out
uninitialised (zeros and denormals) and every attention golden is garbage, while nothing errors. This is what
happened in 2026-11 when the July venv had drifted to 5.14.1. The pinned venv is
`WIP/audio8-tts/realistic-0.6b/.venv-oracle` (freeze in `_logs/venv-oracle.freeze.txt`); re-capturing the Preview
goldens under it reproduces July's integer fixtures exactly. The MLX-side scripts (`pack_goldens.py`,
`build_release_repo.py`, `parity_mlx.py`) keep using the `mlx` venv.

## Caveat carried from the port

`arktts` decodes greedily and its stop head truncates deterministically per (reference, text);
synthetic bakes trip it where 0.99 rescues, and retries never help (AB-L-0075). A GPU rollout can
never be token-exact against a CPU oracle here, because the codec's VQ encode is not code-exact
across devices, so the reference conditioning itself differs (AB-L-0076). Parity is asserted on
the CPU path.
