"""Assemble the publish-ready mlx repo for Audio8-TTS-Preview-0.1b.

Mirrors PROD/Audio8/oracle-capture/build_release_repo.py with one deliberate change:
the codec tensors are COPIED FROM THE 0.6b RELEASE rather than re-derived.

Why copy instead of re-derive: upstream ships the identical codec in both repos (same
LFS oid), which capture_goldens.py step 9 confirmed numerically (max_abs_diff 0.0 on
encode, quantizer decode, and full decode). The 0.6b repo publishes it under Apache-2.0;
the 0.1b repo publishes the same bytes under the revenue-capped Audio8 Community License.
Sourcing our artifact from the Apache-2.0 repo keeps the codec half of the package
permissive, so only the LM carries the community-license gate. Re-deriving would produce
identical tensors with a murkier provenance story.

The script still verifies the copy is bit-identical to what re-deriving would give, so
the provenance claim rests on a check rather than an assumption.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
import os

# Data root: the reference checkpoint, the mlx-audio checkout, the release repo and the
# goldens all live outside this git repo because they are weights and large binaries.
# Override with AUDIO8_WORK when they live elsewhere.
WORK = Path(os.environ.get("AUDIO8_WORK", "/Volumes/Satechi/Development/mlxengine-audio/WIP/audio8-tts"))
sys.path.insert(0, str(WORK / "mlx-audio"))
from mlx_audio.tts.models.arktts import Model, ModelConfig  # noqa: E402

REPO = WORK / "reference" / "Audio8-TTS-Preview-0.1b"
OUT = WORK / "release" / "Audio8-TTS-Preview-0.1b-bf16"
CODEC_RAW = Path(
    "/Volumes/Satechi/Development/mlxengine-audio/PROD/Audio8/oracle-capture/converted/codec_raw.safetensors"
)
RELEASE_06B = Path(
    "/Volumes/Satechi/Development/mlxengine-audio/PROD/Audio8/release/Audio8-TTS-Preview-0.6b-bf16"
)

OUT.mkdir(parents=True, exist_ok=True)

config_dict = json.loads((REPO / "config.json").read_text())
config = ModelConfig.from_dict(config_dict)
assert config.uses_falcon_slow, "expected slow_backbone=falcon_h1"
model = Model(config)

weights = {}
weights.update(mx.load(str(REPO / "model.safetensors")))
weights.update(mx.load(str(CODEC_RAW)))
sanitized = model.sanitize(weights)

# round-trip guard: the converted dict must load strict and pass through sanitize untouched
model.load_weights(list(sanitized.items()), strict=True)
again = model.sanitize(sanitized)
assert again is sanitized or set(again) == set(sanitized), "sanitize not idempotent"

mx.eval(list(sanitized.values()))  # materialize before save — lazy tensors save as zeros
lm = {k: v for k, v in sanitized.items() if k.startswith("model.")}
codec = {k: v for k, v in sanitized.items() if k.startswith("codec.")}

# -- codec: verify the 0.6b artifact matches, then reuse it (see module docstring) -----
published = mx.load(str(RELEASE_06B / "codec.safetensors"))
assert set(published) == set(codec), (
    f"codec key sets differ: {len(published)} published vs {len(codec)} derived"
)
worst = 0.0
for key, derived in codec.items():
    diff = float(mx.abs(published[key].astype(mx.float32) - derived.astype(mx.float32)).max())
    worst = max(worst, diff)
print(f"codec vs published 0.6b artifact: max_abs_diff = {worst:.6g}")
assert worst == 0.0, "published codec is NOT bit-identical — do not claim shared provenance"
shutil.copy2(RELEASE_06B / "codec.safetensors", OUT / "codec.safetensors")
print(f"codec.safetensors copied from the Apache-2.0 0.6b release ({len(codec)} tensors)")

mx.save_safetensors(str(OUT / "model.safetensors"), lm)

config_out = dict(config_dict)
config_out.pop("auto_map", None)
config_out["codec_filename"] = "codec.safetensors"
(OUT / "config.json").write_text(json.dumps(config_out, indent=2) + "\n")
for name in (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
):
    shutil.copy2(REPO / name, OUT / name)
# The 0.1b's own licence travels with the LM half.
shutil.copy2(REPO / "LICENSE", OUT / "LICENSE")

total = sum(f.stat().st_size for f in OUT.iterdir()) / 1e9
print(f"release repo at {OUT} ({total:.2f} GB, {len(lm)} LM + {len(codec)} codec tensors)")

zeros = [k for k, v in sanitized.items() if float(mx.abs(v).max()) == 0.0]
print("all-zero tensors:", zeros or "none")
