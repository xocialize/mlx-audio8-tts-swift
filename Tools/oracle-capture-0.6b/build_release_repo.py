"""Assemble the publish-ready mlx repo: pre-sanitized weights + config + tokenizer."""
import os
import json
import shutil
import sys
from pathlib import Path

# Paths (rooted for this harness's home in the package repo — see README).
# Two roots because the harness's inputs and its artifacts drifted apart historically:
#   AUDIO8_WORK     upstream checkpoint + the mlx-audio checkout  (shared with oracle-capture-0.1b)
#   AUDIO8_06B_DATA goldens / converted / release built by THIS harness
WORK = Path(os.environ.get(
    "AUDIO8_WORK", "/Volumes/Satechi/Development/mlxengine-audio/WIP/audio8-tts"))
DATA = Path(os.environ.get(
    "AUDIO8_06B_DATA", "/Volumes/Satechi/Development/mlxengine-audio/PROD/Audio8"))
HARNESS = DATA / "oracle-capture"

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK / "mlx-audio"))
from mlx_audio.tts.models.arktts import Model, ModelConfig  # noqa: E402

REPO = WORK / "reference" / "Audio8-TTS-Preview-0.6b"
OUT = DATA / "release" / "Audio8-TTS-Preview-0.6b-bf16"
OUT.mkdir(parents=True, exist_ok=True)

config_dict = json.loads((REPO / "config.json").read_text())
config = ModelConfig.from_dict(config_dict)
model = Model(config)

weights = {}
weights.update(mx.load(str(REPO / "model.safetensors")))
weights.update(mx.load(str(HARNESS / "converted" / "codec_raw.safetensors")))
sanitized = model.sanitize(weights)
# round-trip guard: the converted dict must load strict and pass through sanitize untouched
model.load_weights(list(sanitized.items()), strict=True)
again = model.sanitize(sanitized)
assert again is sanitized or set(again) == set(sanitized), "sanitize not idempotent"

mx.eval(list(sanitized.values()))  # materialize before save — lazy tensors save as zeros
lm = {k: v for k, v in sanitized.items() if k.startswith("model.")}
codec = {k: v for k, v in sanitized.items() if k.startswith("codec.")}
mx.save_safetensors(str(OUT / "model.safetensors"), lm)
mx.save_safetensors(str(OUT / "codec.safetensors"), codec)

config_out = dict(config_dict)
config_out.pop("auto_map", None)
config_out["codec_filename"] = "codec.safetensors"
(OUT / "config.json").write_text(json.dumps(config_out, indent=2) + "\n")
for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
    shutil.copy2(REPO / name, OUT / name)

total = sum(f.stat().st_size for f in OUT.iterdir()) / 1e9
print(f"release repo at {OUT} ({total:.2f} GB, {len(lm)} LM + {len(codec)} codec tensors)")

# spot-check no zero-tensors snuck in
zeros = [k for k, v in sanitized.items() if float(mx.abs(v).max()) == 0.0]
print("all-zero tensors:", zeros or "none")
