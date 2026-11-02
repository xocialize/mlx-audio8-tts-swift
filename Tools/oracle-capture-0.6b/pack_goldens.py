"""Pack the .npy goldens into the single arktts_goldens.safetensors bundle audio8-gates reads.

Ints → int32, floats → fp32, keyed by the .npy stem (the keys listed in manifest.json). Layouts stay
channels-FIRST as dumped from torch; the gates transpose at the call site.
"""
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

DATA = Path(os.environ.get(
    "AUDIO8_06B_DATA", "/Volumes/Satechi/Development/mlxengine-audio/PROD/Audio8"))
GOLDENS = DATA / "oracle-capture" / "goldens"

manifest = json.loads((GOLDENS / "manifest.json").read_text())
out = {}
for name in manifest:
    arr = np.load(GOLDENS / f"{name}.npy")
    arr = arr.astype(np.int32) if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool else arr.astype(np.float32)
    out[name] = mx.array(arr)
mx.save_safetensors(str(GOLDENS / "arktts_goldens.safetensors"), out)
print(f"packed {len(out)} tensors → {GOLDENS / 'arktts_goldens.safetensors'}")
