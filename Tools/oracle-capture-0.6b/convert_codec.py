"""One-off: codec.pth (torch pickle) -> codec_raw.safetensors (fp32, keys unchanged).

Weight-norm folding and layout transposes happen later in Model.sanitize();
this only rehosts the tensors out of the pickle format.
"""
import os
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

import torch
from safetensors.torch import save_file

# AUDIO8_06B_CHECKPOINT picks the 0.6b checkpoint: Preview (default) or Realistic — same graph, new LM values.
CHECKPOINT = os.environ.get("AUDIO8_06B_CHECKPOINT", "Audio8-TTS-Preview-0.6b")
REPO = WORK / "reference" / CHECKPOINT
OUT = HARNESS / "converted"
OUT.mkdir(exist_ok=True)

state = torch.load(REPO / "codec.pth", map_location="cpu", weights_only=True)
if "state_dict" in state:
    state = state["state_dict"]
if any("generator." in key for key in state):
    state = {
        key.replace("generator.", ""): value
        for key, value in state.items()
        if "generator." in key
    }
state = {
    key: value.contiguous().float()
    for key, value in state.items()
    if not key.endswith(("freqs_cis", "causal_mask"))
}
print(f"{len(state)} tensors")
for key in list(state)[:5]:
    print(" ", key, tuple(state[key].shape), state[key].dtype)
save_file(state, str(OUT / "codec_raw.safetensors"))
print("saved", OUT / "codec_raw.safetensors")
