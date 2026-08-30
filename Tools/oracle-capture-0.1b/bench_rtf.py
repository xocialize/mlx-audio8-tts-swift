"""Steady-state RTF + peak-memory comparison, 0.1b vs 0.6b, same text and reference.

MEASUREMENTS.md (0.6b, second pass) established that the first run after a load is not
comparable — it measured RTF 1.80 against a steady-state 1.11 on the same input. So the
first run here is a discarded warm-up and the reported figure is the median of the rest.

Deterministic decoding (do_sample=False) so both models emit a fixed number of frames per
run and RTF is not confounded by sampling-length variance. Run one model per process
invocation: `python bench_rtf.py 0.1b` / `python bench_rtf.py 0.6b`, so neither pays the
other's residency.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
import os

# Data root: the reference checkpoint, the mlx-audio checkout, the release repo and the
# goldens all live outside this git repo because they are weights and large binaries.
# Override with AUDIO8_WORK when they live elsewhere.
WORK = Path(os.environ.get("AUDIO8_WORK", "/Volumes/Satechi/Development/mlxengine-audio/WIP/audio8-tts"))
sys.path.insert(0, str(WORK / "mlx-audio"))
from mlx_audio.tts.models.arktts import Model, ModelConfig  # noqa: E402

WHICH = sys.argv[1] if len(sys.argv) > 1 else "0.1b"
RUNS = 4  # 1 warm-up + 3 measured

if WHICH == "0.1b":
    REPO = WORK / "release" / "Audio8-TTS-Preview-0.1b-bf16"
    CODEC = REPO / "codec.safetensors"
else:
    REPO = Path(
        "/Volumes/Satechi/Development/mlxengine-audio/PROD/Audio8/release/Audio8-TTS-Preview-0.6b-bf16"
    )
    CODEC = REPO / "codec.safetensors"

REF_WAV = Path(
    "/Volumes/Satechi/Development/mlxengine-audio/PROD/gepard/oracle-capture/extra_ref_audio/roxy_d1_trim.wav"
)
REF_TEXT = (
    "You know, I was thinking about what you said earlier, and honestly, "
    "I think you might be right about the whole thing."
)
TEXT = (
    "The port is working. Zero shot voice cloning on Apple Silicon, running entirely "
    "in MLX, with the slow autoregressive stack replaced by a hybrid state space model."
)

config = ModelConfig.from_dict(json.loads((REPO / "config.json").read_text()))
model = Model(config)
model.model_path = REPO
weights = {}
weights.update(mx.load(str(REPO / "model.safetensors")))
weights.update(mx.load(str(CODEC)))
model.load_weights(list(model.sanitize(weights).items()), strict=True)
mx.eval(model.parameters())

lm_params = sum(v.size for k, v in mx.load(str(REPO / "model.safetensors")).items())
print(f"=== {WHICH} ({config.slow_backbone}) — LM {lm_params / 1e6:.0f}M params ===")

rtfs, peaks, durs, frames = [], [], [], []
for run in range(RUNS):
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for result in model.generate(
        text=TEXT, ref_audio=str(REF_WAV), ref_text=REF_TEXT,
        do_sample=False, max_tokens=300,
    ):
        wall = time.perf_counter() - t0
        dur = result.samples / config.codec_sample_rate
        tag = "warmup " if run == 0 else "measured"
        print(
            f"  {tag} run{run}: frames {result.token_count:>3}  dur {dur:5.2f}s  "
            f"wall {wall:5.2f}s  RTF {wall / dur:.2f}  peak {result.peak_memory_usage:.2f} GB"
        )
        if run > 0:
            rtfs.append(wall / dur)
            peaks.append(result.peak_memory_usage)
            durs.append(dur)
            frames.append(result.token_count)

print(
    f"\n{WHICH}: steady-state RTF median {statistics.median(rtfs):.2f} "
    f"(min {min(rtfs):.2f}, max {max(rtfs):.2f}) over {len(rtfs)} runs\n"
    f"{WHICH}: peak memory median {statistics.median(peaks):.2f} GB  |  "
    f"{frames[0]} frames, {durs[0]:.2f}s audio"
)
