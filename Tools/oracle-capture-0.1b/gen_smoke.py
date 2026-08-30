"""GPU bf16 e2e smoke for the 0.1b: sampled voice-clone generation.

Loads the PUBLISH-READY release repo (not the reference checkpoint) through the same
Model the mlx-audio loader builds, so this exercises the artifact a user would download:
pre-sanitized weights, codec.safetensors, config with codec_filename rewritten.
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
import os

# Data root: the reference checkpoint, the mlx-audio checkout, the release repo and the
# goldens all live outside this git repo because they are weights and large binaries.
# Override with AUDIO8_WORK when they live elsewhere.
WORK = Path(os.environ.get("AUDIO8_WORK", "/Volumes/Satechi/Development/mlxengine-audio/WIP/audio8-tts"))
sys.path.insert(0, str(WORK / "mlx-audio"))
from mlx_audio.tts.models.arktts import Model, ModelConfig  # noqa: E402

RELEASE = WORK / "release" / "Audio8-TTS-Preview-0.1b-bf16"
REF_WAV = Path(
    "/Volumes/Satechi/Development/mlxengine-audio/PROD/gepard/oracle-capture/extra_ref_audio/roxy_d1_trim.wav"
)
REF_TEXT = (
    "You know, I was thinking about what you said earlier, and honestly, "
    "I think you might be right about the whole thing."
)

config = ModelConfig.from_dict(json.loads((RELEASE / "config.json").read_text()))
model = Model(config)
model.model_path = RELEASE

t0 = time.perf_counter()
weights = {}
weights.update(mx.load(str(RELEASE / "model.safetensors")))
weights.update(mx.load(str(RELEASE / "codec.safetensors")))
sanitized = model.sanitize(weights)
assert sanitized is weights, "pre-sanitized release weights should pass through untouched"
model.load_weights(list(sanitized.items()), strict=True)
mx.eval(model.parameters())
print(f"loaded release repo in {time.perf_counter() - t0:.2f}s (LM bf16, codec fp32), GPU stream")

for result in model.generate(
    text="The port is working. Zero shot voice cloning on Apple Silicon, running entirely in MLX.",
    ref_audio=str(REF_WAV),
    ref_text=REF_TEXT,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=300,
    seed=1234,
):
    audio = np.asarray(result.audio.astype(mx.float32))
    rms = float(np.sqrt(np.mean(audio**2)))
    print(
        f"frames {result.token_count}  dur {result.samples / 44100:.2f}s  "
        f"RTF {result.real_time_factor:.2f}  peak_mem {result.peak_memory_usage:.2f} GB  "
        f"rms {20 * np.log10(rms + 1e-12):.1f} dBFS  "
        f"peak {20 * np.log10(np.abs(audio).max() + 1e-12):.1f} dBFS"
    )
    sf.write(str(WORK / "oracle-capture-0.1b" / "goldens" / "mlx_smoke_bf16.wav"), audio, 44100)
print("saved goldens/mlx_smoke_bf16.wav")
