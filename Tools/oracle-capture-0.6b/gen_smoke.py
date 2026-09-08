"""GPU bf16 e2e smoke: sampled voice-clone generation through Model.generate."""
import os
import json
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

import numpy as np
import mlx.core as mx
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK / "mlx-audio"))
from mlx_audio.tts.models.arktts import Model, ModelConfig  # noqa: E402

REPO = WORK / "reference" / "Audio8-TTS-Preview-0.6b"
REF_WAV = Path(
    "/Volumes/Satechi/Development/mlxengine-audio/PROD/gepard/oracle-capture/extra_ref_audio/roxy_d1_trim.wav"
)
REF_TEXT = (
    "You know, I was thinking about what you said earlier, and honestly, "
    "I think you might be right about the whole thing."
)

config = ModelConfig.from_dict(json.loads((REPO / "config.json").read_text()))
model = Model(config)
model.model_path = REPO
weights = {}
weights.update(mx.load(str(REPO / "model.safetensors")))          # LM bf16 as stored
weights.update(mx.load(str(HARNESS / "converted" / "codec_raw.safetensors")))  # codec fp32
model.load_weights(list(model.sanitize(weights).items()), strict=True)
mx.eval(model.parameters())
print("loaded (LM bf16, codec fp32), GPU stream")

for result in model.generate(
    text="The port is working. Zero shot voice cloning on Apple Silicon, running entirely in MLX.",
    ref_audio=str(REF_WAV),
    ref_text=REF_TEXT,
    temperature=0.7, top_p=0.9, top_k=50, max_tokens=300, seed=1234,
):
    audio = np.asarray(result.audio.astype(mx.float32))
    rms = float(np.sqrt(np.mean(audio**2)))
    print(f"frames {result.token_count}  dur {result.samples/44100:.2f}s  "
          f"RTF {result.real_time_factor:.2f}  peak_mem {result.peak_memory_usage:.2f} GB  "
          f"rms {20*np.log10(rms+1e-12):.1f} dBFS  peak {20*np.log10(np.abs(audio).max()+1e-12):.1f} dBFS")
    sf.write(str(HARNESS / "goldens" / "mlx_smoke_bf16.wav"), audio, 44100)
print("saved goldens/mlx_smoke_bf16.wav")
