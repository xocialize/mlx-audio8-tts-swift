"""PT-oracle vs MLX-port parity, layer -> block -> e2e. CPU stream, fp32."""
from __future__ import annotations

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

mx.set_default_device(mx.cpu)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK / "mlx-audio"))

from mlx_audio.tts.models.arktts import Model, ModelConfig  # noqa: E402
from mlx_audio.tts.models.arktts.arktts import ArkttsModel  # noqa: E402

GOLD = HARNESS / "goldens"
# AUDIO8_06B_CHECKPOINT picks the 0.6b checkpoint: Preview (default) or Realistic — same graph, new LM values.
CHECKPOINT = os.environ.get("AUDIO8_06B_CHECKPOINT", "Audio8-TTS-Preview-0.6b")
REPO = WORK / "reference" / CHECKPOINT

FAILURES = []


def g(name: str) -> np.ndarray:
    return np.load(GOLD / f"{name}.npy")


def check(name: str, mlx_val: mx.array, golden: np.ndarray, tol: float):
    got = np.asarray(mlx_val.astype(mx.float32))
    if got.shape != golden.shape:
        FAILURES.append(f"{name}: SHAPE {got.shape} vs {golden.shape}")
        print(f"  FAIL {name}: shape {got.shape} vs {golden.shape}")
        return
    diff = float(np.max(np.abs(got - golden)))
    denom = float(np.max(np.abs(golden)) + 1e-9)
    status = "ok  " if diff < tol else "FAIL"
    if diff >= tol:
        FAILURES.append(f"{name}: max_abs {diff:.3e} (rel {diff/denom:.3e})")
    print(f"  {status} {name:34s} max_abs {diff:.3e}  (golden max {denom:.3e})")


def to_nlc(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.swapaxes(x, 1, 2))


def main():
    config_dict = json.loads((REPO / "config.json").read_text())
    config = ModelConfig.from_dict(config_dict)
    model = Model(config)
    model.model_path = REPO

    weights = {}
    weights.update(mx.load(str(REPO / "model.safetensors")))
    weights.update(mx.load(str(HARNESS / "converted" / "codec_raw.safetensors")))
    sanitized = model.sanitize(weights)
    model.load_weights(list(sanitized.items()), strict=True)
    # fp32 for parity (oracle ran fp32); rope tables stay bf16 like the reference
    from mlx.utils import tree_map

    model.update(tree_map(lambda a: a.astype(mx.float32) if a.dtype == mx.bfloat16 else a, model.parameters()))
    mx.eval(model.parameters())
    print("weights loaded strict ✓")

    lm: ArkttsModel = model.model
    codec = model.codec

    # ---- unit: rope tables --------------------------------------------------
    check("unit.rope_table_slow", lm._freqs_cis[:32], g("unit.rope_table_slow"), 1e-6)
    check("unit.rope_table_fast", lm._fast_freqs_cis, g("unit.rope_table_fast"), 1e-6)

    # ---- unit: slow/fast blocks --------------------------------------------
    x_slow = mx.array(g("unit.x_slow_in"))
    length = 17
    rope = lm._freqs_cis[mx.arange(length)][None]
    row = mx.arange(length)[:, None]
    col = mx.arange(length)[None, :]
    mask = (col <= row)[None, None]
    check("unit.rmsnorm_out", lm.norm(x_slow), g("unit.rmsnorm_out"), 1e-5)
    check("unit.slow_ffn0_out", lm.layers[0].feed_forward(x_slow), g("unit.slow_ffn0_out"), 1e-4)
    check(
        "unit.slow_attn0_out",
        lm.layers[0].attention(lm.layers[0].attention_norm(x_slow), rope, mask),
        g("unit.slow_attn0_out"), 1e-4,
    )
    check("unit.slow_block0_out", lm.layers[0](x_slow, rope, mask), g("unit.slow_block0_out"), 1e-4)
    x_fast = x_slow[:, :10]
    rope_f = lm._fast_freqs_cis[mx.arange(10)][None]
    mask_f = (mx.arange(10)[None, :] <= mx.arange(10)[:, None])[None, None]
    check("unit.fast_block0_out", lm.fast_layers[0](x_fast, rope_f, mask_f), g("unit.fast_block0_out"), 1e-4)

    # ---- unit: codec --------------------------------------------------------
    x_audio = mx.array(to_nlc(g("unit.x_audio_in")))
    check("unit.codec_encoder_out", codec.encoder(x_audio), to_nlc(g("unit.codec_encoder_out")), 1e-3)
    x_res = mx.array(to_nlc(g("unit.x_res_in")))
    check(
        "unit.codec_resunit_out",
        codec.encoder.block[1].block[0](x_res),
        to_nlc(g("unit.codec_resunit_out")), 1e-4,
    )
    x_wt = mx.array(to_nlc(g("unit.x_wt_in")))
    check(
        "unit.codec_pre_module_out",
        codec.quantizer.pre_module(x_wt),
        to_nlc(g("unit.codec_pre_module_out")), 1e-3,
    )
    codes_unit = mx.array(g("unit.codes_in").astype(np.int64))
    check(
        "unit.codec_qdecode_out",
        codec.quantizer.decode(codes_unit),
        to_nlc(g("unit.codec_qdecode_out")), 1e-3,
    )
    check(
        "unit.codec_full_decode_out",
        codec.decode(codes_unit)[:, None, :].transpose(0, 2, 1),
        to_nlc(g("unit.codec_full_decode_out")), 1e-3,
    )

    # ---- e2e: codec encode on the real reference clip ----------------------
    ref_audio = mx.array(g("ref_audio_44100"))
    ref_len = int(g("proc.reference_audio_lengths")[0])
    codes, code_lengths = codec.encode(ref_audio[None], mx.array([ref_len], dtype=mx.int64))
    mx.eval(codes, code_lengths)
    golden_codes = g("encode.ref_codes").astype(np.int64)
    got_codes = np.asarray(codes)
    agree = float((got_codes == golden_codes).mean())
    print(f"  {'ok  ' if agree == 1.0 else 'WARN'} encode.ref_codes agreement {agree:.4%}")
    if agree < 0.999:
        FAILURES.append(f"encode.ref_codes agreement {agree:.4%}")

    # ---- e2e: prefill parity ------------------------------------------------
    prompt = mx.array(g("prompt.ids").astype(np.int64))
    prompt_mask = mx.array(g("prompt.mask").astype(np.int64))
    check("prefill.embed", lm._embed(prompt), g("prefill.embed"), 1e-4)
    logits, hidden = lm(prompt, prompt_mask)
    check("prefill.logits_last", logits[:, -1], g("prefill.logits_last"), 2e-2)
    check("prefill.hidden_last", hidden[:, -1], g("prefill.hidden_last"), 2e-2)

    # ---- e2e: greedy generation --------------------------------------------
    golden_greedy = g("gen.codes_greedy").astype(np.int64)
    ref_codes_np = [np.asarray(codes[0])]
    ref_lens_np = np.asarray(code_lengths)
    prefix = np.asarray(g("proc.prefix_input_ids")[0], dtype=np.int64)
    suffix = np.asarray(g("proc.suffix_input_ids")[0], dtype=np.int64)
    gen = lm.generate_codes(
        [prefix], [suffix], ref_codes_np, ref_lens_np,
        max_new_tokens=200, do_sample=False,
    )
    mx.eval(gen)
    gen_np = np.asarray(gen)
    print(f"  greedy codes: mlx {gen_np.shape} vs golden {golden_greedy.shape}")
    frames = min(gen_np.shape[-1], golden_greedy.shape[-1])
    agree = float((gen_np[..., :frames] == golden_greedy[..., :frames]).mean())
    print(f"  {'ok  ' if agree > 0.98 else 'WARN'} greedy code agreement {agree:.4%} over {frames} frames")
    if gen_np.shape != golden_greedy.shape or agree < 0.98:
        FAILURES.append(f"greedy: shape {gen_np.shape} vs {golden_greedy.shape}, agree {agree:.4%}")

    # ---- e2e: decode golden codes -> waveform ------------------------------
    wave = codec.decode(mx.array(golden_greedy))
    mx.eval(wave)
    golden_wave = g("gen.waveform")[0]
    got_wave = np.asarray(wave[0])[: golden_wave.shape[0]]
    diff = float(np.max(np.abs(got_wave - golden_wave)))
    rms = float(np.sqrt(np.mean((got_wave - golden_wave) ** 2)))
    print(f"  {'ok  ' if diff < 5e-3 else 'FAIL'} waveform max_abs {diff:.3e} rms {rms:.3e}")
    if diff >= 5e-3:
        FAILURES.append(f"waveform diff {diff:.3e}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("ALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
