"""PT-oracle vs Python-MLX parity for Audio8-TTS-Preview-0.1b (arktts / falcon_h1).

Mirrors PROD/Audio8/oracle-capture/parity_mlx.py (the 0.6b harness) against the 0.1b
goldens. Checks, in order: weight load, the embedding seam, prefill through the Falcon
slow stack, the compact semantic head, the fast AR, the codec, and greedy generation.

Run with the 0.6b venv (mlx + mlx_lm):
    PROD/Audio8/oracle-capture/.venv/bin/python parity_mlx.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)

HERE = Path(__file__).resolve().parent
import os

# Data root: the reference checkpoint, the mlx-audio checkout, the release repo and the
# goldens all live outside this git repo because they are weights and large binaries.
# Override with AUDIO8_WORK when they live elsewhere.
WORK = Path(os.environ.get("AUDIO8_WORK", "/Volumes/Satechi/Development/mlxengine-audio/WIP/audio8-tts"))
sys.path.insert(0, str(WORK / "mlx-audio"))

from mlx_audio.tts.models.arktts import Model, ModelConfig  # noqa: E402
from mlx_audio.tts.models.arktts.arktts import ArkttsModel  # noqa: E402

GOLD = WORK / "oracle-capture-0.1b" / "goldens"
REPO = WORK / "reference" / "Audio8-TTS-Preview-0.1b"
# The codec is bit-identical between the 0.6b and 0.1b (same upstream sha256, verified
# numerically in capture_goldens.py step 9), so the 0.6b's already-converted raw codec
# is reused rather than re-converted.
CODEC_RAW = Path(
    "/Volumes/Satechi/Development/mlxengine-audio/PROD/Audio8/oracle-capture/converted/codec_raw.safetensors"
)
# capture_goldens.py step 9 drives the codec with the 0.6b's saved unit INPUTS (that is
# what makes it a cross-model check), so those inputs live in the 0.6b golden set.
GOLD_06B = Path(
    "/Volumes/Satechi/Development/mlxengine-audio/PROD/Audio8/oracle-capture/goldens"
)

FAILURES: list[str] = []


def g(name: str) -> np.ndarray:
    return np.load(GOLD / f"{name}.npy")


def check(name: str, value, golden: np.ndarray, tol: float) -> None:
    got = np.asarray(value.astype(mx.float32)) if isinstance(value, mx.array) else np.asarray(value)
    if got.shape != golden.shape:
        FAILURES.append(f"{name}: SHAPE {got.shape} vs {golden.shape}")
        print(f"  FAIL {name}: shape {got.shape} vs {golden.shape}")
        return
    diff = float(np.max(np.abs(got - golden)))
    denom = float(np.max(np.abs(golden)) + 1e-9)
    if diff >= tol:
        FAILURES.append(f"{name}: max_abs {diff:.3e} (tol {tol:.0e})")
    print(f"  {'ok  ' if diff < tol else 'FAIL'} {name:32s} max_abs {diff:.3e}  rel {diff/denom:.2e}")


def to_nlc(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.swapaxes(x, 1, 2))


def main() -> None:
    config_dict = json.loads((REPO / "config.json").read_text())
    config = ModelConfig.from_dict(config_dict)
    assert config.uses_falcon_slow, "config.json should select the falcon_h1 slow backbone"
    print(f"slow_backbone={config.slow_backbone}  dim={config.dim}  n_layer={config.n_layer}")

    model = Model(config)
    model.model_path = REPO

    weights = {}
    weights.update(mx.load(str(REPO / "model.safetensors")))
    weights.update(mx.load(str(CODEC_RAW)))
    sanitized = model.sanitize(weights)
    model.load_weights(list(sanitized.items()), strict=True)

    from mlx.utils import tree_map

    model.update(
        tree_map(
            lambda a: a.astype(mx.float32) if a.dtype == mx.bfloat16 else a,
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    print("weights loaded strict ✓\n")

    lm: ArkttsModel = model.model
    codec = model.codec

    # ---- the embedding seam -------------------------------------------------
    print("[embedding seam]")
    prompt = mx.array(g("prompt.ids").astype(np.int64))
    embed_raw = lm._embed(prompt)
    check("embed.raw_unscaled", embed_raw, g("embed.raw_unscaled"), 1e-4)
    check(
        "embed.scaled_correct",
        embed_raw * config.embedding_multiplier,
        g("embed.scaled_correct"),
        1e-4,
    )
    # Guard the trap directly: if a future refactor folds the multiplier into
    # embed_tokens, this assertion is what fails, loudly, instead of the audio quietly
    # degrading. `text_embeddings` must be the RAW table.
    folded = np.asarray(
        (lm.text_embeddings(prompt[:, 0])).astype(mx.float32)
    )
    if np.max(np.abs(folded - g("embed.text_only"))) > 1e-4:
        FAILURES.append("embed_tokens carries a folded multiplier — sanitize regressed")
        print("  FAIL embed_tokens is NOT the raw table (multiplier folded in)")
    else:
        print("  ok   embed_tokens is the raw table (multiplier not folded)")

    # ---- prefill through the Falcon slow stack ------------------------------
    print("\n[prefill]")
    logits, normalized = lm(prompt, mx.array(g("prompt.mask").astype(np.int64)))
    mx.eval(logits, normalized)
    check("prefill.hidden_last", normalized[:, -1], g("prefill.hidden_last"), 2e-2)
    check("prefill.logits_last", logits[:, -1], g("prefill.logits_last"), 5e-2)
    print(f"  semantic logits width = {logits.shape[-1]} (expected {config.codebook_size + 1})")

    # ---- fast AR + heads ----------------------------------------------------
    print("\n[fast AR]")
    x_fast = mx.array(g("unit.x_fast_in"))
    n = config.num_codebooks
    rope_f = lm._fast_freqs_cis[mx.arange(n)][None]
    mask_f = (mx.arange(n)[None, :] <= mx.arange(n)[:, None])[None, None]
    check("unit.fast_block0_out", lm.fast_layers[0](x_fast, rope_f, mask_f), g("unit.fast_block0_out"), 1e-4)
    check("unit.fast_norm_out", lm.fast_norm(x_fast), g("unit.fast_norm_out"), 1e-5)
    check("unit.fast_output_out", lm.fast_output(x_fast), g("unit.fast_output_out"), 1e-4)
    check("unit.rope_table_fast", lm._fast_freqs_cis, g("unit.rope_table_fast"), 1e-6)

    # ---- codec --------------------------------------------------------------
    print("\n[codec]")
    codes_unit = mx.array(np.load(GOLD_06B / "unit.codes_in.npy").astype(np.int64))
    check("unit.codec_qdecode_out", codec.quantizer.decode(codes_unit), to_nlc(g("unit.codec_qdecode_out")), 1e-3)
    check(
        "unit.codec_full_decode_out",
        codec.decode(codes_unit)[:, None, :].transpose(0, 2, 1),
        to_nlc(g("unit.codec_full_decode_out")),
        1e-3,
    )

    ref_audio = mx.array(g("ref_audio_44100"))
    ref_len = int(g("proc.reference_audio_lengths")[0])
    ref_codes, ref_code_lengths = codec.encode(ref_audio[None], mx.array([ref_len], dtype=mx.int64))
    mx.eval(ref_codes, ref_code_lengths)
    agree = float((np.asarray(ref_codes) == g("encode.ref_codes").astype(np.int64)).mean())
    print(f"  {'ok  ' if agree == 1.0 else 'FAIL'} encode.ref_codes agreement {agree:.4%}")
    if agree < 0.999:
        FAILURES.append(f"encode.ref_codes agreement {agree:.4%}")

    # ---- greedy generation --------------------------------------------------
    print("\n[greedy generation]")
    golden_greedy = g("gen.codes_greedy").astype(np.int64)
    gen = lm.generate_codes(
        [np.asarray(g("proc.prefix_input_ids")[0], dtype=np.int64)],
        [np.asarray(g("proc.suffix_input_ids")[0], dtype=np.int64)],
        [np.asarray(ref_codes[0])],
        np.asarray(ref_code_lengths),
        max_new_tokens=200,
        do_sample=False,
    )
    mx.eval(gen)
    gen_np = np.asarray(gen)
    print(f"  codes: mlx {gen_np.shape} vs golden {golden_greedy.shape}")
    frames = min(gen_np.shape[-1], golden_greedy.shape[-1])
    agree = float((gen_np[..., :frames] == golden_greedy[..., :frames]).mean())
    print(f"  {'ok  ' if agree > 0.98 else 'FAIL'} greedy code agreement {agree:.4%} over {frames} frames")
    if gen_np.shape != golden_greedy.shape or agree < 0.98:
        FAILURES.append(f"greedy: shape {gen_np.shape} vs {golden_greedy.shape}, agree {agree:.4%}")

    # ---- decode the golden codes -------------------------------------------
    print("\n[waveform]")
    wave = codec.decode(mx.array(golden_greedy))
    mx.eval(wave)
    golden_wave = g("gen.waveform")[0]
    got_wave = np.asarray(wave[0])[: golden_wave.shape[0]]
    diff = float(np.max(np.abs(got_wave - golden_wave)))
    print(f"  {'ok  ' if diff < 5e-3 else 'FAIL'} waveform max_abs {diff:.3e}")
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
