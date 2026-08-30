"""Prove the Falcon-H1 slow stack in MLX against the PyTorch goldens — and prove the
embedding-multiplier seam, both ways.

This is the de-risking step that runs BEFORE any Swift is written. It answers two
questions that together decide whether the 0.1b port is cheap or expensive:

  Q1. Does the stock `mlx_lm` Falcon-H1 implementation reproduce this checkpoint's
      numerics? If yes, the slow-backbone swap is a wiring job, not a porting job.
      (`mlx-swift-lm`'s FalconH1.swift is the same implementation in Swift, so a green
      Q1 transfers.)

  Q2. Does folding `embedding_multiplier` into `embed_tokens.weight` — what BOTH
      `mlx_lm.sanitize()` and `mlx-swift-lm`'s `sanitize()` do — break arktts?

For Q2 the harness runs the identical stack twice, changing only where the multiplier is
applied, and reports both against the same golden. A port is only correct if CORRECT
passes and FOLDED fails; if FOLDED also passed, the seam wouldn't matter and this whole
concern would be noise.

Run with the 0.6b venv (it has mlx + mlx_lm):
    PROD/Audio8/oracle-capture/.venv/bin/python parity_slow_stack.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

from mlx_lm.models.base import create_attention_mask, create_ssm_mask  # noqa: E402
from mlx_lm.models.falcon_h1 import FalconH1Model, ModelArgs  # noqa: E402

HERE = Path(__file__).resolve().parent
import os

# Data root: the reference checkpoint, the mlx-audio checkout, the release repo and the
# goldens all live outside this git repo because they are weights and large binaries.
# Override with AUDIO8_WORK when they live elsewhere.
WORK = Path(os.environ.get("AUDIO8_WORK", "/Volumes/Satechi/Development/mlxengine-audio/WIP/audio8-tts"))
REPO = WORK / "reference" / "Audio8-TTS-Preview-0.1b"
GOLD = WORK / "oracle-capture-0.1b" / "goldens"

FAILURES: list[str] = []


def g(name: str) -> np.ndarray:
    return np.load(GOLD / f"{name}.npy")


def check(label: str, got, golden: np.ndarray, tol: float, record: bool = True) -> float:
    arr = np.asarray(got.astype(mx.float32)) if isinstance(got, mx.array) else np.asarray(got)
    if arr.shape != golden.shape:
        msg = f"{label}: SHAPE {arr.shape} vs {golden.shape}"
        if record:
            FAILURES.append(msg)
        print(f"  FAIL {label}: shape {arr.shape} vs {golden.shape}")
        return float("inf")
    diff = float(np.max(np.abs(arr - golden)))
    denom = float(np.max(np.abs(golden)) + 1e-9)
    ok = diff < tol
    if not ok and record:
        FAILURES.append(f"{label}: max_abs {diff:.3e} (tol {tol:.0e})")
    print(f"  {'ok  ' if ok else 'FAIL'} {label:32s} max_abs {diff:.3e}  rel {diff/denom:.2e}")
    return diff


def falcon_args_from_arktts(cfg: dict) -> ModelArgs:
    """Mirror of ArkttsModel._build_falcon_config in the reference remote code."""
    return ModelArgs(
        model_type="falcon_h1",
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["dim"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["n_layer"],
        num_attention_heads=cfg["n_head"],
        num_key_value_heads=cfg["n_local_heads"],
        head_dim=cfg["head_dim"],
        rms_norm_eps=cfg["norm_eps"],
        rope_theta=cfg["rope_base"],
        max_position_embeddings=cfg["max_seq_len"],
        attention_bias=cfg["attention_bias"],
        attention_in_multiplier=cfg["attention_in_multiplier"],
        attention_out_multiplier=cfg["attention_out_multiplier"],
        key_multiplier=cfg["key_multiplier"],
        embedding_multiplier=cfg["embedding_multiplier"],
        lm_head_multiplier=cfg["lm_head_multiplier"],
        mlp_bias=cfg["mlp_bias"],
        mlp_multipliers=cfg["mlp_multipliers"],
        mamba_chunk_size=cfg["mamba_chunk_size"],
        mamba_conv_bias=cfg["mamba_conv_bias"],
        mamba_d_conv=cfg["mamba_d_conv"],
        mamba_d_head=cfg["mamba_d_head"],
        mamba_d_ssm=cfg["mamba_d_ssm"],
        mamba_d_state=cfg["mamba_d_state"],
        mamba_expand=cfg["mamba_expand"],
        mamba_n_groups=cfg["mamba_n_groups"],
        mamba_n_heads=cfg["mamba_n_heads"],
        mamba_norm_before_gate=cfg["mamba_norm_before_gate"],
        mamba_proj_bias=cfg["mamba_proj_bias"],
        mamba_rms_norm=cfg["mamba_rms_norm"],
        mamba_use_mlp=cfg["mamba_use_mlp"],
        projectors_bias=cfg["projectors_bias"],
        ssm_in_multiplier=cfg["ssm_in_multiplier"],
        ssm_multipliers=cfg["ssm_multipliers"],
        ssm_out_multiplier=cfg["ssm_out_multiplier"],
        tie_word_embeddings=True,
    )


def sanitize_slow(weights: dict, args: ModelArgs, mup: mx.array, fold_embedding: bool) -> dict:
    """mlx_lm's FalconH1 sanitize, with the embedding fold made switchable.

    `fold_embedding=False` is the arktts-correct variant: the multiplier is applied later,
    to the FULL composite embedding (text + codebooks), not to the lookup table.
    """
    out = {}
    for name, param in weights.items():
        if name.endswith("embed_tokens.weight"):
            if fold_embedding:
                param = param * args.embedding_multiplier
        elif name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
            # NOTE: mlx_lm omits v_proj here; mlx-swift-lm includes it. Harmless for this
            # checkpoint (attention_in_multiplier == 1.0) but a real divergence between the
            # two implementations, worth knowing before trusting either blind.
            param = param * args.attention_in_multiplier
        elif name.endswith("o_proj.weight"):
            param = param * args.attention_out_multiplier
        elif name.endswith("out_proj.weight"):
            param = param * args.ssm_out_multiplier
        elif name.endswith("gate_proj.weight"):
            param = param * args.mlp_multipliers[0]
        elif name.endswith("down_proj.weight"):
            param = param * args.mlp_multipliers[1]
        elif name.endswith("in_proj.weight"):
            param = param * (args.ssm_in_multiplier * mup.astype(param.dtype)[:, None])
        elif "conv1d.weight" in name:
            param = param.transpose(0, 2, 1)
        out[name] = param
    return out


def run_stack(model: FalconH1Model, h: mx.array) -> tuple[mx.array, dict[int, mx.array]]:
    """FalconH1Model.__call__ with the embedding lookup replaced by an injected `h`.

    This is the seam `mlx_lm`/`mlx-swift-lm` do not expose: arktts never feeds token ids
    to the slow stack. Adding an `inputs_embeds`-style entry point upstream is the clean
    fix; this function is what that entry point would do.

    Returns the post-`final_layernorm` hidden AND the raw per-layer outputs, because the
    goldens hook the layers BEFORE the final norm — comparing a normed tensor against an
    un-normed golden is a check that cannot pass and must not be dressed up as one.
    """
    cache = [(None, None)] * len(model.layers)
    mamba_mask = create_ssm_mask(h, cache[0][0])
    attn_mask = create_attention_mask(h, cache[0][1])
    taps: dict[int, mx.array] = {}
    for idx, (layer, c) in enumerate(zip(model.layers, cache)):
        h = layer(h, cache=c, attn_mask=attn_mask, mamba_mask=mamba_mask)
        if idx in (0, 11, 23):
            taps[idx] = h
    return model.final_layernorm(h), taps


def main() -> None:
    cfg = json.loads((REPO / "config.json").read_text())
    args = falcon_args_from_arktts(cfg)
    mult = cfg["embedding_multiplier"]

    non_unity = {
        k: cfg[k]
        for k in (
            "attention_in_multiplier", "attention_out_multiplier", "key_multiplier",
            "embedding_multiplier", "lm_head_multiplier", "ssm_in_multiplier",
            "ssm_out_multiplier",
        )
        if cfg[k] != 1.0
    }
    non_unity.update({k: cfg[k] for k in ("mlp_multipliers", "ssm_multipliers") if set(cfg[k]) != {1.0}})
    print(f"non-unity multipliers in this config: {non_unity}")
    print("  (lm_head_multiplier is unused — arktts has its own semantic_output head)\n")

    raw = mx.load(str(REPO / "model.safetensors"))
    slow = {k[len("slow."):]: v.astype(mx.float32) for k, v in raw.items() if k.startswith("slow.")}
    print(f"slow tensors: {len(slow)}  (renamed slow.* -> bare, for FalconH1Model)")

    model = FalconH1Model(args)
    mup = model._mup_vector
    print(f"mup vector: shape {mup.shape}, all-ones={bool(mx.all(mup == 1.0))}\n")

    # ---- golden inputs ------------------------------------------------------
    embed_raw = mx.array(g("embed.raw_unscaled"))            # (text + codebooks), unscaled
    text_only = mx.array(g("embed.text_only"))
    codebook_sum = mx.array(g("embed.codebook_sum"))
    prompt_ids = mx.array(g("prompt.ids").astype(np.int64))

    results = {}
    for variant, fold in (("CORRECT (multiplier on the composite)", False),
                          ("FOLDED  (mlx_lm/mlx-swift-lm sanitize)", True)):
        print(f"=== {variant} ===")
        model.update(nn.utils.tree_map(lambda x: x, model.parameters()))  # fresh graph
        weights = sanitize_slow(slow, args, mup, fold_embedding=fold)
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        mx.eval(model.parameters())

        if fold:
            # embed_tokens.weight already carries the multiplier, so the text half is
            # scaled and the codebook half is not — exactly the silent bug.
            h = model.embed_tokens(prompt_ids[:, 0]) + codebook_sum
        else:
            h = embed_raw * mult

        seam_golden = g("embed.scaled_wrong_foldedweights" if fold else "embed.scaled_correct")
        check("embed_seam", h, seam_golden, 1e-4, record=not fold)

        out, taps = run_stack(model, h)
        mx.eval(out, *taps.values())

        # Per-layer taps: these are hooked pre-final-norm on the PyTorch side, so they are
        # compared against the raw layer outputs. Tolerance widens with depth because fp32
        # error accumulates through 24 hybrid layers; relative error is the honest metric.
        for idx in sorted(taps):
            check(f"prefill.slow_layer{idx}_out", taps[idx], g(f"prefill.slow_layer{idx}_out"),
                  1e-2 if idx == 0 else 5e-1, record=not fold)
        dh = check("prefill.hidden_last", out[:, -1], g("prefill.hidden_last"), 2e-2, record=not fold)
        results[fold] = dh
        print()

    # ---- verdict ------------------------------------------------------------
    correct_d, folded_d = results[False], results[True]
    print("=" * 72)
    print(f"CORRECT variant  hidden_last max_abs = {correct_d:.3e}")
    print(f"FOLDED  variant  hidden_last max_abs = {folded_d:.3e}")
    if correct_d < 2e-2 and folded_d > 2e-2:
        print("\nVERDICT: mlx_lm's Falcon-H1 reproduces this checkpoint (Q1 green), and the")
        print("         weight-folded seam is measurably WRONG (Q2 confirmed). The Swift port")
        print("         needs an inputs_embeds seam; it must NOT reuse sanitize() verbatim.")
    elif correct_d >= 2e-2:
        print("\nVERDICT: the CORRECT seam does not reproduce the golden — the slow-stack")
        print("         swap is NOT a wiring job. Investigate before costing the port.")
    else:
        print("\nVERDICT: both seams reproduce the golden — the fold is harmless here.")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
