"""Capture PyTorch oracle goldens for the arktts (Audio8-TTS-Preview-0.1b) MLX port.

Adapted from the 0.6b harness (PROD/Audio8/oracle-capture/capture_goldens.py). The 0.1b
differs in exactly one structural way that matters here: the slow AR backbone is a
Falcon-H1 hybrid (Mamba-2 + attention + MLP per layer) rather than a pure-attention
Qwen-style stack. The fast AR, the processor, and the codec are unchanged.

Runs the reference implementation fp32 on CPU (deterministic, do_sample=False) and saves:
  1. e2e fixtures: processor outputs, reference codes, generated codes, waveform
  2. THE EMBEDDING SEAM (see below) — the goldens that pin the port's highest-risk trap
  3. hooked intermediates: falcon slow layers, per-submodule I/O, fast stack, codec stages
  4. a codec cross-check against the 0.6b goldens, proving the shared-codec claim numerically

## Why the embedding seam gets its own fixtures

`mlx-swift-lm`'s FalconH1 `sanitize()` folds `embedding_multiplier` into
`embed_tokens.weight`. That is correct for a plain Falcon-H1 LM, where the embedding
lookup is the only thing entering the stack. It is WRONG here: arktts feeds

    (text_embed + sum_of_ten_codebook_embeds) * embedding_multiplier

so folding the multiplier into `embed_tokens.weight` alone scales the text half and
silently leaves the codebook half at 1.0. Both orderings produce plausible audio of the
right length, so this cannot be caught by listening. `embed.*` below captures each half
and both orderings so the divergence is measurable rather than argued.

All arrays land in goldens/ as .npy plus a manifest.json describing shapes/dtypes.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
import os

# Data root: the reference checkpoint, the mlx-audio checkout, the release repo and the
# goldens all live outside this git repo because they are weights and large binaries.
# Override with AUDIO8_WORK when they live elsewhere.
WORK = Path(os.environ.get("AUDIO8_WORK", "/Volumes/Satechi/Development/mlxengine-audio/WIP/audio8-tts"))
REPO = WORK / "reference" / "Audio8-TTS-Preview-0.1b"
OUT = WORK / "oracle-capture-0.1b" / "goldens"
OUT.mkdir(exist_ok=True)

# The 0.6b goldens, for the shared-codec cross-check.
GOLDENS_06B = Path(
    "/Volumes/Satechi/Development/mlxengine-audio/PROD/Audio8/oracle-capture/goldens"
)

# Same reference clip and prompts as the 0.6b capture, so the two runs are comparable.
REF_WAV = Path(
    "/Volumes/Satechi/Development/mlxengine-audio/PROD/gepard/oracle-capture/extra_ref_audio/roxy_d1_trim.wav"
)
REF_TEXT = (
    "You know, I was thinking about what you said earlier, and honestly, "
    "I think you might be right about the whole thing."
)
TARGET_TEXT = "The quick brown fox jumps over the lazy dog, while the river keeps flowing north."

manifest: dict = {}


def save(name: str, tensor) -> np.ndarray:
    if torch.is_tensor(tensor):
        arr = tensor.detach().to(torch.float32).cpu().numpy()
    else:
        arr = np.asarray(tensor)
    np.save(OUT / f"{name}.npy", arr)
    manifest[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    print(f"  saved {name}: {list(arr.shape)}")
    return arr


def first_tensor(value):
    """Hook outputs are sometimes tuples; take the leading tensor."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value:
        return first_tensor(value[0])
    return None


def main() -> None:
    torch.manual_seed(0)
    sys.path.insert(0, str(REPO))
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(REPO), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(REPO), trust_remote_code=True, dtype=torch.float32).eval()
    cfg = model.config
    print(f"loaded {type(model).__name__}: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"slow backbone: {type(model.slow).__name__}, {len(model.slow.layers)} layers")

    # ---- 1. processor fixtures -------------------------------------------------
    print("\n[1] processor")
    inputs = processor(
        text=[TARGET_TEXT],
        reference_text=[REF_TEXT],
        reference_audio=[str(REF_WAV)],
    )
    for key, value in inputs.items():
        save(f"proc.{key}", value)
    ref_audio_441 = inputs["reference_audio_values"][0, 0, : int(inputs["reference_audio_lengths"][0])]
    save("ref_audio_44100", ref_audio_441)

    # ---- 2. codec encode -------------------------------------------------------
    print("\n[2] codec encode")
    codec = model.load_codec(device="cpu", dtype=torch.float32)
    codec_hooks: dict = {}

    def hook(name):
        def fn(_m, _i, output):
            codec_hooks.setdefault(name, first_tensor(output))
        return fn

    handles = [
        codec.encoder.register_forward_hook(hook("codec.encoder_out")),
        codec.quantizer.downsample.register_forward_hook(hook("codec.downsample_out")),
        codec.quantizer.pre_module.register_forward_hook(hook("codec.pre_module_out")),
        codec.quantizer.post_module.register_forward_hook(hook("codec.post_module_out")),
        codec.quantizer.upsample.register_forward_hook(hook("codec.upsample_out")),
    ]
    ref_codes, ref_code_lengths = model.encode_audio(
        inputs["reference_audio_values"], inputs["reference_audio_lengths"]
    )
    for h in handles:
        h.remove()
    save("encode.ref_codes", ref_codes)
    save("encode.ref_code_lengths", ref_code_lengths)
    for name, tensor in codec_hooks.items():
        save(f"encode.{name}", tensor)

    # ---- 3. prompt build -------------------------------------------------------
    print("\n[3] prompt")
    prompt, prompt_mask = model._prepare_prompt(
        prefix_input_ids=inputs["prefix_input_ids"],
        prefix_attention_mask=inputs["prefix_attention_mask"],
        suffix_input_ids=inputs["suffix_input_ids"],
        suffix_attention_mask=inputs["suffix_attention_mask"],
        reference_codes=ref_codes,
        reference_code_lengths=ref_code_lengths,
    )
    save("prompt.ids", prompt)
    save("prompt.mask", prompt_mask)

    # ---- 4. THE EMBEDDING SEAM -------------------------------------------------
    # Each half of the sum, both orderings of the multiplier, and the value that
    # actually enters the Falcon stack. See the module docstring.
    print("\n[4] embedding seam (the multiplier trap)")
    mult = float(model.slow.embedding_multiplier)
    manifest["_embedding_multiplier"] = mult
    print(f"  embedding_multiplier = {mult!r}")

    with torch.inference_mode():
        text_embed = model.embeddings(prompt[:, 0])
        codebook_parts = [
            model.codebook_embeddings(prompt[:, i + 1] + i * cfg.codebook_size)
            for i in range(cfg.num_codebooks)
        ]
        codebook_sum_raw = torch.stack(codebook_parts, dim=1).sum(dim=1)
        semantic = (prompt[:, 0] >= cfg.semantic_begin_id) & (prompt[:, 0] <= cfg.semantic_end_id)
        codebook_sum = torch.where(semantic.unsqueeze(-1), codebook_sum_raw, 0.0)

        embed_raw = model._embed(prompt)             # (text + codebooks), NO multiplier
        embed_correct = embed_raw * mult             # what actually enters Falcon
        embed_wrong = text_embed * mult + codebook_sum   # what weight-folding would produce

    save("embed.text_only", text_embed)
    save("embed.codebook_sum", codebook_sum)
    save("embed.semantic_positions", semantic.to(torch.int32))
    save("embed.raw_unscaled", embed_raw)
    save("embed.scaled_correct", embed_correct)
    save("embed.scaled_wrong_foldedweights", embed_wrong)

    divergence = float((embed_correct - embed_wrong).abs().max())
    rel = divergence / float(embed_correct.abs().max())
    manifest["_embed_fold_divergence_max_abs"] = divergence
    manifest["_embed_fold_divergence_relative"] = rel
    print(f"  fold-vs-correct divergence: max_abs={divergence:.6g}  relative={rel:.4%}")
    print("  ^ a port that reuses mlx-swift-lm sanitize() verbatim lands on the WRONG one")

    # ---- 5. prefill through the Falcon slow stack ------------------------------
    print("\n[5] prefill")
    layer_outs: dict = {}
    sub_io: dict = {}

    def layer_hook(idx):
        def fn(_m, _i, output):
            layer_outs.setdefault(idx, first_tensor(output))
        return fn

    def sub_hook(name):
        def fn(_m, inp, output):
            if name in sub_io:
                return
            sub_io[name] = (first_tensor(inp), first_tensor(output))
        return fn

    hs = [model.slow.layers[i].register_forward_hook(layer_hook(i)) for i in (0, 11, 23)]
    layer0 = model.slow.layers[0]
    hs += [
        layer0.mamba.register_forward_hook(sub_hook("mamba0")),
        layer0.self_attn.register_forward_hook(sub_hook("selfattn0")),
        layer0.feed_forward.register_forward_hook(sub_hook("ffn0")),
        layer0.input_layernorm.register_forward_hook(sub_hook("inputnorm0")),
        layer0.pre_ff_layernorm.register_forward_hook(sub_hook("preffnorm0")),
        model.slow.final_layernorm.register_forward_hook(sub_hook("finalnorm")),
    ]
    with torch.inference_mode():
        fwd = model(prompt, attention_mask=prompt_mask)
    for h in hs:
        h.remove()

    for idx, tensor in layer_outs.items():
        save(f"prefill.slow_layer{idx}_out", tensor)
    for name, (tin, tout) in sub_io.items():
        if tin is not None:
            save(f"unit.{name}_in", tin)
        if tout is not None:
            save(f"unit.{name}_out", tout)
    save("prefill.hidden_last", fwd.hidden_states[:, -1])
    save("prefill.logits_last", fwd.logits[:, -1])
    save("prefill.logits_full", fwd.logits)
    print(f"  semantic logits width = {fwd.logits.shape[-1]} (codebook_size + 1 = {cfg.codebook_size + 1})")

    # ---- 6. deterministic generation ------------------------------------------
    print("\n[6] greedy generation")
    torch.manual_seed(0)
    with torch.inference_mode():
        codes = model.generate(
            prefix_input_ids=inputs["prefix_input_ids"],
            prefix_attention_mask=inputs["prefix_attention_mask"],
            suffix_input_ids=inputs["suffix_input_ids"],
            suffix_attention_mask=inputs["suffix_attention_mask"],
            reference_codes=ref_codes,
            reference_code_lengths=ref_code_lengths,
            max_new_tokens=200,
            do_sample=False,
        )
    save("gen.codes_greedy", codes)

    # ---- 7. codec decode -------------------------------------------------------
    print("\n[7] codec decode")
    decode_hooks: dict = {}

    def dhook(name):
        def fn(_m, _i, output):
            decode_hooks.setdefault(name, first_tensor(output))
        return fn

    handles = [
        codec.quantizer.post_module.register_forward_hook(dhook("codec.dec_post_module_out")),
        codec.quantizer.upsample.register_forward_hook(dhook("codec.dec_upsample_out")),
    ]
    with torch.inference_mode():
        waveform, lengths = model.decode_audio(codes)
    for h in handles:
        h.remove()
    save("gen.waveform", waveform)
    save("gen.waveform_lengths", lengths)
    for name, tensor in decode_hooks.items():
        save(f"decode.{name}", tensor)
    with torch.inference_mode():
        save("decode.pre_decoder_latent", codec.quantizer.decode(codes.long()))

    # ---- 8. fast AR unit fixtures ---------------------------------------------
    print("\n[8] fast AR units")
    for layer in model.fast_layers:
        layer.attention.kv_cache = None
    rng = np.random.default_rng(42)
    x_fast = rng.standard_normal((1, cfg.num_codebooks, cfg.fast_dim)).astype(np.float32)
    save("unit.x_fast_in", x_fast)
    n = cfg.num_codebooks
    fast_mask = (torch.arange(n)[None, :] <= torch.arange(n)[:, None])[None, None]
    with torch.inference_mode():
        save("unit.fast_block0_out", model.fast_layers[0](
            torch.from_numpy(x_fast), model.fast_freqs_cis[torch.arange(n)][None], fast_mask,
        ))
        save("unit.fast_norm_out", model.fast_norm(torch.from_numpy(x_fast)))
        save("unit.fast_output_out", model.fast_output(torch.from_numpy(x_fast)))
        save("unit.semantic_output_out", model.semantic_output(
            torch.from_numpy(rng.standard_normal((1, 4, cfg.dim)).astype(np.float32))))
    save("unit.rope_table_fast", model.fast_freqs_cis)

    # ---- 9. shared-codec cross-check ------------------------------------------
    # codec.pth has the same sha256 in both repos. Run the 0.6b's saved unit inputs
    # through THIS repo's codec and compare against the 0.6b's saved outputs. This
    # turns "same file hash" into "same numerics", and licences the reuse of the
    # already-converted codec.safetensors artifact.
    print("\n[9] shared-codec cross-check vs 0.6b goldens")
    checks = {}
    try:
        codes_in = torch.from_numpy(np.load(GOLDENS_06B / "unit.codes_in.npy")).long()
        x_audio = torch.from_numpy(np.load(GOLDENS_06B / "unit.x_audio_in.npy"))
        with torch.inference_mode():
            got_qdecode = codec.quantizer.decode(codes_in)
            got_full = codec.decode(codes_in)
            got_enc = codec.encoder(x_audio)
        for label, got, ref_name in [
            ("qdecode", got_qdecode, "unit.codec_qdecode_out.npy"),
            ("full_decode", got_full, "unit.codec_full_decode_out.npy"),
            ("encoder", got_enc, "unit.codec_encoder_out.npy"),
        ]:
            want = np.load(GOLDENS_06B / ref_name)
            g = got.detach().to(torch.float32).cpu().numpy()
            if g.shape != want.shape:
                checks[label] = {"status": "SHAPE_MISMATCH", "got": list(g.shape), "want": list(want.shape)}
                print(f"  {label}: SHAPE MISMATCH {g.shape} vs {want.shape}")
                continue
            max_abs = float(np.abs(g - want).max())
            checks[label] = {"status": "ok", "max_abs_diff": max_abs}
            print(f"  {label}: max_abs_diff = {max_abs:.6g}{'  (BIT-IDENTICAL)' if max_abs == 0.0 else ''}")
        save("unit.codec_qdecode_out", got_qdecode)
        save("unit.codec_full_decode_out", got_full)
        save("unit.codec_encoder_out", got_enc)
    except FileNotFoundError as exc:
        checks["error"] = f"0.6b goldens unavailable: {exc}"
        print(f"  skipped: {exc}")
    manifest["_codec_crosscheck_vs_0.6b"] = checks

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len([k for k in manifest if not k.startswith('_')])} goldens saved to {OUT}")
    print(f"greedy codes: {tuple(codes.shape)}  waveform samples: {int(lengths[0])}")


if __name__ == "__main__":
    main()
