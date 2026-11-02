"""Capture PyTorch oracle goldens for the arktts (Audio8-TTS-Preview-0.6b) MLX port.

Runs the reference implementation fp32 on CPU (deterministic, do_sample=False) and saves:
  1. e2e fixtures: processor outputs, reference codes, generated codes, waveform
  2. hooked intermediates: embed, slow blocks, norm/logits, fast stack, codec stages
  3. unit fixtures: seeded random inputs through isolated sub-modules

All arrays land in goldens/ as .npy plus a manifest.json describing shapes/dtypes.
"""
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
import torch

# AUDIO8_06B_CHECKPOINT picks the 0.6b checkpoint: Preview (default) or Realistic — same graph, new LM values.
CHECKPOINT = os.environ.get("AUDIO8_06B_CHECKPOINT", "Audio8-TTS-Preview-0.6b")
REPO = WORK / "reference" / CHECKPOINT
OUT = HARNESS / "goldens"
OUT.mkdir(exist_ok=True)
REF_WAV = Path(
    "/Volumes/Satechi/Development/mlxengine-audio/PROD/gepard/oracle-capture/extra_ref_audio/roxy_d1_trim.wav"
)
REF_TEXT = (
    "You know, I was thinking about what you said earlier, and honestly, "
    "I think you might be right about the whole thing."
)
TARGET_TEXT = "The quick brown fox jumps over the lazy dog, while the river keeps flowing north."

manifest: dict[str, dict] = {}


def save(name: str, tensor):
    arr = tensor.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    np.save(OUT / f"{name}.npy", arr)
    manifest[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    print(f"  saved {name}: {list(arr.shape)}")


def main():
    torch.manual_seed(0)
    sys.path.insert(0, str(REPO))
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(REPO), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(REPO), trust_remote_code=True, dtype=torch.float32).eval()

    # ---- 1. processor fixtures -------------------------------------------------
    inputs = processor(
        text=[TARGET_TEXT],
        reference_text=[REF_TEXT],
        reference_audio=[str(REF_WAV)],
    )
    for key, value in inputs.items():
        save(f"proc.{key}", value)

    # resampled 44.1k reference waveform as its own fixture (decouples resampler parity)
    ref_audio_441 = inputs["reference_audio_values"][0, 0, : int(inputs["reference_audio_lengths"][0])]
    save("ref_audio_44100", ref_audio_441)

    # ---- 2. codec encode -------------------------------------------------------
    codec = model.load_codec(device="cpu", dtype=torch.float32)

    codec_hooks = {}

    def hook(name):
        def fn(_m, _i, output):
            if name not in codec_hooks:
                codec_hooks[name] = output if torch.is_tensor(output) else output[0]
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

    # ---- 3. prompt build + slow prefill ---------------------------------------
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

    # no-cache forward for prefill parity (hook slow blocks 0, 11, 23)
    embed = model._embed(prompt)
    save("prefill.embed", embed)
    layer_outs = {}

    def layer_hook(idx):
        def fn(_m, _i, output):
            layer_outs[idx] = output
        return fn

    hs = [model.layers[i].register_forward_hook(layer_hook(i)) for i in (0, 11, 23)]
    with torch.inference_mode():
        fwd = model(prompt, attention_mask=prompt_mask)
    for h in hs:
        h.remove()
    for idx, tensor in layer_outs.items():
        save(f"prefill.layer{idx}_out", tensor)
    save("prefill.logits_last", fwd.logits[:, -1])
    save("prefill.hidden_last", fwd.hidden_states[:, -1])

    # ---- 4. deterministic generation (argmax everywhere) ----------------------
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

    # ---- 5. codec decode -------------------------------------------------------
    decode_hooks = {}

    def dhook(name):
        def fn(_m, _i, output):
            decode_hooks.setdefault(name, output if torch.is_tensor(output) else output[0])
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

    # latent right before the decoder for isolation of decoder-only parity
    with torch.inference_mode():
        latent = codec.quantizer.decode(codes[:, :, : int(lengths[0]) // 2048 if False else codes.shape[-1]].long())
    save("decode.pre_decoder_latent", latent)

    # ---- 6. unit fixtures: seeded inputs through isolated sub-modules ----------
    for layer in list(model.layers) + list(model.fast_layers):
        layer.attention.kv_cache = None
    rng = np.random.default_rng(42)

    x_slow = rng.standard_normal((1, 17, 896)).astype(np.float32)
    save("unit.x_slow_in", x_slow)
    xt = torch.from_numpy(x_slow)
    rope = model.freqs_cis[torch.arange(17)][None]
    length = 17
    row = torch.arange(length)[:, None]
    col = torch.arange(length)[None, :]
    mask = (col <= row)[None, None]
    with torch.inference_mode():
        save("unit.slow_block0_out", model.layers[0](xt, rope, mask))
        save("unit.slow_attn0_out", model.layers[0].attention(model.layers[0].attention_norm(xt), rope, mask))
        save("unit.slow_ffn0_out", model.layers[0].feed_forward(xt))
        save("unit.rmsnorm_out", model.norm(xt))
        save("unit.fast_block0_out", model.fast_layers[0](
            torch.from_numpy(rng.standard_normal((1, 10, 896)).astype(np.float32)
                             if False else x_slow[:, :10]),
            model.fast_freqs_cis[torch.arange(10)][None],
            (torch.arange(10)[None, :] <= torch.arange(10)[:, None])[None, None],
        ))
    save("unit.rope_table_slow", model.freqs_cis[:32])
    save("unit.rope_table_fast", model.fast_freqs_cis)

    # codec units
    x_audio = rng.standard_normal((1, 1, 8192)).astype(np.float32) * 0.1
    save("unit.x_audio_in", x_audio)
    xa = torch.from_numpy(x_audio)
    with torch.inference_mode():
        save("unit.codec_encoder_out", codec.encoder(xa))
        first_res = codec.encoder.block[1].block[0]
        x_res_in = rng.standard_normal((1, 64, 256)).astype(np.float32)
        save("unit.x_res_in", x_res_in)
        save("unit.codec_resunit_out", first_res(torch.from_numpy(x_res_in)))
        x_wt_in = rng.standard_normal((1, 1024, 24)).astype(np.float32)
        save("unit.x_wt_in", x_wt_in)
        save("unit.codec_pre_module_out", codec.quantizer.pre_module(torch.from_numpy(x_wt_in)))
        codes_unit = torch.from_numpy(
            np.stack([rng.integers(0, 4096, size=(1, 24)),
                      *[rng.integers(0, 1024, size=(1, 24)) for _ in range(9)]], axis=1)
        ).long()
        save("unit.codes_in", codes_unit)
        save("unit.codec_qdecode_out", codec.quantizer.decode(codes_unit))
        save("unit.codec_full_decode_out", codec.decode(codes_unit))

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} goldens saved to {OUT}")
    print("greedy codes shape:", codes.shape, "waveform samples:", int(lengths[0]))


if __name__ == "__main__":
    main()
