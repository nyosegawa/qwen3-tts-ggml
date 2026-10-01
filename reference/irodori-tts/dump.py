"""Runs the official Irodori-TTS on the CPU in float32 and saves the tensors the C++ port is checked against.

usage: uv run python dump.py <mf|rf> <out dir> <text> <reference.wav> [--seed n]

mf is v4.1-Small-MF with its 4 MeanFlow steps; rf is v4.1-Small with 40 Euler steps and the runtime's
default guidance (text 3.0 and speaker 5.0, each against a branch without it, while t >= 0.5).

The official runtime.synthesize() runs once with its stages wrapped, so what is saved is what it computed.
The finer stages (the text encoder's layers, the speaker encoder, each DiT block of the first step, the
codec's blocks) are then recomputed from those inputs, and the recomputation is asserted to give the same
result as synthesize(). Writes <out dir>/*.npy and meta.json:

  input_ids            the text's tokens with <s>, [N]; the runtime pads them to 256 and masks the rest
  text_layers          ModernBERT's hidden states on the N tokens alone: embeddings then each layer, [26, N, 768]
  text_backbone        ModernBERT's final output on the padded input, [N, 768]
  text_state           the text condition after the projector and its norm, [N, 512]
  ref_wav              the reference as read, mono float32 at 48 kHz
  ref_wav_normalized   the reference after loudness normalization to -16 LUFS
  ref_latent           its DACVAE latent (the encoder's mean), [T, 32]
  ref_encoder          the DACVAE encoder's output before the bottleneck, [1024, T]
  speaker_encoded      the speaker encoder's output on the latent in patches of 4, [T / 4, 768]
  speaker_state        after its norm, with the masked mean prepended, [1 + T / 4, 768]
  duration_features    the 14 features of the text, [14]
  duration_log_frames  the predictor's log(1 + frames), [1]
  noise                the sampler's starting point, [S, 32]
  dit_t                each step's time, [steps]
  dit_delta            mf: each step's interval, [steps]
  dit_branches         rf: the DiT output of each branch (all, no text, no speaker) of the guided steps,
                       [guided steps, 3, S, 32]
  dit_velocity         the velocity of each step, after guidance for rf, [steps, S, 32]
  dit_x                the latent after each step, [steps, S, 32]
  dit_step0_blocks     the output of each DiT block in the first step, [12, branches, S, 1280]
  dit_cond             the timestep condition of each step, [steps, 3840]
  codec_in             the codec input after its bottleneck projection, [1024, S]
  codec_conv_in        after the first convolution, [1536, S]
  codec_block{0..3}    after each upsampling block, [channels, samples so far]
  wav                  the decoded audio before the tail is trimmed, [S * 1920]
  audio                what synthesize() returned, trimmed at the tail
"""

import argparse
import json
import os

import numpy as np
import torch

from irodori_tts import inference_runtime as ir
from irodori_tts.codec import DACVAECodec
from irodori_tts.model import get_timestep_embedding, patch_sequence_with_mask
from irodori_tts.text_normalization import normalize_text
from pins import CODE, CODEC, MODELS, snapshot

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("text")
parser.add_argument("reference")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
torch.set_grad_enabled(False)

ir.SilentCipherWatermarker._load_backend = staticmethod(lambda **_: None)
runtime = ir.InferenceRuntime.from_key(
    ir.RuntimeKey(
        checkpoint=os.path.join(snapshot(MODELS[args.model]), "model.safetensors"),
        model_device="cpu",
        codec_repo=os.path.join(snapshot(CODEC), "weights.pth"),
        codec_device="cpu",
    )
)
model, codec = runtime.model, runtime.codec
saved = {}


def save(name, tensor):
    array = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    saved[name] = array.astype(np.int32 if array.dtype.kind in "iub" else np.float32)


def wrap(owner, name, record):
    original = getattr(owner, name)

    def wrapper(*a, **k):
        out = original(*a, **k)
        record(a, k, out)
        return out

    setattr(owner, name, wrapper)


calls = {"encode_conditions": [], "velocity": [], "normalized": [], "encode_in": [], "decode": []}
normalize_loudness = DACVAECodec._normalize_loudness
DACVAECodec._normalize_loudness = staticmethod(
    lambda *a, **k: calls["normalized"].append(normalize_loudness(*a, **k)) or calls["normalized"][-1])
wrap(codec, "encode_waveform", lambda a, k, out: calls["encode_in"].append((a[0], out)))
wrap(model, "encode_conditions", lambda a, k, out: calls["encode_conditions"].append((k, out)))
wrap(model, "predict_duration_log_frames", lambda a, k, out: saved.update(
    duration_features=k["duration_features"][0].numpy(), duration_log_frames=out.numpy()))
wrap(model, "forward_with_encoded_conditions", lambda a, k, out: calls["velocity"].append((k, out)))
wrap(codec, "decode_latent", lambda a, k, out: calls["decode"].append((a[0], out)))

result = runtime.synthesize(ir.SamplingRequest(text=args.text, ref_wav=args.reference, seed=args.seed))
normalized_text = normalize_text(args.text).strip()

# Text: the runtime pads to 256 tokens; the port runs the valid ones only.
kwargs, (text_state, text_mask, speaker_state, speaker_mask, _, _) = calls["encode_conditions"][-1]
n = int(text_mask[0].sum())
ids = kwargs["text_input_ids"][0, :n]
save("input_ids", ids)
backbone = model.pretrained_text_backbone
padded = backbone(kwargs["text_input_ids"], kwargs["text_mask"])[0, :n]
save("text_backbone", padded)
layers = backbone.backbone(input_ids=ids[None], attention_mask=torch.ones(1, n, dtype=torch.long),
                           output_hidden_states=True, return_dict=True).hidden_states
save("text_layers", torch.stack([h[0] for h in layers]))
unpadded_error = float((layers[-1][0] - padded).abs().max())
state = model.text_norm(model.text_encoder(backbone, kwargs["text_input_ids"], kwargs["text_mask"]))[0, :n]
assert torch.equal(state, text_state[0, :n])
save("text_state", state)

# Reference: loudness, the DACVAE encoder, the speaker encoder.
wav_in, latent = calls["encode_in"][0]
save("ref_wav", wav_in.reshape(-1))
save("ref_wav_normalized", calls["normalized"][0])
padded_wav = codec.model._pad(calls["normalized"][0].reshape(1, 1, -1))
encoded = codec.model.encoder(padded_wav)
assert torch.equal(codec.model.quantizer.in_proj(encoded).chunk(2, dim=1)[0].transpose(1, 2), latent)
save("ref_encoder", encoded[0])
save("ref_latent", latent[0])
patched, patch_mask = patch_sequence_with_mask(latent, torch.ones(latent.shape[:2], dtype=torch.bool),
                                               model.cfg.speaker_patch_size)
speaker_encoded = model.speaker_encoder(patched, patch_mask)
save("speaker_encoded", speaker_encoded[0])
recomputed, _ = model._prepend_masked_mean_token(model.speaker_norm(speaker_encoded), patch_mask)
assert torch.equal(recomputed, speaker_state)
save("speaker_state", speaker_state[0])

# Sampling: every step, and each block of the first.
steps = calls["velocity"]
meanflow = model.cfg.flow_parameterization == "meanflow"
save("noise", steps[0][0]["x_t"][0])
save("dit_t", torch.stack([k["t"][0] for k, _ in steps]))
velocities, branches = [], []
for k, out in steps:
    if out.shape[0] == 1:
        velocities.append(out)
        continue
    # The sampler's own order of operations, so that the latent comes out bit for bit.
    v = out[0:1]
    for scale, branch in zip((3.0, 5.0), out[1:]):
        v = v + scale * (out[0:1] - branch[None])
    velocities.append(v)
    branches.append(out)
if meanflow:
    save("dit_delta", torch.stack([k["delta_t"][0] for k, _ in steps]))
else:
    save("dit_branches", torch.stack(branches))
save("dit_velocity", torch.cat(velocities))
xs, x = [], steps[0][0]["x_t"][0:1]
for i, (k, _) in enumerate(steps):
    t_next = steps[i + 1][0]["t"][0] if i + 1 < len(steps) else torch.zeros_like(k["t"][0])
    x = x + velocities[i] * (t_next - k["t"][0])
    xs.append(x[0])
save("dit_x", torch.stack(xs))
cond = []
for k, _ in steps:
    c = model.cond_module(get_timestep_embedding(k["t"][0:1], model.cfg.timestep_embed_dim))
    if meanflow:
        c = c + model.delta_cond_module(get_timestep_embedding(k["delta_t"][0:1], model.cfg.timestep_embed_dim))
    cond.append(c[0])
save("dit_cond", torch.stack(cond))
blocks = []
hooks = [b.register_forward_hook(lambda m, a, out: blocks.append(out)) for b in model.blocks]
first = model.forward_with_encoded_conditions(**steps[0][0])
for h in hooks:
    h.remove()
assert torch.equal(first, steps[0][1])
save("dit_step0_blocks", torch.stack(blocks))

# Codec: the decoder's stages on the final latent.
z, audio = calls["decode"][0]
assert torch.equal(z, xs[-1][None, : z.shape[1]])
dac = codec.model
h = dac.quantizer.out_proj(z.transpose(1, 2))
save("codec_in", h[0])
h = dac.decoder.model[0](h)
save("codec_conv_in", h[0])
for i, block in enumerate(dac.decoder.model[1:]):
    h = block(h)
    save(f"codec_block{i}", h[0])
h = dac.decoder.watermark(h)
assert torch.equal(h, audio)
save("wav", audio.reshape(-1))
save("audio", result.audio.reshape(-1))

for name, array in saved.items():
    np.save(os.path.join(args.out_dir, f"{name}.npy"), array)
meta = {
    "code": CODE, "model": MODELS[args.model], "codec": CODEC, "seed": args.seed,
    "text": args.text, "normalized_text": normalized_text, "tokens": n,
    "reference": os.path.basename(args.reference),
    "latent_frames": int(z.shape[1]), "audio_samples": int(result.audio.shape[-1]),
    "unpadded_text_max_abs_diff": unpadded_error,
    "messages": result.messages,
}
json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in meta.items() if k not in ("code", "messages")}, ensure_ascii=False))
