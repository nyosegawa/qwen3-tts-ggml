"""Converts Semantic-DACVAE-Japanese-32dim, the codec of Irodori-TTS, to the GGUF the C++ port reads.

usage: uv run python convert_codec.py <out dir> [--type f32|f16]

Writes semantic-dacvae-japanese-32dim-<type>.gguf: the encoder and the mean of the bottleneck, which turn a
reference voice into the latent the speaker encoder reads, and the decoder, which turns a latent into audio.

The weight normalization is folded by the dacvae library itself (remove_weight_norm). Every convolution
weight is stored as numpy [k, out, in] (ne = [in, out, k]), so that tap k is a plain [in, out] matrix: the
C++ side runs a convolution as one matrix product per tap on channel-first activations. A transposed
convolution's weight [in, out, k] is stored the same way as [k, out, in]. A strided
convolution of width 2s is stored as its two halves, each [out, s * in] with the input channel fastest, so
that it runs as two matrix products on the input cut into frames of s samples.
"""

import argparse
import os

import numpy as np
import torch
from dacvae import DACVAE
from gguf import GGUFWriter

from pins import CODEC, snapshot

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("--type", choices=["f32", "f16"], default="f16")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

model = DACVAE.load(os.path.join(snapshot(CODEC), "weights.pth")).eval()
for module in model.modules():
    try:
        torch.nn.utils.remove_weight_norm(module)
    except ValueError:
        pass
encoder_rates = [int(r) for r in model.encoder_rates]
assert int(model.hop_length) == int(np.prod(encoder_rates))

name = CODEC["repository"].split("/")[1]
path = os.path.join(args.out_dir, f"{name.lower()}-{args.type}.gguf")
w = GGUFWriter(path, "dacvae")
w.add_name(name)
w.add_license("MIT")
w.add_source_url(f"https://huggingface.co/{CODEC['repository']}/tree/{CODEC['revision']}")
w.add_uint32("dacvae.sample_rate", int(model.sample_rate))
w.add_uint32("dacvae.hop_length", int(model.hop_length))
w.add_uint32("dacvae.latent_dim", int(model.quantizer.codebook_dim))
w.add_array("dacvae.encoder_rates", encoder_rates)
decoder_rates = [int(r) for r in model.decoder_rates]
w.add_array("dacvae.decoder_rates", decoder_rates)


def array(t):
    return t.detach().float().numpy()


def add(out_name, data, big):
    data = np.ascontiguousarray(data, dtype=np.float32)
    w.add_tensor(out_name, data.astype(np.float16) if big and args.type == "f16" else data)


def conv(prefix, module):
    """A stride-1 Conv1d: weight [out, in, k] stored as [k, out, in]."""
    assert module.stride[0] == 1 and module.padding[0] == (module.kernel_size[0] - 1) * module.dilation[0] // 2
    add(prefix + ".weight", np.transpose(array(module.weight), (2, 0, 1)), True)
    add(prefix + ".bias", array(module.bias), False)


def snake(prefix, module):
    """Snake: x + sin(alpha x)^2 / (alpha + 1e-9), with the reciprocal computed in float32 as dacvae does."""
    alpha = array(module.alpha).reshape(-1)
    add(prefix + ".alpha", alpha, False)
    add(prefix + ".inv_alpha", np.float32(1.0) / (alpha + np.float32(1e-9)), False)


def residual(prefix, unit):
    snake(prefix + ".snake1", unit.block[0])
    conv(prefix + ".conv1", unit.block[1])
    snake(prefix + ".snake2", unit.block[2])
    conv(prefix + ".conv2", unit.block[3])


encoder = model.encoder.block
conv("enc.conv_in", encoder[0])
for i, stride in enumerate(encoder_rates):
    block = encoder[i + 1].block
    p = f"enc.blk.{i}"
    for j in range(3):
        assert block[j].block[1].dilation[0] == 3**j
        residual(f"{p}.res.{j}", block[j])
    snake(p + ".snake", block[3])
    down = block[4]
    assert down.stride[0] == stride and down.kernel_size[0] == 2 * stride and down.padding[0] == stride // 2
    weight = array(down.weight)
    add(p + ".down.first", np.transpose(weight[:, :, :stride], (0, 2, 1)).reshape(weight.shape[0], -1), True)
    add(p + ".down.second", np.transpose(weight[:, :, stride:], (0, 2, 1)).reshape(weight.shape[0], -1), True)
    add(p + ".down.bias", array(down.bias), False)
snake("enc.snake", encoder[len(encoder_rates) + 1])
conv("enc.conv_out", encoder[len(encoder_rates) + 2])

# The bottleneck's mean: the first half of in_proj's channels; Irodori-TTS encodes deterministically.
latent_dim = int(model.quantizer.codebook_dim)
in_proj = model.quantizer.in_proj
add("bottleneck.mean.weight", array(in_proj.weight)[:latent_dim, :, 0], True)
add("bottleneck.mean.bias", array(in_proj.bias)[:latent_dim], False)

# The decoder. Of each block's layers the forward pass uses the Snake, the transposed convolution and the
# residual units of dilation 1, 3 and 9 (the other layers belong to the watermark's path). Irodori-TTS
# replaces the watermark with the first Snake, convolution and tanh of its encoder block.
conv("dec.in_proj", model.quantizer.out_proj)
conv("dec.conv_in", model.decoder.model[0])
for i, stride in enumerate(decoder_rates):
    block = model.decoder.model[i + 1].block
    p = f"dec.blk.{i}"
    snake(p + ".snake", block[0])
    up = block[1]
    assert up.stride[0] == stride and up.kernel_size[0] == 2 * stride and up.padding[0] == stride // 2
    assert stride % 2 == 0 and up.output_padding[0] == 0
    add(p + ".up.weight", np.transpose(array(up.weight), (2, 1, 0)), True)
    add(p + ".up.bias", array(up.bias), False)
    for j, layer in enumerate((4, 5, 8)):
        assert block[layer].block[1].dilation[0] == 3**j
        residual(f"{p}.res.{j}", block[layer])
watermark = model.decoder.wm_model.encoder_block.pre
snake("dec.out_snake", watermark[0])
conv("dec.conv_out", watermark[1])

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path)
