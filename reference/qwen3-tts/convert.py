"""Converts an official Qwen3-TTS 12Hz CustomVoice checkpoint to the GGUF files the C++ port reads.

usage: uv run python convert.py <model dir> <out dir> [--type f16|q8_0] [--codec-type f32|f16]

Writes two files:
  qwen3-tts-<size>-customvoice-<type>.gguf   the talker, the code predictor, the text embedding and the tokenizer
  qwen3-tts-codec-12hz-<codec type>.gguf     the codec decoder, which every size shares

Tensor shapes follow ggml, whose ne[0] is the last numpy axis. A Linear weight [out, in] is stored as is
(ne = [in, out]). Every convolution weight is stored as numpy [k, out, in] (ne = [in, out, k]), so that tap k is
a plain [in, out] matrix: the C++ side runs a convolution as one matrix product per tap on channel-first
activations. A depthwise weight [c, 1, k] is stored as [k, c].
"""

import argparse
import json
import os

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import quantize
from safetensors import safe_open

ARCH_TALKER = "qwen3tts-talker"
ARCH_CODEC = "qwen3tts-codec"

# The BCP 47 tag of each language the checkpoint names. Its dialects have none: a dialect is spoken only by
# its speakers, when the language is Chinese or left to the model.
LANGUAGE_TAGS = {"chinese": "zh", "english": "en", "french": "fr", "german": "de", "italian": "it",
                 "japanese": "ja", "korean": "ko", "portuguese": "pt", "russian": "ru", "spanish": "es"}

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("out_dir")
parser.add_argument("--type", choices=["f16", "q8_0", "f32"], default="q8_0")
parser.add_argument("--codec-type", choices=["f16", "f32"], default="f16")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

config = json.load(open(os.path.join(args.model_dir, "config.json")))
talker_cfg = config["talker_config"]
cp_cfg = talker_cfg["code_predictor_config"]
size = {"0b6": "0.6b", "1b7": "1.7b"}[config["tts_model_size"]]
assert config["tts_model_type"] == "custom_voice", "only CustomVoice checkpoints are supported"
assert config["tokenizer_type"] == "qwen3_tts_tokenizer_12hz"


def load(path):
    f = safe_open(path, "pt")
    return {k: f.get_tensor(k).float().numpy() for k in f.keys()}


def add(writer, name, data, kind):
    """kind: 'matrix' is quantized to the requested type, 'f16' or 'f32' is stored as is."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    if kind == "matrix" and args.type == "q8_0" and data.ndim == 2 and data.shape[-1] % 32 == 0:
        writer.add_tensor(name, quantize(data, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)
    elif kind == "matrix" and args.type in ("f16", "q8_0"):
        writer.add_tensor(name, data.astype(np.float16))
    elif kind == "f16":
        writer.add_tensor(name, data.astype(np.float16))
    else:
        writer.add_tensor(name, data)


# ---------------------------------------------------------------- talker
weights = load(os.path.join(args.model_dir, "model.safetensors"))
talker_path = os.path.join(args.out_dir, f"qwen3-tts-{size}-customvoice-{args.type}.gguf")
w = GGUFWriter(talker_path, ARCH_TALKER)
w.add_name(f"Qwen3-TTS-12Hz-{size.upper()}-CustomVoice")
for key in ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "vocab_size", "text_hidden_size", "text_vocab_size", "num_code_groups",
            "codec_bos_id", "codec_eos_token_id", "codec_pad_id", "codec_think_id", "codec_nothink_id",
            "codec_think_bos_id", "codec_think_eos_id"]:
    w.add_uint32(f"talker.{key}", int(talker_cfg[key]))
w.add_float32("talker.rms_norm_eps", float(talker_cfg["rms_norm_eps"]))
w.add_float32("talker.rope_theta", float(talker_cfg["rope_theta"]))
for key in ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "vocab_size"]:
    w.add_uint32(f"code_predictor.{key}", int(cp_cfg[key]))
w.add_float32("code_predictor.rms_norm_eps", float(cp_cfg["rms_norm_eps"]))
w.add_float32("code_predictor.rope_theta", float(cp_cfg["rope_theta"]))
for key in ["tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id", "im_start_token_id", "im_end_token_id",
            "assistant_token_id"]:
    w.add_uint32(f"text.{key}", int(config[key]))

speakers = sorted(talker_cfg["spk_id"].items(), key=lambda kv: kv[0])
w.add_array("talker.speaker_names", [name for name, _ in speakers])
w.add_array("talker.speaker_ids", [int(i) for _, i in speakers])
w.add_array("talker.speaker_dialects", [talker_cfg["spk_is_dialect"][name] or "" for name, _ in speakers])
languages = sorted(talker_cfg["codec_language_id"].items(), key=lambda kv: kv[0])
dialects = {d for d in talker_cfg["spk_is_dialect"].values() if d}
unknown = [name for name, _ in languages if name not in LANGUAGE_TAGS and name not in dialects]
assert not unknown, f"no BCP 47 tag for the languages {unknown}"
w.add_array("talker.language_names", [name for name, _ in languages])
w.add_array("talker.language_ids", [int(i) for _, i in languages])
w.add_array("talker.language_tags", [LANGUAGE_TAGS.get(name, "") for name, _ in languages])
tags = sorted(LANGUAGE_TAGS[name] for name, _ in languages if name in LANGUAGE_TAGS)
w.add_languages(tags)
w.add_array("speech.languages", tags)
w.add_bool("speech.language_selectable", True)

# The Qwen2 byte-level BPE: tokens in id order, merges in rank order, and the special tokens.
vocab = json.load(open(os.path.join(args.model_dir, "vocab.json"), encoding="utf-8"))
tok_cfg = json.load(open(os.path.join(args.model_dir, "tokenizer_config.json"), encoding="utf-8"))
added = {int(i): v["content"] for i, v in tok_cfg["added_tokens_decoder"].items()}
n_tokens = max(max(vocab.values()), max(added)) + 1
tokens = [""] * n_tokens
for t, i in vocab.items():
    tokens[i] = t
for i, t in added.items():
    tokens[i] = t
merges = [line.rstrip("\n") for line in open(os.path.join(args.model_dir, "merges.txt"), encoding="utf-8")
          if line.strip() and not line.startswith("#version")]
w.add_array("tokenizer.tokens", tokens)
w.add_array("tokenizer.merges", merges)
w.add_array("tokenizer.special_ids", sorted(added))

p = "talker."
add(w, "talker.text_embd", weights[p + "model.text_embedding.weight"], "matrix")
add(w, "talker.text_proj.fc1.weight", weights[p + "text_projection.linear_fc1.weight"], "matrix")
add(w, "talker.text_proj.fc1.bias", weights[p + "text_projection.linear_fc1.bias"], "f32")
add(w, "talker.text_proj.fc2.weight", weights[p + "text_projection.linear_fc2.weight"], "matrix")
add(w, "talker.text_proj.fc2.bias", weights[p + "text_projection.linear_fc2.bias"], "f32")
add(w, "talker.codec_embd", weights[p + "model.codec_embedding.weight"], "matrix")
add(w, "talker.codec_head", weights[p + "codec_head.weight"], "matrix")
add(w, "talker.norm", weights[p + "model.norm.weight"], "f32")


def add_layers(prefix_in, prefix_out, n):
    for i in range(n):
        a = f"{prefix_in}.layers.{i}."
        b = f"{prefix_out}.blk.{i}."
        add(w, b + "attn_norm", weights[a + "input_layernorm.weight"], "f32")
        add(w, b + "ffn_norm", weights[a + "post_attention_layernorm.weight"], "f32")
        add(w, b + "attn_q", weights[a + "self_attn.q_proj.weight"], "matrix")
        add(w, b + "attn_k", weights[a + "self_attn.k_proj.weight"], "matrix")
        add(w, b + "attn_v", weights[a + "self_attn.v_proj.weight"], "matrix")
        add(w, b + "attn_o", weights[a + "self_attn.o_proj.weight"], "matrix")
        add(w, b + "attn_q_norm", weights[a + "self_attn.q_norm.weight"], "f32")
        add(w, b + "attn_k_norm", weights[a + "self_attn.k_norm.weight"], "f32")
        add(w, b + "ffn_gate", weights[a + "mlp.gate_proj.weight"], "matrix")
        add(w, b + "ffn_up", weights[a + "mlp.up_proj.weight"], "matrix")
        add(w, b + "ffn_down", weights[a + "mlp.down_proj.weight"], "matrix")


add_layers(p + "model", "talker", talker_cfg["num_hidden_layers"])
c = p + "code_predictor."
add_layers(c + "model", "cp", cp_cfg["num_hidden_layers"])
add(w, "cp.norm", weights[c + "model.norm.weight"], "f32")
for i in range(talker_cfg["num_code_groups"] - 1):
    add(w, f"cp.codec_embd.{i}", weights[c + f"model.codec_embedding.{i}.weight"], "matrix")
    add(w, f"cp.head.{i}", weights[c + f"lm_head.{i}.weight"], "matrix")
if c + "small_to_mtp_projection.weight" in weights:
    add(w, "cp.in_proj.weight", weights[c + "small_to_mtp_projection.weight"], "matrix")
    add(w, "cp.in_proj.bias", weights[c + "small_to_mtp_projection.bias"], "f32")

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", talker_path)

# ---------------------------------------------------------------- codec decoder
codec_dir = os.path.join(args.model_dir, "speech_tokenizer")
codec_cfg = json.load(open(os.path.join(codec_dir, "config.json")))["decoder_config"]
cw = {k[len("decoder."):]: v for k, v in load(os.path.join(codec_dir, "model.safetensors")).items()
      if k.startswith("decoder.")}
codec_path = os.path.join(args.out_dir, f"qwen3-tts-codec-12hz-{args.codec_type}.gguf")
w = GGUFWriter(codec_path, ARCH_CODEC)
w.add_name("Qwen3-TTS-Tokenizer-12Hz decoder")
for key in ["latent_dim", "codebook_dim", "codebook_size", "decoder_dim", "hidden_size", "intermediate_size",
            "head_dim", "num_attention_heads", "num_key_value_heads", "num_hidden_layers", "num_quantizers",
            "sliding_window"]:
    w.add_uint32(f"codec.{key}", int(codec_cfg[key]))
w.add_float32("codec.rms_norm_eps", float(codec_cfg["rms_norm_eps"]))
w.add_float32("codec.rope_theta", float(codec_cfg["rope_theta"]))
w.add_array("codec.upsample_rates", [int(r) for r in codec_cfg["upsample_rates"]])
w.add_array("codec.upsampling_ratios", [int(r) for r in codec_cfg["upsampling_ratios"]])

big = args.codec_type  # "f16" or "f32" for the large weights; small vectors stay f32


def codebook(prefix):
    usage = cw[prefix + "._codebook.cluster_usage"]
    return cw[prefix + "._codebook.embedding_sum"] / np.maximum(usage, 1e-5)[:, None]


# The first quantizer and the other fifteen each project 256 -> 512 with their own 1x1 conv.
add(w, "codec.vq.first.codebook.0", codebook("quantizer.rvq_first.vq.layers.0"), "f32")
add(w, "codec.vq.first.out_proj", cw["quantizer.rvq_first.output_proj.weight"][:, :, 0], "f32")
for i in range(codec_cfg["num_quantizers"] - 1):
    add(w, f"codec.vq.rest.codebook.{i}", codebook(f"quantizer.rvq_rest.vq.layers.{i}"), "f32")
add(w, "codec.vq.rest.out_proj", cw["quantizer.rvq_rest.output_proj.weight"][:, :, 0], "f32")

def conv(name, weight):
    """Conv1d weight [out, in, k] -> [k, out, in]."""
    add(w, name, np.transpose(weight, (2, 0, 1)), big)


def tconv(name, weight):
    """ConvTranspose1d weight [in, out, k] -> [k, out, in]."""
    add(w, name, np.transpose(weight, (2, 1, 0)), big)


conv("codec.pre_conv.weight", cw["pre_conv.conv.weight"])
add(w, "codec.pre_conv.bias", cw["pre_conv.conv.bias"], "f32")

t = "pre_transformer."
add(w, "codec.tf.in_proj.weight", cw[t + "input_proj.weight"], big)
add(w, "codec.tf.in_proj.bias", cw[t + "input_proj.bias"], "f32")
add(w, "codec.tf.out_proj.weight", cw[t + "output_proj.weight"], big)
add(w, "codec.tf.out_proj.bias", cw[t + "output_proj.bias"], "f32")
add(w, "codec.tf.norm", cw[t + "norm.weight"], "f32")
for i in range(codec_cfg["num_hidden_layers"]):
    a = f"{t}layers.{i}."
    b = f"codec.tf.blk.{i}."
    add(w, b + "attn_norm", cw[a + "input_layernorm.weight"], "f32")
    add(w, b + "ffn_norm", cw[a + "post_attention_layernorm.weight"], "f32")
    add(w, b + "attn_q", cw[a + "self_attn.q_proj.weight"], big)
    add(w, b + "attn_k", cw[a + "self_attn.k_proj.weight"], big)
    add(w, b + "attn_v", cw[a + "self_attn.v_proj.weight"], big)
    add(w, b + "attn_o", cw[a + "self_attn.o_proj.weight"], big)
    add(w, b + "attn_scale", cw[a + "self_attn_layer_scale.scale"], "f32")
    add(w, b + "ffn_gate", cw[a + "mlp.gate_proj.weight"], big)
    add(w, b + "ffn_up", cw[a + "mlp.up_proj.weight"], big)
    add(w, b + "ffn_down", cw[a + "mlp.down_proj.weight"], big)
    add(w, b + "ffn_scale", cw[a + "mlp_layer_scale.scale"], "f32")

for i in range(len(codec_cfg["upsampling_ratios"])):
    a = f"upsample.{i}."
    b = f"codec.up.{i}."
    tconv(b + "tconv.weight", cw[a + "0.conv.weight"])
    add(w, b + "tconv.bias", cw[a + "0.conv.bias"], "f32")
    add(w, b + "dwconv.weight", np.transpose(cw[a + "1.dwconv.conv.weight"][:, 0, :]), "f32")
    add(w, b + "dwconv.bias", cw[a + "1.dwconv.conv.bias"], "f32")
    add(w, b + "norm.weight", cw[a + "1.norm.weight"], "f32")
    add(w, b + "norm.bias", cw[a + "1.norm.bias"], "f32")
    add(w, b + "pw1.weight", cw[a + "1.pwconv1.weight"], big)
    add(w, b + "pw1.bias", cw[a + "1.pwconv1.bias"], "f32")
    add(w, b + "pw2.weight", cw[a + "1.pwconv2.weight"], big)
    add(w, b + "pw2.bias", cw[a + "1.pwconv2.bias"], "f32")
    add(w, b + "gamma", cw[a + "1.gamma"], "f32")


def snake(prefix, name):
    """SnakeBeta: x + inv_beta * sin(alpha * x)^2, with alpha = exp(a) and inv_beta = 1 / (exp(b) + 1e-9)."""
    add(w, name + ".alpha", np.exp(cw[prefix + ".alpha"]), "f32")
    add(w, name + ".inv_beta", 1.0 / (np.exp(cw[prefix + ".beta"]) + 1e-9), "f32")


conv("codec.dec.in_conv.weight", cw["decoder.0.conv.weight"])
add(w, "codec.dec.in_conv.bias", cw["decoder.0.conv.bias"], "f32")
n_blocks = len(codec_cfg["upsample_rates"])
for i in range(n_blocks):
    a = f"decoder.{i + 1}.block."
    b = f"codec.dec.blk.{i}."
    snake(a + "0", b + "snake")
    tconv(b + "tconv.weight", cw[a + "1.conv.weight"])
    add(w, b + "tconv.bias", cw[a + "1.conv.bias"], "f32")
    for j in range(3):
        r = f"{a}{j + 2}."
        s = f"{b}res.{j}."
        snake(r + "act1", s + "snake1")
        conv(s + "conv1.weight", cw[r + "conv1.conv.weight"])
        add(w, s + "conv1.bias", cw[r + "conv1.conv.bias"], "f32")
        snake(r + "act2", s + "snake2")
        conv(s + "conv2.weight", cw[r + "conv2.conv.weight"])
        add(w, s + "conv2.bias", cw[r + "conv2.conv.bias"], "f32")
snake(f"decoder.{n_blocks + 1}", "codec.dec.out_snake")
conv("codec.dec.out_conv.weight", cw[f"decoder.{n_blocks + 2}.conv.weight"])
add(w, "codec.dec.out_conv.bias", cw[f"decoder.{n_blocks + 2}.conv.bias"], "f32")

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", codec_path)
