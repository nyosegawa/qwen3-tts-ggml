"""Converts the official Irodori-TTS v4.1-Small-MF or v4.1-Small checkpoint to the GGUF the C++ port reads.

usage: uv run python convert.py <mf|rf> <out dir> [--type f32|f16|q8_0]

Writes irodori-tts-v4.1-small-mf-<type>.gguf or irodori-tts-v4.1-small-<type>.gguf: the tokenizer, ModernBERT-ja
and the projector that make the text condition.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is
(ne = [in, out]). --type applies to the matrices whose rows are a multiple of 32; norms, biases and the
rest stay float32.
"""

import argparse
import json
import os

import numpy as np
from gguf import GGMLQuantizationType, GGUFValueType, GGUFWriter
from gguf.quants import quantize
from safetensors import safe_open

from pins import MODELS, snapshot

ARCH = "irodori-tts"

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("--type", choices=["f32", "f16", "q8_0"], default="q8_0")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

pin = MODELS[args.model]
model_dir = snapshot(pin)
checkpoint = safe_open(os.path.join(model_dir, "model.safetensors"), "np")
config = json.loads(checkpoint.metadata()["config_json"])
text_config = json.loads(checkpoint.metadata()["text_encoder_config_json"])
assert config["text_encoder_type"] == "pretrained" and text_config["model_type"] == "modernbert"
assert config["pretrained_projector_type"] == "residual_mlp" and config["latent_patch_size"] == 1

name = pin["repository"].split("/")[1]
path = os.path.join(args.out_dir, f"{name.lower()}-{args.type}.gguf")
w = GGUFWriter(path, ARCH)
w.add_name(name)
w.add_license("MIT")
w.add_source_url(f"https://huggingface.co/{pin['repository']}/tree/{pin['revision']}")
w.add_languages(["ja"])
# The languages as BCP 47 tags; Irodori-TTS takes no language with a request.
w.add_array("speech.languages", ["ja"])
w.add_bool("speech.language_selectable", False)
w.add_string("irodori.flow", config["flow_parameterization"] if "flow_parameterization" in config else "rf_velocity")
w.add_float32("irodori.norm_eps", float(config["norm_eps"]))
w.add_uint32("irodori.max_text_tokens", int(config["max_text_len"]))

w.add_uint32("irodori.text.hidden_size", int(text_config["hidden_size"]))
w.add_uint32("irodori.text.intermediate_size", int(text_config["intermediate_size"]))
w.add_uint32("irodori.text.num_layers", int(text_config["num_hidden_layers"]))
w.add_uint32("irodori.text.num_heads", int(text_config["num_attention_heads"]))
w.add_float32("irodori.text.norm_eps", float(text_config["norm_eps"]))
assert text_config["hidden_activation"] == "gelu" and not text_config["norm_bias"]
assert not text_config["attention_bias"] and not text_config["mlp_bias"]
# A local layer attends to the tokens at most local_attention / 2 away on either side.
w.add_uint32("irodori.text.window", int(text_config["local_attention"]) // 2)
w.add_array("irodori.text.layer_global", [int(t == "full_attention") for t in text_config["layer_types"]])
w.add_float32("irodori.text.rope_theta_global", float(text_config["rope_parameters"]["full_attention"]["rope_theta"]))
w.add_float32("irodori.text.rope_theta_local", float(text_config["rope_parameters"]["sliding_attention"]["rope_theta"]))
w.add_uint32("irodori.text.dim", int(config["text_dim"]))

# The SentencePiece Unigram tokenizer, with its scores in float64 as tokenizer.json has them, since the
# Viterbi path compares their sums.
tokenizer = json.load(open(os.path.join(model_dir, "tokenizer", "tokenizer.json"), encoding="utf-8"))
model = tokenizer["model"]
assert model["type"] == "Unigram" and model["byte_fallback"] and tokenizer["normalizer"] is None
assert tokenizer["pre_tokenizer"] == {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": False}
tokens = [piece for piece, _ in model["vocab"]]
for added in tokenizer["added_tokens"]:
    assert tokens[added["id"]] == added["content"] and not added["normalized"]
    assert not (added["lstrip"] or added["rstrip"] or added["single_word"])
tokenizer_config = json.load(open(os.path.join(model_dir, "tokenizer", "tokenizer_config.json"), encoding="utf-8"))
w.add_array("tokenizer.tokens", tokens)
w.add_key_value("tokenizer.scores", [float(score) for _, score in model["vocab"]], GGUFValueType.ARRAY,
                sub_type=GGUFValueType.FLOAT64)
w.add_array("tokenizer.added_ids", [int(added["id"]) for added in tokenizer["added_tokens"]])
w.add_uint32("tokenizer.bos_id", tokens.index(tokenizer_config["bos_token"]))
w.add_uint32("tokenizer.unknown_id", int(model["unk_id"]))


def tensor(key):
    return checkpoint.get_tensor(key).astype(np.float32)


def add(out_name, data, matrix):
    data = np.ascontiguousarray(data, dtype=np.float32)
    if matrix and args.type == "q8_0" and data.ndim == 2 and data.shape[-1] % 32 == 0:
        w.add_tensor(out_name, quantize(data, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)
    elif matrix and args.type in ("f16", "q8_0"):
        w.add_tensor(out_name, data.astype(np.float16))
    else:
        w.add_tensor(out_name, data)


# ModernBERT-ja: the fused query, key and value projection is split, and so is the MLP's input projection
# into the half that goes through GELU and the half that gates it.
b = "pretrained_text_backbone.backbone."
hidden = int(text_config["hidden_size"])
inner = int(text_config["intermediate_size"])
add("text.embd", tensor(b + "embeddings.tok_embeddings.weight"), True)
add("text.embd_norm", tensor(b + "embeddings.norm.weight"), False)
for i in range(int(text_config["num_hidden_layers"])):
    a, o = f"{b}layers.{i}.", f"text.blk.{i}."
    if i > 0:
        add(o + "attn_norm", tensor(a + "attn_norm.weight"), False)
    qkv = tensor(a + "attn.Wqkv.weight")
    add(o + "attn_q", qkv[:hidden], True)
    add(o + "attn_k", qkv[hidden : 2 * hidden], True)
    add(o + "attn_v", qkv[2 * hidden :], True)
    add(o + "attn_out", tensor(a + "attn.Wo.weight"), True)
    add(o + "ffn_norm", tensor(a + "mlp_norm.weight"), False)
    wi = tensor(a + "mlp.Wi.weight")
    add(o + "ffn_act", wi[:inner], True)
    add(o + "ffn_gate", wi[inner:], True)
    add(o + "ffn_down", tensor(a + "mlp.Wo.weight"), True)
add("text.final_norm", tensor(b + "final_norm.weight"), False)

p = "text_encoder."
add("text.proj.weight", tensor(p + "projector.weight"), True)
add("text.proj.bias", tensor(p + "projector.bias"), False)
add("text.proj.res_norm", tensor(p + "residual_norm.weight"), False)
add("text.proj.res_up.weight", tensor(p + "residual_up.weight"), True)
add("text.proj.res_up.bias", tensor(p + "residual_up.bias"), False)
add("text.proj.res_down.weight", tensor(p + "residual_down.weight"), True)
add("text.proj.res_down.bias", tensor(p + "residual_down.bias"), False)
add("text.norm", tensor("text_norm.weight"), False)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path)
