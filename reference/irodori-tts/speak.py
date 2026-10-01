"""Speaks a list of sentences with the official Irodori-TTS runtime and a reference voice, for listening.

usage: uv run python speak.py <model: mf|rf> <prompts.json> <reference.wav> <out dir> [--device mps|cpu] [--seed n]

Writes one 16-bit WAV per sentence, named by its id, and manifest.json with the model, the settings and
the seconds each synthesis took. A first synthesis before the timed ones warms the device up.

SilentCipher's watermark is left out, as the port leaves it out: the runtime applies it whenever the
package is installed, and Irodori-TTS lists the package as a dependency.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time

import soundfile as sf
import torch

from irodori_tts import inference_runtime as ir
from pins import CODE, CODEC, MODELS, snapshot

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("prompts")
parser.add_argument("reference")
parser.add_argument("out_dir")
parser.add_argument("--device", default="mps")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

ir.SilentCipherWatermarker._load_backend = staticmethod(lambda **_: None)

model = MODELS[args.model]
checkpoint = os.path.join(snapshot(model), "model.safetensors")
runtime = ir.InferenceRuntime.from_key(
    ir.RuntimeKey(
        checkpoint=checkpoint,
        model_device=args.device,
        codec_repo=os.path.join(snapshot(CODEC), "weights.pth"),
        codec_device=args.device,
    )
)
assert not runtime.watermarker.ready
steps = 4 if runtime.model_cfg.flow_parameterization == "meanflow" else 40

prompts = json.load(open(args.prompts, encoding="utf-8"))
reference_sha256 = hashlib.sha256(open(args.reference, "rb").read()).hexdigest()
os.makedirs(args.out_dir, exist_ok=True)


def speak(text):
    request = ir.SamplingRequest(text=text, ref_wav=args.reference, seed=args.seed, num_steps=steps)
    t0 = time.perf_counter()
    result = runtime.synthesize(request)
    return result, time.perf_counter() - t0


speak(prompts["prompts"][0]["text"])
sentences = []
for p in prompts["prompts"]:
    result, seconds = speak(p["text"])
    audio = result.audio.squeeze(0).numpy()
    wav = f"{p['id']}.wav"
    sf.write(os.path.join(args.out_dir, wav), audio, result.sample_rate, subtype="PCM_16")
    sentences.append({
        "id": p["id"],
        "text": p["text"],
        "wav": wav,
        "seed": args.seed,
        "seconds": round(seconds, 3),
        "audioSeconds": round(len(audio) / result.sample_rate, 3),
        "stages": {name: round(s, 4) for name, s in result.stage_timings},
    })
    print(f"{p['id']}: {seconds:.2f} s for {len(audio) / result.sample_rate:.2f} s of audio")

chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
manifest = {
    "engine": "Irodori-TTS, official PyTorch runtime",
    "code": CODE,
    "model": model,
    "codec": CODEC,
    "sampler": runtime.model_cfg.flow_parameterization,
    "steps": steps,
    "cfg": None if steps == 4 else {"text": 3.0, "speaker": 5.0, "guidance": "independent", "minT": 0.5, "maxT": 1.0},
    "reference": {"file": os.path.basename(args.reference), "sha256": reference_sha256},
    "referenceNormalizeDb": -16.0,
    "watermark": False,
    "sampleRate": 48000,
    "device": args.device,
    "precision": "fp32",
    "torch": torch.__version__,
    "machine": f"{chip}, macOS {platform.mac_ver()[0]}",
    "prompts": os.path.basename(args.prompts),
    "seconds": "wall-clock time of one runtime.synthesize call, which encodes the reference each time",
    "sentences": sentences,
}
json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("->", args.out_dir)
