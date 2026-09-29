"""Drives qwen3-tts-worker the way ASIST's main process does: waits for ready, sends two requests, cancels
the second after its first chunk, sends a third, and checks each answer. Writes the first answer to a WAV.

usage: python3 tools/worker_smoke.py <worker> <talker.gguf> <codec.gguf> <out.wav> [extra worker args...]
"""

import base64
import json
import subprocess
import sys
import time
import wave

worker, talker, codec, out_wav = sys.argv[1:5]
proc = subprocess.Popen([worker, talker, codec, *sys.argv[5:]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL)
t0 = time.perf_counter()


def read():
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("the worker exited")
        line = line.decode("utf-8").rstrip("\n")
        if line.endswith("\r"):
            raise SystemExit("a line ends with \\r")
        if line.startswith("ASIST_JSON:"):
            return json.loads(line[len("ASIST_JSON:"):])


def send(obj):
    proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    proc.stdin.flush()


ready = read()
assert ready["type"] == "ready", ready
print(f"ready in {time.perf_counter() - t0:.2f} s: rate {ready['sampleRate']}, {len(ready['voices'])} voices, "
      f"{len(ready['languages'])} languages, backend {ready.get('backend')}")

t1 = time.perf_counter()
send({"id": "a", "text": "明日の東京は晴れで、最高気温は二十四度の予報です。", "voice": "ono_anna", "language": "japanese", "speed": 1.0})
send({"id": "b", "text": "これは途中で止める長めの文です。止まったら終わりの知らせは来ません。", "voice": "ryan", "language": "japanese"})
pcm_a, first_a, cancelled_b, seq_b = bytearray(), None, False, []
while True:
    m = read()
    if m["type"] == "chunk" and m["id"] == "a":
        if first_a is None:
            first_a = time.perf_counter() - t1
        pcm_a += base64.b64decode(m["pcm"])
    elif m["type"] == "end" and m["id"] == "a":
        assert m["samples"] * 2 == len(pcm_a), (m, len(pcm_a))
        print(f"a: first chunk {first_a:.3f} s, {m['samples'] / ready['sampleRate']:.2f} s of audio")
    elif m["type"] == "chunk" and m["id"] == "b":
        seq_b.append(m["seq"])
        if not cancelled_b:
            send({"type": "cancel", "id": "b"})
            send({"id": "c", "text": "三つ目です。", "voice": "ono_anna", "language": "japanese"})
            cancelled_b = True
    elif m["type"] == "end" and m["id"] == "b":
        raise SystemExit("b ended although it was cancelled")
    elif m["type"] == "end" and m["id"] == "c":
        print(f"b: {len(seq_b)} chunk(s) before the cancel took effect, no end; c: {m['samples']} samples")
        break
    elif m["type"] in ("error", "fatal"):
        raise SystemExit(f"unexpected {m}")

proc.stdin.close()
proc.wait(timeout=30)
with wave.open(out_wav, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(ready["sampleRate"])
    w.writeframes(bytes(pcm_a))
print("ok, exit", proc.returncode)
