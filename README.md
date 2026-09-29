# qwen3-tts-ggml

Qwen3-TTS 12Hz CustomVoice in C++ on [ggml](https://github.com/ggml-org/ggml), for the named speakers
(`ono_anna`, `ryan`, ...). It decodes audio frame by frame with the same samples as decoding the whole
utterance, so streaming does not add artifacts at the frame boundaries. It targets Metal, Vulkan and
CUDA; so far it has been checked on Metal and on the CPU.

It is written for [ASIST](https://github.com/nyosegawa/asist) and implements only what ASIST uses:

- the talker (a Qwen3 decoder) that predicts the first codebook of each frame,
- the code predictor that predicts the other 15 codebooks,
- the 12Hz codec decoder (RVQ dequantization, sliding-window transformer, ConvNeXt upsampling and the
  SnakeBeta decoder), with each stage's causal state carried from one call to the next,
- the Qwen2 byte-level BPE tokenizer and the CustomVoice prompt,
- sampling as in transformers' `generate()` (temperature, top-k, top-p, repetition penalty).

Voice cloning, VoiceDesign, the codec encoder and the speaker encoder are out of scope.

## Models

Converted GGUF files are on Hugging Face: [sakasegawa/qwen3-tts-ggml](https://huggingface.co/sakasegawa/qwen3-tts-ggml).
A synthesis needs one talker (`qwen3-tts-0.6b-customvoice-q8_0.gguf` or
`qwen3-tts-1.7b-customvoice-q8_0.gguf`) and the codec (`qwen3-tts-codec-12hz-f16.gguf`).

To convert them yourself from the official checkpoints:

```sh
cd reference
uv run python convert.py <Qwen3-TTS-12Hz-1.7B-CustomVoice dir> ../models/gguf --type q8_0 --codec-type f16
```

## Build

```sh
git clone --recurse-submodules https://github.com/nyosegawa/qwen3-tts-ggml.git
cd qwen3-tts-ggml
cmake -B build                  # Metal on macOS, the CPU elsewhere
cmake --build build --config Release -j
```

For Vulkan or CUDA, configure with `-DGGML_VULKAN=ON` (the Vulkan SDK is needed to build) or
`-DGGML_CUDA=ON` instead.

## Use

```sh
build/qwen3-tts <talker.gguf> <codec.gguf> ono_anna japanese "明日の東京は晴れです。" out.wav
```

`qwen3-tts-worker <talker.gguf> <codec.gguf>` reads one JSON request per line on stdin and streams
base64 PCM chunks on stdout, with the protocol of ASIST's Qwen3-TTS worker (see the comment at the top
of `tools/qwen3-tts-worker.cpp`).

## Accuracy

`reference/dump.py` runs the official implementation with greedy decoding and saves the tensors of
every stage; the check tools compare against them.

| Check | Result |
|---|---|
| Codec decoder, whole utterance, CPU, F32 (`codec-check`) | 114 dB SNR against the official decoder |
| Codec decoder, one frame at a time against whole, CPU | 133 dB SNR |
| Codec decoder on Metal | error at -52 dB of the voice, at the level of 16-bit rounding in silence |
| Talker and code predictor, F32, teacher forcing (`talker-check`) | argmax matches on every frame; greedy decode gives the same 54 frames |
| Tokenizer (`tokenizer-check`) | matches the model's `tokenizer.json` on 19 texts |

The tokenizer follows the pre-tokenizer of the `tokenizer.json` that ships with the model. The official
package loads it through transformers 4.57.3 with `fix_mistral_regex=True`, which swaps in Mistral's
pattern; the two differ on Latin words in mixed case, contractions and `/`.

## Speed

On an Apple M5 with Metal and Q8_0 weights, for Japanese sentences:

| Model | First audio | Real-time factor |
|---|---|---|
| 0.6B | 0.05 s | 0.36 |
| 1.7B | 0.08 s | 0.49 |

## License

MIT, see [LICENSE](LICENSE). The model weights are the Qwen team's, under the Apache License 2.0.
