# speech.cpp

The speech models of [ASIST](https://github.com/nyosegawa/asist) in C++ on [ggml](https://github.com/ggml-org/ggml),
run as one worker process that ASIST starts. It targets Metal, Vulkan and CUDA; it is checked on Metal,
on Vulkan (NVIDIA) and on the CPU. It implements only what ASIST uses from each model, and checks every
stage of a port against the official implementation.

| Family | Model | Task | Converted weights |
|---|---|---|---|
| Qwen3-TTS | Qwen3-TTS 12Hz 0.6B and 1.7B CustomVoice | speech synthesis with the named speakers, streamed frame by frame | [sakasegawa/qwen3-tts-ggml](https://huggingface.co/sakasegawa/qwen3-tts-ggml) |
| Irodori-TTS | Irodori-TTS v4.1-Small-MF and v4.1-Small | Japanese speech synthesis in the voice of a reference recording, a sentence at a time, streamed as the codec decodes it | convert them yourself (below) |

## Binaries

[Releases](https://github.com/nyosegawa/speech.cpp/releases) carry the tools built for macOS arm64
(Metal) and Windows x64 (Vulkan), with their SHA-256 sums. The Vulkan build needs no particular driver
version; on the first run the GPU driver compiles its shaders, which takes seconds and is cached by the
driver until it is updated.

## Build

```sh
git clone --recurse-submodules https://github.com/nyosegawa/speech.cpp.git
cd speech.cpp
cmake -B build                  # Metal on macOS, the CPU elsewhere
cmake --build build --config Release -j
```

For Vulkan or CUDA, configure with `-DGGML_VULKAN=ON` (the Vulkan SDK is needed to build) or
`-DGGML_CUDA=ON` instead.

## Layout

- `families/<family>/` runs one architecture of model, whichever weights it is given.
- `tools/` holds the worker, a command-line tool per family, and the checks that compare each stage with
  the official implementation.
- `reference/<model>/` pins the official implementation in a uv environment, converts its weights to GGUF
  and dumps the tensors the checks compare with.

## The worker

`speech-worker` is the process ASIST starts. It reads one JSON request per line on stdin and answers on
stdout, each line prefixed with `ASIST_JSON:`, and runs the family that `general.architecture` of the model
GGUF names.

```sh
speech-worker qwen3-tts-0.6b-customvoice-q8_0.gguf qwen3-tts-codec-12hz-f16.gguf
speech-worker irodori-tts-v4.1-small-mf-f16.gguf semantic-dacvae-japanese-32dim-f32.gguf \
    --voice bright=bright-young-woman-10s.voice.gguf --voice calm=calm-reference.wav
```

| Option | For | Meaning |
|---|---|---|
| `--device NAME`, `gpu`, `cpu` | both | the device as `--devices` names it (`MTL0`, `Vulkan1`), the first GPU (the default) or the CPU |
| `--seed n` | both | the seed of the first request; each later request takes the next one. Without it the seed is random |
| `--ctx n` | Qwen3-TTS | the talker's context in positions (2048, about 160 s of speech) |
| `--voice NAME=FILE` | Irodori-TTS | a voice, repeated for more: a reference WAVE file or a voice file (below). At least one is needed |
| `--steps n` | Irodori-TTS | the sampler's steps: 4 for v4.1-Small-MF and 40 for v4.1-Small unless given |

`speech-worker --devices` prints the devices it can run on, with their memory, and exits.

The messages, one JSON object per line:

| Direction | Message |
|---|---|
| out | `{"type":"ready","model":"Irodori-TTS-v4.1-Small-MF","architecture":"irodori-tts","sampleRate":48000,"streaming":"sentence","voices":["bright","calm"],"languages":["ja"],"languageSelectable":false,"steps":4,"backend":"MTL0"}` |
| in | `{"id":"1","text":"明日の東京は晴れです。","voice":"bright"}`, with `"language"` and `"speed"` optional |
| out | `{"type":"chunk","id":"1","seq":0,"pcm":"<base64 of 16-bit little-endian mono PCM at sampleRate>"}`, one or more |
| out | `{"type":"end","id":"1","samples":278400}` |
| out | `{"type":"error","id":"1","error":"..."}` when a request cannot be spoken |
| out | `{"type":"fatal","error":"..."}` when the worker cannot start |
| in | `{"type":"cancel","id":"1"}`: the request stops between two chunks and sends no `end`; a request cancelled before it starts is dropped |

Requests are served one at a time in arrival order. `speed` is accepted and has no effect.

- **Qwen3-TTS** streams frame by frame (`"streaming":"frame"`, 24 kHz). Its voices are the model's speakers,
  and `languages` lists the names a request's `language` takes (`"auto"` when left out).
- **Irodori-TTS** makes a sentence at once and streams it as the codec decodes it (`"streaming":"sentence"`,
  48 kHz), so a request should be one sentence; a text longer than the model's 256 tokens is refused. Its
  voices are those given with `--voice`. The model is not told a language: `languages` lists the BCP 47
  tags it speaks, and a request's `language`, when given, must be one of them or a region of one (`ja`,
  `ja-JP`).

### Irodori-TTS voices

Irodori-TTS has no voices of its own; it speaks in the voice of a reference. A voice is either:

- a reference WAVE file: 48 kHz (other rates are refused), at most 120 s, 16-, 24- or 32-bit PCM or 32-bit
  float, the channels averaged. The worker normalizes its loudness and encodes it with the codec when it
  starts, as the official runtime does for every request.
- a voice file, which `irodori-tts --make-voice` writes from a reference WAVE file: the reference's codec
  latent in a GGUF that names the codec it was made with (a voice file of another codec is refused).

```sh
irodori-tts --make-voice irodori-tts-v4.1-small-mf-f16.gguf semantic-dacvae-japanese-32dim-f32.gguf \
    bright-young-woman-10s.wav bright-young-woman-10s.voice.gguf --device cpu
```

For the 10.7 s reference bright-young-woman-10s.wav, the voice file is 35 KB against the WAVE file's 1 MB,
and loads in 0.016 s on Metal against 0.72 s to encode the WAVE file (5.05 s on the CPU) on an Apple M5.
Made on the CPU (`--device cpu`), its latent is the official encoder's to 99 dB SNR; on Metal it is 40 dB
(see Accuracy below).

## Qwen3-TTS

It decodes audio frame by frame with the same samples as decoding the whole utterance, so streaming does
not add artifacts at the frame boundaries. Implemented:

- the talker (a Qwen3 decoder) that predicts the first codebook of each frame,
- the code predictor that predicts the other 15 codebooks,
- the 12Hz codec decoder (RVQ dequantization, sliding-window transformer, ConvNeXt upsampling and the
  SnakeBeta decoder), with each stage's causal state carried from one call to the next,
- the Qwen2 byte-level BPE tokenizer and the CustomVoice prompt,
- sampling as in transformers' `generate()` (temperature, top-k, top-p, repetition penalty).

Voice cloning, VoiceDesign, the codec encoder and the speaker encoder are out of scope.

### Models

A synthesis needs one talker (`qwen3-tts-0.6b-customvoice-q8_0.gguf` or
`qwen3-tts-1.7b-customvoice-q8_0.gguf`) and the codec (`qwen3-tts-codec-12hz-f16.gguf`) from
[sakasegawa/qwen3-tts-ggml](https://huggingface.co/sakasegawa/qwen3-tts-ggml). To convert them yourself
from the official checkpoints:

```sh
cd reference/qwen3-tts
uv run python convert.py <Qwen3-TTS-12Hz-1.7B-CustomVoice dir> ../../models/gguf --type q8_0 --codec-type f16
```

### Use

```sh
build/qwen3-tts <talker.gguf> <codec.gguf> ono_anna japanese "明日の東京は晴れです。" out.wav
```

`speech-worker <talker.gguf> <codec.gguf>` runs it behind the worker protocol (above).

### Accuracy

`reference/qwen3-tts/dump.py` runs the official implementation with greedy decoding and saves the tensors
of every stage; the check tools compare against them.

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

### Speed

Q8_0 weights, Japanese sentences, after the shaders are compiled:

| Model | Device | First audio | Real-time factor | VRAM |
|---|---|---|---|---|
| 0.6B | Apple M5, Metal | 0.05 s | 0.35 | |
| 1.7B | Apple M5, Metal | 0.08 s | 0.49 | |
| 0.6B | RTX 2080, Vulkan | 0.07 s | 0.31 | 1.6 GB |
| 1.7B | RTX 2080, Vulkan | 0.08 s | 0.36 | 2.7 GB |

## Irodori-TTS

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) by Aratako: a DiT that makes the 32-dimensional latent of
a 48 kHz codec (Semantic-DACVAE-Japanese-32dim) for a whole sentence, its length set beforehand by a
duration predictor. Implemented, for v4.1-Small-MF (4 MeanFlow steps) and v4.1-Small (Euler steps with the
runtime's guidance, text 3.0 and speaker 5.0 while t ≥ 0.5):

- the official text normalization, with NFKC from Unicode 13.0 as the official runtime's Python has it,
- the SentencePiece Unigram tokenizer with byte fallback, and ModernBERT-ja with its projector,
- the reference's loudness normalization and the codec encoder, in windows of 100 frames,
- the speaker encoder, the duration predictor, the DiT and both samplers,
- the tail cut where the latent goes flat, and the codec decoder, a first window of 12 frames (0.48 s) and
  then 48 at a time, each window giving the samples of decoding the whole latent at once.

Not implemented: captions (VoiceDesign), speaker-inversion embeddings, SilentCipher's watermark,
`duration_scale`, and resampling a reference that is not at 48 kHz. The noise comes from the worker's own
generator, so a seed gives other audio than the same seed in the official runtime.

### Models

The converted weights are not published yet. To convert them from the pinned official checkpoints:

```sh
cd reference/irodori-tts
uv run python convert.py mf ../../models --type f16       # irodori-tts-v4.1-small-mf-f16.gguf, 1.5 GB
uv run python convert.py rf ../../models --type f16       # irodori-tts-v4.1-small-f16.gguf, 1.5 GB
uv run python convert_codec.py ../../models --type f32    # semantic-dacvae-japanese-32dim-f32.gguf, 371 MB
```

`--type` also takes `f32` and `q8_0` (0.8 GB). Qwen3-ASR 1.7B transcribed the 20 sentences of the speed
table below with 2.99% CER in F32 and in F16, and 3.81% in Q8_0, which garbled one phrase.

### Use

```sh
build/irodori-tts irodori-tts-v4.1-small-mf-f16.gguf semantic-dacvae-japanese-32dim-f32.gguf \
    bright-young-woman-10s.voice.gguf "明日の東京は晴れです。" out.wav [--device NAME] [--seed n] [--steps n]
```

### Accuracy

`reference/irodori-tts/dump.py` runs the official implementation on the CPU in float32 with fixed noise and
saves every stage; the check tools compare each stage, given the dump's own inputs, with it. Apple M5:

| Check | CPU, F32 | Metal, F32 |
|---|---|---|
| Normalization and tokenizer, 51 texts (`irodori-text-check`) | all equal | all equal |
| Text condition (`irodori-text-check`) | 118 to 123 dB SNR | 57 to 123 dB |
| Reference latent (`irodori-codec-check`) | 99 dB | 40 dB |
| Speaker condition (`irodori-condition-check`) | 111 dB | 47 dB |
| Predicted length | the official frames on every dump | the same |
| DiT steps, MF and RF (`irodori-dit-check`) | 95 dB or more | 48 dB or more |
| Sampled latent, MF / RF 40 steps | 86 to 122 dB / 109 to 111 dB | 33 to 61 dB / 50 to 54 dB |
| Decoded audio (`irodori-codec-check`) | 119 dB | 47 dB |
| Decoding in windows against at once | equal | equal |
| Whole synthesis from the dump's noise (`irodori-synthesis-check`) | 75 to 110 dB, the same length | 22 to 41 dB, the same length |

Metal's matrix kernel rounds both its inputs to half precision (`kernel_mul_mm_f32_f32` keeps its tiles as
`half`), which is the gap between the two columns; MeanFlow's four large steps carry it into the latent, so
on Metal the audio is the same speech rather than the same waveform. The codec on Metal is checked against
the CPU as well: its error lies 47 dB below the voice, and the quietest tenth of the 20 ms frames stays as
quiet as on the CPU (-77 against -76 dBFS). audio.cpp v0.8.2's Irodori-TTS adds a distorted copy of the voice
14 dB below it on Metal and raises the quiet parts to -60 dBFS.

### Speed

The 20 sentences of speech-bench's prompts/speak-ja-JP.json through `speech-worker` in the voice file above,
one request at a time, after the worker is ready:

| Model | Device | Median first audio | p90 first audio | Real-time factor | Memory |
|---|---|---|---|---|---|
| v4.1-Small-MF F16, 4 steps | Apple M5, Metal | 0.19 s | 0.35 s | 0.15 | 2.2 GB |
| v4.1-Small-MF Q8_0, 4 steps | Apple M5, Metal | 0.19 s | 0.34 s | 0.15 | 1.5 GB |
| v4.1-Small F16, 16 steps | Apple M5, Metal | 0.85 s | 2.25 s | 0.28 | 2.2 GB |
| v4.1-Small F16, 40 steps | Apple M5, Metal | 2.01 s | 5.52 s | 0.49 | 2.2 GB |

Memory is the worker's peak memory footprint with the F32 codec. The first audio comes after the text, the
whole sampler and the codec's first window, so it grows with the sentence.

## License

MIT, see [LICENSE](LICENSE). The model weights are their authors': Qwen3-TTS is the Qwen team's, under the
Apache License 2.0. Irodori-TTS v4.1-Small and v4.1-Small-MF are Aratako's, under the MIT License with the
ethical restrictions of their model cards (no voice cloning without consent, no deepfakes or
misinformation). Semantic-DACVAE-Japanese-32dim is MIT on its card; it derives from
facebook/dacvae-watermarked, whose card says both Apache-2.0 and the SAM License.
