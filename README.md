# qwen3-tts-ggml

Qwen3-TTS 12Hz CustomVoice in C++ on [ggml](https://github.com/ggml-org/ggml), for the named
speakers (`ono_anna`, `ryan`, ...), with a streaming decoder that produces the same samples as
decoding the whole utterance at once. Backends: Metal (macOS arm64), Vulkan and CUDA (Windows x64).

It is written for [ASIST](https://github.com/nyosegawa/asist) and implements only what ASIST uses:

- the talker (Qwen3 decoder, 28 layers for 1.7B) that predicts the first codebook of each frame,
- the code predictor (5 layers) that predicts the other 15 codebooks,
- the 12Hz codec decoder (RVQ dequantization, 8-layer sliding-window transformer, ConvNeXt
  upsampling and the SnakeBeta decoder), run frame by frame with its causal state carried over,
- the Qwen2 BPE tokenizer and the CustomVoice prompt.

Voice cloning, VoiceDesign, the codec encoder and the speaker encoder are out of scope.

The reference is the official Python implementation, [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).
The weights are converted from the official checkpoints (Apache-2.0).

## Status

Under construction.

## Plan

1. Reference: dump the intermediate tensors of the official implementation (prompt embeddings,
   talker logits, code predictor logits, codec decoder stages) for fixed inputs and greedy decoding.
2. Converter: official safetensors to GGUF (F16, Q8_0).
3. Codec decoder, offline then streaming, checked against the reference and against itself
   (streaming output equals offline output).
4. Talker and code predictor with a KV cache, checked against the reference codes under greedy
   decoding.
5. Worker: the JSON-lines protocol of ASIST's `qwen_tts_worker.py`, sampling, cancellation.
6. CI builds for Metal, Vulkan and CUDA, and measurements against MLX on a Mac and against the
   current setup on an RTX 2080.

## License

MIT. See [LICENSE](LICENSE).
