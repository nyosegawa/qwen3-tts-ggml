# Irodori-TTS runs both v4.1-Small-MF and v4.1-Small

Decided 2026-10-01.

## Context

The port aimed at v4.1-Small-MF, the MeanFlow distillation of v4.1-Small that speaks in 4 steps where
v4.1-Small takes 40 with guidance. Before porting, the official runtime spoke the 20 sentences of
speech-bench's prompts/speak-ja-JP.json in the voice of bright-young-woman-10s.wav, and Qwen3-ASR 1.7B
transcribed them:

| Official runtime | Character error rate |
|---|---|
| v4.1-Small, 40 steps, two seeds | 2.16% and 2.16% |
| v4.1-Small-MF, 4 steps, six seeds and noises | 2.99% to 5.80%, 4.8% on average |
| audio.cpp v0.8.2, v4.1-Small at 16 steps, same reference | 2.16% |

Every run had the same 13 errors that are not the model's (他 for ほか, katakana for Slack). MF added one to
three sentences a run with a word that came out wrong (Bluetooth as ウルブドゥス), more often in English words
and long sentences, though some of these may be the recognizer's. MPS and the CPU gave the same
transcriptions from the same noise, so the device was not the cause.

## Decision

Port both. The two share every stage but the sampler, so v4.1-Small adds its Euler steps and its guidance
(a batch of 3 while t >= 0.5) and a second GGUF. The worker takes either model and `--steps`.

## Consequences

Which one ASIST uses by default is left to the measurements: the median first audio is 0.19 s with MF and
0.85 s with v4.1-Small at 16 steps on an Apple M5, and 0.13 s and 0.49 s on an RTX 2080; speech-bench
measures the errors and the likeness to the reference of each.
