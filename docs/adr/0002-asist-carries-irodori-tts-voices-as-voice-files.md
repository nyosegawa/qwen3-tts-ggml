# ASIST carries Irodori-TTS voices as voice files

Decided 2026-10-01.

## Context

Irodori-TTS speaks in the voice of a reference recording, and the worker takes a voice either as the
reference WAVE file or as a voice file: the reference's codec latent in a GGUF that names the codec. ASIST
ships a few fixed voices, made in speech-bench from takes of a described voice (about 10 s each), and could
carry either form.

For bright-young-woman-10s.wav (10.74 s, 48 kHz, 16-bit):

| | WAVE file | Voice file |
|---|---|---|
| Size | 1,031,084 bytes | 34,720 bytes |
| Time to load, Apple M5, Metal | 0.72 s | 0.016 s |
| Time to load, Apple M5, CPU | 5.05 s | 0.028 s |
| Time to load, RTX 2080, Vulkan | 0.43 s | 0.025 s |
| Latent against the official encoder | 99 dB on the CPU, 40 dB on Metal, 33 dB on Vulkan | as made |

The worker loads each voice once when it starts, so a WAVE file delays `ready` by its encoding, for every
voice and on every start. Encoding on Metal rounds the matrix products' inputs to half precision, so the
same WAVE file gives a slightly different latent on each kind of device.

## Decision

ASIST carries voice files, made once on the CPU with `irodori-tts --make-voice ... --device cpu` from the
reference WAVE files that speech-bench makes. The WAVE path stays for making voices and for speech-bench.

The voice file holds the codec latent rather than the speaker encoder's output: the latent depends only on
the codec, so a voice works with both v4.1-Small-MF and v4.1-Small, and the speaker encoder takes
milliseconds.

## Consequences

A voice file names the codec's source and revision and is refused with another codec; a new codec means
making the voices again from their WAVE files, which speech-bench keeps.
