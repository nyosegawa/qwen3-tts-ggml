# The worker names languages with BCP 47 tags

Decided 2026-10-01.

## Context

Irodori-TTS came with its languages as BCP 47 tags (`ja`), while Qwen3-TTS listed the names its checkpoint
uses for them (`japanese`, and `beijing_dialect` and `sichuan_dialect` beside the languages), and a request
had to name a language the way the family listed it. A caller had to know which family it was talking to,
and to read the two dialects, which only two speakers speak, as languages. ASIST had not been released, so
the protocol could still change without a migration.

## Decision

Every family lists its languages in `ready` as BCP 47 tags, and a request's `language` is one of them, a
region or script of one (`ja-JP`, `zh-Hant`), or `auto`; anything else is an error. The tags live in the
GGUF: `general.languages` and `speech.languages` list them, and Qwen3-TTS's `talker.language_tags` gives the
tag of each of the checkpoint's language names, which `reference/qwen3-tts/convert.py` takes from one table.
Qwen3-TTS's dialects get no tag; as in the official implementation, `dylan` and `eric` speak theirs when the
language is `zh` or left to the model. `qwen3-tts` takes a tag as well.

## Consequences

Qwen3-TTS GGUFs converted before this lack `talker.language_tags` and are refused with the reader's missing
key error; the published files were converted again. A request that names a language by the checkpoint's
name (`japanese`) is now refused.
