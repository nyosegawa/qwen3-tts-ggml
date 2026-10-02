# AGENTS.md

## Project

speech.cpp runs the speech models ASIST uses in C++ on [ggml](https://github.com/ggml-org/ggml), on macOS
arm64 with Metal and Windows x64 with Vulkan, as one worker process that ASIST starts. It implements only
what ASIST uses from each model. Each model's official implementation is the reference: every stage of a
port is checked against tensors dumped from it. README.md is the documentation for users; `docs/adr/` keeps
the decisions. Read the relevant implementation and its check before changing behavior.

## Architecture

- `families/<family>/` holds the code of one architecture, whichever weights it is given: `qwen3-tts/`
  runs Qwen3-TTS 0.6B and 1.7B. A family reads its GGUF files and turns text into audio; it knows nothing
  of the worker protocol or the command line.
- `common/` holds what two families use in the same role. Code moves there when a second family needs
  it, not before, and never as a framework for families that do not exist yet.
- `tools/` holds the executables: the worker, a command-line tool per family that speaks a text into a
  WAV file, and one check per ported stage (`*-check.cpp`) that compares the stage with the reference
  dumps.
- The worker chooses the family from `general.architecture` of the model's GGUF. Its protocol (the
  `ASIST_JSON:` lines) is ASIST's contract.
- `reference/<model>/` holds, per model, a uv environment that pins the official code, PyTorch and the rest,
  the conversion of the official weights to GGUF, and the scripts that run the official implementation to
  dump reference tensors. Dumps go to `reference/<model>/out/`.

Keep these boundaries explicit: code does not reach past its module for an operation that belongs to
another one.

## Code

- C++17 with ggml as a git submodule and no other dependency. Each tool is one executable with ggml linked
  in, so that a release is one file per tool.
- The GGUF layout is this repository's own: `reference/<model>/convert.py` defines the tensor names and
  metadata keys, and the C++ reads exactly what it writes. A change of layout changes both in the same
  commit. The model's constants live in the GGUF metadata, not in the C++.
- Do not add fallback behavior; fail loudly rather than degrade silently. A GGUF without a key or tensor,
  a text longer than the model takes, a WAVE format that is not understood and a device that does not
  start all throw with a message; the worker reports them as `error` or `fatal`. Nothing is truncated or
  moved to another device behind the caller's back.
- Fix a defect where its cause is, in a form in which it cannot happen, rather than with a guard for the
  one case that showed it; the code after the fix reads better than before. When a fix is much larger
  than the defect, or needs a choice only the user can make, stop and ask instead of patching.
- Persist each fact in one authoritative place and derive the rest from it.
- Every download is pinned: a Hugging Face repository by revision, a git dependency by commit, Python
  packages by `uv.lock`, the Vulkan SDK in CI by version.
- Extract code only when it creates a coherent responsibility, a reusable boundary or an independently
  checkable unit. Introduce a shared abstraction only after two current implementations show the same
  responsibility with meaningful variation.
- A file approaching 500 lines calls for a review of its responsibilities; split it when a coherent one
  can be extracted, not to meet a line count.
- On Windows, a tool reads its command line as UTF-8, sets stdin and stdout to binary, and defines
  `NOMINMAX` before `windows.h`.
- A path is a UTF-8 string from the command line to the file. A C stream opens it with `ggml_fopen()` and a
  C++ stream through `std::filesystem::u8path()`: `fopen()` and a stream opened on a `std::string` read the
  path in the ANSI code page on Windows, so a path with any character outside ASCII is not found.
- Model weights, reference dumps and audio are never committed; `.gitignore` covers `models/`,
  `reference/*/out/`, `*.gguf` and `*.wav`.

## Comments

- Write comments in English, in full sentences and the present tense. Other languages appear only as
  data, quoted verbatim.
- Say only what the code cannot: the layout of a tensor, a quirk of ggml or of the official implementation
  and the failure it causes, a measured value with its condition and date, a constraint, or why the
  simpler approach was rejected. Do not restate names, narrate steps, or mention history, TODOs, docs,
  tickets or conversations.
- Doc comments (`/** */`) on the declarations in headers and on non-obvious declarations in a file,
  without parameter or return tags. A tool starts with a comment that says what it does and its usage.
- Inside a function, a comment goes on its own line above the code. No trailing comments and no banner
  comments.

## Tests

- The checks are the tests. A check runs one stage on the reference dump's inputs, compares its output
  with the dump, prints the error (SNR, largest difference, argmax agreement), and exits with a failure
  when the error is beyond what the stage's arithmetic explains.
- A failing check points to a defect. A check does not fail on a deliberate change of wording, a name or a
  configured value, and does not only assert that something exists.
- Add or update the check of a stage whose behavior changes, and run it on the CPU in F32 and on the GPU
  before committing. A change to code that runs on the GPU is also run on Vulkan, on the Windows machine.
- A port of a model checks each stage in the order data flows, with the stage's own inputs from the dump,
  so that the first stage that departs is the one reported.
- Checks need weights and dumps that are not in the repository, so CI builds them but does not run them.

## Text

- The output of the tools, README.md and AGENTS.md are English. Japanese and other languages appear only
  as data: texts to speak and model output.
- An error message names what failed and what to do.

## Decisions

`docs/adr/` keeps the decisions the code cannot show, one file each, and the file names say which
behavior each covers. Before changing a behavior, list the folder and read the records whose names cover
it; if the change contradicts one, say so to the user first. Before committing, ask whether the work
settled a choice or turned an approach down for good; if so, the record goes into the same commit.

## Workflow

- Build with `cmake -B build && cmake --build build -j`, and run the checks the change touches before
  committing code.
- Never commit on main. Every change reaches main through a pull request, one coherent unit each: a
  model's stage, a fix, a refactor or a documentation change.
- Commit messages and pull request titles are one English sentence in the imperative, without a prefix
  such as `feat:`; the body says what changed, why, and how it was checked.
- The user merges pull requests, with a squash, once CI passes. An agent merges only when told to.
- A tag `v*` builds the release in CI. Releases and tags are never deleted or moved: ASIST pins them by
  SHA-256.
- Converted GGUF files go to Hugging Face only with the user's approval, one repository per family
  (sakasegawa/qwen3-tts-ggml, sakasegawa/irodori-tts-ggml), with the licenses of what it holds. A changed
  file goes up under the same name, and its card's SHA-256 changes with it.
- When a change alters what a user does or sees (a tool's arguments, the worker protocol, the GGUF
  layout), update README.md in the same change.
