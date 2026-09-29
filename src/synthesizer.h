#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "codec.h"
#include "prompt.h"
#include "sampler.h"
#include "talker.h"
#include "tokenizer.h"

struct SynthesisRequest {
    std::string text;
    std::string speaker;
    std::string language = "auto";
    SamplingParams talker{false, 0.9f, 50, 1.0f, 1.05f};
    SamplingParams code_predictor{false, 0.9f, 50, 1.0f, 1.0f};
    int max_frames = 2048;
    uint64_t seed = 0;
};

/** Called with each piece of 24 kHz audio as it is decoded; returning false stops the synthesis. */
using AudioSink = std::function<bool(const float * samples, size_t n)>;

/** Text in, audio out: the tokenizer, the talker, the code predictor and the codec decoder together. */
class Synthesizer {
public:
    Synthesizer(const std::string & talker_path, const std::string & codec_path, ggml_backend_t backend, int n_ctx);

    /**
     * Speaks `r.text`, decoding the first frame on its own so that audio starts as early as possible and
     * later frames `frames_per_piece` at a time. Returns the number of frames generated.
     */
    int synthesize(const SynthesisRequest & r, const AudioSink & sink, int frames_per_piece = 4);

    int sample_rate() const { return codec_.sample_rate(); }
    const PromptIds & ids() const { return ids_; }

private:
    Talker talker_;
    CodecDecoder codec_;
    Tokenizer tokenizer_;
    PromptIds ids_;
};
