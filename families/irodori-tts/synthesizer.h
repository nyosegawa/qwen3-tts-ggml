#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "codec.h"
#include "dit.h"
#include "duration.h"
#include "sampler.h"
#include "speaker-encoder.h"
#include "text-encoder.h"
#include "tokenizer.h"

namespace irodori {

/** A voice: the latent of its reference, and the speaker condition the model makes of it. */
struct Voice {
    std::vector<float> latent;
    int frames = 0;
    std::vector<float> speaker;
    int speaker_tokens = 0;
};

struct Request {
    std::string text;
    uint64_t seed = 0;
    /** The sampler's steps; 0 takes the model's default (4 for MeanFlow, 40 for RF). */
    int steps = 0;
    /** The sampler's starting point, row-major [frames, latent_dim], instead of noise from the seed. */
    std::vector<float> noise;
};

/** Where the time of one synthesis went, in seconds, and how long the speech is. */
struct Stats {
    double text = 0, sampling = 0, first_audio = 0, codec = 0;
    int tokens = 0, frames = 0;
    size_t samples = 0;
};

/**
 * Text in, 48 kHz audio out, as the official runtime makes it: the normalized text's tokens, the text
 * condition and the predicted length, the sampler from seeded noise, the tail cut where the latent goes
 * flat, and the codec decoding window by window so that the first audio comes before the rest is decoded.
 */
class Synthesizer {
public:
    Synthesizer(const std::string & model_path, const std::string & codec_path, ggml_backend_t backend);
    ~Synthesizer();

    /** A voice from a reference WAVE file or a voice file that save_voice() wrote. */
    Voice load_voice(const std::string & path);
    /** A voice from the latent of its reference, row-major [frames, latent_dim]. */
    Voice voice_from_latent(std::vector<float> latent);
    /** Writes a voice's latent to a GGUF file that load_voice() reads, bound to this codec. */
    void save_voice(const Voice & voice, const std::string & path) const;

    /** Speaks `r.text` in `voice`, passing the audio to `sink` window by window. Returns the samples. */
    size_t synthesize(const Request & r, const Voice & voice, const AudioSink & sink, Stats * stats = nullptr);

    int sample_rate() const { return codec_.sample_rate(); }
    const ModelFile & model() const { return *model_; }
    const Codec & codec() const { return codec_; }

    /** The decoder's first window, in frames; a short one brings the first audio early. */
    int first_window = 12;
    /** The decoder's later windows, in frames. */
    int window = 48;

private:
    ggml_backend_t backend_;
    std::unique_ptr<ModelFile> model_;
    Codec codec_;
    Tokenizer tokenizer_;
    TextEncoder text_;
    SpeakerEncoder speaker_;
    DurationPredictor duration_;
    Dit dit_;
    Sampler sampler_;
    ggml_gallocr_t allocr_ = nullptr;
    double max_reference_seconds_ = 0;
};

/**
 * The official find_flattening_point(): the first frame from which 20 frames (zeros past the end) have a
 * standard deviation under 0.05 and a mean within 0.1 of zero, or the number of frames when none does.
 */
int flattening_point(const std::vector<float> & latent, int frames, int latent_dim);

}  // namespace irodori
