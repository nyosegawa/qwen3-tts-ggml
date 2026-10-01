#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

/** Called with each piece of audio as it is decoded; returning false stops the decoding. */
using AudioSink = std::function<bool(const float * samples, size_t n)>;

/**
 * Semantic-DACVAE-Japanese-32dim, the codec of Irodori-TTS: 48 kHz audio and a 32-dimensional latent at
 * 25 frames a second. Every convolution is non-causal with symmetric padding.
 */
class Codec {
public:
    Codec(const std::string & path, ggml_backend_t backend);
    ~Codec();
    Codec(const Codec &) = delete;
    Codec & operator=(const Codec &) = delete;

    int sample_rate() const { return sample_rate_; }
    int hop() const { return hop_; }
    int latent_dim() const { return latent_dim_; }

    /**
     * The latent, row-major [frames, latent_dim], of mono audio at sample_rate() as the official runtime
     * encodes a reference after normalizing its loudness: padded at the end by reflection to whole frames,
     * then encoded in windows of `window` frames, each with enough frames around it that the frames it keeps
     * are those of encoding the whole at once.
     */
    std::vector<float> encode(const std::vector<float> & audio, int window = 100);

    /** The encoder's output [latent_dim, samples / hop] for `samples` that are a whole number of frames. */
    ggml_tensor * build_encoder(Graph & g, const std::vector<float> & samples) const;

    /**
     * The audio [1, frames * hop] of a latent, row-major [frames, latent_dim], of at least two frames.
     * `stages`, when given, receives the output of the input projection, of the first convolution and of
     * each upsampling block.
     */
    ggml_tensor * build_decoder(Graph & g, const std::vector<float> & latent, std::vector<ggml_tensor *> * stages = nullptr) const;

    /**
     * Decodes the first `samples` samples of a latent, row-major [frames, latent_dim], `first_window`
     * frames first and then `window` at a time, each with enough frames around it that its samples are
     * those of decoding the whole latent at once, and passes each window's samples to `sink`.
     */
    void decode(const std::vector<float> & latent, int64_t samples, int first_window, int window, const AudioSink & sink);

    /** The frames on each side of a window that the encoder's receptive field reaches. */
    static constexpr int kEncoderMargin = 8;
    /** The same for the decoder, whose receptive field reaches about 7.7 frames. */
    static constexpr int kDecoderMargin = 10;

private:
    ggml_backend_t backend_;
    std::unique_ptr<ModelFile> model_;
    ggml_gallocr_t allocr_ = nullptr;
    int sample_rate_ = 0, hop_ = 0, latent_dim_ = 0;
    std::vector<int32_t> encoder_rates_, decoder_rates_;
};
