#pragma once

#include "graph.h"
#include "model-file.h"

/**
 * The duration predictor of v4.1: a stack of SwiGLU blocks over the text condition's tokens, each
 * modulated by the speaker's summary and, without a caption, by the learned null caption. Each token
 * contributes softplus(output) frames.
 */
class DurationPredictor {
public:
    /** `sample_rate` and `hop` are the codec's, which turn the bounds in seconds into frames. */
    DurationPredictor(const ModelFile & m, int sample_rate, int hop);

    /** The predicted frames summed over the tokens, [1], for a text condition [text_dim, n] and a speaker summary [speaker_dim]. */
    ggml_tensor * build(Graph & g, ggml_tensor * text_state, ggml_tensor * speaker_summary) const;

    /**
     * The latent frames the runtime synthesizes for a predicted sum, through its log1p and expm1, its
     * rounding to the nearest even, and its bounds of 0.5 s and 30 s.
     */
    int frames(float predicted_sum) const;

private:
    const ModelFile & m_;
    int layers_, min_frames_, max_frames_;
    float eps_;
};
