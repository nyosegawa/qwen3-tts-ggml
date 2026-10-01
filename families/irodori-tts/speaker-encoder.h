#pragma once

#include <vector>

#include "graph.h"
#include "model-file.h"

/**
 * The speaker condition: a pre-norm transformer over the reference latent in patches of four frames, its
 * output normed, with the mean of its positions prepended as a summary the duration predictor reads.
 */
class SpeakerEncoder {
public:
    explicit SpeakerEncoder(const ModelFile & m);

    /**
     * The speaker condition [dim, 1 + frames / patch] of a reference latent, row-major [frames, latent_dim];
     * the frames after the last whole patch are left out, as the official model leaves them. When
     * `encoded` is given it receives the transformer's output before the norm.
     */
    ggml_tensor * build(Graph & g, const std::vector<float> & latent, ggml_tensor ** encoded = nullptr) const;

    int dim() const { return dim_; }

private:
    const ModelFile & m_;
    int dim_, heads_, layers_, patch_, latent_dim_;
    float eps_, theta_;
};
