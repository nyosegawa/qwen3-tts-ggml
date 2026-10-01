#pragma once

#include <cstdint>
#include <vector>

#include "graph.h"
#include "model-file.h"

/**
 * The text condition: ModernBERT-ja on the tokens, then the residual projector and the norm that map
 * its 768 channels to the DiT's text space.
 *
 * The official runtime pads the tokens to 256 and masks the padding out of every attention; a padded
 * position never reaches a real one, so this runs on the real tokens alone.
 */
class TextEncoder {
public:
    explicit TextEncoder(const ModelFile & m);

    /**
     * The text condition of `ids`, [dim, n]. When `layers` is given it receives ModernBERT's hidden
     * states: the embeddings, then the output of each layer, the last after the final norm.
     */
    ggml_tensor * build(Graph & g, const std::vector<int32_t> & ids, std::vector<ggml_tensor *> * layers = nullptr) const;

    int dim() const { return dim_; }
    int max_tokens() const { return max_tokens_; }

private:
    const ModelFile & m_;
    int hidden_, heads_, layers_, window_, dim_, max_tokens_;
    float eps_, norm_eps_, theta_global_, theta_local_;
    std::vector<int32_t> global_;
};
