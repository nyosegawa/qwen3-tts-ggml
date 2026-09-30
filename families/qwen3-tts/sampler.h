#pragma once

#include <cstdint>
#include <random>
#include <vector>

/** Sampling settings of one stack, with the defaults of the official generate_config. */
struct SamplingParams {
    bool greedy = false;
    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
};

/**
 * Picks a token the way transformers' generate() does: the repetition penalty on the tokens in
 * `history`, the tokens in `banned` removed, then argmax, or temperature, top-k and top-p before a
 * draw.
 */
int32_t sample(std::vector<float> logits, const SamplingParams & p, const std::vector<int32_t> & history,
               const std::vector<bool> & banned, std::mt19937_64 & rng);
