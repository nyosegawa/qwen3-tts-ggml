#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

int32_t sample(std::vector<float> logits, const SamplingParams & p, const std::vector<int32_t> & history,
               const std::vector<bool> & banned, std::mt19937_64 & rng) {
    const int n = (int) logits.size();
    if (p.repetition_penalty != 1.0f) {
        std::vector<bool> seen(n, false);
        for (int32_t t : history) {
            if (t < 0 || t >= n || seen[t]) continue;
            seen[t] = true;
            logits[t] = logits[t] < 0 ? logits[t] * p.repetition_penalty : logits[t] / p.repetition_penalty;
        }
    }
    for (int i = 0; i < n && i < (int) banned.size(); i++) {
        if (banned[i]) logits[i] = -INFINITY;
    }
    if (p.greedy) {
        return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    for (float & l : logits) l /= p.temperature;
    std::vector<int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    const int k = p.top_k > 0 ? std::min(p.top_k, n) : n;
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    order.resize(k);

    const float max_logit = logits[order[0]];
    if (!std::isfinite(max_logit)) throw std::runtime_error("every token is banned");
    std::vector<double> probs(k);
    double sum = 0;
    for (int i = 0; i < k; i++) sum += probs[i] = std::exp((double) logits[order[i]] - max_logit);
    for (double & q : probs) q /= sum;
    if (p.top_p < 1.0f) {
        // Keep the smallest set whose probability reaches top_p, always at least one token.
        double cumulative = 0;
        int keep = 0;
        while (keep < k) {
            cumulative += probs[keep++];
            if (cumulative >= p.top_p) break;
        }
        probs.resize(keep);
        order.resize(keep);
    }
    std::discrete_distribution<int> pick(probs.begin(), probs.end());
    return order[pick(rng)];
}
