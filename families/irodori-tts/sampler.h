#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "dit.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace irodori {

/** One step of a sampler as it ran: its time, the velocity it took, and the latent after it. */
struct SamplerStep {
    int index;
    float t;
    const std::vector<float> & velocity;
    const std::vector<float> & x;
};

/**
 * The official samplers. A MeanFlow model takes `steps` equal steps from time 1 to 0, each told its
 * interval. An RF model takes Euler steps from 0.999 to 0 and, while t is in the model's guidance range,
 * adds the guidance against the branches without the text and without the speaker.
 */
class Sampler {
public:
    Sampler(const Dit & dit, const ModelFile & m, ggml_backend_t backend);
    ~Sampler();
    Sampler(const Sampler &) = delete;
    Sampler & operator=(const Sampler &) = delete;

    /** The times the sampler visits, steps + 1 of them, computed as torch.linspace() computes them. */
    std::vector<float> schedule(int steps) const;

    /** The latent, row-major [frames, latent_dim], reached from `noise` in `steps` steps. */
    std::vector<float> sample(const Conditions & c, std::vector<float> noise, int frames, int steps,
                              const std::function<void(const SamplerStep &)> & observe = {});

    /** The velocity of one step; for RF with guidance, `branches` receives each branch's output. */
    std::vector<float> velocity(const Conditions & c, const std::vector<float> & x, int frames, float t, float t_next,
                                std::vector<float> * branches = nullptr);

    int default_steps() const { return default_steps_; }

private:
    const Dit & dit_;
    ggml_backend_t backend_;
    ggml_gallocr_t allocr_ = nullptr;
    int default_steps_;
    float cfg_text_ = 0, cfg_speaker_ = 0, cfg_min_t_ = 0, cfg_max_t_ = 0;
};

/** Standard normal samples from a seed, the same on every platform (a Mersenne Twister and Box-Muller). */
std::vector<float> gaussian_noise(uint64_t seed, size_t n);

}  // namespace irodori
