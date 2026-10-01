#include "sampler.h"

#include <cmath>
#include <random>

namespace {

/**
 * torch.linspace() on the CPU in float32: the first half from the start, the second from the end. (Its
 * vectorized path can round the last bit of a few points differently; dit_t in the dumps shows the times
 * the official run used.)
 */
std::vector<float> linspace(float start, float end, int n) {
    std::vector<float> out(n);
    const float step = (end - start) / (float) (n - 1);
    for (int i = 0; i < n; i++) out[i] = i < n / 2 ? start + step * (float) i : end - step * (float) (n - i - 1);
    return out;
}

}  // namespace

Sampler::Sampler(const Dit & dit, const ModelFile & m, ggml_backend_t backend) : dit_(dit), backend_(backend) {
    default_steps_ = (int) m.u32("irodori.default_steps");
    if (!dit.meanflow()) {
        cfg_text_ = m.f32("irodori.cfg_text");
        cfg_speaker_ = m.f32("irodori.cfg_speaker");
        cfg_min_t_ = m.f32("irodori.cfg_min_t");
        cfg_max_t_ = m.f32("irodori.cfg_max_t");
    }
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Sampler::~Sampler() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

std::vector<float> Sampler::schedule(int steps) const {
    if (dit_.meanflow()) return linspace(1.0f, 0.0f, steps + 1);
    // RF starts just short of pure noise.
    std::vector<float> t = linspace(0.0f, 1.0f, steps + 1);
    for (float & v : t) v = (1.0f - v) * 0.999f;
    return t;
}

std::vector<float> Sampler::velocity(const Conditions & c, const std::vector<float> & x, int frames, float t, float t_next,
                                     std::vector<float> * branches) {
    const bool guided = !dit_.meanflow() && (double) t >= cfg_min_t_ && (double) t <= cfg_max_t_;
    Graph g;
    ggml_tensor * out = dit_.build(g, x, frames, t, t - t_next, c, guided ? 3 : 1);
    g.output(out);
    g.compute(backend_, allocr_);
    std::vector<float> v = Graph::read(out);
    if (!guided) return v;
    if (branches) *branches = v;
    // The runtime's order: the text's guidance added first, then the speaker's.
    const size_t n = x.size();
    std::vector<float> guided_v(v.begin(), v.begin() + n);
    for (size_t i = 0; i < n; i++) guided_v[i] = guided_v[i] + cfg_text_ * (v[i] - v[n + i]);
    for (size_t i = 0; i < n; i++) guided_v[i] = guided_v[i] + cfg_speaker_ * (v[i] - v[2 * n + i]);
    return guided_v;
}

std::vector<float> Sampler::sample(const Conditions & c, std::vector<float> x, int frames, int steps,
                                   const std::function<void(const SamplerStep &)> & observe) {
    const std::vector<float> times = schedule(steps);
    for (int i = 0; i < steps; i++) {
        const std::vector<float> v = velocity(c, x, frames, times[i], times[i + 1]);
        const float dt = times[i + 1] - times[i];
        for (size_t j = 0; j < x.size(); j++) x[j] = x[j] + v[j] * dt;
        if (observe) observe({i, times[i], v, x});
    }
    return x;
}

std::vector<float> gaussian_noise(uint64_t seed, size_t n) {
    std::mt19937_64 rng(seed);
    std::vector<float> out(n);
    constexpr double kTwoPi = 6.283185307179586;
    for (size_t i = 0; i < n; i += 2) {
        // Uniform on (0, 1]: 53 random bits, offset so that the logarithm never sees zero.
        const double u1 = ((rng() >> 11) + 1.0) * 0x1.0p-53, u2 = (rng() >> 11) * 0x1.0p-53;
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i] = (float) (r * std::cos(kTwoPi * u2));
        if (i + 1 < n) out[i + 1] = (float) (r * std::sin(kTwoPi * u2));
    }
    return out;
}
