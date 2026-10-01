#include "duration.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "layers.h"

DurationPredictor::DurationPredictor(const ModelFile & m, int sample_rate, int hop) : m_(m) {
    layers_ = (int) m.u32("irodori.duration.num_layers");
    eps_ = m.f32("irodori.norm_eps");
    min_frames_ = std::max(1, (int) std::ceil(m.f32("irodori.min_seconds") * sample_rate / (double) hop));
    max_frames_ = std::max(1, (int) std::floor(m.f32("irodori.max_seconds") * sample_rate / (double) hop));
}

ggml_tensor * DurationPredictor::build(Graph & g, ggml_tensor * text_state, ggml_tensor * speaker_summary) const {
    ggml_context * ctx = g.ctx();
    const Layers l{ctx, m_, eps_};
    ggml_tensor * speaker = ggml_silu(ctx, speaker_summary);
    ggml_tensor * caption = ggml_silu(ctx, m_.tensor("duration.null_caption"));
    ggml_tensor * h = l.linear(text_state, "duration.in_proj");
    const int64_t dim = h->ne[0];
    for (int i = 0; i < layers_; i++) {
        const std::string b = "duration.blk." + std::to_string(i) + ".";
        ggml_tensor * mod = ggml_add(ctx, l.linear(speaker, b + "mod"), l.linear(caption, b + "caption_mod"));
        auto part = [&](int k) { return ggml_view_1d(ctx, mod, dim, k * dim * ggml_element_size(mod)); };
        ggml_tensor * x = l.modulate(l.rms(h, b + "norm"), part(0), part(1));
        h = ggml_add(ctx, h, ggml_mul(ctx, l.swiglu(x, b), ggml_tanh(ctx, part(2))));
    }
    ggml_tensor * per_token = ggml_softplus(ctx, l.linear(l.rms(h, "duration.out_norm"), "duration.out_proj"));
    return ggml_sum(ctx, per_token);
}

int DurationPredictor::frames(float predicted_sum) const {
    // The runtime's float32 round trip through log1p and expm1, then Python's round(), which rounds half to even.
    const float predicted = std::expm1(std::log1p(std::max(predicted_sum, 0.0f)));
    const int rounded = (int) std::nearbyint((double) predicted);
    return std::max(min_frames_, std::min(max_frames_, rounded));
}
