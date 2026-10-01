#include "speaker-encoder.h"

#include <stdexcept>
#include <string>

#include "layers.h"

namespace irodori {

/*
 * Each head's query and key get their own RMSNorm, then RoPE that rotates neighbouring pairs of channels
 * (GGML_ROPE_TYPE_NORMAL, the complex multiplication of the official model) across the whole head.
 */

SpeakerEncoder::SpeakerEncoder(const ModelFile & m) : m_(m) {
    dim_ = (int) m.u32("irodori.speaker.dim");
    heads_ = (int) m.u32("irodori.speaker.num_heads");
    layers_ = (int) m.u32("irodori.speaker.num_layers");
    patch_ = (int) m.u32("irodori.speaker.patch_size");
    latent_dim_ = (int) m.u32("irodori.latent_dim");
    eps_ = m.f32("irodori.norm_eps");
    theta_ = m.f32("irodori.rope_theta");
}

ggml_tensor * SpeakerEncoder::build(Graph & g, const std::vector<float> & latent, ggml_tensor ** encoded) const {
    ggml_context * ctx = g.ctx();
    const Layers l{ctx, m_, eps_};
    const int64_t patches = (int64_t) latent.size() / latent_dim_ / patch_;
    if (patches == 0) throw std::runtime_error("the reference is shorter than one patch of the speaker encoder");
    const int head_dim = dim_ / heads_;
    // A patch is its frames one after another, which the row-major latent already is.
    const std::vector<float> whole(latent.begin(), latent.begin() + patches * patch_ * latent_dim_);
    ggml_tensor * x = l.linear(g.input(whole, (int64_t) patch_ * latent_dim_, patches), "speaker.in_proj");
    x = ggml_scale(ctx, x, 1.0f / 6.0f);

    std::vector<int32_t> positions(patches);
    for (int64_t i = 0; i < patches; i++) positions[i] = (int32_t) i;
    ggml_tensor * pos = g.input(positions, patches);
    for (int i = 0; i < layers_; i++) {
        const std::string b = "speaker.blk." + std::to_string(i) + ".";
        ggml_tensor * h = l.rms(x, b + "attn_norm");
        auto heads = [&](const char * w) {
            return ggml_reshape_4d(ctx, mul_mat(ctx, m_.tensor(b + w), h), head_dim, heads_, patches, 1);
        };
        ggml_tensor * q = ggml_mul(ctx, ggml_rms_norm(ctx, heads("attn_q"), eps_), m_.tensor(b + "q_norm"));
        ggml_tensor * k = ggml_mul(ctx, ggml_rms_norm(ctx, heads("attn_k"), eps_), m_.tensor(b + "k_norm"));
        q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, theta_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, theta_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * y = ggml_reshape_2d(ctx, l.attention(q, k, heads("attn_v"), nullptr), dim_, patches);
        y = ggml_mul(ctx, y, ggml_sigmoid(ctx, mul_mat(ctx, m_.tensor(b + "attn_gate"), h)));
        x = ggml_add(ctx, x, mul_mat(ctx, m_.tensor(b + "attn_o"), y));
        x = ggml_add(ctx, x, l.swiglu(l.rms(x, b + "ffn_norm"), b));
    }
    if (encoded) *encoded = x;
    x = l.rms(x, "speaker.norm");
    ggml_tensor * mean = ggml_reshape_2d(ctx, ggml_mean(ctx, ggml_cont(ctx, ggml_transpose(ctx, x))), dim_, 1);
    return ggml_concat(ctx, mean, x, 1);
}

}  // namespace irodori
