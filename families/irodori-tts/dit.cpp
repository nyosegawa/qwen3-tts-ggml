#include "dit.h"

#include <cmath>
#include <string>

#include "layers.h"

namespace irodori {

/*
 * The latent attends to itself, then to the text and the speaker, in one softmax over the concatenated
 * keys. Each head's query and key get their own RMSNorm; RoPE (neighbouring pairs, as the official complex
 * multiplication) turns only the first half of the heads of the latent's own queries and keys, since the
 * official model splits the head axis rather than the channels. The conditions' keys and values carry no
 * position.
 */

Dit::Dit(const ModelFile & m) : m_(m) {
    meanflow_ = m.str("irodori.flow") == "meanflow";
    dim_ = (int) m.u32("irodori.dit.dim");
    heads_ = (int) m.u32("irodori.dit.num_heads");
    layers_ = (int) m.u32("irodori.dit.num_layers");
    timestep_dim_ = (int) m.u32("irodori.dit.timestep_dim");
    latent_dim_ = (int) m.u32("irodori.latent_dim");
    eps_ = m.f32("irodori.norm_eps");
    theta_ = m.f32("irodori.rope_theta");
}

std::vector<float> timestep_embedding(float t, int dim) {
    const int half = dim / 2;
    const float log_base = std::log(10000.0f);
    std::vector<float> out(dim);
    for (int i = 0; i < half; i++) {
        const float frequency = 1000.0f * std::exp(-log_base * (float) i / (float) half);
        const float arg = t * frequency;
        out[i] = std::cos(arg);
        out[half + i] = std::sin(arg);
    }
    return out;
}

ggml_tensor * Dit::build(Graph & g, const std::vector<float> & x_in, int frames, float t, float delta, const Conditions & c,
                         int branches, std::vector<ggml_tensor *> * blocks, ggml_tensor ** cond_out) const {
    ggml_context * ctx = g.ctx();
    const Layers l{ctx, m_, eps_};
    const int head_dim = dim_ / heads_, half = heads_ / 2;
    const int64_t s = frames, n_kv = frames + c.text_tokens + c.speaker_tokens;

    auto mlp3 = [&](const std::string & prefix, ggml_tensor * x) {
        x = ggml_silu(ctx, mul_mat(ctx, m_.tensor(prefix + ".0"), x));
        x = ggml_silu(ctx, mul_mat(ctx, m_.tensor(prefix + ".1"), x));
        return mul_mat(ctx, m_.tensor(prefix + ".2"), x);
    };
    ggml_tensor * cond = mlp3("dit.cond", g.input(timestep_embedding(t, timestep_dim_), timestep_dim_));
    if (meanflow_) cond = ggml_add(ctx, cond, mlp3("dit.delta_cond", g.input(timestep_embedding(delta, timestep_dim_), timestep_dim_)));
    if (cond_out) *cond_out = cond;
    auto chunk = [&](int k) { return ggml_view_1d(ctx, cond, dim_, (size_t) k * dim_ * ggml_element_size(cond)); };

    // Every branch starts from the same latent; they part where their conditions differ.
    std::vector<float> x_rep;
    for (int b = 0; b < branches; b++) x_rep.insert(x_rep.end(), x_in.begin(), x_in.end());
    ggml_tensor * x = l.linear(g.input(x_rep, latent_dim_, s, branches), "dit.in_proj");

    ggml_tensor * text = g.input(c.text, c.text.size() / c.text_tokens, c.text_tokens);
    ggml_tensor * speaker = g.input(c.speaker, c.speaker.size() / c.speaker_tokens, c.speaker_tokens);
    std::vector<int32_t> positions(s);
    for (int64_t i = 0; i < s; i++) positions[i] = (int32_t) i;
    ggml_tensor * pos = g.input(positions, s);
    // Branch 1 leaves out the text's keys and branch 2 the speaker's.
    ggml_tensor * mask = nullptr;
    if (branches > 1) {
        std::vector<float> m((size_t) n_kv * s * branches, 0.0f);
        for (int b = 1; b < branches; b++) {
            const int64_t from = b == 1 ? s : s + c.text_tokens, to = b == 1 ? s + c.text_tokens : n_kv;
            for (int64_t q = 0; q < s; q++)
                for (int64_t k = from; k < to; k++) m[((size_t) b * s + q) * n_kv + k] = -INFINITY;
        }
        mask = g.input(m, n_kv, s, 1, branches);
    }

    // Both AdaLN layers of a block split the same condition into shift, scale and gate, each through its own weights.
    auto adaln = [&](ggml_tensor * x, const std::string & prefix, ggml_tensor ** gate) {
        auto part = [&](const char * name, int k) {
            ggml_tensor * base = chunk(k);
            const std::string p = prefix + "." + name;
            ggml_tensor * low = mul_mat(ctx, m_.tensor(p + ".down"), ggml_silu(ctx, base));
            return ggml_add(ctx, l.linear(low, p + ".up"), base);
        };
        ggml_tensor * shift = part("shift", 0), * scale = part("scale", 1);
        *gate = ggml_tanh(ctx, part("gate", 2));
        return l.modulate(ggml_rms_norm(ctx, x, eps_), shift, scale);
    };
    auto rope_half = [&](ggml_tensor * t4) {
        ggml_tensor * turned = ggml_cont(ctx, ggml_view_4d(ctx, t4, head_dim, half, s, branches, t4->nb[1], t4->nb[2], t4->nb[3], 0));
        ggml_tensor * kept = ggml_view_4d(ctx, t4, head_dim, heads_ - half, s, branches, t4->nb[1], t4->nb[2], t4->nb[3],
                                          (size_t) half * t4->nb[1]);
        turned = ggml_rope_ext(ctx, turned, pos, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, theta_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        return ggml_concat(ctx, turned, ggml_cont(ctx, kept), 1);
    };

    for (int i = 0; i < layers_; i++) {
        const std::string b = "dit.blk." + std::to_string(i) + ".";
        ggml_tensor * gate = nullptr;
        ggml_tensor * h = adaln(x, b + "attn_ada", &gate);
        auto heads = [&](const std::string & w, ggml_tensor * in, int64_t n, int64_t batch) {
            return ggml_reshape_4d(ctx, mul_mat(ctx, m_.tensor(b + w), in), head_dim, heads_, n, batch);
        };
        auto norm = [&](ggml_tensor * t4, const char * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, t4, eps_), m_.tensor(b + w)); };
        ggml_tensor * q = rope_half(norm(heads("attn_q", h, s, branches), "q_norm"));
        ggml_tensor * k_self = rope_half(norm(heads("attn_k", h, s, branches), "k_norm"));
        ggml_tensor * v_self = heads("attn_v", h, s, branches);
        auto repeat = [&](ggml_tensor * t4) {
            return branches == 1 ? t4 : ggml_repeat(ctx, t4, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, t4->ne[0], t4->ne[1], t4->ne[2], branches));
        };
        ggml_tensor * k_text = repeat(norm(heads("attn_k_text", text, c.text_tokens, 1), "k_norm"));
        ggml_tensor * v_text = repeat(heads("attn_v_text", text, c.text_tokens, 1));
        ggml_tensor * k_speaker = repeat(norm(heads("attn_k_speaker", speaker, c.speaker_tokens, 1), "k_norm"));
        ggml_tensor * v_speaker = repeat(heads("attn_v_speaker", speaker, c.speaker_tokens, 1));
        ggml_tensor * k = ggml_concat(ctx, ggml_concat(ctx, k_self, k_text, 2), k_speaker, 2);
        ggml_tensor * v = ggml_concat(ctx, ggml_concat(ctx, v_self, v_text, 2), v_speaker, 2);
        ggml_tensor * y = l.attention(q, k, v, mask);
        y = ggml_mul(ctx, y, ggml_sigmoid(ctx, mul_mat(ctx, m_.tensor(b + "attn_gate"), h)));
        x = ggml_add(ctx, x, ggml_mul(ctx, mul_mat(ctx, m_.tensor(b + "attn_o"), y), gate));

        h = adaln(x, b + "ffn_ada", &gate);
        x = ggml_add(ctx, x, ggml_mul(ctx, l.swiglu(h, b), gate));
        if (blocks) blocks->push_back(x);
    }
    return l.linear(l.rms(x, "dit.out_norm"), "dit.out_proj");
}

}  // namespace irodori
