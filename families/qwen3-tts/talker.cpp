#include "talker.h"

#include <cmath>
#include <stdexcept>

/*
 * Activations are channel-first ([hidden, tokens]). The talker's position embedding is Qwen's
 * multimodal RoPE, but a text-only sequence gives its three sections the same position, which makes
 * it the ordinary rotate-half RoPE (GGML_ROPE_TYPE_NEOX).
 */

struct GraphCtx {
    ggml_context * ctx;
    ggml_cgraph * gf;
};

namespace {

constexpr int kGraphSize = 8192;

GraphCtx new_graph() {
    ggml_init_params params = {ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false),
                               nullptr, true};
    ggml_context * ctx = ggml_init(params);
    return {ctx, ggml_new_graph_custom(ctx, kGraphSize, false)};
}

DecoderShape read_shape(const ModelFile & m, const std::string & p) {
    DecoderShape s;
    s.hidden = (int) m.u32(p + ".hidden_size");
    s.ffn = (int) m.u32(p + ".intermediate_size");
    s.n_layer = (int) m.u32(p + ".num_hidden_layers");
    s.n_head = (int) m.u32(p + ".num_attention_heads");
    s.n_kv_head = (int) m.u32(p + ".num_key_value_heads");
    s.head_dim = (int) m.u32(p + ".head_dim");
    s.rms_eps = m.f32(p + ".rms_norm_eps");
    s.rope_theta = m.f32(p + ".rope_theta");
    return s;
}

ggml_tensor * input_i32(ggml_context * ctx, int64_t n) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_input(t);
    return t;
}

}  // namespace

Talker::Talker(const std::string & path, ggml_backend_t backend, int n_ctx) : backend_(backend), n_ctx_(n_ctx) {
    model_ = std::make_unique<ModelFile>(path, backend);
    const ModelFile & m = *model_;
    talker_ = read_shape(m, "talker");
    cp_ = read_shape(m, "code_predictor");
    n_groups_ = (int) m.u32("talker.num_code_groups");
    vocab_ = (int) m.u32("talker.vocab_size");
    cp_vocab_ = (int) m.u32("code_predictor.vocab_size");

    const int n_tensors = 2 * (talker_.n_layer + cp_.n_layer);
    ggml_init_params params = {ggml_tensor_overhead() * (n_tensors + 4), nullptr, true};
    cache_ctx_ = ggml_init(params);
    for (int l = 0; l < talker_.n_layer; l++) {
        talker_k_.push_back(ggml_new_tensor_2d(cache_ctx_, GGML_TYPE_F16, talker_.n_kv_head * talker_.head_dim, n_ctx));
        talker_v_.push_back(ggml_new_tensor_2d(cache_ctx_, GGML_TYPE_F16, talker_.n_kv_head * talker_.head_dim, n_ctx));
    }
    for (int l = 0; l < cp_.n_layer; l++) {
        cp_k_.push_back(ggml_new_tensor_2d(cache_ctx_, GGML_TYPE_F32, cp_.n_kv_head * cp_.head_dim, n_groups_ + 1));
        cp_v_.push_back(ggml_new_tensor_2d(cache_ctx_, GGML_TYPE_F32, cp_.n_kv_head * cp_.head_dim, n_groups_ + 1));
    }
    cache_buffer_ = ggml_backend_alloc_ctx_tensors(cache_ctx_, backend_);
    if (!cache_buffer_) throw std::runtime_error("cannot allocate the key/value caches");
    ggml_backend_buffer_clear(cache_buffer_, 0);
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Talker::~Talker() {
    if (allocr_) ggml_gallocr_free(allocr_);
    if (cache_buffer_) ggml_backend_buffer_free(cache_buffer_);
    if (cache_ctx_) ggml_free(cache_ctx_);
}

ggml_tensor * Talker::run_stack(GraphCtx & g, const std::string & prefix, const DecoderShape & s, ggml_tensor * x,
                                ggml_tensor * pos, ggml_tensor * mask, std::vector<ggml_tensor *> & k_cache,
                                std::vector<ggml_tensor *> & v_cache, int64_t n_past, int64_t n_tokens) {
    const ModelFile & m = *model_;
    ggml_context * ctx = g.ctx;
    const int64_t kv_dim = (int64_t) s.n_kv_head * s.head_dim;
    const int64_t n_kv = n_past + n_tokens;
    for (int l = 0; l < s.n_layer; l++) {
        const std::string b = prefix + ".blk." + std::to_string(l) + ".";
        ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, x, s.rms_eps), m.tensor(b + "attn_norm"));
        ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, m.tensor(b + "attn_q"), h), s.head_dim, s.n_head, n_tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, m.tensor(b + "attn_k"), h), s.head_dim, s.n_kv_head, n_tokens);
        ggml_tensor * v = ggml_mul_mat(ctx, m.tensor(b + "attn_v"), h);
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, s.rms_eps), m.tensor(b + "attn_q_norm"));
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, s.rms_eps), m.tensor(b + "attn_k_norm"));
        q = ggml_rope_ext(ctx, q, pos, nullptr, s.head_dim, GGML_ROPE_TYPE_NEOX, 0, s.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, s.head_dim, GGML_ROPE_TYPE_NEOX, 0, s.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor * kc = k_cache[l];
        ggml_tensor * vc = v_cache[l];
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, k, kv_dim, n_tokens),
                                                 ggml_view_2d(ctx, kc, kv_dim, n_tokens, kc->nb[1], n_past * kc->nb[1])));
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, v, ggml_view_2d(ctx, vc, kv_dim, n_tokens, vc->nb[1], n_past * vc->nb[1])));

        ggml_tensor * keys = ggml_view_3d(ctx, kc, s.head_dim, s.n_kv_head, n_kv, ggml_row_size(kc->type, s.head_dim), kc->nb[1], 0);
        ggml_tensor * vals = ggml_view_3d(ctx, vc, s.head_dim, s.n_kv_head, n_kv, ggml_row_size(vc->type, s.head_dim), vc->nb[1], 0);
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, keys, 0, 2, 1, 3));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vals, 1, 2, 0, 3));
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kq = ggml_mul_mat(ctx, kh, qh);
        kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / std::sqrt((float) s.head_dim), 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);
        ggml_tensor * o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)),
                                          (int64_t) s.n_head * s.head_dim, n_tokens);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, m.tensor(b + "attn_o"), o));

        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, s.rms_eps), m.tensor(b + "ffn_norm"));
        ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, m.tensor(b + "ffn_gate"), h));
        h = ggml_mul_mat(ctx, m.tensor(b + "ffn_down"), ggml_mul(ctx, gate, ggml_mul_mat(ctx, m.tensor(b + "ffn_up"), h)));
        x = ggml_add(ctx, x, h);
    }
    return x;
}

std::vector<float> Talker::text_embeddings(const std::vector<int32_t> & ids) {
    const ModelFile & m = *model_;
    GraphCtx g = new_graph();
    ggml_tensor * idx = input_i32(g.ctx, (int64_t) ids.size());
    ggml_tensor * x = ggml_get_rows(g.ctx, m.tensor("talker.text_embd"), idx);
    x = ggml_add(g.ctx, ggml_mul_mat(g.ctx, m.tensor("talker.text_proj.fc1.weight"), x), m.tensor("talker.text_proj.fc1.bias"));
    x = ggml_silu(g.ctx, x);
    x = ggml_add(g.ctx, ggml_mul_mat(g.ctx, m.tensor("talker.text_proj.fc2.weight"), x), m.tensor("talker.text_proj.fc2.bias"));
    ggml_set_output(x);
    ggml_build_forward_expand(g.gf, x);
    if (!ggml_gallocr_alloc_graph(allocr_, g.gf)) throw std::runtime_error("cannot allocate the text embedding graph");
    ggml_backend_tensor_set(idx, ids.data(), 0, ids.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend_, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("text embedding failed");
    std::vector<float> out(ggml_nelements(x));
    ggml_backend_tensor_get(x, out.data(), 0, ggml_nbytes(x));
    ggml_free(g.ctx);
    return out;
}

std::vector<float> Talker::codec_embeddings(const std::vector<int32_t> & ids) {
    GraphCtx g = new_graph();
    ggml_tensor * idx = input_i32(g.ctx, (int64_t) ids.size());
    ggml_tensor * x = ggml_get_rows(g.ctx, model_->tensor("talker.codec_embd"), idx);
    ggml_set_output(x);
    ggml_build_forward_expand(g.gf, x);
    if (!ggml_gallocr_alloc_graph(allocr_, g.gf)) throw std::runtime_error("cannot allocate the codec embedding graph");
    ggml_backend_tensor_set(idx, ids.data(), 0, ids.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend_, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("codec embedding failed");
    std::vector<float> out(ggml_nelements(x));
    ggml_backend_tensor_get(x, out.data(), 0, ggml_nbytes(x));
    ggml_free(g.ctx);
    return out;
}

void Talker::prefill(const std::vector<float> & embeds, int n) {
    n_past_ = 0;
    run_talker(&embeds, nullptr, nullptr, n);
}

void Talker::step(const int32_t * codes, const std::vector<float> & extra) {
    run_talker(nullptr, codes, &extra, 1);
}

void Talker::run_talker(const std::vector<float> * embeds, const int32_t * codes, const std::vector<float> * extra,
                        int64_t n) {
    if (n_past_ + n > n_ctx_) throw std::runtime_error("the utterance is longer than the talker's context");
    const ModelFile & m = *model_;
    GraphCtx g = new_graph();
    ggml_context * ctx = g.ctx;

    ggml_tensor * in_embeds = nullptr, * in_extra = nullptr;
    std::vector<ggml_tensor *> in_codes;
    ggml_tensor * x;
    if (embeds) {
        in_embeds = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, talker_.hidden, n);
        ggml_set_input(in_embeds);
        x = in_embeds;
    } else {
        // A frame's input is the sum of its codes' embeddings: the first from the talker's table, the
        // others from the code predictor's, which live in the talker's hidden space.
        for (int q = 0; q < n_groups_; q++) in_codes.push_back(input_i32(ctx, 1));
        x = ggml_get_rows(ctx, m.tensor("talker.codec_embd"), in_codes[0]);
        for (int q = 1; q < n_groups_; q++) {
            x = ggml_add(ctx, x, ggml_get_rows(ctx, m.tensor("cp.codec_embd." + std::to_string(q - 1)), in_codes[q]));
        }
        in_extra = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, talker_.hidden);
        ggml_set_input(in_extra);
        x = ggml_add(ctx, x, in_extra);
    }
    ggml_tensor * pos = input_i32(ctx, n);
    ggml_tensor * mask = nullptr;
    if (n > 1) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_past_ + n, n);
        ggml_set_input(mask);
    }
    x = run_stack(g, "talker", talker_, x, pos, mask, talker_k_, talker_v_, n_past_, n);
    x = ggml_mul(ctx, ggml_rms_norm(ctx, x, talker_.rms_eps), m.tensor("talker.norm"));
    ggml_tensor * last = ggml_view_2d(ctx, x, talker_.hidden, 1, x->nb[1], (n - 1) * x->nb[1]);
    ggml_tensor * hidden = ggml_cont(ctx, last);
    ggml_tensor * logits = ggml_mul_mat(ctx, m.tensor("talker.codec_head"), last);
    ggml_set_output(hidden);
    ggml_set_output(logits);
    ggml_build_forward_expand(g.gf, hidden);
    ggml_build_forward_expand(g.gf, logits);
    if (!ggml_gallocr_alloc_graph(allocr_, g.gf)) throw std::runtime_error("cannot allocate the talker graph");

    if (in_embeds) ggml_backend_tensor_set(in_embeds, embeds->data(), 0, ggml_nbytes(in_embeds));
    for (int q = 0; q < (int) in_codes.size(); q++) ggml_backend_tensor_set(in_codes[q], codes + q, 0, sizeof(int32_t));
    if (in_extra) ggml_backend_tensor_set(in_extra, extra->data(), 0, ggml_nbytes(in_extra));
    std::vector<int32_t> positions(n);
    for (int64_t i = 0; i < n; i++) positions[i] = (int32_t) (n_past_ + i);
    ggml_backend_tensor_set(pos, positions.data(), 0, n * sizeof(int32_t));
    if (mask) {
        std::vector<float> md((n_past_ + n) * n);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n_past_ + n; j++) md[i * (n_past_ + n) + j] = j <= n_past_ + i ? 0.0f : -INFINITY;
        ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(float));
    }
    if (ggml_backend_graph_compute(backend_, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("the talker failed");
    logits_.resize(vocab_);
    hidden_.resize(talker_.hidden);
    ggml_backend_tensor_get(logits, logits_.data(), 0, ggml_nbytes(logits));
    ggml_backend_tensor_get(hidden, hidden_.data(), 0, ggml_nbytes(hidden));
    n_past_ += n;
    ggml_free(ctx);
}

const std::vector<float> & Talker::cp_begin(int32_t code0) {
    cp_past_ = 0;
    return run_cp(code0, 0);
}

const std::vector<float> & Talker::cp_next(int group, int32_t code) {
    return run_cp(code, group);
}

const std::vector<float> & Talker::run_cp(int32_t code, int group) {
    const ModelFile & m = *model_;
    GraphCtx g = new_graph();
    ggml_context * ctx = g.ctx;
    ggml_tensor * in_code = input_i32(ctx, 1);
    ggml_tensor * in_hidden = nullptr;
    ggml_tensor * x;
    int64_t n;
    if (group == 0) {
        // The first step reads two positions: the talker's hidden state and the first code's embedding.
        in_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, talker_.hidden, 1);
        ggml_set_input(in_hidden);
        ggml_tensor * e = ggml_get_rows(ctx, m.tensor("talker.codec_embd"), in_code);
        x = ggml_concat(ctx, in_hidden, e, 1);
        n = 2;
    } else {
        x = ggml_get_rows(ctx, m.tensor("cp.codec_embd." + std::to_string(group - 1)), in_code);
        n = 1;
    }
    if (ggml_tensor * w = m.optional_tensor("cp.in_proj.weight")) {
        x = ggml_add(ctx, ggml_mul_mat(ctx, w, x), m.tensor("cp.in_proj.bias"));
    }
    ggml_tensor * pos = input_i32(ctx, n);
    ggml_tensor * mask = nullptr;
    if (n > 1) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cp_past_ + n, n);
        ggml_set_input(mask);
    }
    x = run_stack(g, "cp", cp_, x, pos, mask, cp_k_, cp_v_, cp_past_, n);
    x = ggml_mul(ctx, ggml_rms_norm(ctx, x, cp_.rms_eps), m.tensor("cp.norm"));
    ggml_tensor * last = ggml_view_2d(ctx, x, cp_.hidden, 1, x->nb[1], (n - 1) * x->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(ctx, m.tensor("cp.head." + std::to_string(group)), last);
    ggml_set_output(logits);
    ggml_build_forward_expand(g.gf, logits);
    if (!ggml_gallocr_alloc_graph(allocr_, g.gf)) throw std::runtime_error("cannot allocate the code predictor graph");

    ggml_backend_tensor_set(in_code, &code, 0, sizeof(int32_t));
    if (in_hidden) ggml_backend_tensor_set(in_hidden, hidden_.data(), 0, ggml_nbytes(in_hidden));
    std::vector<int32_t> positions(n);
    for (int64_t i = 0; i < n; i++) positions[i] = (int32_t) (cp_past_ + i);
    ggml_backend_tensor_set(pos, positions.data(), 0, n * sizeof(int32_t));
    if (mask) {
        std::vector<float> md((cp_past_ + n) * n);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < cp_past_ + n; j++) md[i * (cp_past_ + n) + j] = j <= cp_past_ + i ? 0.0f : -INFINITY;
        ggml_backend_tensor_set(mask, md.data(), 0, md.size() * sizeof(float));
    }
    if (ggml_backend_graph_compute(backend_, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("the code predictor failed");
    cp_logits_.resize(cp_vocab_);
    ggml_backend_tensor_get(logits, cp_logits_.data(), 0, ggml_nbytes(logits));
    cp_past_ += n;
    ggml_free(ctx);
    return cp_logits_;
}
