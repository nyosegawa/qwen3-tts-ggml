#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "model-file.h"

/** The shape of one Qwen3 decoder stack (the talker's or the code predictor's). */
struct DecoderShape {
    int hidden = 0;
    int ffn = 0;
    int n_layer = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int head_dim = 0;
    float rms_eps = 0;
    float rope_theta = 0;
};

/**
 * The talker and its code predictor. The talker reads one embedding per frame and predicts the frame's
 * first code; the code predictor, conditioned on the talker's last hidden state, predicts the other
 * fifteen one by one. Both keep a key/value cache: the talker's spans the utterance, the code
 * predictor's is rebuilt every frame.
 */
class Talker {
public:
    Talker(const std::string & path, ggml_backend_t backend, int n_ctx);
    ~Talker();

    const ModelFile & model() const { return *model_; }
    int num_code_groups() const { return n_groups_; }
    int hidden() const { return talker_.hidden; }
    int vocab() const { return vocab_; }
    int cp_vocab() const { return cp_vocab_; }

    /** Projected text embeddings of `ids`, row-major [ids.size(), hidden]. */
    std::vector<float> text_embeddings(const std::vector<int32_t> & ids);
    /** Talker codec embeddings of `ids`, row-major [ids.size(), hidden]. */
    std::vector<float> codec_embeddings(const std::vector<int32_t> & ids);

    /** Starts an utterance: clears the cache and runs `embeds` ([n, hidden], row-major) through the talker. */
    void prefill(const std::vector<float> & embeds, int n);
    /** Feeds one frame: the sum of its 16 code embeddings plus `extra` ([hidden]). */
    void step(const int32_t * codes, const std::vector<float> & extra);

    /** The codec head's logits for the next frame's first code. */
    const std::vector<float> & logits() const { return logits_; }
    /** The talker's last hidden state after the final norm, which conditions the code predictor. */
    const std::vector<float> & hidden_state() const { return hidden_; }

    /** Code predictor: starts a frame from the talker's hidden state and the first code, and returns the logits of code 1. */
    const std::vector<float> & cp_begin(int32_t code0);
    /** Feeds code `group` (1..14) and returns the logits of code `group + 1`. */
    const std::vector<float> & cp_next(int group, int32_t code);

    int64_t n_past() const { return n_past_; }

private:
    struct Stack;
    ggml_tensor * run_stack(struct GraphCtx & g, const std::string & prefix, const DecoderShape & s, ggml_tensor * x,
                            ggml_tensor * pos, ggml_tensor * mask, std::vector<ggml_tensor *> & k_cache,
                            std::vector<ggml_tensor *> & v_cache, int64_t n_past, int64_t n_tokens);
    void run_talker(const std::vector<float> * embeds, const int32_t * codes, const std::vector<float> * extra, int64_t n);
    const std::vector<float> & run_cp(int32_t code, int group);

    ggml_backend_t backend_;
    std::unique_ptr<ModelFile> model_;
    DecoderShape talker_, cp_;
    int n_groups_ = 0;
    int vocab_ = 0;
    int cp_vocab_ = 0;
    int n_ctx_ = 0;

    ggml_context * cache_ctx_ = nullptr;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    std::vector<ggml_tensor *> talker_k_, talker_v_, cp_k_, cp_v_;
    ggml_gallocr_t allocr_ = nullptr;

    int64_t n_past_ = 0;
    int64_t cp_past_ = 0;
    std::vector<float> logits_, hidden_, cp_logits_;
};
