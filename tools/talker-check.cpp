// Checks the talker and the code predictor against a reference dump.
//
// Teacher forcing feeds the reference codes frame by frame and compares every stage's output, so the
// first stage that departs is the one reported. A free-running greedy decode then shows whether the
// same codes come out.
//
// usage: talker-check <talker.gguf> <reference dir> [gpu|cpu]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>

#include "backend.h"
#include "npy.h"
#include "prompt.h"
#include "sampler.h"
#include "talker.h"

namespace {

struct Worst {
    double rel = 0;
    int argmax_miss = 0;
    int count = 0;

    void add(const float * got, const float * want, int n) {
        double num = 0, den = 0;
        for (int i = 0; i < n; i++) {
            num += (double) (got[i] - want[i]) * (got[i] - want[i]);
            den += (double) want[i] * want[i];
        }
        rel = std::max(rel, std::sqrt(num / std::max(den, 1e-30)));
        if (std::max_element(got, got + n) - got != std::max_element(want, want + n) - want) argmax_miss++;
        count++;
    }
    void print(const char * what) const {
        std::printf("%-26s worst relative error %.2e, argmax differs %d of %d\n", what, rel, argmax_miss, count);
    }
};

std::string meta_field(const std::string & dir, const std::string & key) {
    std::ifstream f(dir + "/meta.json");
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t k = json.find("\"" + key + "\"");
    const size_t a = json.find('"', json.find(':', k) + 1);
    const size_t b = json.find('"', a + 1);
    return json.substr(a + 1, b - a - 1);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <talker.gguf> <reference dir> [gpu|cpu]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    ggml_backend_t backend = init_backend(argc > 3 ? argv[3] : "");
    std::printf("backend: %s\n", ggml_backend_name(backend));
    Talker talker(argv[1], backend, 2048);
    const PromptIds ids(talker.model());

    const Npy input_ids = read_npy(dir + "/input_ids.npy");
    const Npy ref_prefill = read_npy(dir + "/prefill_embeds.npy");
    const Npy ref_logits = read_npy(dir + "/talker_logits.npy");
    const Npy ref_hidden = read_npy(dir + "/talker_hidden.npy");
    const Npy ref_cp = read_npy(dir + "/cp_logits.npy");
    const Npy ref_codes = read_npy(dir + "/codes.npy");
    const std::string speaker = meta_field(dir, "speaker"), language = meta_field(dir, "language");
    const int h = talker.hidden(), n_groups = talker.num_code_groups();
    const int n_frames = (int) ref_codes.shape[0];

    const Prompt prompt = build_prompt(talker, ids, input_ids.i32, speaker, language);
    Worst w_prompt;
    if (prompt.n != ref_prefill.shape[0]) {
        std::printf("prompt has %d rows, the reference %lld\n", prompt.n, (long long) ref_prefill.shape[0]);
        return 1;
    }
    for (int r = 0; r < prompt.n; r++) w_prompt.add(&prompt.embeds[r * h], &ref_prefill.f32[r * h], h);
    w_prompt.print("prompt embeddings");

    // Teacher forcing.
    Worst w_logits, w_hidden, w_cp;
    talker.prefill(prompt.embeds, prompt.n);
    w_logits.add(talker.logits().data(), &ref_logits.f32[0], talker.vocab());
    w_hidden.add(talker.hidden_state().data(), &ref_hidden.f32[0], h);
    for (int f = 0; f < n_frames; f++) {
        const int32_t * codes = &ref_codes.i32[f * n_groups];
        const float * cp_ref = &ref_cp.f32[(size_t) f * (n_groups - 1) * talker.cp_vocab()];
        w_cp.add(talker.cp_begin(codes[0]).data(), cp_ref, talker.cp_vocab());
        for (int g = 1; g < n_groups - 1; g++) {
            w_cp.add(talker.cp_next(g, codes[g]).data(), cp_ref + (size_t) g * talker.cp_vocab(), talker.cp_vocab());
        }
        talker.step(codes, prompt.frame_extra);
        w_logits.add(talker.logits().data(), &ref_logits.f32[(size_t) (f + 1) * talker.vocab()], talker.vocab());
        w_hidden.add(talker.hidden_state().data(), &ref_hidden.f32[(size_t) (f + 1) * h], h);
    }
    w_logits.print("talker logits");
    w_hidden.print("talker hidden state");
    w_cp.print("code predictor logits");

    // Free-running greedy decode with the official talker settings.
    SamplingParams talker_p;
    talker_p.greedy = true;
    talker_p.repetition_penalty = 1.05f;
    SamplingParams cp_p;
    cp_p.greedy = true;
    std::vector<bool> banned(talker.vocab(), false);
    for (int i = talker.vocab() - 1024; i < talker.vocab(); i++) banned[i] = i != ids.codec_eos;
    std::vector<bool> none;
    std::mt19937_64 rng(0);
    std::vector<int32_t> history, codes(n_groups);
    int first_diff = -1, frames = 0;
    talker.prefill(prompt.embeds, prompt.n);
    for (int f = 0; f < 2 * n_frames + 10; f++) {
        std::vector<bool> b = banned;
        if (f < 2) b[ids.codec_eos] = true;
        codes[0] = sample(talker.logits(), talker_p, history, b, rng);
        if (codes[0] == ids.codec_eos) break;
        history.push_back(codes[0]);
        codes[1] = sample(talker.cp_begin(codes[0]), cp_p, {}, none, rng);
        for (int g = 1; g < n_groups - 1; g++) codes[g + 1] = sample(talker.cp_next(g, codes[g]), cp_p, {}, none, rng);
        if (first_diff < 0 && (f >= n_frames || !std::equal(codes.begin(), codes.end(), &ref_codes.i32[f * n_groups]))) {
            first_diff = f;
        }
        talker.step(codes.data(), prompt.frame_extra);
        frames++;
    }
    std::printf("greedy decode: %d frames (reference %d), first frame that differs: %d\n", frames, n_frames, first_diff);
    ggml_backend_free(backend);
    return 0;
}
