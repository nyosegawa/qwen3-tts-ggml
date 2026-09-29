#include "synthesizer.h"

#include <random>
#include <stdexcept>

Synthesizer::Synthesizer(const std::string & talker_path, const std::string & codec_path, ggml_backend_t backend,
                         int n_ctx)
    : talker_(talker_path, backend, n_ctx),
      codec_(codec_path, backend),
      tokenizer_(talker_.model()),
      ids_(talker_.model()) {
    if (codec_.num_quantizers() != talker_.num_code_groups()) {
        throw std::runtime_error("the talker and the codec disagree on the number of codebooks");
    }
}

int Synthesizer::synthesize(const SynthesisRequest & r, const AudioSink & sink, int frames_per_piece) {
    std::vector<int32_t> text_ids = {ids_.im_start, ids_.assistant, ids_.newline};
    const std::vector<int32_t> body = tokenizer_.encode(r.text);
    if (body.empty()) throw std::runtime_error("the text is empty");
    text_ids.insert(text_ids.end(), body.begin(), body.end());
    text_ids.insert(text_ids.end(), {ids_.im_end, ids_.newline, ids_.im_start, ids_.assistant, ids_.newline});
    const Prompt prompt = build_prompt(talker_, ids_, text_ids, r.speaker, r.language);

    const int n_groups = talker_.num_code_groups();
    const int vocab = talker_.vocab();
    // Only the 2048 audio codes and the end of speech may be sampled; the rest of the talker's
    // vocabulary is its control tokens.
    std::vector<bool> banned(vocab, false);
    for (int i = vocab - 1024; i < vocab; i++) banned[i] = i != ids_.codec_eos;
    const std::vector<bool> none;
    std::mt19937_64 rng(r.seed);

    talker_.prefill(prompt.embeds, prompt.n);
    codec_.reset();
    std::vector<int32_t> history, frame(n_groups), pending;
    std::vector<float> audio;
    int frames = 0, pending_frames = 0;
    bool stopped = false;

    auto flush = [&]() {
        if (pending_frames == 0) return;
        audio.clear();
        codec_.decode(pending.data(), pending_frames, audio);
        pending.clear();
        pending_frames = 0;
        if (!sink(audio.data(), audio.size())) stopped = true;
    };

    while (frames < r.max_frames && !stopped) {
        std::vector<bool> b = banned;
        // The official generate() asks for at least two frames before the end of speech.
        if (frames < 2) b[ids_.codec_eos] = true;
        frame[0] = sample(talker_.logits(), r.talker, history, b, rng);
        if (frame[0] == ids_.codec_eos) break;
        history.push_back(frame[0]);
        frame[1] = sample(talker_.cp_begin(frame[0]), r.code_predictor, {}, none, rng);
        for (int g = 1; g < n_groups - 1; g++) {
            frame[g + 1] = sample(talker_.cp_next(g, frame[g]), r.code_predictor, {}, none, rng);
        }
        pending.insert(pending.end(), frame.begin(), frame.end());
        pending_frames++;
        frames++;
        if (frames == 1 || pending_frames >= frames_per_piece) flush();
        if (stopped) break;
        talker_.step(frame.data(), prompt.frame_extra);
    }
    if (!stopped) flush();
    return frames;
}
