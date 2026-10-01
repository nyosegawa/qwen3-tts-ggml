// Speaks a text into a WAV file and reports how long the first audio and the whole took.
//
// usage: qwen3-tts <talker.gguf> <codec.gguf> <speaker> <language> <text> <out.wav> [gpu|cpu] [seed] [--greedy]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "args.h"
#include "backend.h"
#include "synthesizer.h"
#include "wav.h"

namespace {

int run(const std::vector<std::string> & a) {
    const int argc = (int) a.size();
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s <talker.gguf> <codec.gguf> <speaker> <language> <text> <out.wav> [gpu|cpu] [seed] [--greedy]\n", a[0].c_str());
        return 2;
    }
    ggml_backend_t backend = init_backend(argc > 7 ? a[7] : "");
    auto t_load = std::chrono::steady_clock::now();
    Synthesizer synth(a[1], a[2], backend, 4096);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();

    SynthesisRequest r;
    r.speaker = a[3];
    r.language = a[4];
    r.text = a[5];
    r.seed = argc > 8 ? std::stoull(a[8]) : 0;
    if (argc > 9 && a[9] == "--greedy") {
        r.talker.greedy = true;
        r.code_predictor.greedy = true;
    }

    std::vector<float> pcm;
    double first_audio_s = -1;
    const auto t0 = std::chrono::steady_clock::now();
    SynthesisStats stats;
    const int frames = synth.synthesize(r, [&](const float * s, size_t n) {
        if (first_audio_s < 0) first_audio_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        pcm.insert(pcm.end(), s, s + n);
        return true;
    }, 4, &stats);
    const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double audio_s = (double) pcm.size() / synth.sample_rate();
    std::printf("backend %s, load %.2f s, %d frames, %.2f s of audio, first audio %.3f s, total %.3f s, RTF %.3f\n",
                ggml_backend_name(backend), load_s, frames, audio_s, first_audio_s, total_s, total_s / audio_s);
    std::printf("per frame: talker %.1f ms, code predictor %.1f ms, codec %.1f ms (prompt %.1f ms once)\n",
                1000 * stats.talker / frames, 1000 * stats.code_predictor / frames, 1000 * stats.codec / frames,
                1000 * stats.prompt);
    write_wav(a[6], pcm, synth.sample_rate());
    ggml_backend_free(backend);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        return run(utf8_args(argc, argv));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
