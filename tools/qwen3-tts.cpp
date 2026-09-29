// Speaks a text into a WAV file and reports how long the first audio and the whole took.
//
// usage: qwen3-tts <talker.gguf> <codec.gguf> <speaker> <language> <text> <out.wav> [gpu|cpu] [seed] [--greedy]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "backend.h"
#include "synthesizer.h"

namespace {

void write_wav(const std::string & path, const std::vector<float> & pcm, int rate) {
    std::ofstream f(path, std::ios::binary);
    const uint32_t data_size = (uint32_t) pcm.size() * 2;
    auto u32 = [&](uint32_t v) { f.write((const char *) &v, 4); };
    auto u16 = [&](uint16_t v) { f.write((const char *) &v, 2); };
    f.write("RIFF", 4); u32(36 + data_size); f.write("WAVEfmt ", 8);
    u32(16); u16(1); u16(1); u32(rate); u32(rate * 2); u16(2); u16(16);
    f.write("data", 4); u32(data_size);
    for (float s : pcm) {
        const int16_t v = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s)) * 32767.0f);
        f.write((const char *) &v, 2);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s <talker.gguf> <codec.gguf> <speaker> <language> <text> <out.wav> [gpu|cpu] [seed] [--greedy]\n", argv[0]);
        return 2;
    }
    ggml_backend_t backend = init_backend(argc > 7 ? argv[7] : "");
    auto t_load = std::chrono::steady_clock::now();
    Synthesizer synth(argv[1], argv[2], backend, 4096);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();

    SynthesisRequest r;
    r.speaker = argv[3];
    r.language = argv[4];
    r.text = argv[5];
    r.seed = argc > 8 ? std::stoull(argv[8]) : 0;
    if (argc > 9 && std::strcmp(argv[9], "--greedy") == 0) {
        r.talker.greedy = true;
        r.code_predictor.greedy = true;
    }

    std::vector<float> pcm;
    double first_audio_s = -1;
    const auto t0 = std::chrono::steady_clock::now();
    const int frames = synth.synthesize(r, [&](const float * s, size_t n) {
        if (first_audio_s < 0) first_audio_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        pcm.insert(pcm.end(), s, s + n);
        return true;
    });
    const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double audio_s = (double) pcm.size() / synth.sample_rate();
    std::printf("backend %s, load %.2f s, %d frames, %.2f s of audio, first audio %.3f s, total %.3f s, RTF %.3f\n",
                ggml_backend_name(backend), load_s, frames, audio_s, first_audio_s, total_s, total_s / audio_s);
    write_wav(argv[6], pcm, synth.sample_rate());
    ggml_backend_free(backend);
    return 0;
}
