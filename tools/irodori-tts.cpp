// Speaks a text with Irodori-TTS into a WAVE file and reports where the time went, or makes a voice file
// from a reference recording.
//
// usage: irodori-tts <model.gguf> <codec.gguf> <voice> <text> <out.wav> [--device NAME] [--seed n] [--steps n]
//        irodori-tts --make-voice <model.gguf> <codec.gguf> <reference.wav> <voice.gguf> [--device NAME]
//
// A voice is a reference WAVE file (48 kHz, at most 120 s) or a voice file that --make-voice wrote.

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "irodori-tts/synthesizer.h"
#include "wav.h"

using namespace irodori;

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int run(std::vector<std::string> a) {
    std::string device;
    uint64_t seed = 0;
    int steps = 0;
    std::vector<std::string> positional;
    for (size_t i = 1; i < a.size(); i++) {
        if (a[i] == "--device" && i + 1 < a.size()) device = a[++i];
        else if (a[i] == "--seed" && i + 1 < a.size()) seed = std::stoull(a[++i]);
        else if (a[i] == "--steps" && i + 1 < a.size()) steps = std::stoi(a[++i]);
        else positional.push_back(a[i]);
    }
    const bool make_voice = !positional.empty() && positional[0] == "--make-voice";
    if (positional.size() != 5) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <codec.gguf> <voice> <text> <out.wav> [--device NAME] [--seed n] [--steps n]\n"
                     "       %s --make-voice <model.gguf> <codec.gguf> <reference.wav> <voice.gguf> [--device NAME]\n",
                     a[0].c_str(), a[0].c_str());
        return 2;
    }
    const size_t p = make_voice ? 1 : 0;
    ggml_backend_t backend = init_backend(device);
    auto t0 = std::chrono::steady_clock::now();
    Synthesizer synth(positional[p], positional[p + 1], backend);
    std::printf("backend %s, load %.2f s\n", ggml_backend_name(backend), seconds_since(t0));
    t0 = std::chrono::steady_clock::now();
    const Voice voice = synth.load_voice(positional[p + 2]);
    std::printf("voice: %d frames (%.2f s) in %.3f s\n", voice.frames, (double) voice.frames * synth.codec().hop() / synth.sample_rate(),
                seconds_since(t0));
    if (make_voice) {
        synth.save_voice(voice, positional[4]);
        std::printf("wrote %s\n", positional[4].c_str());
        ggml_backend_free(backend);
        return 0;
    }

    Request r;
    r.text = positional[3];
    r.seed = seed;
    r.steps = steps;
    std::vector<float> audio;
    Stats stats;
    t0 = std::chrono::steady_clock::now();
    synth.synthesize(r, voice, [&](const float * s, size_t n) {
        audio.insert(audio.end(), s, s + n);
        return true;
    }, &stats);
    const double total = seconds_since(t0), seconds = (double) audio.size() / synth.sample_rate();
    std::printf("%d tokens, %d frames, %.2f s of audio: first audio %.3f s, total %.3f s, RTF %.3f\n", stats.tokens, stats.frames,
                seconds, stats.first_audio, total, total / seconds);
    std::printf("text and duration %.3f s, sampling %.3f s, codec %.3f s\n", stats.text, stats.sampling, stats.codec);
    write_wav(positional[4], audio, synth.sample_rate());
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
