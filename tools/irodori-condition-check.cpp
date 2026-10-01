// Checks the speaker encoder and the duration predictor of Irodori-TTS against the official implementation
// on every dump of reference/irodori-tts/dump.py, each stage from the dump's own inputs: the reference latent
// for the speaker encoder, and the text and speaker conditions for the duration predictor.
//
// usage: irodori-condition-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "backend.h"
#include "compare.h"
#include "duration.h"
#include "npy.h"
#include "speaker-encoder.h"

using namespace irodori;

namespace {

/** An integer member of the dump's meta.json. */
int meta_int(const std::filesystem::path & dir, const std::string & key) {
    std::ifstream f(dir / "meta.json");
    const std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t at = json.find("\"" + key + "\"");
    if (at == std::string::npos) throw std::runtime_error(key + " is missing from " + (dir / "meta.json").string());
    return std::stoi(json.substr(json.find(':', at) + 1));
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", argv[0]);
        return 2;
    }
    try {
        ggml_backend_t backend = init_backend(argc > 3 ? argv[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        ModelFile model(argv[1], backend);
        const SpeakerEncoder speaker(model);
        // The codec of every Irodori-TTS checkpoint: 48 kHz audio, 1920 samples a frame.
        const DurationPredictor duration(model, 48000, 1920);
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        std::vector<std::filesystem::path> dumps;
        for (const auto & e : std::filesystem::directory_iterator(argv[2])) {
            if (std::filesystem::exists(e.path() / "speaker_state.npy")) dumps.push_back(e.path());
        }
        std::sort(dumps.begin(), dumps.end());
        bool ok = !dumps.empty();
        for (const auto & d : dumps) {
            std::printf("%s\n", d.filename().string().c_str());
            const Npy latent = read_npy((d / "ref_latent.npy").string());
            const Npy encoded_ref = read_npy((d / "speaker_encoded.npy").string());
            const Npy state_ref = read_npy((d / "speaker_state.npy").string());
            {
                Graph g;
                ggml_tensor * encoded = nullptr;
                ggml_tensor * state = speaker.build(g, latent.f32, &encoded);
                g.output(encoded);
                g.output(state);
                g.compute(backend, allocr);
                print_diff("  speaker encoder", compare(Graph::read(encoded), encoded_ref.f32));
                const Diff ds = compare(Graph::read(state), state_ref.f32);
                print_diff("  speaker condition", ds);
                // Measured on an Apple M5: 111 dB on the CPU in F32, 47 dB on Metal, 50 dB with F16 weights
                // and 25 dB with Q8_0. A wrong operation falls far below this bound.
                ok = ok && ds.snr_db > 20;
            }
            const Npy text = read_npy((d / "text_state.npy").string());
            const Npy log_frames = read_npy((d / "duration_log_frames.npy").string());
            Graph g;
            ggml_tensor * text_state = g.input(text.f32, text.shape[1], text.shape[0]);
            const std::vector<float> summary(state_ref.f32.begin(), state_ref.f32.begin() + state_ref.shape[1]);
            ggml_tensor * sum = duration.build(g, text_state, g.input(summary, (int64_t) summary.size()));
            g.output(sum);
            g.compute(backend, allocr);
            const float predicted = Graph::read(sum)[0];
            const int frames = duration.frames(predicted), want = meta_int(d, "latent_frames");
            std::printf("  duration: log(1 + frames) %.6f (official %.6f), %d frames (official %d)\n", std::log1p(predicted),
                        log_frames.f32[0], frames, want);
            // F32 and F16 give the official frames; Q8_0 weights move a 27 s text by one frame (40 ms).
            ok = ok && std::abs(frames - want) <= 1;
        }
        ggml_gallocr_free(allocr);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
