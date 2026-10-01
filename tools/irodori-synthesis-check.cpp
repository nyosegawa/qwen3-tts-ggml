// Checks the whole synthesis of Irodori-TTS against the official runtime: for every dump of
// reference/irodori-tts/dump.py made with the same model, the dump's text spoken in the voice of the dump's
// reference latent from the dump's noise, against the audio the official synthesize() returned, the cut
// at the tail included.
//
// usage: irodori-synthesis-check <model.gguf> <codec.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "backend.h"
#include "compare.h"
#include "flat-json.h"
#include "irodori-tts/synthesizer.h"
#include "npy.h"

using namespace irodori;

namespace {

/** A string member of the dump's meta.json, and the object member `outer` before it when given. */
std::string meta_string(const std::filesystem::path & dir, const std::string & key, const std::string & outer = "") {
    std::ifstream f(dir / "meta.json");
    const std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t at = json.find("\"" + key + "\"", outer.empty() ? 0 : json.find("\"" + outer + "\""));
    if (at == std::string::npos) throw std::runtime_error(key + " is missing from " + (dir / "meta.json").string());
    at = json.find('"', json.find(':', at) + 1);
    return flat_json::parse_string(json, at);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <codec.gguf> <reference out dir> [gpu|cpu|device name]\n", argv[0]);
        return 2;
    }
    try {
        ggml_backend_t backend = init_backend(argc > 4 ? argv[4] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        Synthesizer synth(argv[1], argv[2], backend);
        const std::string source = synth.model().str("general.source.url");
        std::vector<std::filesystem::path> dumps;
        for (const auto & e : std::filesystem::directory_iterator(argv[3])) {
            if (std::filesystem::exists(e.path() / "audio.npy") &&
                source.find("huggingface.co/" + meta_string(e.path(), "repository", "model") + "/tree/") != std::string::npos) {
                dumps.push_back(e.path());
            }
        }
        std::sort(dumps.begin(), dumps.end());
        bool ok = !dumps.empty();
        for (const auto & d : dumps) {
            const Voice voice = synth.voice_from_latent(read_npy((d / "ref_latent.npy").string()).f32);
            const Npy official = read_npy((d / "audio.npy").string());
            Request r;
            r.text = meta_string(d, "text");
            r.noise = read_npy((d / "noise.npy").string()).f32;
            std::vector<float> audio;
            Stats stats;
            try {
                synth.synthesize(r, voice, [&](const float * s, size_t n) {
                    audio.insert(audio.end(), s, s + n);
                    return true;
                }, &stats);
            } catch (const std::exception & e) {
                // Q8_0 weights can move the predicted length by a frame, and the dump's noise then does not fit.
                std::printf("%s: %s\n", d.filename().string().c_str(), e.what());
                ok = false;
                continue;
            }
            const Diff da = compare(audio, official.f32);
            std::printf("%s: %zu samples (official %zu), %d frames\n", d.filename().string().c_str(), audio.size(), official.f32.size(),
                        stats.frames);
            print_diff("  audio against the official", da);
            std::printf("  first audio %.3f s: text and duration %.3f s, sampling %.3f s, codec %.3f s in all\n", stats.first_audio,
                        stats.text, stats.sampling, stats.codec);
            // Measured on an Apple M5: 75 to 110 dB on the CPU in F32. On Metal the sampler carries the
            // half-precision rounding of Metal's matrix kernel into the latent, and the audio lies 22 to 41 dB
            // from the official (11 to 29 dB with Q8_0 weights): the same speech, not the same waveform. A
            // wrong stage gives a few dB.
            ok = ok && audio.size() == official.f32.size() && da.snr_db > 10;
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
