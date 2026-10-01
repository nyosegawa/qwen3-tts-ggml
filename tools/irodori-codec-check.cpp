// Checks the codec of Irodori-TTS against the official implementation on a dump of
// reference/irodori-tts/dump.py: reading the reference WAVE file, normalizing its loudness, and encoding it,
// each from the dump's own input; then encoding in windows against encoding at once, and the whole path
// from the file.
//
// usage: irodori-codec-check <codec.gguf> <dump dir> <reference.wav> [gpu|cpu|device name]

#include <chrono>
#include <cstdio>
#include <string>

#include "backend.h"
#include "codec.h"
#include "compare.h"
#include "loudness.h"
#include "npy.h"
#include "reference.h"
#include "wav.h"

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <codec.gguf> <dump dir> <reference.wav> [gpu|cpu|device name]\n", argv[0]);
        return 2;
    }
    try {
        const std::string dir = argv[2];
        ggml_backend_t backend = init_backend(argc > 4 ? argv[4] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        Codec codec(argv[1], backend);
        const Npy ref_wav = read_npy(dir + "/ref_wav.npy");
        const Npy ref_normalized = read_npy(dir + "/ref_wav_normalized.npy");
        const Npy ref_latent = read_npy(dir + "/ref_latent.npy");
        bool ok = true;

        const std::vector<float> read = read_wav(argv[3]).mono();
        const Diff dr = compare(read, ref_wav.f32);
        print_diff("reading the WAVE file", dr);
        ok = ok && read.size() == ref_wav.f32.size() && dr.max_abs == 0;

        const std::vector<float> normalized = normalize_loudness(ref_wav.f32, codec.sample_rate(), kReferenceLufs);
        const Diff dn = compare(normalized, ref_normalized.f32);
        print_diff("loudness normalization", dn);
        ok = ok && dn.snr_db > 80;

        auto t0 = std::chrono::steady_clock::now();
        const std::vector<float> windowed = codec.encode(ref_normalized.f32);
        const double windowed_s = seconds_since(t0);
        const Diff de = compare(windowed, ref_latent.f32);
        print_diff("encoder, in windows of 100 frames", de);
        const int64_t frames = (int64_t) ref_latent.shape[0];
        t0 = std::chrono::steady_clock::now();
        const std::vector<float> whole = codec.encode(ref_normalized.f32, (int) frames);
        const double whole_s = seconds_since(t0);
        print_diff("encoder, at once", compare(whole, ref_latent.f32));
        const Diff dw = compare(windowed, whole);
        print_diff("in windows against at once", dw);
        std::printf("encoding %.2f s of audio: %.3f s in windows, %.3f s at once\n",
                    (double) ref_normalized.f32.size() / codec.sample_rate(), windowed_s, whole_s);
        // Measured on an Apple M5: 99 dB on the CPU in F32, 40 dB on Metal, whose matrix kernel rounds its
        // inputs to half precision, and 33 dB with F16 weights on the CPU. In windows and at once agree to
        // the bit on both. A frame that a window's edge reaches would differ by far more.
        ok = ok && windowed.size() == ref_latent.f32.size() && de.snr_db > 30 && dw.snr_db > 60;

        t0 = std::chrono::steady_clock::now();
        const std::vector<float> from_file = encode_reference(codec, argv[3], 120.0);
        std::printf("reference file to latent: %.3f s\n", seconds_since(t0));
        const Diff df = compare(from_file, ref_latent.f32);
        print_diff("reference file to latent", df);
        ok = ok && df.snr_db > 30;

        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
