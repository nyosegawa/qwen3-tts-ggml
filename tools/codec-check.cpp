// Decodes the codes of a reference dump and compares the samples with the official decoder's, then
// decodes them again in pieces and compares those with the whole decode.
//
// usage: codec-check <codec.gguf> <reference dir> [gpu|cpu] [out.wav]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

#include "backend.h"
#include "codec.h"
#include "npy.h"

namespace {

struct Diff {
    double max_abs = 0;
    double snr_db = 0;
};

Diff compare(const std::vector<float> & got, const std::vector<float> & want) {
    Diff d;
    double signal = 0, noise = 0;
    const size_t n = std::min(got.size(), want.size());
    for (size_t i = 0; i < n; i++) {
        const double e = (double) got[i] - want[i];
        d.max_abs = std::max(d.max_abs, std::fabs(e));
        signal += (double) want[i] * want[i];
        noise += e * e;
    }
    d.snr_db = noise == 0 ? INFINITY : 10 * std::log10(signal / noise);
    return d;
}

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

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <codec.gguf> <reference dir> [gpu|cpu] [out.wav]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    ggml_backend_t backend = init_backend(argc > 3 ? argv[3] : "");
    std::printf("backend: %s\n", ggml_backend_name(backend));
    CodecDecoder codec(argv[1], backend);

    const Npy codes = read_npy(dir + "/codes.npy");
    const Npy wav = read_npy(dir + "/wav.npy");
    const int n_frames = (int) codes.shape[0];

    std::vector<float> whole;
    codec.reset();
    auto t0 = std::chrono::steady_clock::now();
    codec.decode(codes.i32.data(), n_frames, whole);
    const double whole_s = seconds_since(t0);
    const Diff dw = compare(whole, wav.f32);
    std::printf("whole: %zu samples (reference %lld) in %.3f s, max |diff| %.2e, SNR %.1f dB\n", whole.size(),
                (long long) wav.size(), whole_s, dw.max_abs, dw.snr_db);

    int status = 0;
    for (int piece : {1, 3}) {
        std::vector<float> pieces;
        codec.reset();
        t0 = std::chrono::steady_clock::now();
        for (int f = 0; f < n_frames; f += piece) {
            codec.decode(codes.i32.data() + (size_t) f * codec.num_quantizers(), std::min(piece, n_frames - f), pieces);
        }
        const double pieces_s = seconds_since(t0);
        const Diff dp = compare(pieces, whole);
        std::printf("in pieces of %d frame(s): %.3f s (%.1f ms per frame), max |diff| vs whole %.2e, SNR %.1f dB\n",
                    piece, pieces_s, 1000 * pieces_s / n_frames * piece / piece, dp.max_abs, dp.snr_db);
        if (pieces.size() != whole.size() || dp.snr_db < 60) status = 1;
    }
    if (whole.size() != (size_t) wav.size() || dw.snr_db < 40) status = 1;
    if (argc > 4) write_wav(argv[4], whole, codec.sample_rate());
    ggml_backend_free(backend);
    return status;
}
