// Checks the codec of Irodori-TTS against the official implementation on a dump of
// reference/irodori-tts/dump.py, each stage from the dump's own input.
//
// Encoder: reading the reference WAVE file, normalizing its loudness, encoding it, encoding in windows
// against at once, and the whole path from the file.
// Decoder: each stage on the dump's latent, decoding in windows against at once, and, when the device is
// not the CPU, the same decoding on the CPU: how far below the voice the device's error lies, and how loud
// the quietest parts are on each, where a device that mishandles the decoder shows its error first.
//
// usage: irodori-codec-check <codec.gguf> <dump dir> <reference.wav> [gpu|cpu|device name]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>

#include "backend.h"
#include "codec.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "loudness.h"
#include "npy.h"
#include "reference.h"
#include "wav.h"

using namespace irodori;

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int meta_int(const std::string & dir, const std::string & key) {
    std::ifstream f(dir + "/meta.json");
    const std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t at = json.find("\"" + key + "\"");
    if (at == std::string::npos) throw std::runtime_error(key + " is missing from " + dir + "/meta.json");
    return std::stoi(json.substr(json.find(':', at) + 1));
}

/** A [rows, cols] row-major array as [cols, rows]: the dump's [channels, time] in ggml's channel-first order. */
std::vector<float> transpose(const Npy & a) {
    const size_t rows = (size_t) a.shape[0], cols = (size_t) a.shape[1];
    std::vector<float> out(a.f32.size());
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++) out[c * rows + r] = a.f32[r * cols + c];
    return out;
}

double dbfs(double power) { return 10 * std::log10(std::max(power, 1e-20)); }

/** The mean power, in dBFS, of the 20 ms frames of `signal` at `frames`. */
double level(const std::vector<float> & signal, const std::vector<size_t> & frames, size_t width) {
    double sum = 0;
    for (size_t f : frames)
        for (size_t i = f * width; i < (f + 1) * width; i++) sum += (double) signal[i] * signal[i];
    return dbfs(sum / (double) (frames.size() * width));
}

bool check_encoder(Codec & codec, const std::string & dir, const std::string & wav_path) {
    const Npy ref_wav = read_npy(dir + "/ref_wav.npy");
    const Npy ref_normalized = read_npy(dir + "/ref_wav_normalized.npy");
    const Npy ref_latent = read_npy(dir + "/ref_latent.npy");
    bool ok = true;

    const std::vector<float> read = read_wav(wav_path).mono();
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
    print_diff("encoder in windows against at once", dw);
    std::printf("encoding %.2f s of audio: %.3f s in windows, %.3f s at once\n",
                (double) ref_normalized.f32.size() / codec.sample_rate(), windowed_s, whole_s);
    // Measured on an Apple M5: 99 dB on the CPU in F32, 40 dB on Metal, whose matrix kernel rounds its
    // inputs to half precision, and 33 dB with F16 weights on the CPU. In windows and at once agree to
    // the bit on both. A frame that a window's edge reaches would differ by far more.
    ok = ok && windowed.size() == ref_latent.f32.size() && de.snr_db > 30 && dw.snr_db > 60;

    t0 = std::chrono::steady_clock::now();
    const std::vector<float> from_file = encode_reference(codec, wav_path, 120.0);
    std::printf("reference file to latent: %.3f s\n", seconds_since(t0));
    const Diff df = compare(from_file, ref_latent.f32);
    print_diff("reference file to latent", df);
    return ok && df.snr_db > 30;
}

/** The decoder's output for the whole latent at once. */
std::vector<float> decode_whole(Codec & codec, ggml_backend_t backend, const std::vector<float> & latent,
                                std::vector<std::vector<float>> * stages = nullptr) {
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    Graph g;
    std::vector<ggml_tensor *> outputs;
    ggml_tensor * out = codec.build_decoder(g, latent, stages ? &outputs : nullptr);
    g.output(out);
    for (ggml_tensor * t : outputs) g.output(t);
    g.compute(backend, allocr);
    if (stages)
        for (ggml_tensor * t : outputs) stages->push_back(Graph::read(t));
    std::vector<float> audio = Graph::read(out);
    ggml_gallocr_free(allocr);
    return audio;
}

bool check_decoder(Codec & codec, ggml_backend_t backend, const std::string & codec_path, const std::string & dir) {
    const Npy xs = read_npy(dir + "/dit_x.npy");
    const int frames = meta_int(dir, "latent_frames");
    const size_t n = (size_t) frames * codec.latent_dim();
    const std::vector<float> latent(xs.f32.end() - (ptrdiff_t) (xs.shape[1] * xs.shape[2]),
                                    xs.f32.end() - (ptrdiff_t) (xs.shape[1] * xs.shape[2]) + (ptrdiff_t) n);
    const Npy official = read_npy(dir + "/wav.npy");
    bool ok = true;

    std::vector<std::vector<float>> stages;
    auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> whole = decode_whole(codec, backend, latent, &stages);
    const double whole_s = seconds_since(t0);
    const char * names[] = {"codec_in", "codec_conv_in", "codec_block0", "codec_block1", "codec_block2", "codec_block3"};
    for (size_t i = 0; i < stages.size(); i++) {
        print_diff(std::string("decoder ") + names[i], compare(stages[i], transpose(read_npy(dir + "/" + names[i] + ".npy"))));
    }
    const Diff dw = compare(whole, official.f32);
    print_diff("decoded audio against the official", dw);
    // Measured on an Apple M5: 119 dB on the CPU in F32 and 47 dB on Metal.
    ok = ok && whole.size() == official.f32.size() && dw.snr_db > 30;

    std::vector<float> windowed;
    double first_s = -1;
    t0 = std::chrono::steady_clock::now();
    codec.decode(latent, (int64_t) whole.size(), 12, 48, [&](const float * s, size_t count) {
        if (first_s < 0) first_s = seconds_since(t0);
        windowed.insert(windowed.end(), s, s + count);
        return true;
    });
    const double windowed_s = seconds_since(t0);
    const Diff dv = compare(windowed, whole);
    print_diff("decoder in windows against at once", dv);
    const double audio_s = (double) whole.size() / codec.sample_rate();
    std::printf("decoding %.2f s of audio: first 12 frames in %.3f s, all in windows %.3f s, at once %.3f s\n", audio_s,
                first_s, windowed_s, whole_s);
    ok = ok && windowed.size() == whole.size() && dv.snr_db > 60;

    if (std::string(ggml_backend_name(backend)) != "CPU") {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        Codec cpu_codec(codec_path, cpu);
        const std::vector<float> reference = decode_whole(cpu_codec, cpu, latent);
        const Diff dc = compare(whole, reference);
        std::printf("%s against the CPU: error %.1f dB below the voice, max |diff| %.2e\n", ggml_backend_name(backend), dc.snr_db,
                    dc.max_abs);
        // The quietest tenth of the 20 ms frames, as the CPU decodes them.
        const size_t width = (size_t) codec.sample_rate() / 50, count = reference.size() / width;
        std::vector<std::pair<double, size_t>> levels;
        for (size_t f = 0; f < count; f++) levels.push_back({level(reference, {f}, width), f});
        std::sort(levels.begin(), levels.end());
        std::vector<size_t> quiet;
        for (size_t i = 0; i < std::max<size_t>(1, count / 10); i++) quiet.push_back(levels[i].second);
        std::vector<float> difference(whole.size());
        for (size_t i = 0; i < whole.size(); i++) difference[i] = whole[i] - reference[i];
        std::printf("quietest tenth of the 20 ms frames: CPU %.1f dBFS, %s %.1f dBFS, official %.1f dBFS, their difference %.1f dBFS\n",
                    level(reference, quiet, width), ggml_backend_name(backend), level(whole, quiet, width),
                    level(official.f32, quiet, width), level(difference, quiet, width));
        ggml_backend_free(cpu);
        // Measured on an Apple M5: Metal's error lies 47 dB below the voice and its quietest frames are as
        // quiet as the CPU's (-76 to -78 dBFS). A decoder that adds a distorted copy of the voice, as audio.cpp
        // v0.8.2's Irodori-TTS does on Metal (14 dB below the voice, quiet parts raised from -76 to -60 dBFS),
        // fails both bounds.
        const double raised = level(whole, quiet, width) - level(reference, quiet, width);
        ok = ok && dc.snr_db > 30 && raised < 6;
    }
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <codec.gguf> <dump dir> <reference.wav> [gpu|cpu|device name]\n", argv[0]);
        return 2;
    }
    try {
        ggml_backend_t backend = init_backend(argc > 4 ? argv[4] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        bool ok;
        {
            Codec codec(argv[1], backend);
            ok = check_encoder(codec, argv[2], argv[3]);
            ok = check_decoder(codec, backend, argv[1], argv[2]) && ok;
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
