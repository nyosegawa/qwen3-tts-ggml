// Checks the text stage of Irodori-TTS against the official implementation: the normalization and the
// tokenizer on the cases of reference/irodori-tts/text_cases.py, then the text encoder, layer by layer, on
// the tokens of every dump that has them (dump.py's, and text_cases.py's texts longer than the local
// window). The text encoder is given the dump's tokens, so its error is its own.
//
// usage: irodori-text-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "backend.h"
#include "compare.h"
#include "irodori-tts/text-encoder.h"
#include "irodori-tts/text-normalizer.h"
#include "irodori-tts/tokenizer.h"
#include "npy.h"

using namespace irodori;

namespace {

std::string from_hex(const std::string & hex) {
    std::string s;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) s += (char) std::stoi(hex.substr(i, 2), nullptr, 16);
    return s;
}

/** The text cases that fail the normalization or the tokenizer, of all the cases in `path`. */
int check_cases(const Tokenizer & tokenizer, const std::string & path, int & total) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string line;
    int failed = 0;
    while (std::getline(f, line)) {
        const size_t a = line.find('\t'), b = line.find('\t', a + 1);
        const std::string text = from_hex(line.substr(0, a)), want_text = from_hex(line.substr(a + 1, b - a - 1));
        std::vector<int32_t> want;
        for (size_t i = b + 1; i < line.size();) {
            size_t j = line.find(' ', i);
            if (j == std::string::npos) j = line.size();
            if (j > i) want.push_back(std::stoi(line.substr(i, j - i)));
            i = j + 1;
        }
        total++;
        const std::string got_text = normalize_text(text);
        const std::vector<int32_t> got = got_text.empty() ? std::vector<int32_t>{} : tokenizer.encode(got_text);
        if (got_text != want_text || got != want) {
            failed++;
            std::printf("MISMATCH: %s\n  normalized: %s\n  want:       %s\n  ids:", text.c_str(), got_text.c_str(),
                        want_text.c_str());
            for (int32_t id : got) std::printf(" %d", id);
            std::printf("\n  want ids:");
            for (int32_t id : want) std::printf(" %d", id);
            std::printf("\n");
        }
    }
    return failed;
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
        const Tokenizer tokenizer(model);
        const TextEncoder encoder(model);
        const std::filesystem::path dir = argv[2];

        int total = 0;
        const int failed = check_cases(tokenizer, (dir / "text" / "text-cases.tsv").string(), total);
        std::printf("normalization and tokenizer: %d of %d cases match\n", total - failed, total);
        bool ok = failed == 0;

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        std::vector<std::filesystem::path> dumps;
        for (const auto & e : std::filesystem::recursive_directory_iterator(dir)) {
            if (e.is_regular_file() && e.path().filename() == "text_layers.npy") dumps.push_back(e.path().parent_path());
        }
        std::sort(dumps.begin(), dumps.end());
        double worst = INFINITY;
        for (const auto & d : dumps) {
            const Npy ids = read_npy((d / "input_ids.npy").string());
            const Npy layers = read_npy((d / "text_layers.npy").string());
            const Npy state = read_npy((d / "text_state.npy").string());
            Graph g;
            std::vector<ggml_tensor *> hidden;
            ggml_tensor * out = encoder.build(g, ids.i32, &hidden);
            g.output(out);
            for (ggml_tensor * h : hidden) g.output(h);
            g.compute(backend, allocr);
            const size_t n = ids.i32.size(), width = (size_t) layers.shape[2];
            Diff worst_layer;
            int worst_index = 0;
            for (size_t l = 0; l < hidden.size(); l++) {
                const Diff dl = compare(Graph::read(hidden[l]).data(), &layers.f32[l * n * width], n * width);
                if (dl.snr_db < worst_layer.snr_db) {
                    worst_layer = dl;
                    worst_index = (int) l;
                }
            }
            const Diff ds = compare(Graph::read(out), state.f32);
            std::printf("%s (%zu tokens)\n", d.filename().string().c_str(), n);
            print_diff("  worst ModernBERT layer " + std::to_string(worst_index), worst_layer);
            print_diff("  text condition", ds);
            worst = std::min(worst, ds.snr_db);
        }
        ggml_gallocr_free(allocr);
        // Measured on an Apple M5: F32 weights give 118 dB on the CPU and 57 dB on Metal, whose matrix
        // kernels round to half precision; F16 weights 57 dB and Q8_0 weights 37 dB. A wrong operation
        // falls far below this bound.
        if (dumps.empty() || worst < 30) ok = false;
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
