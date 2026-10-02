// Compares the tokenizer with the ids of the model's own tokenizer.json (reference/tokenizer_cases.py).
//
// usage: tokenizer-check <talker.gguf> <cases.tsv>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "args.h"
#include "backend.h"
#include "qwen3-tts/tokenizer.h"

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <talker.gguf> <cases.tsv>\n", args[0].c_str());
        return 2;
    }
    ggml_backend_t backend = init_backend("cpu");
    ModelFile model(args[1], backend);
    Tokenizer tokenizer(model);

    std::ifstream f(std::filesystem::u8path(args[2]));
    std::string line;
    int total = 0, failed = 0;
    while (std::getline(f, line)) {
        const size_t tab = line.find('\t');
        const std::string hex = line.substr(0, tab);
        std::string text;
        for (size_t i = 0; i + 1 < hex.size(); i += 2) text += (char) std::stoi(hex.substr(i, 2), nullptr, 16);
        std::vector<int32_t> want;
        std::istringstream ids(line.substr(tab + 1));
        for (int32_t id; ids >> id;) want.push_back(id);

        const std::vector<int32_t> got = tokenizer.encode(text);
        total++;
        if (got != want) {
            failed++;
            std::printf("MISMATCH: %s\n  want:", text.c_str());
            for (int32_t id : want) std::printf(" %d", id);
            std::printf("\n  got: ");
            for (int32_t id : got) std::printf(" %d", id);
            std::printf("\n  pieces:");
            for (const std::string & p : tokenizer.pre_tokenize(text)) std::printf(" [%s]", p.c_str());
            std::printf("\n");
        }
    }
    std::printf("%d of %d cases match\n", total - failed, total);
    ggml_backend_free(backend);
    return failed == 0 ? 0 : 1;
}
