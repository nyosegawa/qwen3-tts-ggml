// Checks the DiT and the sampler of Irodori-TTS against the official implementation on the dumps of
// reference/irodori-tts/dump.py made with the same model (MF or RF): the timestep condition and every block
// of the first step, then every step's velocity (after guidance for RF) from the dump's own latent before
// it, then the whole sampler from the dump's noise.
//
// usage: irodori-dit-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "backend.h"
#include "compare.h"
#include "npy.h"
#include "sampler.h"

namespace {

/** The dump's model repository, from its meta.json. */
std::string meta_model(const std::filesystem::path & dir) {
    std::ifstream f(dir / "meta.json");
    const std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t at = json.find("\"repository\"", json.find("\"model\""));
    const size_t a = json.find('"', json.find(':', at) + 1), b = json.find('"', a + 1);
    return json.substr(a + 1, b - a - 1);
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
        const Dit dit(model);
        Sampler sampler(dit, model, backend);
        const std::string source = model.str("general.source.url");
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        std::vector<std::filesystem::path> dumps;
        for (const auto & e : std::filesystem::directory_iterator(argv[2])) {
            if (std::filesystem::exists(e.path() / "dit_velocity.npy") &&
                source.find("huggingface.co/" + meta_model(e.path()) + "/tree/") != std::string::npos) {
                dumps.push_back(e.path());
            }
        }
        std::sort(dumps.begin(), dumps.end());
        bool ok = !dumps.empty();
        for (const auto & d : dumps) {
            auto npy = [&](const char * f) { return read_npy((d / f).string()); };
            const Npy text = npy("text_state.npy"), speaker = npy("speaker_state.npy"), noise = npy("noise.npy");
            const Npy times = npy("dit_t.npy"), velocity = npy("dit_velocity.npy"), xs = npy("dit_x.npy");
            const Npy cond = npy("dit_cond.npy"), blocks = npy("dit_step0_blocks.npy");
            const Conditions c{text.f32, (int) text.shape[0], speaker.f32, (int) speaker.shape[0]};
            const int frames = (int) noise.shape[0], steps = (int) times.shape[0];
            const size_t n = (size_t) frames * dit.latent_dim();
            std::printf("%s (%d frames, %d steps)\n", d.filename().string().c_str(), frames, steps);

            const std::vector<float> schedule = sampler.schedule(steps);
            double schedule_error = 0;
            for (int i = 0; i < steps; i++) schedule_error = std::max(schedule_error, (double) std::fabs(schedule[i] - times.f32[i]));
            std::printf("  schedule: largest difference %.1e\n", schedule_error);

            const int branches = dit.meanflow() ? 1 : 3;
            const float t0 = times.f32[0], t1 = schedule[1];
            Graph g;
            std::vector<ggml_tensor *> hidden;
            ggml_tensor * cond_out = nullptr;
            g.output(dit.build(g, noise.f32, frames, t0, t0 - t1, c, branches, &hidden, &cond_out));
            g.output(cond_out);
            for (ggml_tensor * h : hidden) g.output(h);
            g.compute(backend, allocr);
            print_diff("  timestep condition, step 0", compare(Graph::read(cond_out).data(), cond.f32.data(), (size_t) cond.shape[1]));
            Diff worst_block;
            const size_t block_size = blocks.f32.size() / hidden.size();
            for (size_t l = 0; l < hidden.size(); l++) {
                const Diff db = compare(Graph::read(hidden[l]).data(), &blocks.f32[l * block_size], block_size);
                if (db.snr_db < worst_block.snr_db) worst_block = db;
            }
            print_diff("  worst block of step 0", worst_block);

            Diff worst_step;
            int worst_index = 0;
            for (int i = 0; i < steps; i++) {
                const std::vector<float> x(i == 0 ? noise.f32 : std::vector<float>(xs.f32.begin() + (i - 1) * n, xs.f32.begin() + i * n));
                const std::vector<float> v = sampler.velocity(c, x, frames, times.f32[i], i + 1 < steps ? times.f32[i + 1] : 0.0f);
                const Diff dv = compare(v.data(), &velocity.f32[i * n], n);
                if (dv.snr_db < worst_step.snr_db) {
                    worst_step = dv;
                    worst_index = i;
                }
            }
            print_diff("  worst step, step " + std::to_string(worst_index), worst_step);

            const auto start = std::chrono::steady_clock::now();
            const std::vector<float> final_x = sampler.sample(c, noise.f32, frames, steps);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const Diff df = compare(final_x.data(), &xs.f32[(steps - 1) * n], n);
            print_diff("  sampled latent", df);
            std::printf("  sampling took %.3f s\n", seconds);
            // Measured on an Apple M5 for the MF dumps: 86 to 122 dB on the CPU in F32 and 33 to 61 dB on Metal,
            // whose matrix kernel rounds its inputs to half precision, which four large steps amplify; 21 to
            // 41 dB with Q8_0 weights. RF reaches 109 dB on the CPU and 50 dB on Metal. A wrong operation
            // gives a few dB.
            ok = ok && df.snr_db > 15;
        }
        ggml_gallocr_free(allocr);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
