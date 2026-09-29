// A JSON-lines worker with the protocol of ASIST's resources/qwen_tts_worker.py, so that ASIST can run
// it in place of the mlx-audio one.
//
// Reads one JSON object per line from stdin and answers on stdout, each line prefixed with `ASIST_JSON:`.
//   in : {"id": "...", "text": "...", "voice": "ono_anna", "language": "japanese", "speed": 1.0}
//        {"type": "cancel", "id": "..."}
//   out: {"type": "ready", "sampleRate": 24000, "voices": [...], "languages": [...]}
//        {"type": "chunk", "id": "...", "seq": 0, "pcm": "<base64 int16le mono>"}
//        {"type": "end", "id": "...", "samples": 123456}
//        {"type": "error", "id": "...", "error": "..."}
//        {"type": "fatal", "error": "..."}
//
// Requests are served one at a time in arrival order. A cancel takes effect between two chunks, and a
// request cancelled before it starts is dropped. `speed` is accepted and has no effect, as with
// mlx-audio, whose Qwen3-TTS does not support it either.
//
// `--devices` instead prints the devices ggml can run on and exits, so that the caller can tell whether
// the machine has a GPU and how much memory it has before starting a worker:
//   {"type": "devices", "devices": [{"name": "Vulkan0", "description": "NVIDIA GeForce RTX 2080",
//                                    "kind": "gpu", "memoryTotal": 8589934592, "memoryFree": 7516192768}]}
//
// usage: qwen3-tts-worker <talker.gguf> <codec.gguf> [--backend gpu|cpu] [--ctx n] [--seed n]
//        qwen3-tts-worker --devices

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "backend.h"
#include "flat-json.h"
#include "synthesizer.h"

namespace {

std::mutex out_mutex;

void emit(const std::string & json) {
    std::lock_guard<std::mutex> lock(out_mutex);
    std::fwrite("ASIST_JSON:", 1, 11, stdout);
    std::fwrite(json.data(), 1, json.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

std::string base64(const uint8_t * data, size_t n) {
    static const char * table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t) data[i] << 16 | (i + 1 < n ? (uint32_t) data[i + 1] << 8 : 0) | (i + 2 < n ? data[i + 2] : 0);
        out += table[(v >> 18) & 63];
        out += table[(v >> 12) & 63];
        out += i + 1 < n ? table[(v >> 6) & 63] : '=';
        out += i + 2 < n ? table[v & 63] : '=';
    }
    return out;
}

std::string json_array(const std::vector<std::string> & items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); i++) out += (i ? "," : "") + json_string(items[i]);
    return out + "]";
}

struct Inbox {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<FlatJson> requests;
    std::set<std::string> cancelled;
    bool closed = false;

    bool is_cancelled(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        return cancelled.count(id) > 0;
    }
    void forget(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled.erase(id);
    }
};

void read_requests(Inbox & inbox) {
    std::string line;
    while (std::getline(std::cin, line)) {
        FlatJson message;
        try {
            message = parse_flat_json(line);
        } catch (const std::exception &) {
            continue;
        }
        std::lock_guard<std::mutex> lock(inbox.mutex);
        if (message["type"] == "cancel") {
            inbox.cancelled.insert(message["id"]);
        } else {
            inbox.requests.push_back(message);
            inbox.ready.notify_one();
        }
    }
    std::lock_guard<std::mutex> lock(inbox.mutex);
    inbox.closed = true;
    inbox.ready.notify_one();
}

}  // namespace

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    quiet_ggml_logs();
    if (argc == 2 && !std::strcmp(argv[1], "--devices")) {
        std::string list;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            const char * kind = type == GGML_BACKEND_DEVICE_TYPE_GPU ? "gpu" : type == GGML_BACKEND_DEVICE_TYPE_IGPU ? "igpu" : "cpu";
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            list += (i ? "," : "") + std::string("{\"name\":") + json_string(ggml_backend_dev_name(dev)) +
                    ",\"description\":" + json_string(ggml_backend_dev_description(dev)) + ",\"kind\":\"" + kind +
                    "\",\"memoryTotal\":" + std::to_string(total) + ",\"memoryFree\":" + std::to_string(free) + "}";
        }
        emit("{\"type\":\"devices\",\"devices\":[" + list + "]}");
        return 0;
    }
    if (argc < 3) {
        emit("{\"type\":\"fatal\",\"error\":\"expected the talker and codec GGUF paths\"}");
        return 2;
    }
    std::string backend_name;
    // 2048 positions hold a prompt and about 160 s of speech; the talker's cache is 0.24 GB at this size.
    int n_ctx = 2048;
    uint64_t seed = std::random_device{}();
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--backend")) backend_name = argv[i + 1];
        else if (!std::strcmp(argv[i], "--ctx")) n_ctx = std::stoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--seed")) seed = std::stoull(argv[i + 1]);
    }

    ggml_backend_t backend = nullptr;
    std::unique_ptr<Synthesizer> synth;
    std::vector<std::string> voices, languages;
    try {
        backend = init_backend(backend_name);
        synth = std::make_unique<Synthesizer>(argv[1], argv[2], backend, n_ctx);
        voices = synth->ids().speaker_names;
        languages = synth->ids().language_names;
        std::sort(voices.begin(), voices.end());
        std::sort(languages.begin(), languages.end());
        if (voices.empty()) throw std::runtime_error("the model has no preset voices");
        // The first synthesis compiles the GPU kernels, which on Vulkan takes seconds for every new shape, so
        // a short and a longer text run through the prompt, the talker, the code predictor and the codec at
        // the sizes speech uses before "ready".
        for (const auto & [text, frames] : std::vector<std::pair<std::string, int>>{
                 {"あ", 4}, {"明日の東京は晴れで、最高気温は二十四度の予報です。", 40}}) {
            SynthesisRequest warmup;
            warmup.text = text;
            warmup.speaker = voices[0];
            warmup.max_frames = frames;
            synth->synthesize(warmup, [](const float *, size_t) { return true; });
        }
    } catch (const std::exception & e) {
        emit("{\"type\":\"fatal\",\"error\":" + json_string(e.what()) + "}");
        return 1;
    }
    emit("{\"type\":\"ready\",\"sampleRate\":" + std::to_string(synth->sample_rate()) + ",\"voices\":" + json_array(voices) +
         ",\"languages\":" + json_array(languages) + ",\"backend\":" + json_string(ggml_backend_name(backend)) + "}");

    Inbox inbox;
    std::thread reader(read_requests, std::ref(inbox));
    reader.detach();

    for (;;) {
        FlatJson request;
        {
            std::unique_lock<std::mutex> lock(inbox.mutex);
            inbox.ready.wait(lock, [&] { return !inbox.requests.empty() || inbox.closed; });
            if (inbox.requests.empty()) break;
            request = inbox.requests.front();
            inbox.requests.pop_front();
        }
        const std::string id = request["id"];
        try {
            if (inbox.is_cancelled(id)) {
                inbox.forget(id);
                continue;
            }
            SynthesisRequest r;
            r.text = request["text"];
            r.speaker = request["voice"];
            r.language = request.count("language") && !request["language"].empty() ? request["language"] : "auto";
            r.seed = seed++;
            size_t samples = 0;
            int seq = 0;
            bool cancelled = false;
            std::vector<int16_t> pcm;
            synth->synthesize(r, [&](const float * s, size_t n) {
                if (inbox.is_cancelled(id)) {
                    cancelled = true;
                    return false;
                }
                pcm.resize(n);
                for (size_t i = 0; i < n; i++) pcm[i] = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
                emit("{\"type\":\"chunk\",\"id\":" + json_string(id) + ",\"seq\":" + std::to_string(seq++) + ",\"pcm\":\"" +
                     base64((const uint8_t *) pcm.data(), n * sizeof(int16_t)) + "\"}");
                samples += n;
                return true;
            });
            if (!cancelled) emit("{\"type\":\"end\",\"id\":" + json_string(id) + ",\"samples\":" + std::to_string(samples) + "}");
        } catch (const std::exception & e) {
            emit("{\"type\":\"error\",\"id\":" + json_string(id) + ",\"error\":" + json_string(e.what()) + "}");
        }
        inbox.forget(id);
    }
    synth.reset();
    ggml_backend_free(backend);
    return 0;
}
