// The JSON-lines worker ASIST runs for its local speech synthesis. It runs Qwen3-TTS or Irodori-TTS, chosen
// by general.architecture of the model's GGUF.
//
// Reads one JSON object per line from stdin and answers on stdout, each line prefixed with `ASIST_JSON:`.
//   in : {"id": "...", "text": "...", "voice": "...", "language": "...", "speed": 1.0}
//        {"type": "cancel", "id": "..."}
//   out: {"type": "ready", "model": "...", "architecture": "...", "sampleRate": 24000, "streaming": "frame",
//         "voices": [...], "languages": [...], "languageSelectable": true, "backend": "MTL0"}
//        {"type": "chunk", "id": "...", "seq": 0, "pcm": "<base64 int16le mono>"}
//        {"type": "end", "id": "...", "samples": 123456}
//        {"type": "error", "id": "...", "error": "..."}
//        {"type": "fatal", "error": "..."}
//
// Every family lists its languages as BCP 47 tags, and a request's "language", when given, must be one of
// them or a region or script of one ("ja", "ja-JP"); "auto" or no language leaves the choice to the model.
// Qwen3-TTS streams frame by frame ("streaming": "frame"); its voices are the model's speakers, and the
// language goes into its prompt. Irodori-TTS makes a sentence at once and streams it as the codec decodes it
// ("streaming": "sentence"), so a request is one sentence; its voices are the ones given with --voice, the
// model is not told the language ("languageSelectable": false), and its "steps" is the sampler's.
//
// Requests are served one at a time in arrival order. A cancel takes effect between two chunks, and for
// Irodori-TTS also between two of the sampler's steps, before the first chunk; a cancelled request sends no
// end, and one cancelled before it starts is dropped. `speed` is accepted and has no effect.
//
// `--devices` instead prints the devices ggml can run on and exits, so that the caller can tell whether
// the machine has a GPU and how much memory it has before starting a worker:
//   {"type": "devices", "devices": [{"name": "Vulkan0", "description": "NVIDIA GeForce RTX 2080",
//                                    "kind": "gpu", "memoryTotal": 8589934592, "memoryFree": 7516192768}]}
//
// usage: speech-worker <model.gguf> <codec.gguf> [--device NAME|gpu|cpu] [--seed n]
//                      [--ctx n]                               (Qwen3-TTS)
//                      [--voice NAME=FILE]... [--steps n]      (Irodori-TTS; FILE is a WAVE or voice file)
//        speech-worker --devices

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
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "args.h"
#include "backend.h"
#include "flat-json.h"
#include "model-file.h"
#include "worker/engine.h"

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

void list_devices() {
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
}

/** The options after the two model files; anything else on the command line throws. */
WorkerOptions parse_options(const std::vector<std::string> & a, std::string & device, uint64_t & seed) {
    WorkerOptions o;
    o.model = a[1];
    o.codec = a[2];
    for (size_t i = 3; i < a.size(); i++) {
        const std::string & key = a[i];
        if (i + 1 >= a.size()) throw std::runtime_error(key + " needs a value");
        const std::string & value = a[++i];
        if (key == "--device" || key == "--backend") device = value;
        else if (key == "--seed") seed = std::stoull(value);
        else if (key == "--ctx") o.context = std::stoi(value);
        else if (key == "--steps") o.steps = std::stoi(value);
        else if (key == "--voice") {
            const size_t eq = value.find('=');
            if (eq == std::string::npos || eq == 0) throw std::runtime_error("--voice takes NAME=FILE");
            o.voices.push_back({value.substr(0, eq), value.substr(eq + 1)});
        } else {
            throw std::runtime_error("unknown option " + key);
        }
    }
    return o;
}

}  // namespace

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    configure_ggml();
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() == 2 && args[1] == "--devices") {
        list_devices();
        return 0;
    }
    if (args.size() < 3) {
        emit("{\"type\":\"fatal\",\"error\":\"expected the model and codec GGUF paths\"}");
        return 2;
    }

    ggml_backend_t backend = nullptr;
    std::unique_ptr<Engine> engine;
    std::string device;
    uint64_t seed = std::random_device{}();
    try {
        const WorkerOptions options = parse_options(args, device, seed);
        const std::string architecture = gguf_architecture(options.model);
        backend = init_backend(device);
        if (architecture == "qwen3tts-talker") engine = make_qwen3_tts(options, backend);
        else if (architecture == "irodori-tts") engine = make_irodori_tts(options, backend);
        else throw std::runtime_error(options.model + " is a model of " + architecture + ", which this worker does not run");
    } catch (const std::exception & e) {
        emit("{\"type\":\"fatal\",\"error\":" + json_string(e.what()) + "}");
        return 1;
    }
    emit("{\"type\":\"ready\"," + engine->describe() + ",\"backend\":" + json_string(ggml_backend_name(backend)) + "}");

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
        const Cancelled cancelled = [&] { return inbox.is_cancelled(id); };
        try {
            if (cancelled()) {
                inbox.forget(id);
                continue;
            }
            size_t samples = 0;
            int seq = 0;
            std::vector<int16_t> pcm;
            engine->speak(request, seed++, [&](const float * s, size_t n) {
                pcm.resize(n);
                for (size_t i = 0; i < n; i++) pcm[i] = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
                emit("{\"type\":\"chunk\",\"id\":" + json_string(id) + ",\"seq\":" + std::to_string(seq++) + ",\"pcm\":\"" +
                     base64((const uint8_t *) pcm.data(), n * sizeof(int16_t)) + "\"}");
                samples += n;
            }, cancelled);
            if (!cancelled()) emit("{\"type\":\"end\",\"id\":" + json_string(id) + ",\"samples\":" + std::to_string(samples) + "}");
        } catch (const std::exception & e) {
            emit("{\"type\":\"error\",\"id\":" + json_string(id) + ",\"error\":" + json_string(e.what()) + "}");
        }
        inbox.forget(id);
    }
    engine.reset();
    ggml_backend_free(backend);
    return 0;
}
