#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "flat-json.h"
#include "ggml-backend.h"

/** Called with each piece of audio as it is synthesized; returning false stops the synthesis. */
using PcmSink = std::function<bool(const float * samples, size_t n)>;

/** The worker's options that a family reads. */
struct WorkerOptions {
    std::string model, codec;
    /** Qwen3-TTS: the talker's context in positions. */
    int context = 2048;
    /** Irodori-TTS: the voices, each a name and a WAVE or voice file; and the sampler's steps, 0 for the model's default. */
    std::vector<std::pair<std::string, std::string>> voices;
    int steps = 0;
};

/** One family of models behind the worker protocol. */
class Engine {
public:
    virtual ~Engine() = default;

    /** The members of the ready message that describe the model, as JSON members without braces. */
    virtual std::string describe() const = 0;

    /** Synthesizes one request; a request the family cannot take throws, which the worker reports as an error. */
    virtual void speak(const FlatJson & request, uint64_t seed, const PcmSink & sink) = 0;
};

std::unique_ptr<Engine> make_qwen3_tts(const WorkerOptions & options, ggml_backend_t backend);
std::unique_ptr<Engine> make_irodori_tts(const WorkerOptions & options, ggml_backend_t backend);

/** A JSON array of strings. */
inline std::string json_array(const std::vector<std::string> & items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); i++) out += (i ? "," : "") + json_string(items[i]);
    return out + "]";
}
