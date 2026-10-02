#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"
#include "qwen3-tts/synthesizer.h"

namespace {

class Qwen3TtsEngine : public Engine {
public:
    Qwen3TtsEngine(const WorkerOptions & options, ggml_backend_t backend)
        : synth_(options.model, options.codec, backend, options.context) {
        voices_ = synth_.ids().speaker_names;
        languages_ = synth_.languages();
        std::sort(voices_.begin(), voices_.end());
        std::sort(languages_.begin(), languages_.end());
        if (voices_.empty()) throw std::runtime_error("the model has no preset voices");
        // The first synthesis compiles the GPU kernels, which on Vulkan takes seconds for every new shape, so a
        // short and a longer text run through the prompt, the talker, the code predictor and the codec at the
        // sizes speech uses.
        for (const auto & [text, frames] : std::vector<std::pair<std::string, int>>{
                 {"あ", 4}, {"明日の東京は晴れで、最高気温は二十四度の予報です。", 40}}) {
            SynthesisRequest warmup;
            warmup.text = text;
            warmup.speaker = voices_[0];
            warmup.max_frames = frames;
            synth_.synthesize(warmup, [](const float *, size_t) { return true; });
        }
    }

    std::string describe() const override {
        return "\"model\":" + json_string(synth_.talker_name()) + ",\"architecture\":\"qwen3tts-talker\",\"sampleRate\":" +
               std::to_string(synth_.sample_rate()) + ",\"streaming\":\"frame\",\"voices\":" + json_array(voices_) +
               ",\"languages\":" + json_array(languages_) + ",\"languageSelectable\":true";
    }

    void speak(const FlatJson & request, uint64_t seed, const PcmSink & sink, const Cancelled & cancelled) override {
        SynthesisRequest r;
        r.text = value(request, "text");
        r.speaker = value(request, "voice");
        const std::string language = value(request, "language");
        r.language = language.empty() ? "auto" : language;
        r.seed = seed;
        // Qwen3-TTS passes audio after its first frame and then every four frames, so it stops at the sink.
        synth_.synthesize(r, [&](const float * s, size_t n) {
            if (cancelled()) return false;
            sink(s, n);
            return true;
        });
    }

private:
    static std::string value(const FlatJson & request, const char * key) {
        const auto it = request.find(key);
        return it == request.end() ? "" : it->second;
    }

    Synthesizer synth_;
    std::vector<std::string> voices_, languages_;
};

}  // namespace

std::unique_ptr<Engine> make_qwen3_tts(const WorkerOptions & options, ggml_backend_t backend) {
    if (!options.voices.empty() || options.steps != 0) throw std::runtime_error("--voice and --steps are for Irodori-TTS models");
    return std::make_unique<Qwen3TtsEngine>(options, backend);
}
