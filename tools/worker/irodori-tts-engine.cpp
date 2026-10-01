#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "irodori-tts/synthesizer.h"
#include "language.h"

namespace {

class IrodoriTtsEngine : public Engine {
public:
    IrodoriTtsEngine(const WorkerOptions & options, ggml_backend_t backend) : synth_(options.model, options.codec, backend), steps_(options.steps) {
        if (options.voices.empty()) {
            throw std::runtime_error("an Irodori-TTS model has no voices of its own; give one with --voice NAME=FILE");
        }
        for (const auto & [name, path] : options.voices) {
            if (voices_.count(name)) throw std::runtime_error("two voices are named " + name);
            voices_[name] = synth_.load_voice(path);
            names_.push_back(name);
        }
        languages_ = synth_.model().str_array("speech.languages");
        // The first synthesis compiles the GPU kernels, which on Vulkan takes seconds; a short and a longer
        // text run through every stage and both sizes of the decoder's windows.
        for (const char * text : {"あ。", "明日の東京は晴れで、最高気温は二十四度の予報です。"}) {
            irodori::Request warmup;
            warmup.text = text;
            warmup.steps = steps_;
            synth_.synthesize(warmup, voices_.begin()->second, [](const float *, size_t) { return true; });
        }
    }

    std::string describe() const override {
        return "\"model\":" + json_string(synth_.model().str("general.name")) + ",\"architecture\":\"irodori-tts\",\"sampleRate\":" +
               std::to_string(synth_.sample_rate()) + ",\"streaming\":\"sentence\",\"voices\":" + json_array(names_) +
               ",\"languages\":" + json_array(languages_) + ",\"languageSelectable\":false,\"steps\":" +
               std::to_string(steps_ > 0 ? steps_ : (int) synth_.model().u32("irodori.default_steps"));
    }

    void speak(const FlatJson & request, uint64_t seed, const PcmSink & sink) override {
        const auto voice = voices_.find(value(request, "voice"));
        if (voice == voices_.end()) throw std::runtime_error("no voice is named \"" + value(request, "voice") + "\"");
        // The model is not told a language; a request may still name one, which must be one it speaks.
        const std::string language = value(request, "language");
        if (!language.empty() && language != "auto" && !speaks(language)) {
            std::string list;
            for (const std::string & l : languages_) list += (list.empty() ? "" : ", ") + l;
            throw std::runtime_error("Irodori-TTS speaks " + list + ", not " + language);
        }
        irodori::Request r;
        r.text = value(request, "text");
        r.seed = seed;
        r.steps = steps_;
        synth_.synthesize(r, voice->second, sink);
    }

private:
    static std::string value(const FlatJson & request, const char * key) {
        const auto it = request.find(key);
        return it == request.end() ? "" : it->second;
    }

    /** Whether a BCP 47 tag names one of the model's languages. */
    bool speaks(const std::string & tag) const {
        for (const std::string & l : languages_) {
            if (bcp47_matches(tag, l)) return true;
        }
        return false;
    }

    irodori::Synthesizer synth_;
    int steps_;
    std::map<std::string, irodori::Voice> voices_;
    std::vector<std::string> names_, languages_;
};

}  // namespace

std::unique_ptr<Engine> make_irodori_tts(const WorkerOptions & options, ggml_backend_t backend) {
    return std::make_unique<IrodoriTtsEngine>(options, backend);
}
