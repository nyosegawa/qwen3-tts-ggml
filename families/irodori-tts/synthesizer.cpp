#include "synthesizer.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "gguf.h"
#include "reference.h"
#include "text-normalizer.h"

namespace irodori {

namespace {

constexpr const char * kVoiceArchitecture = "irodori-tts-voice";

/** Adds the seconds from its construction to its destruction to `total`. */
struct Timer {
    double & total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~Timer() { total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};

bool starts_with_riff(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4] = {};
    f.read(magic, 4);
    return std::memcmp(magic, "RIFF", 4) == 0;
}

}  // namespace

Synthesizer::Synthesizer(const std::string & model_path, const std::string & codec_path, ggml_backend_t backend)
    : backend_(backend),
      model_(std::make_unique<ModelFile>(model_path, backend)),
      codec_(codec_path, backend),
      tokenizer_(*model_),
      text_(*model_),
      speaker_(*model_),
      duration_(*model_, codec_.sample_rate(), codec_.hop()),
      dit_(*model_),
      sampler_(dit_, *model_, backend) {
    if (model_->str("general.architecture") != "irodori-tts") throw std::runtime_error(model_path + " is not an Irodori-TTS model");
    if ((int) model_->u32("irodori.latent_dim") != codec_.latent_dim()) {
        throw std::runtime_error("the model and the codec disagree on the latent's dimension");
    }
    max_reference_seconds_ = model_->f32("irodori.max_reference_seconds");
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Synthesizer::~Synthesizer() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

Voice Synthesizer::voice_from_latent(std::vector<float> latent) {
    Voice v;
    v.frames = (int) (latent.size() / codec_.latent_dim());
    v.latent = std::move(latent);
    Graph g;
    ggml_tensor * state = speaker_.build(g, v.latent);
    g.output(state);
    g.compute(backend_, allocr_);
    v.speaker = Graph::read(state);
    v.speaker_tokens = (int) state->ne[1];
    return v;
}

Voice Synthesizer::load_voice(const std::string & path) {
    if (starts_with_riff(path)) return voice_from_latent(encode_reference(codec_, path, max_reference_seconds_));
    ModelFile file(path, backend_);
    if (file.str("general.architecture") != kVoiceArchitecture) throw std::runtime_error(path + " is neither a WAVE file nor a voice file");
    if (file.str("voice.codec") != codec_.source()) {
        throw std::runtime_error(path + " was made with the codec " + file.str("voice.codec") + ", not " + codec_.source());
    }
    ggml_tensor * t = file.tensor("latent");
    if (t->type != GGML_TYPE_F32 || t->ne[0] != codec_.latent_dim()) throw std::runtime_error(path + " holds no latent of this codec");
    std::vector<float> latent(ggml_nelements(t));
    ggml_backend_tensor_get(t, latent.data(), 0, ggml_nbytes(t));
    return voice_from_latent(std::move(latent));
}

void Synthesizer::save_voice(const Voice & voice, const std::string & path) const {
    const size_t bytes = voice.latent.size() * sizeof(float);
    ggml_init_params params = {ggml_tensor_overhead() + bytes + 64, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, codec_.latent_dim(), voice.frames);
    ggml_set_name(t, "latent");
    std::memcpy(t->data, voice.latent.data(), bytes);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", kVoiceArchitecture);
    gguf_set_val_str(g, "voice.codec", codec_.source().c_str());
    gguf_add_tensor(g, t);
    const bool written = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    if (!written) throw std::runtime_error("cannot write " + path);
}

size_t Synthesizer::synthesize(const Request & r, const Voice & voice, const AudioSink & sink, Stats * stats) {
    Stats local;
    Stats & st = stats ? *stats : local;
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> text_state;
    float predicted = 0;
    int tokens = 0;
    {
        Timer t{st.text};
        const std::string text = normalize_text(r.text);
        if (text.empty()) throw std::runtime_error("the text is empty after normalization");
        const std::vector<int32_t> ids = tokenizer_.encode(text);
        tokens = (int) ids.size();
        if (tokens > text_.max_tokens()) {
            throw std::runtime_error("the text is " + std::to_string(tokens) + " tokens long and Irodori-TTS takes at most " +
                                     std::to_string(text_.max_tokens()) + "; split it into sentences");
        }
        Graph g;
        ggml_tensor * state = text_.build(g, ids);
        const std::vector<float> summary(voice.speaker.begin(), voice.speaker.begin() + speaker_.dim());
        ggml_tensor * sum = duration_.build(g, state, g.input(summary, speaker_.dim()));
        g.output(state);
        g.output(sum);
        g.compute(backend_, allocr_);
        text_state = Graph::read(state);
        predicted = Graph::read(sum)[0];
    }
    const int frames = duration_.frames(predicted);
    std::vector<float> x;
    {
        Timer t{st.sampling};
        const Conditions c{text_state, tokens, voice.speaker, voice.speaker_tokens};
        const size_t n = (size_t) frames * codec_.latent_dim();
        if (!r.noise.empty() && r.noise.size() != n) throw std::runtime_error("the given noise does not have the predicted length");
        x = sampler_.sample(c, r.noise.empty() ? gaussian_noise(r.seed, n) : r.noise, frames, r.steps > 0 ? r.steps : sampler_.default_steps());
    }
    const int flat = flattening_point(x, frames, codec_.latent_dim());
    int64_t samples = (int64_t) frames * codec_.hop();
    if (flat > 0) samples = std::min(samples, (int64_t) flat * codec_.hop());

    size_t emitted = 0;
    {
        Timer t{st.codec};
        codec_.decode(x, samples, first_window, window, [&](const float * s, size_t n) {
            if (emitted == 0) st.first_audio = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            emitted += n;
            return sink(s, n);
        });
    }
    st.tokens = tokens;
    st.frames = frames;
    st.samples = emitted;
    return emitted;
}

int flattening_point(const std::vector<float> & latent, int frames, int latent_dim) {
    constexpr int kWindow = 20;
    for (int i = 0; i < frames; i++) {
        double sum = 0, squares = 0;
        for (int f = i; f < i + kWindow; f++) {
            for (int d = 0; d < latent_dim; d++) {
                const double v = f < frames ? latent[(size_t) f * latent_dim + d] : 0.0;
                sum += v;
                squares += v * v;
            }
        }
        const double n = (double) kWindow * latent_dim, mean = sum / n;
        const double deviation = std::sqrt(std::max(0.0, squares / n - mean * mean));
        if (deviation < 0.05 && std::fabs(mean) < 0.1) return i;
    }
    return frames;
}

}  // namespace irodori
