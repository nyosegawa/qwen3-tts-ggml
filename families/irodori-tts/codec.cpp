#include "codec.h"

#include <algorithm>
#include <stdexcept>

namespace irodori {

/*
 * Activations are channel-first ([channels, samples], ne0 = channels), so a convolution of width K runs as
 * K matrix products, one per tap, over views of its input shifted in time. A strided convolution of width
 * 2s cuts its input into frames of s samples, which a reshape of the channel-first layout gives for free,
 * and runs as two matrix products over consecutive frames. A transposed convolution of width 2s and stride
 * s padded by s / 2 runs as one matrix product that gives every tap of every input frame; an output frame
 * is then the middle s taps of its own input frame plus the last s / 2 taps of the frame before it (in its
 * first half) and the first s / 2 taps of the frame after it (in its second half).
 */

namespace {

struct Convolutions {
    Graph & g;
    const ModelFile & m;

    ggml_context * ctx() const { return g.ctx(); }

    /** `x` with `left` and `right` samples of silence around it. */
    ggml_tensor * pad(ggml_tensor * x, int64_t left, int64_t right) {
        if (left > 0) x = ggml_concat(ctx(), g.zeros(x->ne[0], left), x, 1);
        if (right > 0) x = ggml_concat(ctx(), x, g.zeros(x->ne[0], right), 1);
        return x;
    }

    /** A stride-1 convolution with symmetric padding; weight ne = [in, out, k]. */
    ggml_tensor * conv(ggml_tensor * x, const std::string & name, int dilation = 1) {
        ggml_tensor * w = m.tensor(name + ".weight");
        const int64_t in = w->ne[0], out = w->ne[1], k_width = w->ne[2], t = x->ne[1];
        ggml_tensor * y = nullptr;
        if (k_width == 1) {
            y = mul_mat(ctx(), ggml_reshape_2d(ctx(), w, in, out), x);
        } else {
            const int64_t p = (k_width - 1) * dilation / 2;
            ggml_tensor * xp = pad(x, p, p);
            for (int64_t k = 0; k < k_width; k++) {
                ggml_tensor * wk = ggml_view_2d(ctx(), w, in, out, w->nb[1], k * w->nb[2]);
                ggml_tensor * xk = ggml_view_2d(ctx(), xp, in, t, xp->nb[1], k * dilation * xp->nb[1]);
                ggml_tensor * yk = mul_mat(ctx(), wk, xk);
                y = y ? ggml_add(ctx(), y, yk) : yk;
            }
        }
        return ggml_add(ctx(), y, m.tensor(name + ".bias"));
    }

    /** A convolution of width 2s and stride s padded by s / 2 on each side, its taps stored in halves. */
    ggml_tensor * down(ggml_tensor * x, const std::string & name, int stride) {
        const int64_t c = x->ne[0], t = x->ne[1], n = t / stride;
        ggml_tensor * frames = ggml_reshape_2d(ctx(), pad(x, stride / 2, stride / 2), c * stride, n + 1);
        ggml_tensor * first = ggml_view_2d(ctx(), frames, c * stride, n, frames->nb[1], 0);
        ggml_tensor * second = ggml_view_2d(ctx(), frames, c * stride, n, frames->nb[1], frames->nb[1]);
        ggml_tensor * y = ggml_add(ctx(), mul_mat(ctx(), m.tensor(name + ".first"), first),
                                   mul_mat(ctx(), m.tensor(name + ".second"), second));
        return ggml_add(ctx(), y, m.tensor(name + ".bias"));
    }

    /** The transposed convolution of width 2s, stride s and padding s / 2; weight ne = [in, out, 2s]. */
    ggml_tensor * up(ggml_tensor * x, const std::string & name, int stride) {
        ggml_tensor * w = m.tensor(name + ".weight");
        const int64_t in = w->ne[0], out = w->ne[1], k_width = w->ne[2], frames = x->ne[1], half = stride / 2;
        ggml_tensor * z = ggml_reshape_3d(ctx(), mul_mat(ctx(), ggml_reshape_2d(ctx(), w, in, out * k_width), x), out, k_width, frames);
        auto taps = [&](int64_t first_tap, int64_t taps_n, int64_t first_frame, int64_t frames_n) {
            return ggml_cont(ctx(), ggml_view_3d(ctx(), z, out, taps_n, frames_n, z->nb[1], z->nb[2],
                                                 first_tap * z->nb[1] + first_frame * z->nb[2]));
        };
        ggml_tensor * own = taps(half, stride, 0, frames);
        ggml_tensor * edge = ggml_reshape_3d(ctx(), g.zeros(out, half), out, half, 1);
        ggml_tensor * before = ggml_concat(ctx(), edge, taps(stride + half, half, 0, frames - 1), 2);
        ggml_tensor * after = ggml_concat(ctx(), taps(0, half, 1, frames - 1), edge, 2);
        ggml_tensor * y = ggml_add(ctx(), own, ggml_concat(ctx(), before, after, 1));
        return ggml_add(ctx(), ggml_reshape_2d(ctx(), y, out, stride * frames), m.tensor(name + ".bias"));
    }

    /** Snake: x + sin(alpha x)^2 / (alpha + 1e-9). */
    ggml_tensor * snake(ggml_tensor * x, const std::string & name) {
        ggml_tensor * s = ggml_sin(ctx(), ggml_mul(ctx(), x, m.tensor(name + ".alpha")));
        return ggml_add(ctx(), x, ggml_mul(ctx(), ggml_sqr(ctx(), s), m.tensor(name + ".inv_alpha")));
    }

    /** A residual unit: Snake, a width-7 convolution, Snake, a width-1 convolution, and the skip. */
    ggml_tensor * residual(ggml_tensor * x, const std::string & name, int dilation) {
        ggml_tensor * h = conv(snake(x, name + ".snake1"), name + ".conv1", dilation);
        h = conv(snake(h, name + ".snake2"), name + ".conv2");
        return ggml_add(ctx(), x, h);
    }
};

}  // namespace

Codec::Codec(const std::string & path, ggml_backend_t backend) : backend_(backend) {
    model_ = std::make_unique<ModelFile>(path, backend);
    sample_rate_ = (int) model_->u32("dacvae.sample_rate");
    hop_ = (int) model_->u32("dacvae.hop_length");
    latent_dim_ = (int) model_->u32("dacvae.latent_dim");
    encoder_rates_ = model_->i32_array("dacvae.encoder_rates");
    decoder_rates_ = model_->i32_array("dacvae.decoder_rates");
    source_ = model_->str("general.source.url");
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Codec::~Codec() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

ggml_tensor * Codec::build_encoder(Graph & g, const std::vector<float> & samples) const {
    Convolutions l{g, *model_};
    const int64_t n = (int64_t) samples.size();
    if (n % hop_ != 0) throw std::runtime_error("the encoder takes a whole number of frames");
    ggml_tensor * x = l.conv(g.input(samples, 1, n), "enc.conv_in");
    for (size_t i = 0; i < encoder_rates_.size(); i++) {
        const std::string b = "enc.blk." + std::to_string(i);
        for (int j = 0, dilation = 1; j < 3; j++, dilation *= 3) x = l.residual(x, b + ".res." + std::to_string(j), dilation);
        x = l.down(l.snake(x, b + ".snake"), b + ".down", encoder_rates_[i]);
    }
    x = l.conv(l.snake(x, "enc.snake"), "enc.conv_out");
    return ggml_add(g.ctx(), mul_mat(g.ctx(), model_->tensor("bottleneck.mean.weight"), x),
                    model_->tensor("bottleneck.mean.bias"));
}

std::vector<float> Codec::encode(const std::vector<float> & audio, int window) {
    const int64_t length = (int64_t) audio.size();
    const int64_t remainder = length % hop_;
    std::vector<float> padded = audio;
    if (remainder) {
        // torch's reflect padding, which repeats the samples before the last one in reverse.
        if (hop_ - remainder >= length) throw std::runtime_error("the reference is shorter than one codec frame");
        for (int64_t i = 0; i < hop_ - remainder; i++) padded.push_back(audio[length - 2 - i]);
    }
    const int64_t frames = (int64_t) padded.size() / hop_;
    std::vector<float> latent((size_t) (frames * latent_dim_));
    for (int64_t a = 0; a < frames; a += window) {
        const int64_t b = std::min(frames, a + window);
        const int64_t from = std::max<int64_t>(0, a - kEncoderMargin), to = std::min(frames, b + kEncoderMargin);
        Graph g;
        ggml_tensor * out = build_encoder(g, std::vector<float>(padded.begin() + from * hop_, padded.begin() + to * hop_));
        g.output(out);
        g.compute(backend_, allocr_);
        const std::vector<float> got = Graph::read(out);
        std::copy(got.begin() + (a - from) * latent_dim_, got.begin() + (b - from) * latent_dim_, latent.begin() + a * latent_dim_);
    }
    return latent;
}

ggml_tensor * Codec::build_decoder(Graph & g, const std::vector<float> & latent, std::vector<ggml_tensor *> * stages) const {
    Convolutions l{g, *model_};
    const int64_t frames = (int64_t) latent.size() / latent_dim_;
    if (frames < 2) throw std::runtime_error("the decoder takes at least two frames at once");
    ggml_tensor * x = l.conv(g.input(latent, latent_dim_, frames), "dec.in_proj");
    if (stages) stages->push_back(x);
    x = l.conv(x, "dec.conv_in");
    if (stages) stages->push_back(x);
    for (size_t i = 0; i < decoder_rates_.size(); i++) {
        const std::string b = "dec.blk." + std::to_string(i);
        x = l.up(l.snake(x, b + ".snake"), b + ".up", decoder_rates_[i]);
        for (int j = 0, dilation = 1; j < 3; j++, dilation *= 3) x = l.residual(x, b + ".res." + std::to_string(j), dilation);
        if (stages) stages->push_back(x);
    }
    return ggml_tanh(g.ctx(), l.conv(l.snake(x, "dec.out_snake"), "dec.conv_out"));
}

void Codec::decode(const std::vector<float> & latent, int64_t samples, int first_window, int window, const AudioSink & sink) {
    const int64_t frames = (int64_t) latent.size() / latent_dim_;
    const int64_t needed = std::min(frames, (samples + hop_ - 1) / hop_);
    int64_t emitted = 0;
    for (int64_t a = 0; a < needed;) {
        const int64_t b = std::min(needed, a + (a == 0 ? first_window : window));
        // The margin, and at least two frames, so that the window is long enough to decode.
        int64_t from = std::max<int64_t>(0, a - kDecoderMargin), to = std::min(frames, b + kDecoderMargin);
        if (to - from < 2) from = std::max<int64_t>(0, to - 2);
        Graph g;
        ggml_tensor * out = build_decoder(g, std::vector<float>(latent.begin() + from * latent_dim_, latent.begin() + to * latent_dim_));
        g.output(out);
        g.compute(backend_, allocr_);
        const std::vector<float> audio = Graph::read(out);
        const int64_t first = (a - from) * hop_, count = std::min((b - a) * hop_, samples - emitted);
        emitted += count;
        if (!sink(audio.data() + first, (size_t) count)) return;
        a = b;
    }
}

}  // namespace irodori
