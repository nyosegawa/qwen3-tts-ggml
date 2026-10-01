#include "codec.h"

#include <algorithm>
#include <stdexcept>

/*
 * Activations are channel-first ([channels, samples], ne0 = channels), so a convolution of width K runs as
 * K matrix products, one per tap, over views of its input shifted in time. A strided convolution of width
 * 2s cuts its input into frames of s samples, which a reshape of the channel-first layout gives for free,
 * and runs as two matrix products over consecutive frames.
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
