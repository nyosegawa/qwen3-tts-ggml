#include "loudness.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Biquad {
    double b0, b1, b2, a1, a2;

    void run(std::vector<double> & x) const {
        double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        for (double & v : x) {
            const double y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = v;
            y2 = y1;
            y1 = y;
            v = y;
        }
    }
};

/** pyloudnorm's RBJ high shelf, normalized by a0. */
Biquad high_shelf(double gain_db, double q, double fc, double rate) {
    const double a = std::pow(10.0, gain_db / 40.0), w0 = 2.0 * kPi * (fc / rate), alpha = std::sin(w0) / (2.0 * q);
    const double c = std::cos(w0), s = 2 * std::sqrt(a) * alpha;
    const double a0 = (a + 1) - (a - 1) * c + s;
    return {a * ((a + 1) + (a - 1) * c + s) / a0, -2 * a * ((a - 1) + (a + 1) * c) / a0,
            a * ((a + 1) + (a - 1) * c - s) / a0, 2 * ((a - 1) - (a + 1) * c) / a0, ((a + 1) - (a - 1) * c - s) / a0};
}

/** pyloudnorm's RBJ high pass, normalized by a0. */
Biquad high_pass(double q, double fc, double rate) {
    const double w0 = 2.0 * kPi * (fc / rate), alpha = std::sin(w0) / (2.0 * q), c = std::cos(w0);
    const double a0 = 1 + alpha;
    return {(1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0, -2 * c / a0, (1 - alpha) / a0};
}

}  // namespace

double integrated_loudness(const std::vector<float> & samples, int sample_rate) {
    std::vector<double> x(samples.begin(), samples.end());
    const size_t min_length = (size_t) (0.5 * sample_rate);
    if (x.size() < min_length) x.resize(min_length, 0.0);
    high_shelf(4.0, 1.0 / std::sqrt(2.0), 1500.0, sample_rate).run(x);
    high_pass(0.5, 38.0, sample_rate).run(x);

    const size_t block = (size_t) (0.4 * sample_rate), step = (size_t) (0.4 * sample_rate * 0.25);
    const size_t n_blocks = (std::max(x.size(), block) - block + step - 1) / step + 1;
    x.resize((n_blocks - 1) * step + block, 0.0);
    std::vector<double> power(n_blocks), level(n_blocks);
    for (size_t b = 0; b < n_blocks; b++) {
        double sum = 0;
        for (size_t i = b * step; i < b * step + block; i++) sum += x[i] * x[i];
        power[b] = sum / (0.4 * sample_rate);
        level[b] = -0.691 + 10.0 * std::log10(power[b]);
    }
    auto gated_mean = [&](double threshold_absolute, double threshold_relative) {
        double sum = 0;
        size_t count = 0;
        for (size_t b = 0; b < n_blocks; b++) {
            if (level[b] > threshold_absolute && level[b] > threshold_relative) {
                sum += power[b];
                count++;
            }
        }
        return count ? sum / (double) count : 0.0;
    };
    const double relative = -0.691 + 10.0 * std::log10(gated_mean(-70.0, -INFINITY)) - 10.0;
    const double lufs = -0.691 + 10.0 * std::log10(gated_mean(-70.0, relative));
    return std::isfinite(lufs) ? std::max(lufs, -70.0) : -70.0;
}

std::vector<float> normalize_loudness(const std::vector<float> & samples, int sample_rate, double target_lufs) {
    const float gain = (float) std::exp((target_lufs - integrated_loudness(samples, sample_rate)) * std::log(10.0) / 20.0);
    std::vector<float> out(samples.size());
    float peak = 0;
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = samples[i] * gain;
        peak = std::max(peak, std::fabs(out[i]));
    }
    if (peak > 1.0f) {
        for (float & v : out) v *= 1.0f / peak;
    }
    return out;
}
