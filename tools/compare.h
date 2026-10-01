#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

/** How far a result is from its reference: the largest difference, and the reference's power over the error's. */
struct Diff {
    double max_abs = 0;
    double snr_db = INFINITY;
    size_t n = 0;
};

inline Diff compare(const float * got, const float * want, size_t n) {
    Diff d;
    d.n = n;
    double signal = 0, noise = 0;
    for (size_t i = 0; i < n; i++) {
        const double e = (double) got[i] - want[i];
        d.max_abs = std::max(d.max_abs, std::fabs(e));
        signal += (double) want[i] * want[i];
        noise += e * e;
    }
    d.snr_db = noise == 0 ? INFINITY : 10 * std::log10(signal / noise);
    return d;
}

inline Diff compare(const std::vector<float> & got, const std::vector<float> & want) {
    return compare(got.data(), want.data(), std::min(got.size(), want.size()));
}

inline void print_diff(const std::string & what, const Diff & d) {
    std::printf("%-34s SNR %6.1f dB, max |diff| %.2e\n", what.c_str(), d.snr_db, d.max_abs);
}
