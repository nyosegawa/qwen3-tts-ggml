#pragma once

#include <vector>

/**
 * The integrated loudness of mono audio in LUFS (ITU-R BS.1770-4) as audiotools measures it on the CPU:
 * pyloudnorm's K-weighting filters, 400 ms blocks every 100 ms with the last one padded with silence,
 * absolute and relative gating, and a floor of -70. Audio shorter than 0.5 s is measured padded to 0.5 s.
 */
double integrated_loudness(const std::vector<float> & samples, int sample_rate);

/**
 * What the official runtime does to a reference before encoding it: a gain to `target_lufs`, then a
 * scale down when a sample exceeds 1.
 */
std::vector<float> normalize_loudness(const std::vector<float> & samples, int sample_rate, double target_lufs);
