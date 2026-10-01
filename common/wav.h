#pragma once

#include <string>
#include <vector>

/** A WAVE file's samples as float, channels interleaved, scaled as torchaudio.load() scales them. */
struct Wav {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;

    /** The channels averaged into one. */
    std::vector<float> mono() const;
};

/** Reads 16-, 24- or 32-bit integer PCM or 32-bit float WAVE; anything else throws. */
Wav read_wav(const std::string & path);

/** Writes mono 16-bit PCM, clipping to [-1, 1]. */
void write_wav(const std::string & path, const std::vector<float> & samples, int sample_rate);
