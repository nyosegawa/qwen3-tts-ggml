#pragma once

#include <string>
#include <vector>

#include "codec.h"

namespace irodori {

/** The loudness the official runtime gives a reference before encoding it, in LUFS. */
constexpr double kReferenceLufs = -16.0;

/**
 * The latent of a reference voice in a WAVE file, row-major [frames, latent_dim], as the official runtime
 * makes it: the channels averaged, the loudness normalized, then encoded. The file must be at the codec's
 * sample rate and at most `max_seconds` long; the runtime would resample or cut it, and this throws.
 */
std::vector<float> encode_reference(Codec & codec, const std::string & wav_path, double max_seconds);

}  // namespace irodori
