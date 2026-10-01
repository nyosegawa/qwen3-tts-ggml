#include "reference.h"

#include <stdexcept>

#include "loudness.h"
#include "wav.h"

namespace irodori {

std::vector<float> encode_reference(Codec & codec, const std::string & wav_path, double max_seconds) {
    const Wav wav = read_wav(wav_path);
    if (wav.sample_rate != codec.sample_rate()) {
        throw std::runtime_error(wav_path + " is at " + std::to_string(wav.sample_rate) + " Hz; a reference voice must be at " +
                                 std::to_string(codec.sample_rate()) + " Hz");
    }
    const std::vector<float> mono = wav.mono();
    const double seconds = (double) mono.size() / wav.sample_rate;
    if (seconds > max_seconds) {
        throw std::runtime_error(wav_path + " is " + std::to_string(seconds) + " s long; a reference voice is at most " +
                                 std::to_string(max_seconds) + " s");
    }
    return codec.encode(normalize_loudness(mono, wav.sample_rate, kReferenceLufs));
}

}  // namespace irodori
