#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "model-file.h"

namespace irodori {

/**
 * The SentencePiece Unigram tokenizer of ModernBERT-ja, as Hugging Face tokenizers runs its tokenizer.json:
 * the added tokens are split out of the text first, spaces become "▁", the Viterbi path over the pieces'
 * scores picks the pieces, and a character no piece covers becomes its UTF-8 bytes.
 */
class Tokenizer {
public:
    explicit Tokenizer(const ModelFile & m);

    /** The ids of normalized text, after <s>, as the runtime's batch_encode() gives them. */
    std::vector<int32_t> encode(const std::string & text) const;

private:
    void encode_piece(const std::string & piece, std::vector<int32_t> & ids) const;

    std::vector<std::string> tokens_;
    std::vector<double> scores_;
    std::unordered_map<std::string, int32_t> ids_;
    std::vector<int32_t> added_;
    size_t longest_ = 0;
    double unknown_score_ = 0;
    int32_t bos_ = 0, unknown_ = 0;
    int32_t bytes_[256];
};

}  // namespace irodori
