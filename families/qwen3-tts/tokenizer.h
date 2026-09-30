#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "model-file.h"

/**
 * The Qwen2 byte-level BPE with the pre-tokenizer of the tokenizer.json that ships with the model.
 *
 * transformers 4.57.3, which the official package pins, replaces that pre-tokenizer with Mistral's when
 * the model is loaded with fix_mistral_regex=True, as the package does; the two split text of Latin
 * letters in mixed case, contractions and '/' differently. This follows tokenizer.json, which is what
 * the model was trained with. The text is expected in NFC; the caller normalizes it.
 */
class Tokenizer {
public:
    explicit Tokenizer(const ModelFile & m);

    /** The ids of `text`, with special tokens written in the text taken literally. */
    std::vector<int32_t> encode(const std::string & text) const;

    /** The pieces the pre-tokenizer splits `text` into, as UTF-8. */
    std::vector<std::string> pre_tokenize(const std::string & text) const;

private:
    std::vector<int32_t> bpe(const std::string & piece) const;

    std::unordered_map<std::string, int32_t> vocab_;
    std::map<std::pair<std::string, std::string>, int> ranks_;
    std::string byte_to_unicode_[256];
};
