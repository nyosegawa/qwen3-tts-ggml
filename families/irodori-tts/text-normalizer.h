#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace irodori {

/** The code points of UTF-8 text; text that is not valid UTF-8 throws. */
std::vector<uint32_t> decode_utf8(const std::string & text);

std::string encode_utf8(const std::vector<uint32_t> & code_points);

/** NFKC of `text`, with the tables of Unicode 13.0, the version of the Python the official runtime runs on. */
std::vector<uint32_t> nfkc(const std::vector<uint32_t> & text);

/**
 * The official runtime's normalize_text() followed by its strip(): symbol replacements, removal of
 * brackets that enclose the whole text, NFKC, and "..." written as "…".
 */
std::string normalize_text(const std::string & text);

}  // namespace irodori
