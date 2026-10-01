#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace {

/** The Unigram path ends at a byte offset; `start` is where its last piece starts, -1 while unreached. */
struct PathEnd {
    int32_t id = -1;
    double score = 0;
    int64_t start = -1;
};

/** Hugging Face's penalty under the lowest piece score for a character no piece covers. */
constexpr double kUnknownPenalty = 10.0;

size_t utf8_length(unsigned char c) {
    return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
}

}  // namespace

Tokenizer::Tokenizer(const ModelFile & m) {
    tokens_ = m.str_array("tokenizer.tokens");
    scores_ = m.f64_array("tokenizer.scores");
    added_ = m.i32_array("tokenizer.added_ids");
    bos_ = (int32_t) m.u32("tokenizer.bos_id");
    unknown_ = (int32_t) m.u32("tokenizer.unknown_id");
    if (scores_.size() != tokens_.size()) throw std::runtime_error("the tokenizer has a score for each token");
    double lowest = 0;
    for (size_t i = 0; i < tokens_.size(); i++) {
        ids_.emplace(tokens_[i], (int32_t) i);
        longest_ = std::max(longest_, tokens_[i].size());
        lowest = std::min(lowest, scores_[i]);
    }
    unknown_score_ = lowest - kUnknownPenalty;
    for (int b = 0; b < 256; b++) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02X>", b);
        const auto it = ids_.find(name);
        if (it == ids_.end()) throw std::runtime_error(std::string("the tokenizer has no byte token ") + name);
        bytes_[b] = it->second;
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids = {bos_};
    std::string piece;
    auto flush = [&]() {
        if (!piece.empty()) encode_piece(piece, ids);
        piece.clear();
    };
    for (size_t i = 0; i < text.size();) {
        // The leftmost and then longest added token, matched on the text before anything else.
        int32_t added = -1;
        for (int32_t id : added_) {
            const std::string & t = tokens_[id];
            if (text.compare(i, t.size(), t) == 0 && (added < 0 || t.size() > tokens_[added].size())) added = id;
        }
        if (added >= 0) {
            flush();
            ids.push_back(added);
            i += tokens_[added].size();
        } else if (text[i] == ' ') {
            piece += "\xE2\x96\x81";
            i++;
        } else {
            piece += text[i++];
        }
    }
    flush();
    return ids;
}

void Tokenizer::encode_piece(const std::string & s, std::vector<int32_t> & ids) const {
    const size_t n = s.size();
    std::vector<PathEnd> best(n + 1);
    best[0].start = 0;
    for (size_t start = 0; start < n;) {
        const size_t char_length = utf8_length((unsigned char) s[start]);
        const double till_here = best[start].score;
        bool covered = false;
        for (size_t length = 1; length <= longest_ && start + length <= n; length++) {
            const auto it = ids_.find(s.substr(start, length));
            if (it == ids_.end()) continue;
            PathEnd & end = best[start + length];
            const double score = till_here + scores_[it->second];
            if (end.start < 0 || score > end.score) end = {it->second, score, (int64_t) start};
            if (length == char_length) covered = true;
        }
        if (!covered) {
            PathEnd & end = best[start + char_length];
            const double score = till_here + unknown_score_;
            if (end.start < 0 || score > end.score) end = {unknown_, score, (int64_t) start};
        }
        start += char_length;
    }
    // Back from the end; consecutive unknown characters are fused into one piece, then spelled in bytes.
    std::vector<std::pair<std::string, int32_t>> pieces;
    for (size_t end = n; end > 0;) {
        const PathEnd & p = best[end];
        const std::string text = s.substr((size_t) p.start, end - (size_t) p.start);
        if (p.id == unknown_ && !pieces.empty() && pieces.back().second == unknown_) {
            pieces.back().first = text + pieces.back().first;
        } else {
            pieces.push_back({text, p.id});
        }
        end = (size_t) p.start;
    }
    for (auto it = pieces.rbegin(); it != pieces.rend(); ++it) {
        if (it->second != unknown_) {
            ids.push_back(it->second);
            continue;
        }
        const auto known = ids_.find(it->first);
        if (known != ids_.end()) {
            ids.push_back(known->second);
            continue;
        }
        for (unsigned char b : it->first) ids.push_back(bytes_[b]);
    }
}
