#include "text-normalizer.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace irodori {

namespace {

struct Decomposition {
    uint32_t code_point;
    uint32_t offset;
    uint32_t length;
};

struct CombiningClassRange {
    uint32_t first, last;
    uint8_t combining_class;
};

struct Composition {
    uint32_t first, second, composite;
};

struct CodepointRange {
    uint32_t first, last;
};

#include "unicode-data.inc"

constexpr uint32_t kHangulFirst = 0xAC00, kLeadFirst = 0x1100, kVowelFirst = 0x1161, kTrailFirst = 0x11A7;
constexpr uint32_t kLeads = 19, kVowels = 21, kTrails = 28, kSyllables = kLeads * kVowels * kTrails;

int combining_class(uint32_t cp) {
    const auto * end = std::end(kCombiningClasses);
    const auto * it = std::upper_bound(std::begin(kCombiningClasses), end, cp,
                                       [](uint32_t c, const CombiningClassRange & r) { return c < r.first; });
    if (it == std::begin(kCombiningClasses)) return 0;
    --it;
    return cp <= it->last ? it->combining_class : 0;
}

bool is_whitespace(uint32_t cp) {
    for (const CodepointRange & r : kWhitespace) {
        if (cp >= r.first && cp <= r.last) return true;
    }
    return false;
}

void decompose(uint32_t cp, std::vector<uint32_t> & out) {
    if (cp >= kHangulFirst && cp < kHangulFirst + kSyllables) {
        const uint32_t s = cp - kHangulFirst;
        out.push_back(kLeadFirst + s / (kVowels * kTrails));
        out.push_back(kVowelFirst + s % (kVowels * kTrails) / kTrails);
        if (s % kTrails) out.push_back(kTrailFirst + s % kTrails);
        return;
    }
    const auto * end = std::end(kDecompositions);
    const auto * it = std::lower_bound(std::begin(kDecompositions), end, cp,
                                       [](const Decomposition & d, uint32_t c) { return d.code_point < c; });
    if (it == end || it->code_point != cp) {
        out.push_back(cp);
        return;
    }
    out.insert(out.end(), kDecompositionPool + it->offset, kDecompositionPool + it->offset + it->length);
}

/** The canonical composite of a starter and the character after it, or 0 when they do not compose. */
uint32_t compose(uint32_t a, uint32_t b) {
    if (a >= kLeadFirst && a < kLeadFirst + kLeads && b >= kVowelFirst && b < kVowelFirst + kVowels) {
        return kHangulFirst + ((a - kLeadFirst) * kVowels + (b - kVowelFirst)) * kTrails;
    }
    if (a >= kHangulFirst && a < kHangulFirst + kSyllables && (a - kHangulFirst) % kTrails == 0 && b > kTrailFirst &&
        b < kTrailFirst + kTrails) {
        return a + (b - kTrailFirst);
    }
    const auto * end = std::end(kCompositions);
    const auto * it = std::lower_bound(std::begin(kCompositions), end, std::make_pair(a, b),
                                       [](const Composition & c, const std::pair<uint32_t, uint32_t> & k) {
                                           return c.first < k.first || (c.first == k.first && c.second < k.second);
                                       });
    return it != end && it->first == a && it->second == b ? it->composite : 0;
}

void replace_all(std::vector<uint32_t> & text, const std::vector<uint32_t> & from, const std::vector<uint32_t> & to) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < text.size();) {
        if (i + from.size() <= text.size() && std::equal(from.begin(), from.end(), text.begin() + i)) {
            out.insert(out.end(), to.begin(), to.end());
            i += from.size();
        } else {
            out.push_back(text[i++]);
        }
    }
    text = std::move(out);
}

std::vector<uint32_t> u32(const char * utf8) { return decode_utf8(utf8); }

/** strip_outer_brackets(): removes a pair of brackets as long as it encloses the whole text. */
void strip_outer_brackets(std::vector<uint32_t> & text) {
    static const std::vector<std::pair<uint32_t, uint32_t>> pairs = {
        {0x300C, 0x300D}, {0x300E, 0x300F}, {0xFF08, 0xFF09}, {0x3010, 0x3011}, {'(', ')'}};
    while (text.size() >= 2) {
        const uint32_t open = text.front(), close = text.back();
        const auto pair = std::find_if(pairs.begin(), pairs.end(), [&](const auto & p) { return p.first == open; });
        if (pair == pairs.end() || pair->second != close) return;
        int depth = 0;
        bool encloses_all = true;
        for (size_t i = 0; i < text.size(); i++) {
            if (text[i] == open) depth++;
            else if (text[i] == close) depth--;
            if (depth == 0 && i + 1 < text.size()) {
                encloses_all = false;
                break;
            }
        }
        if (!encloses_all || depth != 0) return;
        text = std::vector<uint32_t>(text.begin() + 1, text.end() - 1);
    }
}

}  // namespace

std::vector<uint32_t> decode_utf8(const std::string & s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        const int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (len == 0 || i + len > s.size()) throw std::runtime_error("the text is not valid UTF-8");
        uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
        for (int k = 1; k < len; k++) {
            const unsigned char cc = (unsigned char) s[i + k];
            if ((cc >> 6) != 2) throw std::runtime_error("the text is not valid UTF-8");
            cp = (cp << 6) | (cc & 0x3F);
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

std::string encode_utf8(const std::vector<uint32_t> & code_points) {
    std::string s;
    for (uint32_t cp : code_points) {
        if (cp < 0x80) {
            s += (char) cp;
        } else if (cp < 0x800) {
            s += (char) (0xC0 | (cp >> 6));
            s += (char) (0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            s += (char) (0xE0 | (cp >> 12));
            s += (char) (0x80 | ((cp >> 6) & 0x3F));
            s += (char) (0x80 | (cp & 0x3F));
        } else {
            s += (char) (0xF0 | (cp >> 18));
            s += (char) (0x80 | ((cp >> 12) & 0x3F));
            s += (char) (0x80 | ((cp >> 6) & 0x3F));
            s += (char) (0x80 | (cp & 0x3F));
        }
    }
    return s;
}

std::vector<uint32_t> nfkc(const std::vector<uint32_t> & text) {
    std::vector<uint32_t> d;
    for (uint32_t cp : text) decompose(cp, d);
    // Canonical ordering: each run of non-starters sorted by combining class, keeping the order of equals.
    for (size_t i = 0; i < d.size();) {
        if (combining_class(d[i]) == 0) {
            i++;
            continue;
        }
        size_t j = i;
        while (j < d.size() && combining_class(d[j]) != 0) j++;
        std::stable_sort(d.begin() + i, d.begin() + j,
                         [](uint32_t a, uint32_t b) { return combining_class(a) < combining_class(b); });
        i = j;
    }
    std::vector<uint32_t> out;
    size_t starter = SIZE_MAX;
    // The combining class of the last character kept after the starter, -1 when the starter is the last.
    int last_class = -1;
    for (uint32_t cp : d) {
        const int cc = combining_class(cp);
        if (starter != SIZE_MAX && (last_class == -1 || last_class < cc)) {
            if (const uint32_t composite = compose(out[starter], cp)) {
                out[starter] = composite;
                continue;
            }
        }
        if (cc == 0) {
            starter = out.size();
            last_class = -1;
        } else {
            last_class = cc;
        }
        out.push_back(cp);
    }
    return out;
}

std::string normalize_text(const std::string & raw) {
    std::vector<uint32_t> text = decode_utf8(raw);
    static const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> simple = {
        {u32("\t"), {}}, {u32("[n]"), {}}, {u32("\\[n\\]"), {}}, {u32("　"), {}}, {u32("？"), u32("?")},
        {u32("！"), u32("!")}, {u32("♥"), u32("♡")}, {u32("●"), u32("○")}, {u32("◯"), u32("○")},
        {u32("〇"), u32("○")}};
    for (const auto & [from, to] : simple) replace_all(text, from, to);

    static const std::vector<uint32_t> removed = u32(";▼♀♂《》≪≫①②③④⑤⑥");
    auto is_dash = [](uint32_t c) {
        return c == 0x02D7 || (c >= 0x2010 && c <= 0x2015) || c == 0x2043 || c == 0x2212 || c == 0x23AF ||
               c == 0x23E4 || c == 0x2500 || c == 0x2501 || c == 0x2E3A || c == 0x2E3B;
    };
    text.erase(std::remove_if(text.begin(), text.end(), [&](uint32_t c) {
                   return std::find(removed.begin(), removed.end(), c) != removed.end() || is_dash(c);
               }),
               text.end());
    for (uint32_t & c : text) {
        if (c == 0xFF5E || c == 0x301C) c = 0x30FC;
    }
    // A run of three or more ellipses becomes two.
    std::vector<uint32_t> runs;
    for (size_t i = 0; i < text.size();) {
        size_t j = i;
        while (j < text.size() && text[j] == 0x2026) j++;
        if (j - i >= 3) {
            runs.push_back(0x2026);
            runs.push_back(0x2026);
            i = j;
        } else if (j > i) {
            runs.insert(runs.end(), text.begin() + i, text.begin() + j);
            i = j;
        } else {
            runs.push_back(text[i++]);
        }
    }
    text = std::move(runs);

    strip_outer_brackets(text);
    text = nfkc(text);
    replace_all(text, u32("..."), {0x2026});
    replace_all(text, u32(".."), {0x2026});

    size_t begin = 0, end = text.size();
    while (begin < end && is_whitespace(text[begin])) begin++;
    while (end > begin && is_whitespace(text[end - 1])) end--;
    return encode_utf8(std::vector<uint32_t>(text.begin() + begin, text.begin() + end));
}

}  // namespace irodori
