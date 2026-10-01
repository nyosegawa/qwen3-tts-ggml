#include "language.h"

#include <cctype>

bool bcp47_matches(const std::string & tag, const std::string & language) {
    if (tag.size() < language.size()) return false;
    for (size_t i = 0; i < language.size(); i++) {
        if (std::tolower((unsigned char) tag[i]) != std::tolower((unsigned char) language[i])) return false;
    }
    return tag.size() == language.size() || tag[language.size()] == '-';
}
