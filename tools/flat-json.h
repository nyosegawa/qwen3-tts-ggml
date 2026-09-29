#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>

/**
 * A flat JSON object of string and number members, which is all the worker protocol sends. Nested
 * values, arrays, true, false and null are rejected rather than skipped.
 */
using FlatJson = std::map<std::string, std::string>;

namespace flat_json {

inline void append_utf8(std::string & out, uint32_t cp) {
    if (cp < 0x80) out += (char) cp;
    else if (cp < 0x800) { out += (char) (0xC0 | (cp >> 6)); out += (char) (0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += (char) (0xE0 | (cp >> 12)); out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F)); }
    else { out += (char) (0xF0 | (cp >> 18)); out += (char) (0x80 | ((cp >> 12) & 0x3F)); out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F)); }
}

inline std::string parse_string(const std::string & s, size_t & i) {
    if (s[i] != '"') throw std::runtime_error("expected a string");
    std::string out;
    for (i++; i < s.size(); i++) {
        const char c = s[i];
        if (c == '"') { i++; return out; }
        if (c != '\\') { out += c; continue; }
        if (++i >= s.size()) break;
        switch (s[i]) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                auto hex4 = [&](size_t at) { return (uint32_t) std::stoul(s.substr(at, 4), nullptr, 16); };
                uint32_t cp = hex4(i + 1);
                i += 4;
                if (cp >= 0xD800 && cp < 0xDC00 && i + 6 < s.size() && s[i + 1] == '\\' && s[i + 2] == 'u') {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (hex4(i + 3) - 0xDC00);
                    i += 6;
                }
                append_utf8(out, cp);
                break;
            }
            default: throw std::runtime_error("bad escape in a string");
        }
    }
    throw std::runtime_error("unterminated string");
}

}  // namespace flat_json

inline FlatJson parse_flat_json(const std::string & s) {
    FlatJson out;
    size_t i = 0;
    auto skip = [&]() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++; };
    skip();
    if (i >= s.size() || s[i] != '{') throw std::runtime_error("expected an object");
    i++;
    skip();
    if (i < s.size() && s[i] == '}') return out;
    while (i < s.size()) {
        skip();
        const std::string key = flat_json::parse_string(s, i);
        skip();
        if (i >= s.size() || s[i] != ':') throw std::runtime_error("expected ':'");
        i++;
        skip();
        if (i < s.size() && s[i] == '"') {
            out[key] = flat_json::parse_string(s, i);
        } else {
            const size_t start = i;
            while (i < s.size() && (std::isdigit((unsigned char) s[i]) || s[i] == '-' || s[i] == '+' || s[i] == '.' || s[i] == 'e' || s[i] == 'E')) i++;
            if (i == start) throw std::runtime_error("member " + key + " is neither a string nor a number");
            out[key] = s.substr(start, i - start);
        }
        skip();
        if (i < s.size() && s[i] == ',') { i++; continue; }
        if (i < s.size() && s[i] == '}') return out;
        throw std::runtime_error("expected ',' or '}'");
    }
    throw std::runtime_error("unterminated object");
}

/** A JSON string literal of `s`, which is UTF-8. */
inline std::string json_string(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out + "\"";
}
