#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

/** A little-endian float32 or int32 .npy array in C order, as numpy.save writes it. */
struct Npy {
    std::vector<int64_t> shape;
    std::vector<float> f32;
    std::vector<int32_t> i32;

    int64_t size() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

inline Npy read_npy(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + " is not a .npy file");
    uint8_t version[2];
    f.read((char *) version, 2);
    uint32_t header_len = 0;
    if (version[0] == 1) {
        uint16_t n;
        f.read((char *) &n, 2);
        header_len = n;
    } else {
        f.read((char *) &header_len, 4);
    }
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    if (header.find("'fortran_order': False") == std::string::npos) throw std::runtime_error(path + " is not in C order");

    Npy npy;
    const size_t open = header.find('(', header.find("'shape'"));
    const size_t close = header.find(')', open);
    std::string dims = header.substr(open + 1, close - open - 1);
    size_t start = 0;
    while (start < dims.size()) {
        size_t comma = dims.find(',', start);
        if (comma == std::string::npos) comma = dims.size();
        std::string d = dims.substr(start, comma - start);
        if (d.find_first_not_of(" ") != std::string::npos) npy.shape.push_back(std::stoll(d));
        start = comma + 1;
    }
    if (header.find("'<f4'") != std::string::npos) {
        npy.f32.resize(npy.size());
        f.read((char *) npy.f32.data(), npy.size() * 4);
    } else if (header.find("'<i4'") != std::string::npos) {
        npy.i32.resize(npy.size());
        f.read((char *) npy.i32.data(), npy.size() * 4);
    } else {
        throw std::runtime_error(path + " is neither float32 nor int32");
    }
    if (!f) throw std::runtime_error("cannot read the data of " + path);
    return npy;
}
