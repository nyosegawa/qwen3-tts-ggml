#include "model-file.h"

#include <cstdio>
#include <stdexcept>

namespace {

/** fseek() takes a long, which is 32 bits on Windows, so it cannot reach a tensor past 2 GiB there. */
bool seek(FILE * f, uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(f, (int64_t) offset, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t) offset, SEEK_SET) == 0;
#endif
}

}  // namespace

std::string gguf_architecture(const std::string & path) {
    gguf_init_params params = {/*no_alloc =*/true, /*ctx =*/nullptr};
    gguf_context * gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf) throw std::runtime_error("cannot read GGUF: " + path);
    const int64_t id = gguf_find_key(gguf, "general.architecture");
    const std::string arch = id >= 0 && gguf_get_kv_type(gguf, id) == GGUF_TYPE_STRING ? gguf_get_val_str(gguf, id) : "";
    gguf_free(gguf);
    if (arch.empty()) throw std::runtime_error(path + " names no general.architecture");
    return arch;
}

ModelFile::ModelFile(const std::string & path, ggml_backend_t backend) : path_(path) {
    gguf_init_params params = {/*no_alloc =*/true, /*ctx =*/&ctx_};
    gguf_ = gguf_init_from_file(path.c_str(), params);
    if (!gguf_) {
        throw std::runtime_error("cannot read GGUF: " + path);
    }
    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buffer_) {
        throw std::runtime_error("cannot allocate the weights of " + path);
    }
    ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * f = ggml_fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    const uint64_t data_offset = gguf_get_data_offset(gguf_);
    std::vector<uint8_t> staging;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_); i++) {
        const char * name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor * t = ggml_get_tensor(ctx_, name);
        const size_t size = ggml_nbytes(t);
        staging.resize(size);
        if (!seek(f, data_offset + gguf_get_tensor_offset(gguf_, i)) || std::fread(staging.data(), 1, size, f) != size) {
            std::fclose(f);
            throw std::runtime_error(std::string("cannot read tensor ") + name + " of " + path);
        }
        ggml_backend_tensor_set(t, staging.data(), 0, size);
    }
    std::fclose(f);
}

ModelFile::~ModelFile() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
    if (ctx_) ggml_free(ctx_);
    if (gguf_) gguf_free(gguf_);
}

ggml_tensor * ModelFile::optional_tensor(const std::string & name) const {
    return ggml_get_tensor(ctx_, name.c_str());
}

ggml_tensor * ModelFile::tensor(const std::string & name) const {
    ggml_tensor * t = optional_tensor(name);
    if (!t) {
        throw std::runtime_error("tensor " + name + " is missing from " + path_);
    }
    return t;
}

int64_t ModelFile::key_id(const std::string & key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) {
        throw std::runtime_error("key " + key + " is missing from " + path_);
    }
    return id;
}

uint32_t ModelFile::u32(const std::string & key) const {
    return gguf_get_val_u32(gguf_, key_id(key));
}

float ModelFile::f32(const std::string & key) const {
    return gguf_get_val_f32(gguf_, key_id(key));
}

std::string ModelFile::str(const std::string & key) const {
    const int64_t id = key_id(key);
    if (gguf_get_kv_type(gguf_, id) != GGUF_TYPE_STRING) throw std::runtime_error("key " + key + " is not a string");
    return gguf_get_val_str(gguf_, id);
}

std::vector<double> ModelFile::f64_array(const std::string & key) const {
    const int64_t id = key_id(key);
    if (gguf_get_arr_type(gguf_, id) != GGUF_TYPE_FLOAT64) {
        throw std::runtime_error("key " + key + " is not an array of float64");
    }
    const size_t n = gguf_get_arr_n(gguf_, id);
    const double * data = (const double *) gguf_get_arr_data(gguf_, id);
    return std::vector<double>(data, data + n);
}

std::vector<int32_t> ModelFile::i32_array(const std::string & key) const {
    const int64_t id = key_id(key);
    const size_t n = gguf_get_arr_n(gguf_, id);
    const gguf_type type = gguf_get_arr_type(gguf_, id);
    std::vector<int32_t> out(n);
    const void * data = gguf_get_arr_data(gguf_, id);
    for (size_t i = 0; i < n; i++) {
        switch (type) {
            case GGUF_TYPE_INT32: out[i] = ((const int32_t *) data)[i]; break;
            case GGUF_TYPE_UINT32: out[i] = (int32_t) ((const uint32_t *) data)[i]; break;
            case GGUF_TYPE_INT64: out[i] = (int32_t) ((const int64_t *) data)[i]; break;
            case GGUF_TYPE_UINT64: out[i] = (int32_t) ((const uint64_t *) data)[i]; break;
            default: throw std::runtime_error("key " + key + " is not an integer array");
        }
    }
    return out;
}

std::vector<std::string> ModelFile::str_array(const std::string & key) const {
    const int64_t id = key_id(key);
    if (gguf_get_arr_type(gguf_, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("key " + key + " is not a string array");
    }
    const size_t n = gguf_get_arr_n(gguf_, id);
    std::vector<std::string> out(n);
    for (size_t i = 0; i < n; i++) {
        out[i] = gguf_get_arr_str(gguf_, id, i);
    }
    return out;
}
