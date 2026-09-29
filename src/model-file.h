#pragma once

#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

/** A GGUF file whose tensors live in one backend buffer, and whose metadata stays readable. */
class ModelFile {
public:
    ModelFile(const std::string & path, ggml_backend_t backend);
    ~ModelFile();
    ModelFile(const ModelFile &) = delete;
    ModelFile & operator=(const ModelFile &) = delete;

    /** The tensor of that name; a missing one throws, since every caller needs it. */
    ggml_tensor * tensor(const std::string & name) const;
    /** The tensor of that name, or null when the checkpoint does not have it. */
    ggml_tensor * optional_tensor(const std::string & name) const;

    uint32_t u32(const std::string & key) const;
    float f32(const std::string & key) const;
    std::vector<int32_t> i32_array(const std::string & key) const;
    std::vector<std::string> str_array(const std::string & key) const;

    const std::string & path() const { return path_; }

private:
    int64_t key_id(const std::string & key) const;

    std::string path_;
    gguf_context * gguf_ = nullptr;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};
