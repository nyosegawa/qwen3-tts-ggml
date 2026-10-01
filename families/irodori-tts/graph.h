#pragma once

#include <cstdint>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

/**
 * A matrix product that asks for float32 accumulation. Vulkan otherwise accumulates in half precision on
 * GPUs that have it; the CPU and Metal ignore the request (Metal rounds the inputs of its matrix kernel to
 * half precision either way).
 */
inline ggml_tensor * mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * t = ggml_mul_mat(ctx, a, b);
    ggml_prec_set_acc(t, GGML_PREC_F32);
    return t;
}

/**
 * One computation: a ggml context and graph whose inputs carry their data from the moment they are made,
 * so that the code that builds a stage also says what goes into it. The data is uploaded when the graph
 * is computed, after its tensors have been allocated.
 */
class Graph {
public:
    explicit Graph(int max_nodes = 32768);
    ~Graph();
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;

    ggml_context * ctx() const { return ctx_; }

    ggml_tensor * input(const std::vector<float> & data, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1);
    ggml_tensor * input(const std::vector<int32_t> & data, int64_t ne0);
    /** A float32 input of zeros, for padding with a concatenation where a backend pads only on the right. */
    ggml_tensor * zeros(int64_t ne0, int64_t ne1);

    /** Marks `t` as a result to read back after compute(). */
    void output(ggml_tensor * t);

    /** Allocates the graph with `allocr`, uploads the inputs and computes it on `backend`. */
    void compute(ggml_backend_t backend, ggml_gallocr_t allocr);

    /** A result, as float32 in ggml order (ne0 fastest). */
    static std::vector<float> read(const ggml_tensor * t);

private:
    struct Upload {
        ggml_tensor * tensor;
        std::vector<uint8_t> bytes;
    };

    ggml_context * ctx_ = nullptr;
    ggml_cgraph * gf_ = nullptr;
    std::vector<Upload> uploads_;
};
