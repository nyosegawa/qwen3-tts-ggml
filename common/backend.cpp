#include "backend.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "ggml-cpu.h"

namespace {

/** ggml's own messages, warnings and errors only; its informational lines run to dozens per start. */
void log_warnings(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

}  // namespace

void configure_ggml() {
    ggml_log_set(log_warnings, nullptr);
#ifdef __APPLE__
    // ggml v0.25.3's kernel_mul_mm on the tensor API, which Metal uses by default on M5, M6, A19 and A20,
    // writes past its output when the output has 64 modulo 128 columns and so corrupts the next tensor: a
    // codec window of 64 frames decodes to noise on an M5. ggml reads the variable when it first lists devices.
    setenv("GGML_METAL_TENSOR_DISABLE", "1", 1);
#endif
}

ggml_backend_t init_backend(const std::string & name) {
    configure_ggml();
    if (!name.empty() && name != "gpu" && name != "cpu") {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (!dev) throw std::runtime_error("no device is named " + name);
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) throw std::runtime_error("cannot start the device " + name);
        return backend;
    }
    if (name != "cpu") {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (!backend) {
                    throw std::runtime_error(std::string("cannot start the GPU backend ") + ggml_backend_dev_name(dev));
                }
                return backend;
            }
        }
        if (name == "gpu") {
            throw std::runtime_error("no GPU backend was found");
        }
    }
    return ggml_backend_cpu_init();
}
