#include "backend.h"

#include <cstdio>
#include <stdexcept>

#include "ggml-cpu.h"

namespace {

/** ggml's own messages, warnings and errors only; its informational lines run to dozens per start. */
void log_warnings(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

}  // namespace

void quiet_ggml_logs() {
    ggml_log_set(log_warnings, nullptr);
}

ggml_backend_t init_backend(const std::string & name) {
    quiet_ggml_logs();
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
