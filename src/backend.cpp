#include "backend.h"

#include <stdexcept>

#include "ggml-cpu.h"

ggml_backend_t init_backend(const std::string & name) {
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
