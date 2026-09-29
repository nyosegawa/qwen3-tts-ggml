#pragma once

#include <string>

#include "ggml-backend.h"

/** The first GPU backend (Metal, Vulkan or CUDA, whichever was built), or the CPU when `name` is "cpu" or no GPU exists. */
ggml_backend_t init_backend(const std::string & name);
