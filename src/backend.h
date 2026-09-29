#pragma once

#include <string>

#include "ggml-backend.h"

/** Lets only ggml's warnings and errors through to stderr; call it before touching any device. */
void quiet_ggml_logs();

/** The first GPU backend (Metal, Vulkan or CUDA, whichever was built), or the CPU when `name` is "cpu" or no GPU exists. */
ggml_backend_t init_backend(const std::string & name);
