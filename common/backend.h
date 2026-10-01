#pragma once

#include <string>

#include "ggml-backend.h"

/**
 * Sets ggml up for this process: only its warnings and errors reach stderr, and Metal does not run matrix
 * products through Metal 4's tensor API. Call it before touching any device.
 */
void configure_ggml();

/**
 * The backend of the device named `name` as `--devices` lists it (MTL0, Vulkan0, Vulkan1, CPU), "cpu"
 * for the CPU, or the first GPU (Metal, Vulkan or CUDA, whichever was built) when `name` is empty or
 * "gpu". A name no device has, or "gpu" on a machine without one, throws.
 */
ggml_backend_t init_backend(const std::string & name);
