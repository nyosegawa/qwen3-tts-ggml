cmake_minimum_required(VERSION 3.20)
project(qwen3-tts-ggml C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(ggml)

add_library(qwen3tts
    src/model-file.cpp
    src/backend.cpp
    src/codec.cpp
    src/talker.cpp
    src/prompt.cpp
    src/sampler.cpp
    src/tokenizer.cpp
    src/synthesizer.cpp
)
target_include_directories(qwen3tts PUBLIC src)
target_link_libraries(qwen3tts PUBLIC ggml)

add_executable(codec-check tools/codec-check.cpp)
target_link_libraries(codec-check PRIVATE qwen3tts)

add_executable(talker-check tools/talker-check.cpp)
target_link_libraries(talker-check PRIVATE qwen3tts)

add_executable(tokenizer-check tools/tokenizer-check.cpp)
target_link_libraries(tokenizer-check PRIVATE qwen3tts)

add_executable(qwen3-tts tools/qwen3-tts.cpp)
target_link_libraries(qwen3-tts PRIVATE qwen3tts)
