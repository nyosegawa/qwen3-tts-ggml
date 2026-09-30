cmake_minimum_required(VERSION 3.20)
project(qwen3-tts-ggml C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# One executable per tool, with ggml linked in, so that a release is a single file per tool.
option(BUILD_SHARED_LIBS "Build ggml as shared libraries" OFF)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(ggml)

add_library(qwen3tts
    families/qwen3-tts/model-file.cpp
    families/qwen3-tts/backend.cpp
    families/qwen3-tts/codec.cpp
    families/qwen3-tts/talker.cpp
    families/qwen3-tts/prompt.cpp
    families/qwen3-tts/sampler.cpp
    families/qwen3-tts/tokenizer.cpp
    families/qwen3-tts/synthesizer.cpp
)
target_include_directories(qwen3tts PUBLIC families/qwen3-tts)
target_link_libraries(qwen3tts PUBLIC ggml)

add_executable(codec-check tools/codec-check.cpp)
target_link_libraries(codec-check PRIVATE qwen3tts)

add_executable(talker-check tools/talker-check.cpp)
target_link_libraries(talker-check PRIVATE qwen3tts)

add_executable(tokenizer-check tools/tokenizer-check.cpp)
target_link_libraries(tokenizer-check PRIVATE qwen3tts)

add_executable(qwen3-tts tools/qwen3-tts.cpp)
target_link_libraries(qwen3-tts PRIVATE qwen3tts)
if (WIN32)
    target_link_libraries(qwen3-tts PRIVATE shell32)
endif()

add_executable(qwen3-tts-worker tools/qwen3-tts-worker.cpp)
target_link_libraries(qwen3-tts-worker PRIVATE qwen3tts)
