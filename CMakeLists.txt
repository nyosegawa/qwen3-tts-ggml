cmake_minimum_required(VERSION 3.20)
project(speech-cpp C CXX)

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

# Source files are UTF-8; MSVC otherwise reads them in the system code page.
if (MSVC)
    add_compile_options(/utf-8)
endif()

add_library(speechcommon
    common/model-file.cpp
    common/backend.cpp
    common/wav.cpp
)
target_include_directories(speechcommon PUBLIC common)
target_link_libraries(speechcommon PUBLIC ggml)

add_library(qwen3tts
    families/qwen3-tts/codec.cpp
    families/qwen3-tts/talker.cpp
    families/qwen3-tts/prompt.cpp
    families/qwen3-tts/sampler.cpp
    families/qwen3-tts/tokenizer.cpp
    families/qwen3-tts/synthesizer.cpp
)
target_include_directories(qwen3tts PUBLIC families/qwen3-tts)
target_link_libraries(qwen3tts PUBLIC speechcommon)

add_library(irodoritts
    families/irodori-tts/graph.cpp
    families/irodori-tts/text-normalizer.cpp
    families/irodori-tts/tokenizer.cpp
    families/irodori-tts/text-encoder.cpp
    families/irodori-tts/loudness.cpp
    families/irodori-tts/codec.cpp
    families/irodori-tts/reference.cpp
    families/irodori-tts/speaker-encoder.cpp
    families/irodori-tts/duration.cpp
    families/irodori-tts/dit.cpp
    families/irodori-tts/sampler.cpp
)
target_include_directories(irodoritts PUBLIC families/irodori-tts)
target_link_libraries(irodoritts PUBLIC speechcommon)

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

add_executable(irodori-text-check tools/irodori-text-check.cpp)
target_link_libraries(irodori-text-check PRIVATE irodoritts)

add_executable(irodori-codec-check tools/irodori-codec-check.cpp)
target_link_libraries(irodori-codec-check PRIVATE irodoritts)

add_executable(irodori-condition-check tools/irodori-condition-check.cpp)
target_link_libraries(irodori-condition-check PRIVATE irodoritts)

add_executable(irodori-dit-check tools/irodori-dit-check.cpp)
target_link_libraries(irodori-dit-check PRIVATE irodoritts)
