#pragma once

#include <string>
#include <vector>

#include "talker.h"

/** The ids a CustomVoice prompt is built from, read from the talker's GGUF. */
struct PromptIds {
    int32_t tts_bos = 0, tts_eos = 0, tts_pad = 0;
    int32_t im_start = 0, im_end = 0, assistant = 0, newline = 198;
    int32_t codec_bos = 0, codec_eos = 0, codec_pad = 0;
    int32_t think = 0, nothink = 0, think_bos = 0, think_eos = 0;
    std::vector<std::string> speaker_names, speaker_dialects, language_names;
    /** The BCP 47 tag of each of language_names, or "" for a dialect, which only its speakers speak. */
    std::vector<std::string> language_tags;
    std::vector<int32_t> speaker_ids, language_ids;

    explicit PromptIds(const ModelFile & m);
    /** The codec id of a named speaker; an unknown name throws. */
    int32_t speaker(const std::string & name) const;
    /** The codec id of a language by the model's name, or -1 for "auto"; an unknown name throws. */
    int32_t language(const std::string & name) const;
    /**
     * The model's name of the language a BCP 47 tag names (`ja`, `ja-JP`), or "auto" for "auto" and ""; a tag
     * of a language the model does not speak throws.
     */
    std::string language_name(const std::string & tag) const;
    /** The dialect a speaker speaks when the language is Chinese or auto, or "" for none. */
    std::string dialect(const std::string & speaker) const;
};

/** What the talker needs to speak one text: the prefill embeddings and the vector added to every frame. */
struct Prompt {
    std::vector<float> embeds;
    int n = 0;
    std::vector<float> frame_extra;
};

/**
 * Builds the CustomVoice prompt of the official generate() in its non-streaming mode, which puts the
 * whole text in the prefill. `text_ids` are the tokens of `<|im_start|>assistant\n{text}<|im_end|>\n
 * <|im_start|>assistant\n`, and `language` is the model's name of a language (`japanese`) or "auto".
 */
Prompt build_prompt(Talker & talker, const PromptIds & ids, const std::vector<int32_t> & text_ids,
                    const std::string & speaker, const std::string & language);
