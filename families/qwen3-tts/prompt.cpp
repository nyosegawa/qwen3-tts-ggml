#include "prompt.h"

#include <algorithm>
#include <stdexcept>

#include "language.h"

PromptIds::PromptIds(const ModelFile & m) {
    tts_bos = (int32_t) m.u32("text.tts_bos_token_id");
    tts_eos = (int32_t) m.u32("text.tts_eos_token_id");
    tts_pad = (int32_t) m.u32("text.tts_pad_token_id");
    im_start = (int32_t) m.u32("text.im_start_token_id");
    im_end = (int32_t) m.u32("text.im_end_token_id");
    assistant = (int32_t) m.u32("text.assistant_token_id");
    codec_bos = (int32_t) m.u32("talker.codec_bos_id");
    codec_eos = (int32_t) m.u32("talker.codec_eos_token_id");
    codec_pad = (int32_t) m.u32("talker.codec_pad_id");
    think = (int32_t) m.u32("talker.codec_think_id");
    nothink = (int32_t) m.u32("talker.codec_nothink_id");
    think_bos = (int32_t) m.u32("talker.codec_think_bos_id");
    think_eos = (int32_t) m.u32("talker.codec_think_eos_id");
    speaker_names = m.str_array("talker.speaker_names");
    speaker_ids = m.i32_array("talker.speaker_ids");
    speaker_dialects = m.str_array("talker.speaker_dialects");
    language_names = m.str_array("talker.language_names");
    language_ids = m.i32_array("talker.language_ids");
    language_tags = m.str_array("talker.language_tags");
}

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

int32_t PromptIds::speaker(const std::string & name) const {
    const auto it = std::find(speaker_names.begin(), speaker_names.end(), lower(name));
    if (it == speaker_names.end()) throw std::runtime_error("unknown speaker: " + name);
    return speaker_ids[it - speaker_names.begin()];
}

int32_t PromptIds::language(const std::string & name) const {
    if (lower(name) == "auto") return -1;
    const auto it = std::find(language_names.begin(), language_names.end(), lower(name));
    if (it == language_names.end()) throw std::runtime_error("unknown language: " + name);
    return language_ids[it - language_names.begin()];
}

std::string PromptIds::language_name(const std::string & tag) const {
    if (tag.empty() || lower(tag) == "auto") return "auto";
    std::vector<std::string> spoken;
    for (size_t i = 0; i < language_tags.size(); i++) {
        if (language_tags[i].empty()) continue;
        if (bcp47_matches(tag, language_tags[i])) return language_names[i];
        spoken.push_back(language_tags[i]);
    }
    std::sort(spoken.begin(), spoken.end());
    std::string list;
    for (const std::string & s : spoken) list += (list.empty() ? "" : ", ") + s;
    throw std::runtime_error("Qwen3-TTS speaks " + list + ", not " + tag);
}

std::string PromptIds::dialect(const std::string & name) const {
    const auto it = std::find(speaker_names.begin(), speaker_names.end(), lower(name));
    return it == speaker_names.end() ? "" : speaker_dialects[it - speaker_names.begin()];
}

Prompt build_prompt(Talker & talker, const PromptIds & ids, const std::vector<int32_t> & text_ids,
                    const std::string & speaker, const std::string & language) {
    const size_t n_text_ids = text_ids.size();
    // The template's 3 leading tokens are the role, and its 5 trailing ones close it and open the answer.
    if (n_text_ids < 9) throw std::runtime_error("the text is empty");
    const int h = talker.hidden();

    int32_t language_id = ids.language(language);
    const std::string lang = lower(language);
    const std::string dialect = ids.dialect(speaker);
    if ((lang == "chinese" || lang == "auto") && !dialect.empty()) language_id = ids.language(dialect);

    std::vector<int32_t> codec_ids;
    if (language_id < 0) codec_ids = {ids.nothink, ids.think_bos, ids.think_eos};
    else codec_ids = {ids.think, ids.think_bos, language_id, ids.think_eos};
    codec_ids.push_back(ids.speaker(speaker));
    codec_ids.push_back(ids.codec_pad);
    codec_ids.push_back(ids.codec_bos);
    const size_t n_codec = codec_ids.size();

    // Text rows: role (3), tts_pad, tts_bos, the text tokens, tts_eos.
    std::vector<int32_t> text_rows(text_ids.begin(), text_ids.begin() + 3);
    text_rows.push_back(ids.tts_pad);
    text_rows.push_back(ids.tts_bos);
    text_rows.insert(text_rows.end(), text_ids.begin() + 3, text_ids.end() - 5);
    text_rows.push_back(ids.tts_eos);
    const std::vector<float> te = talker.text_embeddings(text_rows);
    const std::vector<float> ce = talker.codec_embeddings(codec_ids);
    const float * pad = &te[3 * h];
    const float * bos = &te[4 * h];
    const float * body = &te[5 * h];
    const size_t n_body = n_text_ids - 8;
    const float * eos = &te[(5 + n_body) * h];
    const float * codec_pad = &ce[(n_codec - 2) * h];
    const float * codec_bos = &ce[(n_codec - 1) * h];

    Prompt p;
    auto row = [&](const float * a, const float * b) {
        for (int i = 0; i < h; i++) p.embeds.push_back(a[i] + (b ? b[i] : 0.0f));
        p.n++;
    };
    for (int r = 0; r < 3; r++) row(&te[r * h], nullptr);
    // tts_pad over the think tags, language and speaker, then tts_bos over codec_pad.
    for (size_t c = 0; c + 1 < n_codec; c++) row(c + 2 < n_codec ? pad : bos, &ce[c * h]);
    for (size_t t = 0; t < n_body; t++) row(body + t * h, codec_pad);
    row(eos, codec_pad);
    row(pad, codec_bos);
    p.frame_extra.assign(pad, pad + h);
    return p;
}
