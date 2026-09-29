"""Writes tokenizer cases for tools/tokenizer-check: one line per text, the text's UTF-8 in hex, a tab,
and the ids the model's own tokenizer.json gives.

usage: uv run python tokenizer_cases.py <model dir> <out.tsv>
"""

import sys
import unicodedata

from transformers import AutoTokenizer

model_dir, out_path = sys.argv[1:3]
# Loaded without fix_mistral_regex, so the pre-tokenizer is the one in tokenizer.json.
tok = AutoTokenizer.from_pretrained(model_dir)

texts = [
    "明日の東京は晴れで、最高気温は二十四度の予報です。",
    "ドル円は今、百四十八円二十銭前後で、昨日より少し円安に動いています。",
    "GitHubのプルリクエストのCIが落ちているか確認して",
    "ASISTの設定でVOICEVOXの話者を変えたい",
    "It's 3:45 PM, and I'll be there in 10 minutes—don't worry!",
    "WE'RE HERE. They've gone. I'd say it's fine; she'll see.",
    "https://asist-agent.com/en/privacy/ and C:\\Users\\name\\file.txt",
    "  leading spaces,   inner   spaces, and trailing spaces   ",
    "line one\nline two\r\n\nline four\n",
    "tabs\tand\t\tmore\t",
    "数字123と４５６、1,234.56円、2026年9月29日",
    "絵文字😀👍🏽と記号★☆※、「かぎかっこ」『二重』",
    "Ünïcödé café naïve résumé — Ελληνικά Русский 한국어 中文",
    "e\u0301 combining and ｶﾀｶﾅ half-width",
    "a",
    " ",
    "\n",
    "!!!???...",
    "x'y 'tis O'Neil ' spaced ' quote",
]
with open(out_path, "w") as f:
    for t in texts:
        t = unicodedata.normalize("NFC", t)
        ids = tok(t, add_special_tokens=False)["input_ids"]
        f.write(t.encode("utf-8").hex() + "\t" + " ".join(map(str, ids)) + "\n")
print(len(texts), "cases ->", out_path)
