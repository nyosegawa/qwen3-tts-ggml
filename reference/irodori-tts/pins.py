"""The pinned weights every script in this folder runs, and the commit of the code pyproject.toml pins."""

import importlib.metadata
import json

from huggingface_hub import snapshot_download

_installed = json.loads(importlib.metadata.distribution("irodori-tts").read_text("direct_url.json"))
CODE = {"repository": _installed["url"], "commit": _installed["vcs_info"]["commit_id"]}
MODELS = {
    "mf": {"repository": "Aratako/Irodori-TTS-v4.1-Small-MF", "revision": "ccc78f5d480b6e51b69b2d5042a14c4da04fea6e"},
    "rf": {"repository": "Aratako/Irodori-TTS-v4.1-Small", "revision": "2b28324dc263ed5e6638b3cf3dd94c82ead07b4b"},
}
CODEC = {"repository": "Aratako/Semantic-DACVAE-Japanese-32dim", "revision": "47376ee24834d7a05a48ebabfe3cde29b3c5e214"}


def snapshot(pin):
    """The local folder of a pinned Hugging Face repository, downloaded on first use."""
    return snapshot_download(pin["repository"], revision=pin["revision"])
