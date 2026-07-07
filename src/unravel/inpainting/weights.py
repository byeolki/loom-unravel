from __future__ import annotations

from pathlib import Path

from ..cache import fetch

MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"


def lama_checkpoint_path() -> Path:
    return fetch(MODEL_URL, "big-lama.pt", subdir="inpainting")
