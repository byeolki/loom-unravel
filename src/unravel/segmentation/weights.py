from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from ..cache import cache_root

REPO_ID = "skytnt/anime-seg"
FILENAME = "isnetis.onnx"


def onnx_model_path() -> Path:
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            cache_dir=str(cache_root() / "segmentation" / "hf"),
        )
    )
