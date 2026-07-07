from __future__ import annotations

import os
import urllib.request
from pathlib import Path


def cache_root() -> Path:
    override = os.environ.get("UNRAVEL_CACHE_DIR")
    root = Path(override) if override else Path.home() / ".cache" / "unravel"
    root.mkdir(parents=True, exist_ok=True)
    return root


def fetch(url: str, filename: str, subdir: str) -> Path:
    target_dir = cache_root() / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    if not target_path.exists():
        tmp_path = target_path.with_suffix(target_path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.rename(target_path)
    return target_path
