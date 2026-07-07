from __future__ import annotations

from pathlib import Path

from ..cache import fetch

CASCADE_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
CHECKPOINT_URL = "https://drive.google.com/uc?export=download&id=1NckKw7elDjQTllRxttO87WY7cnQwdMqz"


def cascade_path() -> Path:
    return fetch(CASCADE_URL, "lbpcascade_animeface.xml", subdir="landmarks")


def checkpoint_path() -> Path:
    return fetch(CHECKPOINT_URL, "checkpoint_landmark_191116.pth.tar", subdir="landmarks")
