from __future__ import annotations

import numpy as np

from ..geometry import mask_bbox
from .inpainter import DiffusionInpainter

OCCLUDED_BY: dict[str, tuple[str, ...]] = {
    "face": ("eyebrow_l", "eyebrow_r", "eye_l", "eye_r", "mouth", "hair_front"),
}

PART_PROMPTS: dict[str, str] = {
    "face": "smooth anime character skin, plain face, no eyes, no eyebrows, no mouth",
}

BBOX_PADDING = 8


def apply_inpainting(
    rgb: np.ndarray, masks: dict[str, np.ndarray], inpainter: DiffusionInpainter
) -> dict[str, np.ndarray]:
    rgb_by_label = {label: rgb for label in masks}

    for label, occluder_labels in OCCLUDED_BY.items():
        base_mask = masks[label]
        occluders = np.zeros_like(base_mask)
        for occluder_label in occluder_labels:
            occluders |= masks[occluder_label]
        hole = base_mask & occluders
        if not hole.any():
            continue

        bbox = mask_bbox(base_mask, padding=BBOX_PADDING)
        if bbox is None:
            continue
        x, y, w, h = bbox

        crop_rgb = rgb[y : y + h, x : x + w]
        crop_hole = hole[y : y + h, x : x + w]
        inpainted_crop = inpainter.inpaint(crop_rgb, crop_hole, PART_PROMPTS[label])

        patched = rgb.copy()
        patched[y : y + h, x : x + w] = np.where(crop_hole[:, :, None], inpainted_crop, crop_rgb)
        rgb_by_label[label] = patched

    return rgb_by_label
