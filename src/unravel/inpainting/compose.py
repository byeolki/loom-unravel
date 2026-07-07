from __future__ import annotations

import numpy as np

from ..geometry import mask_bbox
from ..parts.depth import DEPTH_ORDER
from .inpainter import LamaInpainter

BBOX_PADDING = 8


def occluders_of(label: str) -> tuple[str, ...]:
    own_depth = DEPTH_ORDER[label]
    return tuple(other for other, depth in DEPTH_ORDER.items() if depth > own_depth)


def apply_inpainting(
    rgb: np.ndarray, masks: dict[str, np.ndarray], inpainter: LamaInpainter
) -> dict[str, np.ndarray]:
    rgb_by_label = {label: rgb for label in masks}

    for label in masks:
        occluder_labels = occluders_of(label)
        if not occluder_labels:
            continue

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
        inpainted_crop = inpainter.inpaint(crop_rgb, crop_hole)

        patched = rgb_by_label[label].copy()
        patched[y : y + h, x : x + w] = np.where(crop_hole[:, :, None], inpainted_crop, crop_rgb)
        rgb_by_label[label] = patched

    return rgb_by_label
